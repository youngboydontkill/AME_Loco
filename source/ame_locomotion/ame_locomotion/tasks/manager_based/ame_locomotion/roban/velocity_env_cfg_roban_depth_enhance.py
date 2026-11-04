"""RobanS17 migration of AME-Kuavo-S54-Depth-Enhance."""

import copy
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import RayCasterCfg, TiledCameraCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from ame_locomotion.tasks.manager_based.ame_locomotion import mdp
from ame_locomotion.tasks.manager_based.ame_locomotion.s54.velocity_env_cfg_s54_depth import (
    DepthObservationsCfg,
    KuavoS54DepthSceneCfg,
)
from ame_locomotion.tasks.manager_based.ame_locomotion.s54.velocity_env_cfg_s54_depth_enhance import (
    CNNMLPDepthRLEventCfg,
    EnhancedDepthObservationsCfg,
    MixedEnhancedDepthObservationsCfg,
    KuavoS54EnhanceEnvCfg,
    KuavoS54DepthEnhanceEnvCfg_PLAY,
)
from kuavo.assets.roban_s17 import ROBAN_S17_CFG
from .rough_terrain_cfg import ROBAN_ROUGH_TERRAINS_CFG


# The all-zero S17 pose from the generic asset config leaves the feet below the
# ground at a 0.8 m root height. Use the standing pose shipped by LejuLab for
# this locomotion task so reset states start from a physically valid stance.
_ROBAN_STANDING_INIT = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.77),
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
        "waist_yaw_joint": 0.0,
        "leg_l1_joint": -0.05033,
        "leg_l2_joint": -0.0164,
        "leg_l3_joint": -0.0233,
        "leg_l4_joint": 0.155,
        "leg_l5_joint": -0.1,
        "leg_l6_joint": 0.0,
        "leg_r1_joint": 0.05033,
        "leg_r2_joint": 0.0164,
        "leg_r3_joint": -0.0233,
        "leg_r4_joint": 0.155,
        "leg_r5_joint": -0.1,
        "leg_r6_joint": 0.0,
        "zarm_l1_joint": 0.2,
        "zarm_l2_joint": 0.1,
        "zarm_l3_joint": -0.2,
        "zarm_l4_joint": -0.6,
        "zarm_r1_joint": 0.2,
        "zarm_r2_joint": -0.1,
        "zarm_r3_joint": 0.2,
        "zarm_r4_joint": -0.6,
    },
    joint_vel={".*": 0.0},
)


def _roban_robot_cfg() -> ArticulationCfg:
    return ROBAN_S17_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=_ROBAN_STANDING_INIT,
    )


# The S17 URDF already pitches ``camera_base`` down by 0.27152 rad.  Apply the
# remaining fixed pitch so the optical axis is about 40 degrees below level.
_ROBAN_CAMERA_BASE_PITCH = 0.27152
_ROBAN_CAMERA_PITCH = math.radians(40.0) - _ROBAN_CAMERA_BASE_PITCH
_ROBAN_CAMERA_PITCH_QUAT = (
    math.cos(_ROBAN_CAMERA_PITCH / 2.0),
    0.0,
    math.sin(_ROBAN_CAMERA_PITCH / 2.0),
    0.0,
)
_ROBAN_CAMERA_ROS_QUAT = (0.5, -0.5, 0.5, -0.5)
# Quaternion product q_pitch * q_ros, in Isaac's (w, x, y, z) convention.
_ROBAN_CAMERA_ROT = (
    _ROBAN_CAMERA_PITCH_QUAT[0] * _ROBAN_CAMERA_ROS_QUAT[0]
    - _ROBAN_CAMERA_PITCH_QUAT[2] * _ROBAN_CAMERA_ROS_QUAT[2],
    _ROBAN_CAMERA_PITCH_QUAT[0] * _ROBAN_CAMERA_ROS_QUAT[1]
    + _ROBAN_CAMERA_PITCH_QUAT[2] * _ROBAN_CAMERA_ROS_QUAT[3],
    _ROBAN_CAMERA_PITCH_QUAT[0] * _ROBAN_CAMERA_ROS_QUAT[2]
    + _ROBAN_CAMERA_PITCH_QUAT[2] * _ROBAN_CAMERA_ROS_QUAT[0],
    _ROBAN_CAMERA_PITCH_QUAT[0] * _ROBAN_CAMERA_ROS_QUAT[3]
    - _ROBAN_CAMERA_PITCH_QUAT[2] * _ROBAN_CAMERA_ROS_QUAT[1],
)


@configclass
class RobanDepthSceneCfg(KuavoS54DepthSceneCfg):
    """S17 scene with the physical head camera as the depth sensor."""

    robot: ArticulationCfg = _roban_robot_cfg()
    Feet_L_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/leg_l6_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.2, 0.05]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    Feet_R_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/leg_r6_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.05, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.2, 0.05]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    depth_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera_base/depth_camera",
        update_period=0.02,
        update_latest_camera_pose=False,
        height=42,
        width=42,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.5335,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
        # The optical axis is fixed at approximately 40 degrees below level.
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=_ROBAN_CAMERA_ROT,
            convention="ros",
        ),
    )


def _apply_roban_name_mapping(cfg) -> None:
    """Apply S17 body, joint, action, and contact selectors."""

    cfg.scene.robot = _roban_robot_cfg()
    if cfg.scene.height_scanner is not None:
        cfg.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/waist_yaw_link"

    for event_name in ("add_base_mass", "base_com", "base_external_force_torque"):
        event = getattr(cfg.events, event_name, None)
        if event is not None:
            event.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base_link")

    cfg.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=["leg_[lr][1-5]_link", "base_link", "zarm_.*_link"]
    )
    foot_body = "leg_[lr]6_link"
    for reward_name in ("feet_air_time", "feet_air_time_variance", "feet_stumble"):
        reward = getattr(cfg.rewards, reward_name, None)
        if reward is not None:
            reward.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=foot_body)
    for reward_name in ("feet_slide", "feet_too_near"):
        reward = getattr(cfg.rewards, reward_name, None)
        if reward is not None:
            reward.params["asset_cfg"] = SceneEntityCfg("robot", body_names=foot_body)
            if reward_name == "feet_slide":
                reward.params["sensor_cfg"] = SceneEntityCfg("contact_forces", body_names=foot_body)

    if getattr(cfg.rewards, "joint_coordination", None) is not None:
        cfg.rewards.joint_coordination.params["coord_joints"] = [
            ["leg_l3_joint", "zarm_r1_joint"],
            ["leg_r3_joint", "zarm_l1_joint"],
        ]
    if hasattr(cfg.rewards, "joint_deviation_hip"):
        cfg.rewards.joint_deviation_hip.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["leg_[lr][2,3]_joint"]
        )
    if hasattr(cfg.rewards, "joint_deviation_arms"):
        cfg.rewards.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["zarm_[lr][1-4]_joint"]
        )
    if hasattr(cfg.rewards, "joint_deviation_arm_shoulder_roll"):
        cfg.rewards.joint_deviation_arm_shoulder_roll.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["zarm_[lr]2_joint"]
        )
    if hasattr(cfg.rewards, "joint_deviation_waists"):
        cfg.rewards.joint_deviation_waists.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["waist_yaw_joint"]
        )

    # LejuLab's RobanS17 task uses orientation failure as the fall detector and
    # has no base-contact termination.  The inherited Kuavo base-contact term
    # fires during S17 initialization because fixed head/body collision links
    # can report a transient force before the first settled physics frame.
    cfg.terminations.base_contact = None
    cfg.terminations.bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.8, "asset_cfg": SceneEntityCfg("robot")},
    )
    if getattr(cfg.scene, "visualize_cam", None) is not None:
        cfg.scene.visualize_cam.prim_path = "{ENV_REGEX_NS}/Robot/waist_yaw_link/visualize_cam"

    joint_order = ROBAN_S17_CFG.preserve_joint_order
    cfg.actions.joint_pos.joint_names = joint_order.joint_names
    cfg.actions.joint_pos.scale = 0.25
    # Match LejuLab's RobanS17 action mapping explicitly.  The current URDF
    # happens to use this same natural order, but leaving this implicit would
    # let a future URDF/importer order change silently remap policy outputs.
    cfg.actions.joint_pos.preserve_order = True
    # Restore the RobanS17 proprioceptive scales/noise ranges from LejuLab;
    # the inherited G1 defaults scale angular and joint velocities differently.
    policy = cfg.observations.policy
    policy.base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    policy.projected_gravity = ObsTerm(
        func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
    )
    policy.velocity_commands = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "base_velocity"}
    )
    policy.joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        noise=Unoise(n_min=-0.05, n_max=0.05),
        params={"asset_cfg": joint_order},
    )
    policy.joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        noise=Unoise(n_min=-1.5, n_max=1.5),
        params={"asset_cfg": joint_order},
    )
    policy.actions = ObsTerm(func=mdp.last_action)
    critic = cfg.observations.critic
    critic.base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    critic.base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    critic.projected_gravity = ObsTerm(func=mdp.projected_gravity)
    critic.velocity_commands = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "base_velocity"}
    )
    critic.joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": joint_order})
    critic.joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": joint_order})
    critic.actions = ObsTerm(func=mdp.last_action)
    for group_name in ("policy", "critic"):
        group = getattr(cfg.observations, group_name, None)
        if group is not None:
            for term_name in ("joint_pos", "joint_pos_rel", "joint_vel", "joint_vel_rel"):
                term = getattr(group, term_name, None)
                if term is not None:
                    term.params["asset_cfg"] = joint_order


@configclass
class RobanS17RewardsCfg:
    """Reward terms aligned with LejuLab-Train's RobanS17 rough task."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    dof_power_l2 = RewTerm(func=mdp.joint_power_l2, weight=-5.0e-6)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-3.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[lr][1-6]_joint", "zarm_.*_joint"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["leg_[lr][1-5]_link", "base_link", "zarm_.*_link"]
            ),
            "threshold": 1.0,
        },
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_clip,
        weight=10.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[lr]6_link"),
            "threshold_min": 0.2,
            "threshold_max": 0.5,
            "command_threshold": 0.1,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[lr]6_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[lr]6_link"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.feet_clearance_reward,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[lr]6_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[lr]6_link"),
            "target_height": 0.13,
            "std": 0.05,
        },
    )
    feet_descent_velocity = RewTerm(
        func=mdp.feet_descent_velocity,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[lr]6_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names="leg_[lr]6_link"),
            "height_threshold": 0.10,
        },
    )
    alternating_contacts = RewTerm(
        func=mdp.alternating_contacts,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[lr]6_link"),
            "max_stance_time": 0.4,
            "command_name": "base_velocity",
            "command_threshold": 0.1,
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[lr][2,3]_joint"])},
    )
    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["leg_[lr]3_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["zarm_.*"])},
    )
    joint_deviation_arm_shouler_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["zarm_[lr]2_joint"])},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    contact_force = RewTerm(
        func=mdp.contact_forces,
        weight=-0.001,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[lr]6_link"),
            "threshold": 270.0,
            "violation_max": 270.0,
        },
    )
    stand_still_without_cmd = RewTerm(
        func=mdp.stand_still_without_cmd_nominal,
        weight=-0.2,
        params={"command_name": "base_velocity"},
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={"target_height": 0.8, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class RobanS17EventCfg(CNNMLPDepthRLEventCfg):
    """RobanS17 dynamics randomization aligned with LejuLab-Train."""

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": ROBAN_S17_CFG.preserve_joint_order,
            "pos_distribution_params": (-0.1, 0.1),
            "operation": "add",
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    scale_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["leg_.*_link", "zarm_.*_link"]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    scale_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    scale_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint"),
            "friction_distribution_params": (0.8, 1.2),
            "armature_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.7, 0.7), "y": (-0.7, 0.7), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (0.0, 0.1),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.3, 0.3),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.1, 0.1), "velocity_range": (0.0, 0.0)},
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque_stochastic,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": {"x": (-1000.0, 1000.0), "y": (-1000.0, 1000.0), "z": (-600.0, 600.0)},
            "torque_range": {"x": (-20.0, 20.0), "y": (-20.0, 20.0), "z": (-15.0, 15.0)},
            "probability": 0.002,
        },
    )
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 2.0),
            "dynamic_friction_range": (0.3, 2.0),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    push_robot = None


@configclass
class RobanDepthEnhanceEnvCfg(KuavoS54EnhanceEnvCfg):
    """Training configuration for AME-Roban-Depth-Enhance."""

    scene: RobanDepthSceneCfg = RobanDepthSceneCfg(num_envs=2048, env_spacing=2.5)
    rewards: RobanS17RewardsCfg = RobanS17RewardsCfg()
    events: RobanS17EventCfg = RobanS17EventCfg()

    def __post_init__(self):
        super().__post_init__()
        if "Raw" in type(self).__name__:
            self.observations = RobanDepthRawObservationsCfg()
        else:
            self.observations = RobanMixedDepthObservationsCfg()
        _apply_roban_name_mapping(self)
        self.scene.terrain.terrain_generator = copy.deepcopy(ROBAN_ROUGH_TERRAINS_CFG)
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.5,
        )
        self.scene.height_scanner = None
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.contact_forces.update_period = self.sim.dt
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.episode_length_s = 20.0
        # Keep the Roban command space simple: forward speed and yaw rate only.
        # Disable heading mode so the sampled yaw-rate command is not replaced
        # by a random heading controller target.
        command = self.commands.base_velocity
        command.resampling_time_range = (5.0, 10.0)
        command.rel_standing_envs = 0.1
        command.heading_command = False
        command.rel_heading_envs = 0.0
        command.ranges.lin_vel_x = (0.3, 1.0)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (-0.3, 0.3)
        command.ranges.heading = (0.0, 0.0)


@configclass
class RobanDepthEnhanceEnvCfg_PLAY(KuavoS54DepthEnhanceEnvCfg_PLAY):
    """Playback configuration for AME-Roban-Depth-Enhance."""

    scene: RobanDepthSceneCfg = RobanDepthSceneCfg(num_envs=50, env_spacing=2.5)
    rewards: RobanS17RewardsCfg = RobanS17RewardsCfg()
    events: RobanS17EventCfg = RobanS17EventCfg()

    def __post_init__(self):
        super().__post_init__()
        if "Raw" in type(self).__name__:
            self.observations = RobanDepthRawObservationsCfg()
        else:
            self.observations = RobanEnhancedDepthObservationsCfg()
        _apply_roban_name_mapping(self)
        self.scene.terrain.terrain_generator = copy.deepcopy(ROBAN_ROUGH_TERRAINS_CFG)
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.height_scanner = None
        # Match training evaluation starts with the level-zero terrain.  The
        # generic Kuavo play config leaves this as ``None``, which lets a
        # single play environment spawn on an arbitrary curriculum row and
        # can make a partially trained checkpoint fall immediately.
        self.scene.terrain.max_init_terrain_level = 0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
        self.events.physics_material = None
        self.events.add_joint_default_pos.params["pos_distribution_params"] = (0.0, 0.0)
        self.events.add_base_mass = None
        self.events.scale_link_mass = None
        self.events.base_com = None
        self.events.scale_actuator_gains = None
        self.events.scale_joint_parameters = None
        self.events.reset_robot_joints = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.randomize_depth_camera = None
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.episode_length_s = 1.0e9
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        command = self.commands.base_velocity
        command.heading_command = False
        command.rel_heading_envs = 0.0
        command.ranges.lin_vel_x = (1.0, 1.0)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        command.ranges.heading = (0.0, 0.0)


@configclass
class RobanDepthCriticCfg(DepthObservationsCfg.CriticCfg):
    """LejuLab privileged state terms plus the AME depth image."""

    joint_torques = ObsTerm(
        func=mdp.joint_torques,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint")},
    )
    joint_accs = ObsTerm(
        func=mdp.joint_accs,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint")},
    )
    feet_lin_vel = ObsTerm(func=mdp.feet_lin_vel, params={"asset_cfg": SceneEntityCfg("robot", body_names="leg_[lr]6_link")})
    feet_contact_force = ObsTerm(
        func=mdp.feet_contact_force,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[lr]6_link")},
    )
    base_mass_rel = ObsTerm(
        func=mdp.rigid_body_masses,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")},
    )
    rigid_body_material = ObsTerm(
        func=mdp.rigid_body_material,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="leg_[lr]6_link")},
    )
    base_com = ObsTerm(func=mdp.base_com, params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")})
    action_delay = ObsTerm(func=mdp.action_delay, params={"actuators_names": "motor"})
    push_force = ObsTerm(func=mdp.push_force, params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")})
    push_torque = ObsTerm(func=mdp.push_torque, params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link")})
    feet_heights = ObsTerm(
        func=mdp.feet_heights_bipeds,
        params={"sensor_cfg1": SceneEntityCfg("Feet_L_scanner"), "sensor_cfg2": SceneEntityCfg("Feet_R_scanner")},
    )
    feet_air_times = ObsTerm(
        func=mdp.feet_air_time_obs,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[lr]6_link")},
    )


@configclass
class RobanDepthRawObservationsCfg(DepthObservationsCfg):
    """Raw depth policy with LejuLab's privileged critic features."""

    @configclass
    class CriticCfg(RobanDepthCriticCfg):
        pass

    critic: CriticCfg = CriticCfg()


@configclass
class RobanMixedDepthObservationsCfg(MixedEnhancedDepthObservationsCfg):
    """Mixed enhanced/raw policy depth with the aligned critic features."""

    @configclass
    class CriticCfg(RobanDepthCriticCfg):
        pass

    critic: CriticCfg = CriticCfg()


@configclass
class RobanEnhancedDepthObservationsCfg(EnhancedDepthObservationsCfg):
    """Enhanced play depth with the aligned critic features."""

    @configclass
    class CriticCfg(RobanDepthCriticCfg):
        pass

    critic: CriticCfg = CriticCfg()


@configclass
class RobanDepthRawEnvCfg(RobanDepthEnhanceEnvCfg):
    """Training configuration for RobanS17 with unenhanced depth input."""

    observations: RobanDepthRawObservationsCfg = RobanDepthRawObservationsCfg()


@configclass
class RobanDepthRawEnvCfg_PLAY(RobanDepthEnhanceEnvCfg_PLAY):
    """Playback configuration for RobanS17 with unenhanced depth input."""

    observations: RobanDepthRawObservationsCfg = RobanDepthRawObservationsCfg()
