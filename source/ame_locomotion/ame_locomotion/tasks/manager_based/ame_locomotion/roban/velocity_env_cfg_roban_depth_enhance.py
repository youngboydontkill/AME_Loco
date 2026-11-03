"""RobanS17 migration of AME-Kuavo-S54-Depth-Enhance."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from ame_locomotion.tasks.manager_based.ame_locomotion import mdp
from ame_locomotion.tasks.manager_based.ame_locomotion.s54.velocity_env_cfg_s54_depth import (
    DepthObservationsCfg,
    KuavoS54DepthSceneCfg,
)
from ame_locomotion.tasks.manager_based.ame_locomotion.s54.velocity_env_cfg_s54_depth_enhance import (
    KuavoS54EnhanceEnvCfg,
    KuavoS54DepthEnhanceEnvCfg_PLAY,
    KuavoS54EnhanceRewardsCfg,
)
from kuavo.assets.roban_s17 import ROBAN_S17_CFG


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
            event.params["asset_cfg"] = SceneEntityCfg("robot", body_names="waist_yaw_link")

    cfg.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names="(?!leg_[lr][56]_link$).*"
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
            "robot", joint_names=["leg_[lr][12]_joint"]
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
    for group_name in ("policy", "critic"):
        group = getattr(cfg.observations, group_name, None)
        if group is not None:
            for term_name in ("joint_pos", "joint_pos_rel", "joint_vel", "joint_vel_rel"):
                term = getattr(group, term_name, None)
                if term is not None:
                    term.params["asset_cfg"] = joint_order


@configclass
class RobanS17RewardsCfg(KuavoS54EnhanceRewardsCfg):
    """AME rewards adjusted for the S17 arm mechanics and gait conventions."""

    # Keep the whole-arm regularizer at the RobanS17 reference weight.
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["zarm_[lr][1-4]_joint"])},
    )

    # The shoulder-roll axes are especially prone to a visually large unilateral
    # lift, so match LejuLab's dedicated S17 penalty.
    joint_deviation_arm_shoulder_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["zarm_[lr]2_joint"])},
    )

    # Roban's walking task uses explicit foot-phase shaping to make forward
    # commands produce alternating steps instead of a stationary posture.
    alternating_contacts = RewTerm(
        func=mdp.alternating_contacts,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="leg_[lr]6_link"),
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "max_stance_time": 0.4,
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
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-5.0,
        params={"target_height": 0.77, "asset_cfg": SceneEntityCfg("robot")},
    )
    stand_still_without_cmd = RewTerm(
        func=mdp.stand_still_without_cmd,
        weight=-0.2,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["leg_[lr][1-6]_joint", "zarm_[lr][1-4]_joint"]
            ),
        },
    )


@configclass
class RobanDepthEnhanceEnvCfg(KuavoS54EnhanceEnvCfg):
    """Training configuration for AME-Roban-Depth-Enhance."""

    scene: RobanDepthSceneCfg = RobanDepthSceneCfg(num_envs=2048, env_spacing=2.5)
    rewards: RobanS17RewardsCfg = RobanS17RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # LejuLab does not use the G1 cross-body coordination term. On S17 it
        # can pull a shoulder away from its standing pose to match a leg joint.
        self.rewards.joint_coordination = None
        _apply_roban_name_mapping(self)
        self.scene.height_scanner = None
        self.scene.terrain.max_init_terrain_level = 0
        self.events.push_robot = None
        # Keep the Roban command space simple: forward speed and yaw rate only.
        # Disable heading mode so the sampled yaw-rate command is not replaced
        # by a random heading controller target.
        command = self.commands.base_velocity
        command.heading_command = False
        command.rel_heading_envs = 0.0
        command.ranges.lin_vel_x = (0.3, 1.0)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (-0.3, 0.3)
        command.ranges.heading = (0.0, 0.0)
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time_variance.weight = -0.7
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5


@configclass
class RobanDepthEnhanceEnvCfg_PLAY(KuavoS54DepthEnhanceEnvCfg_PLAY):
    """Playback configuration for AME-Roban-Depth-Enhance."""

    scene: RobanDepthSceneCfg = RobanDepthSceneCfg(num_envs=50, env_spacing=2.5)
    rewards: RobanS17RewardsCfg = RobanS17RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.joint_coordination = None
        _apply_roban_name_mapping(self)
        self.scene.height_scanner = None
        command = self.commands.base_velocity
        command.heading_command = False
        command.rel_heading_envs = 0.0
        command.ranges.lin_vel_x = (1.0, 1.0)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        command.ranges.heading = (0.0, 0.0)


@configclass
class RobanRawDepthObservationsCfg(DepthObservationsCfg):
    """Raw camera depth observations with the same 42x42 interface."""


@configclass
class RobanDepthRawEnvCfg(RobanDepthEnhanceEnvCfg):
    """Training configuration for RobanS17 with unenhanced depth input."""

    observations: RobanRawDepthObservationsCfg = RobanRawDepthObservationsCfg()


@configclass
class RobanDepthRawEnvCfg_PLAY(RobanDepthEnhanceEnvCfg_PLAY):
    """Playback configuration for RobanS17 with unenhanced depth input."""

    observations: RobanRawDepthObservationsCfg = RobanRawDepthObservationsCfg()
