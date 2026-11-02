"""RobanS17 migration of AME-Kuavo-S54-Depth-Enhance."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from ame_locomotion.tasks.manager_based.ame_locomotion.s54.velocity_env_cfg_s54_depth import (
    KuavoS54DepthSceneCfg,
)
from ame_locomotion.tasks.manager_based.ame_locomotion.s54.velocity_env_cfg_s54_depth_enhance import (
    KuavoS54DepthEnhanceEnvCfg,
    KuavoS54DepthEnhanceEnvCfg_PLAY,
    KuavoS54EnhanceRewardsCfg,
)
from kuavo.assets.roban_s17 import ROBAN_S17_CFG


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

    robot: ArticulationCfg = ROBAN_S17_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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

    cfg.scene.robot = ROBAN_S17_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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

    if hasattr(cfg.rewards, "joint_coordination"):
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
    if hasattr(cfg.rewards, "joint_deviation_waists"):
        cfg.rewards.joint_deviation_waists.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["waist_yaw_joint"]
        )

    cfg.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=["base_link", "waist_yaw_link"]
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
class RobanDepthEnhanceEnvCfg(KuavoS54DepthEnhanceEnvCfg):
    """Training configuration for AME-Roban-Depth-Enhance."""

    scene: RobanDepthSceneCfg = RobanDepthSceneCfg(num_envs=2048, env_spacing=2.5)
    rewards: KuavoS54EnhanceRewardsCfg = KuavoS54EnhanceRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_roban_name_mapping(self)
        self.scene.height_scanner = None
        self.scene.terrain.max_init_terrain_level = 0
        self.events.push_robot = None
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5


@configclass
class RobanDepthEnhanceEnvCfg_PLAY(KuavoS54DepthEnhanceEnvCfg_PLAY):
    """Playback configuration for AME-Roban-Depth-Enhance."""

    scene: RobanDepthSceneCfg = RobanDepthSceneCfg(num_envs=50, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        _apply_roban_name_mapping(self)
        self.scene.height_scanner = None
