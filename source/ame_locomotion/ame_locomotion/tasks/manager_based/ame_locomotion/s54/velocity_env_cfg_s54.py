"""Kuavo S54 adaptation of the AME G1 rough-terrain task."""

import importlib

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from kuavo.assets.kuavo_s54 import KUAVO_S54_CFG


_g1_cfg = importlib.import_module(
    "ame_locomotion.tasks.manager_based.ame_locomotion.29dof.velocity_env_cfg_29dof"
)


def _make_scene(num_envs: int) -> _g1_cfg.MySceneCfg:
    """Create the G1 scene with the S54 articulation and upper-body scan frame."""
    scene = _g1_cfg.MySceneCfg(num_envs=num_envs, env_spacing=2.5)
    scene.robot = KUAVO_S54_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/waist_yaw_link"
    return scene


def _apply_s54_name_mapping(cfg) -> None:
    """Replace every G1-specific joint/body selector with its S54 equivalent."""
    cfg.scene.robot = KUAVO_S54_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/waist_yaw_link"

    for event_name in ("add_base_mass", "base_com", "base_external_force_torque"):
        event = getattr(cfg.events, event_name, None)
        if event is not None:
            event.params["asset_cfg"] = SceneEntityCfg(
                "robot", body_names="waist_yaw_link"
            )

    cfg.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=["(?!leg_[lr][56]_link$).*"]
    )
    foot_body = "leg_[lr]6_link"
    cfg.rewards.feet_air_time.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=foot_body
    )
    cfg.rewards.feet_air_time_variance.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=foot_body
    )
    cfg.rewards.feet_slide.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=foot_body
    )
    cfg.rewards.feet_slide.params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=foot_body
    )
    cfg.rewards.feet_stumble.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces", body_names=foot_body
    )
    cfg.rewards.feet_too_near.params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=foot_body
    )

    cfg.rewards.joint_coordination.params["coord_joints"] = [
        ["leg_l3_joint", "zarm_r1_joint"],
        ["leg_r3_joint", "zarm_l1_joint"],
    ]
    cfg.rewards.joint_deviation_hip.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=["leg_[lr][12]_joint"]
    )
    cfg.rewards.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=["zarm_[lr][1-7]_joint"]
    )
    cfg.rewards.joint_deviation_waists.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=["waist_yaw_joint"]
    )

    cfg.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces",
        body_names=[
            "base_link",
            "waist_yaw_link",
        ],
    )

    visualize_cam = getattr(cfg.scene, "visualize_cam", None)
    if visualize_cam is not None:
        visualize_cam.prim_path = (
            "{ENV_REGEX_NS}/Robot/waist_yaw_link/visualize_cam"
        )


@configclass
class KuavoS54RoughEnvCfg(_g1_cfg.G1RoughEnvCfg):
    """Training configuration for AME on Kuavo S54."""

    scene: _g1_cfg.MySceneCfg = _make_scene(num_envs=2048)

    def __post_init__(self):
        super().__post_init__()
        _apply_s54_name_mapping(self)
        # S54 has 13 collision shapes on each foot. With 2048 environments and
        # self-collision enabled, PhysX's default 2**26-byte GPU narrowphase
        # stack can overflow and drop contacts. Match Isaac Lab's contact-rich
        # task setting and retain enough headroom above the reported minimum.
        self.sim.physx.gpu_collision_stack_size = 2**28


@configclass
class KuavoS54RoughEnvCfg_PLAY(_g1_cfg.G1RoughEnvCfg_PLAY):
    """Playback configuration for AME on Kuavo S54."""

    scene: _g1_cfg.MySceneCfg = _make_scene(num_envs=2048)

    def __post_init__(self):
        super().__post_init__()
        _apply_s54_name_mapping(self)
        self.sim.physx.gpu_collision_stack_size = 2**28
