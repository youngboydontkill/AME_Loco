"""S54 playback configuration with frozen Student depth enhancement."""
import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ame_locomotion.tasks.manager_based.ame_locomotion import mdp

from .velocity_env_cfg_s54_depth import (
    DepthObservationsCfg,
    KuavoS54DepthRoughEnvCfg,
    KuavoS54DepthRoughEnvCfg_PLAY,
    KuavoS54DepthSceneCfg,
)
from . import velocity_env_cfg_s54 as s54_cfg
from ..mdp.student_depth import EDGE_STUDENT_CHECKPOINT


@configclass
class DepthEnhanceEventCfg(s54_cfg._g1_cfg.EventCfg):
    """Reset events for the depth enhancement domain randomization."""

    randomize_depth_camera = EventTerm(
        func=mdp.randomize_camera_position,
        mode="reset",
        params={
            "sensor_cfg": SceneEntityCfg("depth_camera"),
            "position_range": {
                "x": (-0.00, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
            },
        },
    )


@configclass
class EnhancedDepthObservationsCfg(DepthObservationsCfg):
    """Replace only the policy depth term; retain the original critic input."""

    @configclass
    class PolicyCfg(DepthObservationsCfg.PolicyCfg):
        depth_image = ObsTerm(
            func=mdp.student_depth_image,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "checkpoint_path": EDGE_STUDENT_CHECKPOINT,
                "noise_std": 0.02,
                "dropout_probability": 0.02,
            },
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class MixedEnhancedDepthObservationsCfg(DepthObservationsCfg):
    """Mix Student-enhanced and raw noisy depth for actor training."""

    @configclass
    class PolicyCfg(DepthObservationsCfg.PolicyCfg):
        depth_image = ObsTerm(
            func=mdp.mixed_student_depth_image,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "checkpoint_path": EDGE_STUDENT_CHECKPOINT,
                "enhance_probability": 0.7,
                "noise_std": 0.0,
                "dropout_probability": 0.0,
            },
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class KuavoS54EnhanceEnvCfg(KuavoS54DepthRoughEnvCfg):
    """Training configuration for mixed raw/Student depth inputs."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: MixedEnhancedDepthObservationsCfg = MixedEnhancedDepthObservationsCfg()
    events: DepthEnhanceEventCfg = DepthEnhanceEventCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class KuavoS54DepthEnhanceEnvCfg_PLAY(KuavoS54DepthRoughEnvCfg_PLAY):
    """Playback configuration for AME-Kuavo-S54-Depth-Play-Enhance-v0."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=50, env_spacing=2.5)
    observations: EnhancedDepthObservationsCfg = EnhancedDepthObservationsCfg()
    events: DepthEnhanceEventCfg = DepthEnhanceEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (1, 1)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # self.events.reset_base.params["pose_range"]["yaw"] = (-math.pi / 2, math.pi / 2)


@configclass
class DepthDistillationObservationsCfg(EnhancedDepthObservationsCfg):
    """Enhanced depth for the student and the original height scan for its teacher."""

    @configclass
    class TeacherCfg(s54_cfg._g1_cfg.ObservationsCfg.PolicyCfg):
        pass

    @configclass
    class TeacherCriticCfg(s54_cfg._g1_cfg.ObservationsCfg.CriticCfg):
        pass

    teacher: TeacherCfg = TeacherCfg()
    teacher_critic: TeacherCriticCfg = TeacherCriticCfg()


@configclass
class KuavoS54DepthDistillEnvCfg(KuavoS54DepthRoughEnvCfg):
    """Training environment that exposes both student and teacher observations."""

    observations: DepthDistillationObservationsCfg = DepthDistillationObservationsCfg()
    events: DepthEnhanceEventCfg = DepthEnhanceEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner = s54_cfg._make_scene(self.scene.num_envs).height_scanner


@configclass
class KuavoS54DepthDistillEnvCfg_PLAY(KuavoS54DepthRoughEnvCfg_PLAY):
    """Play configuration for evaluating the distilled depth student."""

    observations: DepthDistillationObservationsCfg = DepthDistillationObservationsCfg()
    events: DepthEnhanceEventCfg = DepthEnhanceEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner = s54_cfg._make_scene(self.scene.num_envs).height_scanner
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)


@configclass
class RawDepthDistillationObservationsCfg(DepthDistillationObservationsCfg):
    """Raw camera depth for the student, retaining matching proprioceptive terms."""

    @configclass
    class PolicyCfg(DepthObservationsCfg.PolicyCfg):
        pass

    policy: PolicyCfg = PolicyCfg()


@configclass
class KuavoS54DepthRawDistillEnvCfg_PLAY(KuavoS54DepthRoughEnvCfg_PLAY):
    """Play configuration for testing the student with unenhanced camera depth."""

    observations: RawDepthDistillationObservationsCfg = RawDepthDistillationObservationsCfg()
    events: DepthEnhanceEventCfg = DepthEnhanceEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner = s54_cfg._make_scene(self.scene.num_envs).height_scanner
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
