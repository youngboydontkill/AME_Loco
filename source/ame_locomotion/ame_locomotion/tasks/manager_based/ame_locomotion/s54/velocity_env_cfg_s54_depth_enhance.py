"""S54 playback configuration with frozen Student depth enhancement."""
import math

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
from ..mdp.student_depth import EDGE_STUDENT_CHECKPOINT


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
                "noise_std": 0.02,
                "dropout_probability": 0.02,
            },
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class KuavoS54EnhanceEnvCfg(KuavoS54DepthRoughEnvCfg):
    """Training configuration for mixed raw/Student depth inputs."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: MixedEnhancedDepthObservationsCfg = MixedEnhancedDepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class KuavoS54DepthEnhanceEnvCfg_PLAY(KuavoS54DepthRoughEnvCfg_PLAY):
    """Playback configuration for AME-Kuavo-S54-Depth-Play-Enhance-v0."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=50, env_spacing=2.5)
    observations: EnhancedDepthObservationsCfg = EnhancedDepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (1, 1)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # self.events.reset_base.params["pose_range"]["yaw"] = (-math.pi / 2, math.pi / 2)
