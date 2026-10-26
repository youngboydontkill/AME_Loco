"""S54 playback configuration with frozen Student depth enhancement."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ame_locomotion.tasks.manager_based.ame_locomotion import mdp

from .velocity_env_cfg_s54_depth import (
    DepthObservationsCfg,
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
            },
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class KuavoS54DepthEnhanceEnvCfg_PLAY(KuavoS54DepthRoughEnvCfg_PLAY):
    """Playback configuration for AME-Kuavo-S54-Depth-Play-Enhance-v0."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=50, env_spacing=2.5)
    observations: EnhancedDepthObservationsCfg = EnhancedDepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
