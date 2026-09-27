"""Kuavo S54 depth task using the frozen Stable-Sim2Real Student."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ame_locomotion.tasks.manager_based.ame_locomotion import mdp

from .velocity_env_cfg_s54_depth import (
    DepthObservationsCfg,
    KuavoS54DepthRoughEnvCfg,
    KuavoS54DepthRoughEnvCfg_PLAY,
    KuavoS54DepthSceneCfg,
    _make_easier_depth_terrain,
)
from ..mdp.student_depth import DEFAULT_STUDENT_CHECKPOINT


@configclass
class EnhancedDepthObservationsCfg(DepthObservationsCfg):
    """Keep the original proprioception and replace only the depth term."""

    @configclass
    class PolicyCfg(DepthObservationsCfg.PolicyCfg):
        depth_image = ObsTerm(
            func=mdp.student_depth_image,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "checkpoint_path": DEFAULT_STUDENT_CHECKPOINT,
            },
        )
        concatenate_terms = True

    @configclass
    class CriticCfg(DepthObservationsCfg.CriticCfg):
        depth_image = ObsTerm(
            func=mdp.student_depth_image,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "checkpoint_path": DEFAULT_STUDENT_CHECKPOINT,
            },
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class KuavoS54DepthEnhancementEnvCfg(KuavoS54DepthRoughEnvCfg):
    """AME-Kuavo-S54-Depth-enhancement training configuration."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: EnhancedDepthObservationsCfg = EnhancedDepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = _make_easier_depth_terrain(
            self.scene.terrain.terrain_generator
        )


@configclass
class KuavoS54DepthEnhancementEnvCfg_PLAY(KuavoS54DepthRoughEnvCfg_PLAY):
    """Playback configuration for AME-Kuavo-S54-Depth-enhancement."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=50, env_spacing=2.5)
    observations: EnhancedDepthObservationsCfg = EnhancedDepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
