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
            func=mdp.student_depth_confidence_image,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "checkpoint_path": DEFAULT_STUDENT_CHECKPOINT,
            },
        )
        concatenate_terms = True

    @configclass
    class CriticCfg(DepthObservationsCfg.CriticCfg):
        depth_image = ObsTerm(
            func=mdp.student_depth_confidence_image,
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


@configclass
class PrivilegedStudentDepthObservationsCfg(DepthObservationsCfg):
    """Actor sees mixed enhanced depth; Critic also sees Student prior maps."""

    @configclass
    class PolicyCfg(DepthObservationsCfg.PolicyCfg):
        depth_image = ObsTerm(
            func=mdp.student_depth_mixed_image,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "checkpoint_path": DEFAULT_STUDENT_CHECKPOINT,
                "enhance_probability": 0.8,
            },
        )

    @configclass
    class CriticCfg(DepthObservationsCfg.CriticCfg):
        depth_image = ObsTerm(
            func=mdp.student_depth_prior_mixed_image,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "checkpoint_path": DEFAULT_STUDENT_CHECKPOINT,
                "enhance_probability": 0.8,
            },
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class KuavoS54DepthEnhancementPrivilegedEnvCfg(KuavoS54DepthRoughEnvCfg):
    """Asymmetric Student depth Actor with privileged auxiliary-map Critic."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: PrivilegedStudentDepthObservationsCfg = PrivilegedStudentDepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = _make_easier_depth_terrain(
            self.scene.terrain.terrain_generator
        )


@configclass
class KuavoS54DepthEnhancementPrivilegedEnvCfg_PLAY(KuavoS54DepthRoughEnvCfg_PLAY):
    """Playback configuration for the asymmetric Student task."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=50, env_spacing=2.5)
    observations: PrivilegedStudentDepthObservationsCfg = PrivilegedStudentDepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = _make_easier_depth_terrain(
            self.scene.terrain.terrain_generator
        )


@configclass
class PrivilegedCriticDepthObservationsCfg(EnhancedDepthObservationsCfg):
    """Deployable Student map for the Actor; Isaac height scan for the Critic."""

    @configclass
    class CriticCfg(EnhancedDepthObservationsCfg.CriticCfg):
        height_scan = ObsTerm(
            func=mdp.elevation_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": False},
        )

    critic: CriticCfg = CriticCfg()


@configclass
class KuavoS54DepthPrivilegedCriticEnvCfg(KuavoS54DepthRoughEnvCfg):
    """Asymmetric Student Actor with privileged terrain-height Critic input."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: PrivilegedCriticDepthObservationsCfg = PrivilegedCriticDepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = _make_easier_depth_terrain(
            self.scene.terrain.terrain_generator
        )


@configclass
class KuavoS54DepthPrivilegedCriticEnvCfg_PLAY(KuavoS54DepthRoughEnvCfg_PLAY):
    """Playback preserves the trained Critic shape; actions use the Student Actor."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=50, env_spacing=2.5)
    observations: PrivilegedCriticDepthObservationsCfg = PrivilegedCriticDepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = _make_easier_depth_terrain(
            self.scene.terrain.terrain_generator
        )


@configclass
class GeometryEnhancedDepthObservationsCfg(EnhancedDepthObservationsCfg):
    """Student depth plus confidence, edge and foothold map channels."""

    @configclass
    class PolicyCfg(EnhancedDepthObservationsCfg.PolicyCfg):
        depth_image = ObsTerm(
            func=mdp.student_depth_geometry_image,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "checkpoint_path": DEFAULT_STUDENT_CHECKPOINT,
            },
        )

    @configclass
    class CriticCfg(EnhancedDepthObservationsCfg.CriticCfg):
        depth_image = ObsTerm(
            func=mdp.student_depth_geometry_image,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "checkpoint_path": DEFAULT_STUDENT_CHECKPOINT,
            },
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class KuavoS54DepthGeometryEnhancementEnvCfg(KuavoS54DepthRoughEnvCfg):
    """Geometry-head Student observation ablation for the S54 task."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: GeometryEnhancedDepthObservationsCfg = GeometryEnhancedDepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = _make_easier_depth_terrain(
            self.scene.terrain.terrain_generator
        )


@configclass
class KuavoS54DepthGeometryEnhancementEnvCfg_PLAY(KuavoS54DepthRoughEnvCfg_PLAY):
    """Playback configuration for the geometry-head Student task."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=50, env_spacing=2.5)
    observations: GeometryEnhancedDepthObservationsCfg = GeometryEnhancedDepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = _make_easier_depth_terrain(
            self.scene.terrain.terrain_generator
        )


@configclass
class SensorRandomizedDepthObservationsCfg(DepthObservationsCfg):
    """Single-channel raw depth with per-step sensor randomization."""

    @configclass
    class PolicyCfg(DepthObservationsCfg.PolicyCfg):
        depth_image = ObsTerm(
            func=mdp.sensor_randomized_depth_image,
            params={"sensor_cfg": SceneEntityCfg("depth_camera"), "noise_std": 0.02, "dropout_probability": 0.02},
        )

    @configclass
    class CriticCfg(DepthObservationsCfg.CriticCfg):
        depth_image = ObsTerm(
            func=mdp.sensor_randomized_depth_image,
            params={"sensor_cfg": SceneEntityCfg("depth_camera"), "noise_std": 0.02, "dropout_probability": 0.02},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class KuavoS54DepthSensorRandomizedEnvCfg(KuavoS54DepthRoughEnvCfg):
    """Raw depth with sensor randomization for the Appendix-A baseline."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: SensorRandomizedDepthObservationsCfg = SensorRandomizedDepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = _make_easier_depth_terrain(
            self.scene.terrain.terrain_generator
        )


@configclass
class KuavoS54DepthSensorRandomizedEnvCfg_PLAY(KuavoS54DepthRoughEnvCfg_PLAY):
    """Playback configuration for the sensor-randomized depth baseline."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=50, env_spacing=2.5)
    observations: SensorRandomizedDepthObservationsCfg = SensorRandomizedDepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = _make_easier_depth_terrain(
            self.scene.terrain.terrain_generator
        )


@configclass
class StudentDepthOnlyObservationsCfg(DepthObservationsCfg):
    """New Student enhanced depth with auxiliary map channels omitted."""

    @configclass
    class PolicyCfg(DepthObservationsCfg.PolicyCfg):
        depth_image = ObsTerm(
            func=mdp.student_depth_image,
            params={"sensor_cfg": SceneEntityCfg("depth_camera"), "checkpoint_path": DEFAULT_STUDENT_CHECKPOINT},
        )

    @configclass
    class CriticCfg(DepthObservationsCfg.CriticCfg):
        depth_image = ObsTerm(
            func=mdp.student_depth_image,
            params={"sensor_cfg": SceneEntityCfg("depth_camera"), "checkpoint_path": DEFAULT_STUDENT_CHECKPOINT},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class KuavoS54DepthStudentEnvCfg(KuavoS54DepthRoughEnvCfg):
    """New Gemini Student depth-only baseline."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: StudentDepthOnlyObservationsCfg = StudentDepthOnlyObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = _make_easier_depth_terrain(
            self.scene.terrain.terrain_generator
        )


@configclass
class KuavoS54DepthStudentEnvCfg_PLAY(KuavoS54DepthRoughEnvCfg_PLAY):
    """Playback configuration for the new Student depth-only baseline."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=50, env_spacing=2.5)
    observations: StudentDepthOnlyObservationsCfg = StudentDepthOnlyObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = _make_easier_depth_terrain(
            self.scene.terrain.terrain_generator
        )
