"""RSL-RL runner configuration for the Student-enhanced S54 depth task."""

from isaaclab.utils import configclass

from .kuavo_s54_depth_ame_rsl_rl_ppo_cfg import (
    KuavoS54DepthAMEPPORunnerCfg,
    KuavoS54DepthActorCriticCfg,
)


@configclass
class KuavoS54DepthConfidenceActorCriticCfg(KuavoS54DepthActorCriticCfg):
    """CNN input contains enhanced depth and confidence as two channels."""

    map_scan_dim: tuple[int, int, int] = (42, 42, 2)


@configclass
class KuavoS54DepthEnhancementAMEPPORunnerCfg(KuavoS54DepthAMEPPORunnerCfg):
    """PPO settings for AME-Kuavo-S54-Depth-enhancement."""

    experiment_name = "kuavo_s54_depth_enhancement"
    policy = KuavoS54DepthConfidenceActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )


@configclass
class KuavoS54DepthPrivilegedCriticAMEPPORunnerCfg(KuavoS54DepthEnhancementAMEPPORunnerCfg):
    """Student depth/confidence Actor with a privileged height-scan Critic."""

    experiment_name = "kuavo_s54_depth_privileged_critic"


@configclass
class KuavoS54DepthEnhancementPrivilegedActorCriticCfg(KuavoS54DepthActorCriticCfg):
    """Actor uses depth while Critic uses depth plus four Student priors."""

    actor_map_scan_dim: tuple[int, int, int] = (42, 42, 1)
    critic_map_scan_dim: tuple[int, int, int] = (42, 42, 5)


@configclass
class KuavoS54DepthEnhancementPrivilegedAMEPPORunnerCfg(KuavoS54DepthAMEPPORunnerCfg):
    """PPO settings for asymmetric Student depth and privileged priors."""

    experiment_name = "kuavo_s54_depth_enhancement_privileged"
    policy = KuavoS54DepthEnhancementPrivilegedActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )


@configclass
class KuavoS54DepthEnhancementPrivilegedV2ActorCriticCfg(KuavoS54DepthActorCriticCfg):
    """Actor receives depth; Critic additionally receives one edge map."""

    actor_map_scan_dim: tuple[int, int, int] = (42, 42, 1)
    critic_map_scan_dim: tuple[int, int, int] = (42, 42, 2)


@configclass
class KuavoS54DepthEnhancementPrivilegedV2AMEPPORunnerCfg(
    KuavoS54DepthAMEPPORunnerCfg
):
    """PPO settings for the edge-privileged S54 v2 task."""

    experiment_name = "kuavo_s54_depth_enhancement_privileged_v2"
    policy = KuavoS54DepthEnhancementPrivilegedV2ActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )


@configclass
class KuavoS54DepthGeometryActorCriticCfg(KuavoS54DepthConfidenceActorCriticCfg):
    """CNN input contains depth, confidence, edge and foothold channels."""

    map_scan_dim: tuple[int, int, int] = (42, 42, 4)


@configclass
class KuavoS54DepthGeometryEnhancementAMEPPORunnerCfg(KuavoS54DepthEnhancementAMEPPORunnerCfg):
    """PPO settings for the geometry-head Student ablation."""

    experiment_name = "kuavo_s54_depth_geometry_enhancement"
    policy = KuavoS54DepthGeometryActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )


@configclass
class KuavoS54DepthStudentAMEPPORunnerCfg(KuavoS54DepthAMEPPORunnerCfg):
    """PPO settings for depth-only inference from the new Student."""

    experiment_name = "kuavo_s54_depth_student"
    max_iterations = 80000
    save_interval = 2000


@configclass
class KuavoS54DepthSensorRandomizedAMEPPORunnerCfg(KuavoS54DepthAMEPPORunnerCfg):
    """PPO settings for the raw sensor-randomized depth baseline."""

    experiment_name = "kuavo_s54_depth_sensor_randomized"
    max_iterations = 80000
    save_interval = 2000
