"""GRU AMP-PPO configuration for Kuavo S54 enhanced depth."""

from isaaclab.utils import configclass

from .kuavo_s54_depth_ame_rsl_rl_ppo_cfg import (
    KuavoS54DepthAMEPPORunnerCfg,
    KuavoS54DepthActorCriticCfg,
)
from .kuavo_s54_depth_amp_rsl_rl_ppo_cfg import (
    KuavoS54DepthAMPAlgorithmCfg,
    KuavoS54DepthAMPPOCfg,
)


@configclass
class KuavoS54DepthGRUActorCriticCfg(KuavoS54DepthActorCriticCfg):
    """Policy settings for five-frame depth and proprioceptive GRU encoding."""

    class_name: str = "ActorCriticGRU"
    gru_hidden_size: int = 256


@configclass
class KuavoS54DepthGRUAMPAlgorithmCfg(KuavoS54DepthAMPAlgorithmCfg):
    velocity_loss_coef: float = 0.1


@configclass
class KuavoS54DepthGRUAMPPOCfg(KuavoS54DepthAMPPOCfg):
    """Runner settings for GRU-Kuavo-S54-Depth-Enhance-AMP-v0."""

    experiment_name = "kuavo_s54_depth_enhance_gru_amp"
    algorithm = KuavoS54DepthGRUAMPAlgorithmCfg()
    policy = KuavoS54DepthGRUActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )


@configclass
class KuavoS54DepthGRUPlayRunnerCfg(KuavoS54DepthAMEPPORunnerCfg):
    """Playback runner settings without constructing the AMP discriminator."""

    experiment_name = "kuavo_s54_depth_enhance_gru_amp"
    policy = KuavoS54DepthGRUActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
