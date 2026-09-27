"""RSL-RL runner configuration for the Kuavo S54 depth task."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from .kuavo_s54_ame_rsl_rl_ppo_cfg import KuavoS54AMEPPORunnerCfg


@configclass
class KuavoS54DepthActorCriticCfg(RslRlPpoActorCriticCfg):
    """Actor-critic settings for a single-channel 42x42 depth image."""

    class_name: str = "ActorCriticEncoder"
    map_scan_dim: tuple[int, int, int] = (42, 42, 1)
    cnn_downsample: bool = True


@configclass
class KuavoS54DepthAMEPPORunnerCfg(KuavoS54AMEPPORunnerCfg):
    """PPO settings for AME-Kuavo-S54-Depth-v0."""

    experiment_name = "kuavo_s54_depth_ame"
    max_iterations = 80000
    save_interval = 2000

    policy = KuavoS54DepthActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
