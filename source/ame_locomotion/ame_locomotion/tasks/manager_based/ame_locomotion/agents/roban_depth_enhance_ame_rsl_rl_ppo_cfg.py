"""RSL-RL runner configuration for RobanS17 depth enhancement."""

from isaaclab.utils import configclass

from .kuavo_s54_depth_ame_rsl_rl_ppo_cfg import KuavoS54DepthActorCriticCfg, KuavoS54DepthAMEPPORunnerCfg


@configclass
class RobanDepthActorCriticCfg(KuavoS54DepthActorCriticCfg):
    """Actor-critic settings for the 21-DOF RobanS17 policy."""


@configclass
class RobanDepthEnhanceAMEPPORunnerCfg(KuavoS54DepthAMEPPORunnerCfg):
    """PPO settings for AME-Roban-Depth-Enhance-v0."""

    experiment_name = "roban_depth_enhance"
    policy = RobanDepthActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
