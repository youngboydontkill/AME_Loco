"""AMP-PPO configuration for the Kuavo S54 enhanced-depth task."""

import os

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from .kuavo_s54_depth_ame_rsl_rl_ppo_cfg import (
    KuavoS54DepthActorCriticCfg,
    KuavoS54DepthAMEPPORunnerCfg,
)


@configclass
class KuavoS54DepthAMPAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "PPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.008
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    amp_cfg: dict = {
        "motion_file": os.environ.get("AME_AMP_MOTION_FILE", "data/amp/kuavo_s54_walk.npz"),
        "learning_rate": 1.0e-4,
        "batch_size": 4096,
        "num_updates": 2,
        "replay_size": 100_000,
        "gradient_penalty_coef": 5.0,
        "reward_coef": 0.5,
    }


@configclass
class KuavoS54DepthAMPPOCfg(KuavoS54DepthAMEPPORunnerCfg):
    """Runner settings for AME-Kuavo-S54-Depth-Enhance-AMP-v0."""

    experiment_name = "kuavo_s54_depth_enhance_amp"
    algorithm = KuavoS54DepthAMPAlgorithmCfg()
    policy = KuavoS54DepthActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
