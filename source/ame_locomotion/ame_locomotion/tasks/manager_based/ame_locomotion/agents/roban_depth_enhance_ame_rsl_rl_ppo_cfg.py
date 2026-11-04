"""RSL-RL runner configuration for RobanS17 depth enhancement."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from .ame_rsl_rl_ppo_cfg import AMEPpoAlgorithmCfg
from .kuavo_s54_depth_ame_rsl_rl_ppo_cfg import KuavoS54DepthActorCriticCfg, KuavoS54DepthAMEPPORunnerCfg


@configclass
class RobanDepthActorCriticCfg(KuavoS54DepthActorCriticCfg):
    """Actor-critic settings for the 21-DOF RobanS17 policy."""


@configclass
class RobanDepthEnhanceAMEPPORunnerCfg(KuavoS54DepthAMEPPORunnerCfg):
    """PPO settings for AME-Roban-Depth-Enhance-v0."""

    experiment_name = "roban_depth_enhance"
    # Match LejuLab's RobanS17WalkPPORunnerCfg training horizon and checkpoint
    # cadence. The depth encoder and AME mixed-precision path remain task-specific.
    max_iterations = 60000
    save_interval = 500
    num_steps_per_env = 24
    # Roban command/reward changes require a fresh optimizer state.
    resume = False
    load_run = None
    load_checkpoint = None
    policy = RobanDepthActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = AMEPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RobanDepthRawAMEPPORunnerCfg(RobanDepthEnhanceAMEPPORunnerCfg):
    """PPO settings for the raw-depth RobanS17 task."""

    experiment_name = "roban_depth_raw"


@configclass
class RobanDepthMLPActorCriticCfg(RslRlPpoActorCriticCfg):
    """CNN depth encoder and MLP fusion policy for 42x42 depth inputs."""

    class_name: str = "ActorCriticCNNMLP"
    map_scan_dim: tuple[int, int, int] = (42, 42, 1)
    cnn_output_dim: int = 64
    cnn_downsample: bool = True


@configclass
class RobanDepthRawMLPAMEPPORunnerCfg(RobanDepthRawAMEPPORunnerCfg):
    """PPO settings for MLP-Roban-Depth-Raw-v0."""

    experiment_name = "mlp_roban_depth_raw"
    policy = RobanDepthMLPActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )


@configclass
class RobanDepthEnhanceMLPAMEPPORunnerCfg(RobanDepthEnhanceAMEPPORunnerCfg):
    """PPO settings for MLP-Roban-Depth-Enhance-v0."""

    experiment_name = "mlp_roban_depth_enhance"
    policy = RobanDepthMLPActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
