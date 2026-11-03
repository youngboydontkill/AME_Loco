"""PPO fine-tuning configuration for the five-frame depth student."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from .ame_rsl_rl_ppo_cfg import AMEPpoAlgorithmCfg
from .kuavo_s54_ame_rsl_rl_ppo_cfg import KuavoS54AMEPPORunnerCfg


@configclass
class KuavoS54DepthStudentActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticCNNProps"
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    init_noise_std: float = 0.1
    student_depth_group: str = "depth"
    student_props_group: str = "props"
    critic_group: str = "teacher_critic"
    teacher_group: str = "teacher"
    teacher_map_scan_dim: tuple[int, int, int] = (33, 21, 3)
    depth_feature_dim: int = 128
    props_hidden_dims: list[int] = [512, 256]
    props_feature_dim: int = 128


@configclass
class KuavoS54DepthStudentPPOAlgorithmCfg(AMEPpoAlgorithmCfg):
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
    bc_loss_coef: float = 0.2


@configclass
class KuavoS54DepthStudentPPORunnerCfg(KuavoS54AMEPPORunnerCfg):
    """PPO fine-tuning starting from a MLP-5props DAgger student checkpoint."""

    experiment_name: str = "kuavo_s54_depth_student_ppo_finetune"
    max_iterations: int = 20000
    save_interval: int = 500
    obs_groups: dict[str, list[str]] = {
        "policy": ["depth", "props"],
        "critic": ["teacher_critic"],
    }
    policy: KuavoS54DepthStudentActorCriticCfg = KuavoS54DepthStudentActorCriticCfg()
    algorithm: KuavoS54DepthStudentPPOAlgorithmCfg = KuavoS54DepthStudentPPOAlgorithmCfg()
