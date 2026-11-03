"""RSL-RL configuration for height-scan to enhanced-depth distillation."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


@configclass
class KuavoS54DepthDistillPolicyCfg:
    class_name: str = "StudentTeacherEncoder"
    student_obs_normalization: bool = False
    teacher_obs_normalization: bool = False
    student_hidden_dims: list[int] = [512, 256, 128]
    teacher_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    init_noise_std: float = 0.1
    teacher_map_scan_dim: tuple[int, int, int] = (33, 21, 3)


@configclass
class KuavoS54DistillationAlgorithmCfg:
    class_name: str = "DAgger"
    use_mixed_precision: bool = True
    num_learning_epochs: int = 1
    gradient_length: int = 1
    learning_rate: float = 1.0e-3
    max_grad_norm: float = 1.0
    loss_type: str = "mse"
    optimizer: str = "adam"


@configclass
class KuavoS54DepthDistillRunnerCfg(RslRlBaseRunnerCfg):
    class_name: str = "DistillationRunner"
    num_steps_per_env: int = 24
    max_iterations: int = 20000
    save_interval: int = 500
    experiment_name: str = "kuavo_s54_depth_enhance_dagger"
    obs_groups: dict[str, list[str]] = {"policy": ["policy"], "teacher": ["teacher"]}
    policy: KuavoS54DepthDistillPolicyCfg = KuavoS54DepthDistillPolicyCfg()
    algorithm: KuavoS54DistillationAlgorithmCfg = KuavoS54DistillationAlgorithmCfg()


@configclass
class KuavoS54DepthCNNMLPPolicyCfg:
    class_name: str = "StudentTeacherCNNMLP"
    student_depth_group: str = "depth"
    student_props_group: str = "props"
    critic_group: str = "teacher_critic"
    teacher_map_scan_dim: tuple[int, int, int] = (33, 21, 3)
    depth_feature_dim: int = 128
    props_hidden_dims: list[int] = [512, 256]
    props_feature_dim: int = 128
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    velocity_hidden_dims: list[int] = [128]
    activation: str = "elu"
    init_noise_std: float = 0.1


@configclass
class KuavoS54DepthCNNMLPAlgorithmCfg(KuavoS54DistillationAlgorithmCfg):
    velocity_loss_coef: float = 0.1
    gamma: float = 0.99
    lam: float = 0.95
    value_learning_rate: float = 1.0e-3
    value_num_learning_epochs: int = 1


@configclass
class KuavoS54DepthCNNMLPRunnerCfg(KuavoS54DepthDistillRunnerCfg):
    experiment_name: str = "kuavo_s54_depth_enhance_dagger_cnn_5props"
    obs_groups: dict[str, list[str]] = {
        "policy": ["depth", "props"],
        "teacher": ["teacher"],
    }
    policy: KuavoS54DepthCNNMLPPolicyCfg = KuavoS54DepthCNNMLPPolicyCfg()
    algorithm: KuavoS54DepthCNNMLPAlgorithmCfg = KuavoS54DepthCNNMLPAlgorithmCfg()
