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
