"""Runner configuration for S54 height-scan teacher to depth student distillation."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class KuavoS54DepthDistillPolicyCfg(RslRlDistillationStudentTeacherCfg):
    class_name: str = "AMEStudentTeacher"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    student_obs_normalization: bool = False
    teacher_obs_normalization: bool = False
    student_hidden_dims: list[int] = [512, 256, 128]
    teacher_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    student_map_scan_dim: tuple[int, int, int] = (42, 42, 1)
    teacher_map_scan_dim: tuple[int, int, int] = (33, 21, 3)
    student_cnn_downsample: bool = True
    teacher_cnn_downsample: bool = True
    student_mha_dim: int = 64
    teacher_mha_dim: int = 64
    student_num_heads: int = 16
    teacher_num_heads: int = 16
    student_attach_global: bool = False
    teacher_attach_global: bool = False


@configclass
class KuavoS54DepthDistillRunnerCfg(RslRlDistillationRunnerCfg):
    """Distillation hyperparameters for AME-Kuavo-S54-Depth-Distill-v0."""

    experiment_name = "kuavo_s54_depth_distill"
    teacher_checkpoint: str | None = None
    num_steps_per_env = 24
    max_iterations = 40000
    save_interval = 500
    obs_groups = {
        "policy": ["policy"],
        "teacher": ["teacher"],
    }
    policy = KuavoS54DepthDistillPolicyCfg()
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
        optimizer="adam",
        loss_type="mse",
    )
