"""PPO fine-tuning configuration for the five-frame depth student."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

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
    depth_feature_dim: int = 128
    props_hidden_dims: list[int] = [512, 256]
    props_feature_dim: int = 128


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
