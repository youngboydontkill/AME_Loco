"""RSL-RL runner configuration for mixed S54 depth enhancement training."""

from isaaclab.utils import configclass

from .kuavo_s54_depth_ame_rsl_rl_ppo_cfg import KuavoS54DepthAMEPPORunnerCfg


@configclass
class KuavoS54EnhanceAMEPPORunnerCfg(KuavoS54DepthAMEPPORunnerCfg):
    """PPO settings for AME-Kuavo-S54-Enhance-v0."""

    experiment_name = "kuavo_s54_enhance"
