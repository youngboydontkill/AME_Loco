"""RSL-RL runner configuration for the Student-enhanced S54 depth task."""

from isaaclab.utils import configclass

from .kuavo_s54_depth_ame_rsl_rl_ppo_cfg import KuavoS54DepthAMEPPORunnerCfg


@configclass
class KuavoS54DepthEnhancementAMEPPORunnerCfg(KuavoS54DepthAMEPPORunnerCfg):
    """PPO settings for AME-Kuavo-S54-Depth-enhancement."""

    experiment_name = "kuavo_s54_depth_enhancement"
