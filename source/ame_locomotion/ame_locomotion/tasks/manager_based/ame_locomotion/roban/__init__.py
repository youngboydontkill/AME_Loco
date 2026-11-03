"""AME RobanS17 depth-enhanced locomotion tasks."""

import gymnasium as gym

from ame_locomotion.tasks.manager_based.ame_locomotion import agents


gym.register(
    id="AME-Roban-Depth-Enhance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_roban_depth_enhance:RobanDepthEnhanceEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.roban_depth_enhance_ame_rsl_rl_ppo_cfg:RobanDepthEnhanceAMEPPORunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Roban-Depth-Enhance-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_roban_depth_enhance:RobanDepthEnhanceEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.roban_depth_enhance_ame_rsl_rl_ppo_cfg:RobanDepthEnhanceAMEPPORunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Roban-Depth-Raw-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_roban_depth_enhance:RobanDepthRawEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.roban_depth_enhance_ame_rsl_rl_ppo_cfg:RobanDepthRawAMEPPORunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Roban-Depth-Raw-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_roban_depth_enhance:RobanDepthRawEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.roban_depth_enhance_ame_rsl_rl_ppo_cfg:RobanDepthRawAMEPPORunnerCfg"
        ),
    },
)
