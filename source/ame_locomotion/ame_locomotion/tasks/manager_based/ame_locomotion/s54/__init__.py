import gymnasium as gym

from ame_locomotion.tasks.manager_based.ame_locomotion import agents


gym.register(
    id="AME-Kuavo-S54-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_s54:KuavoS54RoughEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_ame_rsl_rl_ppo_cfg:KuavoS54AMEPPORunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_s54:KuavoS54RoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_ame_rsl_rl_ppo_cfg:KuavoS54AMEPPORunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Depth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_s54_depth:KuavoS54DepthRoughEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_ame_rsl_rl_ppo_cfg:KuavoS54DepthAMEPPORunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Depth-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_s54_depth:KuavoS54DepthRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_ame_rsl_rl_ppo_cfg:KuavoS54DepthAMEPPORunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Depth-Distill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_s54_depth_distill:KuavoS54DepthDistillEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_distill_cfg:KuavoS54DepthDistillRunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Depth-Distill-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_distill:KuavoS54DepthDistillEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_distill_cfg:KuavoS54DepthDistillRunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Depth-enhancement",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhancement:KuavoS54DepthEnhancementEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_enhancement_ppo_cfg:"
            "KuavoS54DepthEnhancementAMEPPORunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Depth-enhancement-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhancement:"
            "KuavoS54DepthEnhancementEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_enhancement_ppo_cfg:"
            "KuavoS54DepthEnhancementAMEPPORunnerCfg"
        ),
    },
)
