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
    id="AME-Kuavo-S54-Depth-Play-Enhance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhance:KuavoS54DepthEnhanceEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_ame_rsl_rl_ppo_cfg:KuavoS54DepthAMEPPORunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Enhance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhance:KuavoS54EnhanceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_enhance_ame_rsl_rl_ppo_cfg:"
            "KuavoS54EnhanceAMEPPORunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Depth-Enhance-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhance:KuavoS54DepthEnhanceAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_amp_rsl_rl_ppo_cfg:KuavoS54DepthAMPPOCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Depth-Enhance-AMP-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhance:KuavoS54DepthEnhanceAMPEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_ame_rsl_rl_ppo_cfg:KuavoS54DepthAMEPPORunnerCfg"
        ),
    },
)

gym.register(
    id="GRU-Kuavo-S54-Depth-Enhance-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhance:KuavoS54DepthEnhanceGRUAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_gru_amp_rsl_rl_ppo_cfg:KuavoS54DepthGRUAMPPOCfg"
        ),
    },
)

gym.register(
    id="GRU-Kuavo-S54-Depth-Enhance-AMP-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhance:KuavoS54DepthEnhanceGRUAMPEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_gru_amp_rsl_rl_ppo_cfg:KuavoS54DepthGRUPlayRunnerCfg"
        ),
    },
)

gym.register(
    id="GRU-Kuavo-S54-Depth-AMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhance:KuavoS54DepthGRURawAMPEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_gru_amp_rsl_rl_ppo_cfg:KuavoS54DepthGRURawAMPPOCfg"
        ),
    },
)

gym.register(
    id="GRU-Kuavo-S54-Depth-AMP-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhance:KuavoS54DepthGRURawAMPEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_gru_amp_rsl_rl_ppo_cfg:KuavoS54DepthGRURawPlayRunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Depth-Enhance-DAgger-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhance:KuavoS54DepthDistillEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_distill_cfg:KuavoS54DepthDistillRunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Depth-Enhance-DAgger-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhance:KuavoS54DepthDistillEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_distill_cfg:KuavoS54DepthDistillRunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Depth-DAgger-Raw-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhance:KuavoS54DepthRawDistillEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_distill_cfg:KuavoS54DepthDistillRunnerCfg"
        ),
    },
)
