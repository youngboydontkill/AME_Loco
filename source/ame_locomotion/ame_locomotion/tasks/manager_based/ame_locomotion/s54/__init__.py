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

gym.register(
    id="AME-Kuavo-S54-Depth-enhancement-privileged",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhancement:"
            "KuavoS54DepthEnhancementPrivilegedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_enhancement_ppo_cfg:"
            "KuavoS54DepthEnhancementPrivilegedAMEPPORunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Depth-enhancement-privileged-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhancement:"
            "KuavoS54DepthEnhancementPrivilegedEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_enhancement_ppo_cfg:"
            "KuavoS54DepthEnhancementPrivilegedAMEPPORunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Depth-privileged-critic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhancement:KuavoS54DepthPrivilegedCriticEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_enhancement_ppo_cfg:"
            "KuavoS54DepthPrivilegedCriticAMEPPORunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Depth-privileged-critic-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhancement:KuavoS54DepthPrivilegedCriticEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_enhancement_ppo_cfg:"
            "KuavoS54DepthPrivilegedCriticAMEPPORunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Depth-geometry-enhancement",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhancement:"
            "KuavoS54DepthGeometryEnhancementEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_enhancement_ppo_cfg:"
            "KuavoS54DepthGeometryEnhancementAMEPPORunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Depth-geometry-enhancement-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhancement:"
            "KuavoS54DepthGeometryEnhancementEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_enhancement_ppo_cfg:"
            "KuavoS54DepthGeometryEnhancementAMEPPORunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Depth-student",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhancement:KuavoS54DepthStudentEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_enhancement_ppo_cfg:"
            "KuavoS54DepthStudentAMEPPORunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Depth-student-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhancement:KuavoS54DepthStudentEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_enhancement_ppo_cfg:"
            "KuavoS54DepthStudentAMEPPORunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Depth-sensor-randomized",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhancement:KuavoS54DepthSensorRandomizedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_enhancement_ppo_cfg:"
            "KuavoS54DepthSensorRandomizedAMEPPORunnerCfg"
        ),
    },
)

gym.register(
    id="AME-Kuavo-S54-Depth-sensor-randomized-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.velocity_env_cfg_s54_depth_enhancement:KuavoS54DepthSensorRandomizedEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.kuavo_s54_depth_enhancement_ppo_cfg:"
            "KuavoS54DepthSensorRandomizedAMEPPORunnerCfg"
        ),
    },
)
