"""S54 depth student environment with a privileged height-scan teacher observation."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ame_locomotion.tasks.manager_based.ame_locomotion import mdp

from .velocity_env_cfg_s54 import KuavoS54RoughEnvCfg, KuavoS54RoughEnvCfg_PLAY
from .velocity_env_cfg_s54_depth import (
    DepthObservationsCfg,
    KuavoS54DepthSceneCfg,
    _make_easier_depth_terrain,
)


@configclass
class DistillationObservationsCfg(DepthObservationsCfg):
    """Depth policy observations plus the original height-scan teacher input."""

    @configclass
    class TeacherCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.elevation_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": False},
        )
        concatenate_terms = True

    teacher: TeacherCfg = TeacherCfg()


@configclass
class KuavoS54DepthDistillEnvCfg(KuavoS54RoughEnvCfg):
    """Train a depth student against a frozen S54 height-scan teacher."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: DistillationObservationsCfg = DistillationObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = _make_easier_depth_terrain(self.scene.terrain.terrain_generator)
        self.scene.terrain.max_init_terrain_level = 0
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.push_robot = None
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5


@configclass
class KuavoS54DepthDistillEnvCfg_PLAY(KuavoS54RoughEnvCfg_PLAY):
    """Playback configuration for a trained depth student."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=50, env_spacing=2.5)
    observations: DistillationObservationsCfg = DistillationObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
