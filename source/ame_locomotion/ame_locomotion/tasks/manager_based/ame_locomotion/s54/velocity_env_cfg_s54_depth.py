"""Kuavo S54 AME task with a 42x42 depth-camera observation."""

import copy
import importlib

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from ame_locomotion.tasks.manager_based.ame_locomotion import mdp
from ame_locomotion.tasks.manager_based.ame_locomotion.terrains.loco_hf_terrains_cfg import (
    HfStonesBridgeTerrainCfg,
)

from . import velocity_env_cfg_s54 as s54_cfg


_g1_cfg = importlib.import_module(
    "ame_locomotion.tasks.manager_based.ame_locomotion.29dof.velocity_env_cfg_29dof"
)


def _make_easier_depth_terrain(terrain_generator):
    """Reduce obstacle severity while preserving the terrain mix."""
    terrain_generator = copy.deepcopy(terrain_generator)
    sub_terrains = terrain_generator.sub_terrains

    # Training and PLAY configurations intentionally use different terrain
    # subsets. Apply the common adjustments only when the corresponding
    # terrain exists so that PLAY can keep its compact terrain mix.
    if "pyramid_stairs" in sub_terrains:
        sub_terrains["pyramid_stairs"].step_height_range = (0.03, 0.12)
    if "pyramid_stairs_inv" in sub_terrains:
        sub_terrains["pyramid_stairs_inv"].step_height_range = (0.03, 0.12)
    if "boxes" in sub_terrains:
        sub_terrains["boxes"].grid_width = 0.35
        sub_terrains["boxes"].grid_height_range = (0.03, 0.12)
    if "random_rough" in sub_terrains:
        sub_terrains["random_rough"].noise_range = (0.01, 0.06)
    for name in ("hf_pyramid_slope", "hf_pyramid_slope_inv"):
        if name in sub_terrains:
            sub_terrains[name].slope_range = (0.0, 0.25)
    if "hf_steppingstones" in sub_terrains:
        sub_terrains["hf_steppingstones"].stone_width_range = (0.35, 0.60)
        sub_terrains["hf_steppingstones"].stone_distance_range = (0.03, 0.15)
        sub_terrains["hf_steppingstones"].holes_depth = -1.0
    if "hf_gaps" in sub_terrains:
        sub_terrains["hf_gaps"].gap_width_range = (0.06, 0.25)
        sub_terrains["hf_gaps"].ground_width_range = (0.60, 0.60)
        sub_terrains["hf_gaps"].gap_depth = -1.0

    # Reserve part of the training distribution for the bridge terrain. This
    # is scoped to depth training instead of changing other locomotion tasks.
    if "random_rough" in sub_terrains:
        sub_terrains["random_rough"].proportion = 0.05
    for name in ("hf_pyramid_slope", "hf_pyramid_slope_inv"):
        if name in sub_terrains:
            sub_terrains[name].proportion = 0.05
    if "stonebridge" not in sub_terrains:
        sub_terrains["stonebridge"] = HfStonesBridgeTerrainCfg(
            proportion=0.15,
            stone_height_max=0.03,
            stone_width_range=(0.35, 0.60),
            stone_length_range=(0.60, 1.00),
            stone_distance_range=(0.03, 0.15),
            stone_lateral_distance_range=(0.0, 0.0),
            holes_depth=-1.0,
            platform_width=2.0,
            border_width=0.25,
        )

    return terrain_generator


@configclass
class KuavoS54DepthSceneCfg(_g1_cfg.MySceneCfg):
    """S54 scene with a forward-facing depth camera."""

    depth_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/waist_yaw_link/depth_camera",
        update_period=0.02,
        # Pose refresh performs an extra transform update for every camera on
        # every render. Training only consumes depth pixels, so keep it off in
        # the shared scene config. The PLAY config enables it below for tools
        # that unproject depth pixels into world coordinates.
        update_latest_camera_pose=False,
        height=42,
        width=42,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            # Match the 68-degree vertical FOV used by the source MuJoCo camera.
            focal_length=15.5335,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0987, 0.0, -0.028449),
            rot=(0.624338, 0.331967, -0.331967, -0.624338),
            # This pose comes from a MuJoCo camera, whose optical frame follows
            # OpenGL (-Z forward, +Y up). Treating it as ROS reverses the view
            # direction and makes the camera look into the robot's waist.
            convention="opengl",
        ),
    )


@configclass
class DepthObservationsCfg(_g1_cfg.ObservationsCfg):
    """Policy and critic observations for the depth task."""

    @configclass
    class PolicyCfg(_g1_cfg.ObservationsCfg.PolicyCfg):
        height_scan = None
        depth_image = ObsTerm(
            func=mdp.depth_image,
            params={"sensor_cfg": SceneEntityCfg("depth_camera")},
        )

    @configclass
    class CriticCfg(_g1_cfg.ObservationsCfg.CriticCfg):
        height_scan = None
        depth_image = ObsTerm(
            func=mdp.depth_image,
            params={"sensor_cfg": SceneEntityCfg("depth_camera")},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class KuavoS54DepthRoughEnvCfg(s54_cfg.KuavoS54RoughEnvCfg):
    """Training configuration for AME on Kuavo S54 with depth input."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: DepthObservationsCfg = DepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        if self.observations.critic.height_scan is None:
            self.scene.height_scanner = None
        self.scene.terrain.terrain_generator = _make_easier_depth_terrain(
            self.scene.terrain.terrain_generator
        )
        # Bootstrap the more complex visual policy on the lowest terrain row.
        # Successful environments will still advance through the normal curriculum.
        self.scene.terrain.max_init_terrain_level = 0
        # Do not inject destabilizing reset velocities while the visual policy
        # is learning basic locomotion from level-zero terrain.
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.push_robot = None
        # A wider kernel gives a useful gradient when the initial policy has
        # large velocity error; this matches the successful S54 depth run.
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5


@configclass
class KuavoS54DepthRoughEnvCfg_PLAY(s54_cfg.KuavoS54RoughEnvCfg_PLAY):
    """Playback configuration for AME on Kuavo S54 with depth input."""

    scene: KuavoS54DepthSceneCfg = KuavoS54DepthSceneCfg(num_envs=50, env_spacing=2.5)
    observations: DepthObservationsCfg = DepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        if self.observations.critic.height_scan is None:
            self.scene.height_scanner = None
        self.scene.terrain.terrain_generator = _make_easier_depth_terrain(
            self.scene.terrain.terrain_generator
        )
        self.scene.depth_camera.update_latest_camera_pose = True
