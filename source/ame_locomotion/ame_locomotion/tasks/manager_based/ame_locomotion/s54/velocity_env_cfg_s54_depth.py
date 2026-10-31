"""Kuavo S54 AME task with a 42x42 depth-camera observation."""

import importlib

import math
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from ame_locomotion.tasks.manager_based.ame_locomotion import mdp

from . import velocity_env_cfg_s54 as s54_cfg


_g1_cfg = importlib.import_module(
    "ame_locomotion.tasks.manager_based.ame_locomotion.29dof.velocity_env_cfg_29dof"
)


@configclass
class KuavoS54DepthSceneCfg(_g1_cfg.MySceneCfg):
    """S54 scene with a forward-facing depth camera."""

    depth_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/waist_yaw_link/depth_camera",
        update_period=0.02,
        # The camera is rigidly attached to the robot.  Keep its world pose in
        # sync with each rendered depth image so consumers that unproject the
        # image (e.g. attention visualization) use the current camera pose.
        update_latest_camera_pose=False,  # 可视化的时候才打开，不然增大训练环境的计算量
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
        self.scene.height_scanner = None
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
        self.commands.base_velocity.ranges.lin_vel_x = (0, 1)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # self.events.reset_base.params["pose_range"]["yaw"] = (-math.pi / 2, math.pi / 2)


        self.scene.height_scanner = None
