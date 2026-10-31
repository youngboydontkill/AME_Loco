from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def amp_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    end_effector_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return noise-free body-frame motion features for the AMP discriminator."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_w = asset.data.root_pos_w.unsqueeze(1)
    relative_pos_w = asset.data.body_pos_w[:, end_effector_cfg.body_ids] - root_pos_w
    relative_vel_w = (
        asset.data.body_lin_vel_w[:, end_effector_cfg.body_ids]
        - asset.data.root_lin_vel_w.unsqueeze(1)
        - torch.cross(
            asset.data.root_ang_vel_w.unsqueeze(1).expand_as(relative_pos_w), relative_pos_w, dim=-1
        )
    )
    root_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, len(end_effector_cfg.body_ids), -1).reshape(-1, 4)
    end_effector_pos_b = quat_apply_inverse(root_quat, relative_pos_w.reshape(-1, 3)).view(env.num_envs, -1)
    end_effector_vel_b = quat_apply_inverse(root_quat, relative_vel_w.reshape(-1, 3)).view(env.num_envs, -1)
    return torch.cat(
        (
            asset.data.root_lin_vel_b,
            asset.data.root_ang_vel_b,
            asset.data.projected_gravity_b,
            asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids],
            asset.data.joint_vel[:, asset_cfg.joint_ids],
            end_effector_pos_b,
            end_effector_vel_b,
        ),
        dim=-1,
    )


def ray_hits_s(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Get the 3D coordinates of scan points in the sensor's local coordinate 
    frame.

    This function converts the ray hit points from world coordinates to
    sensor-local coordinates. The returned coordinates represent the relative
    positions of scan points w.r.t. the sensor frame.

    Args:
        env: The environment containing the sensor.
        sensor_cfg: The SceneEntity configuration for the sensor.

    Returns:
        A flattened tensor containing the 3D coordinates of scan points in 
        sensor frame. The original shape (N, B, 3) is flattened to (N*B*3,) 
        for compatibility with other observation terms.
    """
    
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    relative_pos_w = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    sensor_quat = sensor.data.quat_w  # (N, 4)
    N, B, _ = relative_pos_w.shape
   
    # Handle different sensor alignments
    alignment = getattr(sensor.cfg, "ray_alignment", "base")
    if alignment == "yaw":
        sensor_quat = yaw_quat(sensor_quat)

    sensor_quat_expanded = (sensor_quat.unsqueeze(1).expand(N, B, 4).reshape(N*B, 4)).to(torch.float)
    relative_pos_w_reshaped = relative_pos_w.reshape(N*B, 3)
    sensor_coords = quat_apply_inverse(sensor_quat_expanded, relative_pos_w_reshaped)
    sensor_coords = sensor_coords.reshape(N, B, 3)

    # print("ray_hits_s:", torch.round(sensor_coords * 100) / 100)
    # print(sensor_coords.reshape(N, B*3))

    if torch.isnan(sensor_coords).any() or torch.isinf(sensor_coords).any():
        print(f"Warning: ray_hits_s contains NaN or Inf: {sensor_coords}")
        sensor_coords = torch.nan_to_num(sensor_coords)
        
    return sensor_coords.reshape(N, B*3)


def elevation_map(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, noise: bool = False) -> torch.Tensor:
    
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    relative_pos_w = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    sensor_quat = sensor.data.quat_w  # (N, 4)
    N, B, _ = relative_pos_w.shape
   
    # Handle different sensor alignments
    alignment = getattr(sensor.cfg, "ray_alignment", "base")
    if alignment == "yaw":
        sensor_quat = yaw_quat(sensor_quat)

    sensor_quat_expanded = (sensor_quat.unsqueeze(1).expand(N, B, 4).reshape(N*B, 4)).to(torch.float)
    relative_pos_w_reshaped = relative_pos_w.reshape(N*B, 3)
    sensor_coords = quat_apply_inverse(sensor_quat_expanded, relative_pos_w_reshaped)
    sensor_coords = sensor_coords.reshape(N, B, 3)

    if torch.isnan(sensor_coords).any() or torch.isinf(sensor_coords).any():
        # print(f"Warning: elevation_map contains NaN or Inf: {sensor_coords}")
        sensor_coords = torch.nan_to_num(sensor_coords)

    if noise:
        # Initialize buffers for shift and delayed observation
        # if getattr(env, "_elevation_map_shift", None) is None:
        #     env._elevation_map_shift = torch.zeros((env.num_envs, 2), device=env.device)
        if getattr(env, "_elevation_map_offset", None) is None or env._elevation_map_offset.shape != (N, 1):
            env._elevation_map_offset = torch.zeros((N, 1), device=env.device)
        # if getattr(env, "_last_elevation_map", None) is None or env._last_elevation_map.shape != (N, B * 3):
        #     env._last_elevation_map = torch.zeros((N, B * 3), device=env.device)

        # Resample x and y shift for reset environments (-3cm to 3cm)
        if hasattr(env, "reset_buf"):
            reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                # env._elevation_map_shift[reset_env_ids] = torch.rand((len(reset_env_ids), 2), device=env.device) * 0.06 - 0.03
                env._elevation_map_offset[reset_env_ids] = torch.rand((len(reset_env_ids), 1), device=env.device) * 0.1 - 0.05

        # Apply x, y shift
        # sensor_coords[..., :2] += env._elevation_map_shift.unsqueeze(1)

        # Add Gaussian noise to each height value
        height_noise = torch.randn_like(sensor_coords[..., 2]) * 0.03 # std dev of 3 cm
        sensor_coords[..., 2] += height_noise  
        # Add a global offset noise to simulate sensor initialization error
        offset_noise = env._elevation_map_offset
        sensor_coords[..., 2] += offset_noise
        
    # Clip height values
    sensor_coords[..., 2] = torch.clamp(sensor_coords[..., 2], min=-1.2, max=0.0)

    current_map = sensor_coords.reshape(N, B * 3)

    return current_map


def depth_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    min_depth: float = 0.0,
    max_depth: float = 5.0,
    image_size: tuple[int, int] = (42, 42),
) -> torch.Tensor:
    """Return a clipped and normalized depth image as a flat observation.

    Invalid camera pixels are treated as out-of-range pixels and therefore map
    to ``+1`` after normalization. This gives the policy a stable value for
    pixels where the renderer reports NaN or infinity.
    """
    if max_depth <= min_depth:
        raise ValueError(f"max_depth must be greater than min_depth, got {min_depth} and {max_depth}")

    camera = env.scene.sensors[sensor_cfg.name]
    depth = camera.data.output["distance_to_image_plane"]
    if depth.ndim == 4:
        if depth.shape[-1] != 1:
            raise ValueError(f"Expected a single-channel depth image, got shape {tuple(depth.shape)}")
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"Expected depth image with shape (N, H, W), got {tuple(depth.shape)}")
    if tuple(depth.shape[-2:]) != image_size:
        raise ValueError(f"Expected depth image size {image_size}, got {tuple(depth.shape[-2:])}")

    depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=min_depth)
    depth = torch.clamp(depth, min=min_depth, max=max_depth)
    depth = 2.0 * (depth - min_depth) / (max_depth - min_depth) - 1.0
    return depth.reshape(depth.shape[0], -1)
