"""Event terms for AME locomotion tasks."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op


def randomize_camera_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg,
    position_range: dict[str, tuple[float, float]],
) -> None:
    """Randomize a camera's position relative to its attached body on reset.

    The configured camera offset is used as the base on every reset, so the
    perturbation does not accumulate across episode resets.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=sensor._device)
    env_ids = env_ids.to(device=sensor._device)

    ranges = torch.tensor(
        [position_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")],
        device=sensor._device,
        dtype=torch.float32,
    )
    position_offset = math_utils.sample_uniform(
        ranges[:, 0],
        ranges[:, 1],
        (len(env_ids), 3),
        device=sensor._device,
    )
    camera_offset_pos = torch.tensor(
        sensor.cfg.offset.pos,
        device=sensor._device,
        dtype=position_offset.dtype,
    ).expand(len(env_ids), -1)
    camera_local_pos = camera_offset_pos + position_offset
    _, camera_local_quat = sensor._view.get_local_poses(indices=env_ids)
    sensor._view.set_local_poses(
        translations=camera_local_pos,
        orientations=camera_local_quat,
        indices=env_ids,
    )


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float],
    operation: str = "add",
) -> None:
    """Randomize nominal joint targets as in LejuLab's Roban task."""
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(asset.data, "default_joint_pos_nominal"):
        asset.data.default_joint_pos_nominal = asset.data.default_joint_pos[0].clone()
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    joint_ids = asset_cfg.joint_ids
    if joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = torch.as_tensor(joint_ids, dtype=torch.long, device=asset.device)
    positions = asset.data.default_joint_pos.clone()
    positions = _randomize_prop_by_op(
        positions,
        pos_distribution_params,
        env_ids,
        joint_ids,
        operation=operation,
        distribution="uniform",
    )
    if isinstance(joint_ids, slice):
        asset.data.default_joint_pos[env_ids, joint_ids] = positions[env_ids, joint_ids]
        action_offset = positions[env_ids]
    else:
        asset.data.default_joint_pos[env_ids[:, None], joint_ids] = positions[env_ids[:, None], joint_ids]
        action_offset = positions[env_ids][:, joint_ids]
    env.action_manager.get_term("joint_pos")._offset[env_ids, :] = action_offset


def apply_external_force_torque_stochastic(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    force_range: dict[str, tuple[float, float]],
    torque_range: dict[str, tuple[float, float]],
    probability: float,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Apply rare per-environment force pulses used by the LejuLab Roban task."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    asset._external_force_b[env_ids] = 0.0
    asset._external_torque_b[env_ids] = 0.0
    selected = env_ids[torch.rand(env_ids.shape, device=env_ids.device) < probability]
    if selected.numel() == 0:
        return
    count = asset.num_bodies if asset_cfg.body_ids == slice(None) else len(asset_cfg.body_ids)
    size = (len(selected), count, 3)
    force_bounds = torch.tensor([force_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")], device=asset.device)
    torque_bounds = torch.tensor([torque_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")], device=asset.device)
    forces = math_utils.sample_uniform(force_bounds[:, 0], force_bounds[:, 1], size, asset.device)
    torques = math_utils.sample_uniform(torque_bounds[:, 0], torque_bounds[:, 1], size, asset.device)
    asset.set_external_force_and_torque(forces, torques, env_ids=selected, body_ids=asset_cfg.body_ids)
