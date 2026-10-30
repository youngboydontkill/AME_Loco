"""Event terms for AME locomotion tasks."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


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
