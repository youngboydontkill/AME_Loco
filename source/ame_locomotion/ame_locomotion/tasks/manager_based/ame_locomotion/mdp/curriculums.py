"""Curriculum helpers for AME locomotion tasks."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter


def terrain_progress_diagnostics(
    env,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """Report the quantities used by :func:`terrain_levels_vel`.

    This term deliberately does not modify terrain origins.  It is placed before
    the actual terrain curriculum term so the reported distance and level are
    the pre-update values for the environments that are being reset.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    required_distance = torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = (distance < required_distance) & ~move_up

    # CurriculumManager accepts tensors in a returned dictionary and logs each
    # key as Curriculum/terrain_progress_diagnostics/<key>.
    return {
        "distance_mean": distance.mean(),
        "distance_max": distance.max(),
        "required_distance_mean": required_distance.mean(),
        "move_up_fraction": move_up.float().mean(),
        "move_down_fraction": move_down.float().mean(),
        "level_mean": terrain.terrain_levels[env_ids].float().mean(),
    }
