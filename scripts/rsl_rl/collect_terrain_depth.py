# Copyright (c) 2026 AME locomotion contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Collect high-resolution depth frames while the S54 policy traverses one terrain."""

import argparse
import copy
import json
import sys
from dataclasses import fields
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "logs/rsl_rl/kuavo_s54_enhance/2026-10-28_10-26-57/model_51000.pt"
DEFAULT_OUTPUT = REPO_ROOT / "depth_data"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--terrain", type=str, default="random_rough", help="Sub-terrain name to generate.")
parser.add_argument("--list-terrains", action="store_true", help="List supported terrain names and exit.")
parser.add_argument(
    "--show-terrain-params",
    action="store_true",
    help="Print the selected terrain's configured parameters and exit.",
)
parser.add_argument(
    "--terrain-params",
    type=str,
    default="{}",
    metavar="JSON",
    help='Override any supported geometry fields, e.g. \'{"noise_range":[0.03,0.12]}\'.',
)
parser.add_argument(
    "--stone-distance-range",
    type=float,
    nargs=2,
    metavar=("MIN_M", "MAX_M"),
    help="Override stepping-stone center spacing range in meters, e.g. 0.10 0.30.",
)
parser.add_argument(
    "--stone-width-range",
    type=float,
    nargs=2,
    metavar=("MIN_M", "MAX_M"),
    help="Override stepping-stone width range in meters, e.g. 0.20 0.50.",
)
parser.add_argument(
    "--stone-height-max",
    type=float,
    help="Override maximum stepping-stone height in meters.",
)
parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps to collect.")
parser.add_argument("--save-interval", type=int, default=1, help="Save one frame every N simulation steps.")
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

from ame_locomotion.tasks.manager_based.ame_locomotion.terrains.finetune_terrain_cfg import (
    FINETUNE_ROUGH_TERRAINS_CFG,
)
from ame_locomotion.tasks.manager_based.ame_locomotion.terrains.terrain_cfg import ROUGH_TERRAINS_CFG

TERRAIN_SOURCES = (ROUGH_TERRAINS_CFG, FINETUNE_ROUGH_TERRAINS_CFG)


def get_terrain_template(name: str):
    return next((source.sub_terrains[name] for source in TERRAIN_SOURCES if name in source.sub_terrains), None)


if args_cli.list_terrains:
    for terrain_name in sorted(set(ROUGH_TERRAINS_CFG.sub_terrains) | set(FINETUNE_ROUGH_TERRAINS_CFG.sub_terrains)):
        print(terrain_name)
    raise SystemExit(0)

if args_cli.show_terrain_params:
    template = get_terrain_template(args_cli.terrain)
    if template is None:
        raise SystemExit(f"Unknown terrain {args_cli.terrain!r}; use --list-terrains to see available names.")
    for field in fields(template):
        if field.name != "function":
            print(f"{field.name}={getattr(template, field.name)!r}")
    raise SystemExit(0)

sys.argv = [sys.argv[0]] + hydra_args
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.sensors import TiledCameraCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import ame_locomotion.tasks  # noqa: F401


@hydra_task_config("AME-Kuavo-S54-Depth-Play-v0", "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    if args_cli.steps < 1 or args_cli.save_interval < 1:
        raise ValueError("--steps and --save-interval must be positive")
    if not args_cli.checkpoint.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {args_cli.checkpoint}")

    terrain_cfg = env_cfg.scene.terrain.terrain_generator
    terrain_template = get_terrain_template(args_cli.terrain)
    if terrain_template is None:
        known = sorted(set(ROUGH_TERRAINS_CFG.sub_terrains) | set(FINETUNE_ROUGH_TERRAINS_CFG.sub_terrains))
        raise ValueError(f"Unknown terrain {args_cli.terrain!r}. Available: {', '.join(known)}")

    terrain_template = copy.deepcopy(terrain_template)
    geometry_overrides = {
        "stone_distance_range": args_cli.stone_distance_range,
        "stone_width_range": args_cli.stone_width_range,
        "stone_height_max": args_cli.stone_height_max,
    }
    try:
        custom_overrides = json.loads(args_cli.terrain_params)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--terrain-params must be a JSON object: {exc}") from exc
    if not isinstance(custom_overrides, dict):
        raise ValueError("--terrain-params must be a JSON object")
    geometry_overrides.update(custom_overrides)
    for parameter, value in geometry_overrides.items():
        if value is None:
            continue
        if not hasattr(terrain_template, parameter):
            raise ValueError(f"Terrain {args_cli.terrain!r} does not support parameter {parameter!r}.")
        if isinstance(value, list):
            if len(value) != 2 or not all(isinstance(item, (int, float)) for item in value):
                raise ValueError(f"{parameter} must be a numeric pair, got {value!r}")
            if value[1] < value[0]:
                raise ValueError(f"Invalid {parameter}: maximum must be >= minimum, got {value}")
            value = tuple(value)
        elif isinstance(value, (int, float)) and value < 0 and parameter not in {"holes_depth", "gap_depth"}:
            raise ValueError(f"Invalid {parameter}: expected a non-negative value, got {value}")
        setattr(terrain_template, parameter, value)

    terrain_cfg.sub_terrains = {args_cli.terrain: terrain_template}
    terrain_cfg.num_rows = max(terrain_cfg.num_rows, 1)
    terrain_cfg.num_cols = max(terrain_cfg.num_cols, 1)
    env_cfg.scene.num_envs = 1
    env_cfg.scene.terrain.max_init_terrain_level = terrain_cfg.num_rows - 1
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device

    # Keep the policy's 42x42 camera and attach an independent 640x480 archive camera.
    env_cfg.scene.depth_capture_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/waist_yaw_link/depth_capture_camera",
        update_period=0.02,
        update_latest_camera_pose=False,  # 可视化的时候才打开，不然增大训练环境的计算量
        height=480,
        width=640,
        data_types=["distance_to_image_plane"],
        spawn=env_cfg.scene.depth_camera.spawn,
        offset=copy.deepcopy(env_cfg.scene.depth_camera.offset),
    )

    raw_env = gym.make("AME-Kuavo-S54-Depth-Play-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    output_dir = args_cli.output.expanduser().resolve() / args_cli.terrain
    output_dir.mkdir(parents=True, exist_ok=True)
    obs = env.get_observations()
    camera = env.unwrapped.scene.sensors["depth_capture_camera"]
    saved = 0
    try:
        for step in range(args_cli.steps):
            with torch.inference_mode():
                result = policy(obs)
                actions = result[0] if isinstance(result, (tuple, list)) else result
                obs, *_ = env.step(actions)

                if (step + 1) % args_cli.save_interval == 0:
                    depth = camera.data.output["distance_to_image_plane"][0]
                    if depth.ndim == 3 and depth.shape[-1] == 1:
                        depth = depth[..., 0]
                    frame = torch.nan_to_num(depth, nan=5.0, posinf=5.0, neginf=0.0)
                    frame = frame.clamp(0.0, 5.0).to(torch.float32).cpu().numpy()
                    if frame.shape != (480, 640):
                        raise RuntimeError(f"Expected a 480x640 depth frame, got {frame.shape}")
                    saved += 1
                    np.save(output_dir / f"frame_{saved:06d}_{args_cli.terrain}_depth.npy", frame)

            if step % 100 == 0:
                print(f"[INFO] step {step + 1}/{args_cli.steps}, saved {saved} frames")

    finally:
        env.close()
    print(f"[INFO] Saved {saved} float32 depth frames to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
