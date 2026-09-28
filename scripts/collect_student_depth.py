#!/usr/bin/env python3
"""Collect raw and Student-enhanced depth images from Isaac Lab scenes.

The script creates a vectorized S54 depth environment, advances it with zero
actions, and writes one sample for every environment at each capture step.
Each sample contains the raw camera depth, the frozen Student output in metres,
their difference, and a side-by-side PNG visualization.

新增采集脚本：

  - /home/hitcsc/YX/AME_LCOO/scripts/collect_student_depth.py
  - /home/hitcsc/YX/AME_LCOO/doc/student_depth_collection.md

  脚本功能：

  - 默认任务：AME-Kuavo-S54-Depth-enhancement
  - 默认展开 S54 的 8 类 terrain
  - 采集原始相机深度、Student 深度、差值和有效像素 mask
  conda activate isaacsim5.1

  python scripts/collect_student_depth.py \
    --num_envs 64 \
    --steps 200 \
    --capture_interval 10 \
    --max_captures 5 \
    --headless \
    --output_dir outputs/student_depth

  可视化图位于：

  outputs/student_depth/env_0000/step_000010_comparison.png

  图像从左到右为：原始深度、Student 深度、Student - raw 差值。
已改为用你指定的 model_62000.pt 控制机器人：策略每步根据观测生成关节动作，环境中的前进指令固定为 0.5 m/s。采集任务默认使用该策略对应的原始深度 S54 任务；Student 仍单独负责生成并保存增强深度图。

脚本支持用 --policy_checkpoint 更换策略、用 --forward_speed 调整速度，並会把 checkpoint 路径和速度写入 manifest。
文档中的说明和命令也已同步更新，见 scripts/collect_student_depth.py 和 doc/student_depth_collection.md。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

# Keep local robot assets importable when this file is launched as
# ``python scripts/collect_student_depth.py`` rather than as a module.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "source"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    type=str,
    default="AME-Kuavo-S54-Depth-v0",
    help="Registered Isaac Lab task. The raw-depth S54 task matches the locomotion policy by default.",
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel scene instances.")
parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps to run.")
parser.add_argument(
    "--capture_interval",
    type=int,
    default=10,
    help="Capture every N simulation steps (the first capture is after N steps).",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="outputs/student_depth",
    help="Directory for arrays, PNGs, and manifest.json.",
)
parser.add_argument(
    "--max_captures",
    type=int,
    default=None,
    help="Optional maximum number of capture rounds. By default all requested steps are used.",
)
parser.add_argument(
    "--all_terrains",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Generate one terrain-generator column per S54 terrain type (default: enabled).",
)
parser.add_argument(
    "--terrain_rows",
    type=int,
    default=2,
    help="Number of difficulty rows when --all_terrains is enabled.",
)
parser.add_argument("--terrain_seed", type=int, default=1234, help="Seed for terrain generation.")
parser.add_argument(
    "--policy_checkpoint",
    type=str,
    default="logs/rsl_rl/kuavo_s54_depth_ame/2026-09-21_16-53-12/model_62000.pt",
    help="RSL-RL locomotion policy checkpoint used to control the robot.",
)
parser.add_argument(
    "--forward_speed",
    type=float,
    default=0.5,
    help="Constant forward velocity command in m/s.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# TiledCamera sensors are required even for headless collection.
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import ame_locomotion.tasks  # noqa: E402,F401
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from ame_locomotion.tasks.manager_based.ame_locomotion.s54.velocity_env_cfg_s54_depth import (  # noqa: E402
    _make_easier_depth_terrain,
)
from ame_locomotion.tasks.manager_based.ame_locomotion.mdp.student_depth import (  # noqa: E402
    student_depth_image,
)
from ame_locomotion.tasks.manager_based.ame_locomotion.terrains.terrain_cfg import (  # noqa: E402
    ROUGH_TERRAINS_CFG,
)
from ame_locomotion.tasks.manager_based.ame_locomotion.agents.kuavo_s54_depth_ame_rsl_rl_ppo_cfg import (  # noqa: E402
    KuavoS54DepthAMEPPORunnerCfg,
)


IMAGE_SIZE = (42, 42)
MIN_DEPTH = 0.05
MAX_DEPTH = 5.0


def _as_depth_image(depth: torch.Tensor) -> np.ndarray:
    """Convert camera output to a CPU float array with shape [N, 42, 42]."""
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3 or tuple(depth.shape[-2:]) != IMAGE_SIZE:
        raise ValueError(f"Expected camera depth [N, 42, 42], got {tuple(depth.shape)}")
    return depth.detach().float().cpu().numpy()


def _terrain_metadata(env, env_id: int) -> dict:
    """Read optional terrain identifiers without depending on a terrain implementation."""
    terrain = getattr(env.scene, "terrain", None)
    metadata = {}
    for name in ("terrain_levels", "terrain_types"):
        values = getattr(terrain, name, None) if terrain is not None else None
        if values is not None:
            value = values[env_id]
            metadata[name.removeprefix("terrain_")] = int(value.item() if hasattr(value, "item") else value)

    if "type" in metadata and terrain is not None:
        generator_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        names = list(getattr(generator_cfg, "sub_terrains", {}).keys())
        if 0 <= metadata["type"] < len(names):
            metadata["type_name"] = names[metadata["type"]]

    origins = getattr(terrain, "env_origins", None) if terrain is not None else None
    if origins is not None:
        origin = origins[env_id].detach().float().cpu().tolist()
        metadata["origin"] = [float(item) for item in origin]
    return metadata


def _colorize(depth: np.ndarray, invalid: np.ndarray | None = None) -> np.ndarray:
    """Make a compact blue-to-yellow depth image without a matplotlib dependency."""
    clipped = np.clip(np.nan_to_num(depth, nan=MAX_DEPTH, posinf=MAX_DEPTH, neginf=MIN_DEPTH), MIN_DEPTH, MAX_DEPTH)
    value = (clipped - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
    # A simple perceptual-ish blue -> cyan -> yellow ramp.
    stops = np.asarray(
        [[35, 35, 150], [30, 170, 220], [245, 235, 60]], dtype=np.float32
    )
    scaled = value * (len(stops) - 1)
    low = np.floor(scaled).astype(np.int32).clip(0, len(stops) - 2)
    high = low + 1
    fraction = (scaled - low)[..., None]
    rgb = stops[low] * (1.0 - fraction) + stops[high] * fraction
    if invalid is not None:
        rgb[np.asarray(invalid, dtype=bool)] = (255, 255, 255)
    return np.asarray(rgb, dtype=np.uint8)


def _comparison_png(raw: np.ndarray, student: np.ndarray, difference: np.ndarray) -> np.ndarray:
    """Return an RGB image containing raw, Student, and signed difference panels."""
    from PIL import Image, ImageDraw

    raw_invalid = ~np.isfinite(raw) | (raw < MIN_DEPTH) | (raw > 4.9)
    raw_rgb = _colorize(raw, raw_invalid)
    student_rgb = _colorize(student)
    diff_scale = max(0.05, float(np.percentile(np.abs(difference), 99)))
    diff_value = np.clip(difference / diff_scale, -1.0, 1.0)
    diff_rgb = np.empty((*difference.shape, 3), dtype=np.uint8)
    positive = np.clip(diff_value, 0.0, 1.0)[..., None]
    negative = np.clip(-diff_value, 0.0, 1.0)[..., None]
    diff_rgb[:] = 245
    diff_rgb = np.where(positive > 0, diff_rgb * (1.0 - positive) + np.array([210, 50, 45]) * positive, diff_rgb)
    diff_rgb = np.where(negative > 0, diff_rgb * (1.0 - negative) + np.array([45, 80, 210]) * negative, diff_rgb)
    diff_rgb = np.asarray(diff_rgb, dtype=np.uint8)

    panels = [Image.fromarray(panel).resize((252, 252), Image.Resampling.NEAREST) for panel in (raw_rgb, student_rgb, diff_rgb)]
    canvas = Image.new("RGB", (252 * 3, 280), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(zip(("raw depth (m)", "Student depth (m)", "Student - raw (m)"), panels)):
        x = index * 252
        canvas.paste(panel, (x, 28))
        draw.text((x + 5, 7), title, fill="black")
    return np.asarray(canvas)


def _save_capture(output_dir: Path, env, env_id: int, step: int, camera_cfg: SceneEntityCfg) -> dict:
    camera = env.scene.sensors[camera_cfg.name]
    raw = _as_depth_image(camera.data.output["distance_to_image_plane"])
    # This is the exact observation function used by the enhancement task.
    student_normalized = student_depth_image(env, camera_cfg)
    student = ((student_normalized.reshape(-1, *IMAGE_SIZE) + 1.0) * (MAX_DEPTH / 2.0)).detach().float().cpu().numpy()
    raw_valid = np.isfinite(raw) & (raw >= MIN_DEPTH) & (raw <= 4.9)
    raw_for_difference = np.where(raw_valid, raw, 4.9)
    difference = student - raw_for_difference

    env_dir = output_dir / f"env_{env_id:04d}"
    env_dir.mkdir(parents=True, exist_ok=True)
    stem = env_dir / f"step_{step:06d}"
    np.save(f"{stem}_raw.npy", raw[env_id].astype(np.float32))
    np.save(f"{stem}_raw_valid.npy", raw_valid[env_id].astype(np.bool_))
    np.save(f"{stem}_student.npy", student[env_id].astype(np.float32))
    np.save(f"{stem}_difference.npy", difference[env_id].astype(np.float32))
    from PIL import Image

    Image.fromarray(_comparison_png(raw[env_id], student[env_id], difference[env_id])).save(f"{stem}_comparison.png")
    return {
        "env_id": env_id,
        "step": step,
        "raw_file": str(stem.relative_to(output_dir).with_name(stem.name + "_raw.npy")),
        "student_file": str(stem.relative_to(output_dir).with_name(stem.name + "_student.npy")),
        "raw_valid_file": str(stem.relative_to(output_dir).with_name(stem.name + "_raw_valid.npy")),
        "difference_file": str(stem.relative_to(output_dir).with_name(stem.name + "_difference.npy")),
        "comparison_file": str(stem.relative_to(output_dir).with_name(stem.name + "_comparison.png")),
        "raw_min": float(np.nanmin(raw[env_id])) if np.isfinite(raw[env_id]).any() else None,
        "raw_max": float(np.nanmax(raw[env_id])) if np.isfinite(raw[env_id]).any() else None,
        "student_min": float(np.min(student[env_id])),
        "student_max": float(np.max(student[env_id])),
        **_terrain_metadata(env, env_id),
    }


def main() -> None:
    if args_cli.steps < 1 or args_cli.capture_interval < 1 or args_cli.num_envs < 1:
        raise ValueError("--steps, --capture_interval, and --num_envs must be positive")
    if args_cli.all_terrains and args_cli.terrain_rows < 1:
        raise ValueError("--terrain_rows must be positive")

    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.forward_speed, args_cli.forward_speed)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    if args_cli.all_terrains:
        if getattr(env_cfg.scene.terrain, "terrain_type", None) != "generator":
            raise ValueError("--all_terrains requires a generator-based terrain scene")
        terrain_generator = _make_easier_depth_terrain(copy.deepcopy(ROUGH_TERRAINS_CFG))
        terrain_generator.curriculum = True
        terrain_generator.seed = args_cli.terrain_seed
        terrain_generator.num_rows = args_cli.terrain_rows
        # Curriculum mode maps columns to sub-terrain types according to their
        # proportions; one column per type makes all eight S54 terrains visible.
        terrain_generator.num_cols = len(terrain_generator.sub_terrains)
        env_cfg.scene.terrain.terrain_generator = terrain_generator
        env_cfg.scene.terrain.max_init_terrain_level = args_cli.terrain_rows - 1
        if args_cli.num_envs < terrain_generator.num_cols:
            raise ValueError(
                f"--num_envs must be at least {terrain_generator.num_cols} for --all_terrains"
            )
    policy_checkpoint = Path(args_cli.policy_checkpoint).expanduser().resolve()
    if not policy_checkpoint.is_file():
        raise FileNotFoundError(f"Locomotion policy checkpoint not found: {policy_checkpoint}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    num_envs = int(getattr(base_env, "num_envs", args_cli.num_envs))
    runner_cfg = KuavoS54DepthAMEPPORunnerCfg()
    env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=base_env.device)
    runner.load(str(policy_checkpoint))
    policy = runner.get_inference_policy(device=base_env.device)
    print(f"[INFO] locomotion policy: {policy_checkpoint}", flush=True)
    print(f"[INFO] forward command: {args_cli.forward_speed:.2f} m/s", flush=True)
    camera_cfg = SceneEntityCfg("depth_camera")
    manifest = {
        "task": args_cli.task,
        "num_envs": num_envs,
        "steps": int(args_cli.steps),
        "capture_interval": int(args_cli.capture_interval),
        "policy_checkpoint": str(policy_checkpoint),
        "commanded_forward_speed_mps": float(args_cli.forward_speed),
        "depth_range_m": [MIN_DEPTH, MAX_DEPTH],
        "all_terrains": bool(args_cli.all_terrains),
        "terrain_rows": int(args_cli.terrain_rows) if args_cli.all_terrains else None,
        "terrain_seed": int(args_cli.terrain_seed) if args_cli.all_terrains else None,
        "samples": [],
    }

    try:
        obs = env.get_observations()
        capture_count = 0
        for step in range(1, args_cli.steps + 1):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                policy_output = policy(obs)
                actions = policy_output[0] if isinstance(policy_output, (tuple, list)) else policy_output
                obs, _, _, _ = env.step(actions)
            if step % args_cli.capture_interval != 0:
                continue
            for env_id in range(num_envs):
                manifest["samples"].append(_save_capture(output_dir, base_env, env_id, step, camera_cfg))
            capture_count += 1
            print(f"[INFO] captured step {step}: {num_envs} environments", flush=True)
            if args_cli.max_captures is not None and capture_count >= args_cli.max_captures:
                break
    finally:
        env.close()
        manifest["num_samples"] = len(manifest["samples"])
        with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=True, indent=2)
        print(f"[INFO] wrote {manifest['num_samples']} samples to {output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
