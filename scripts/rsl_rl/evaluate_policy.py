"""Run a fixed number of locomotion episodes and report outcome metrics."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Prefer IsaacSim's C++ runtime when this script is run from a conda env.
if os.environ.get("CONDA_PREFIX"):
    os.environ["LD_LIBRARY_PATH"] = os.environ["CONDA_PREFIX"] + ":" + os.environ.get("LD_LIBRARY_PATH", "")

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--episodes", type=int, default=128)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--group", type=str, default=None, help="Comparison group label for seed aggregation.")
parser.add_argument("--output", required=True)
parser.add_argument("--recovery_eval", action="store_true", help="Apply a fixed base-velocity perturbation and measure recovery.")
parser.add_argument("--perturbation_step", type=int, default=100, help="Episode control step at which to perturb the base.")
parser.add_argument("--lateral_velocity_kick", type=float, default=0.8, help="World-frame x velocity increment in m/s.")
parser.add_argument("--recovery_error_threshold", type=float, default=0.35, help="Planar body-frame command tracking error threshold in m/s.")
parser.add_argument("--recovery_hold_seconds", type=float, default=0.5, help="Continuous time below threshold required for recovery.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint is required")
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[evaluate_policy] IsaacSim app launched", flush=True)

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import ame_locomotion.tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def _latest_forces(sensor_data: torch.Tensor) -> torch.Tensor:
    while sensor_data.ndim > 3:
        sensor_data = sensor_data[:, -1]
    return sensor_data


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    print(f"[evaluate_policy] configuring task={args_cli.task} episodes={args_cli.episodes} num_envs={args_cli.num_envs}", flush=True)
    if args_cli.episodes <= 0 or args_cli.num_envs <= 0:
        raise ValueError("episodes and num_envs must be positive")
    if args_cli.perturbation_step < 1 or args_cli.lateral_velocity_kick <= 0:
        raise ValueError("recovery perturbation step and velocity kick must be positive")
    if args_cli.recovery_error_threshold <= 0 or args_cli.recovery_hold_seconds <= 0:
        raise ValueError("recovery threshold and hold time must be positive")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    agent_cfg.seed = args_cli.seed

    print("[evaluate_policy] creating environment", flush=True)
    base_env = gym.make(args_cli.task, cfg=env_cfg)
    print("[evaluate_policy] environment created", flush=True)
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations()

        raw_env = env.unwrapped
        contact = raw_env.scene.sensors.get("contact_forces")
        robot = raw_env.scene["robot"]
        body_names = list(robot.body_names)
        contact_body_names = list(contact.body_names) if contact is not None else []
        feet = [
            i for i, name in enumerate(contact_body_names)
            if re.search(r"leg_[lr]6_link$", name)
        ]
        excluded = re.compile(r"leg_[lr][56]_link$")
        collision_bodies = [
            i for i, name in enumerate(contact_body_names) if not excluded.search(name)
        ]
        robot_feet = [
            body_names.index(contact_body_names[index])
            for index in feet if contact_body_names[index] in body_names
        ]
        if contact is None or not feet or not collision_bodies or len(robot_feet) != len(feet):
            raise RuntimeError("Could not resolve contact sensor, foot bodies, or collision bodies")

        device = env.unwrapped.device
        episode_return = torch.zeros(args_cli.num_envs, device=device)
        episode_length = torch.zeros(args_cli.num_envs, device=device, dtype=torch.long)
        collision_seen = torch.zeros(args_cli.num_envs, device=device, dtype=torch.bool)
        foot_contact_steps = torch.zeros(args_cli.num_envs, device=device)
        foot_slip_steps = torch.zeros(args_cli.num_envs, device=device)
        perturbation_applied = torch.zeros(args_cli.num_envs, device=device, dtype=torch.bool)
        recovery_stable_steps = torch.zeros(args_cli.num_envs, device=device, dtype=torch.long)
        recovery_time = torch.full((args_cli.num_envs,), -1.0, device=device)
        control_dt = float(raw_env.step_dt)
        recovery_hold_steps = max(1, int(np.ceil(args_cli.recovery_hold_seconds / control_dt)))
        records: list[dict[str, float | int | bool]] = []

        while len(records) < args_cli.episodes:
            if args_cli.recovery_eval:
                kick_ids = torch.nonzero(
                    (episode_length == args_cli.perturbation_step) & ~perturbation_applied,
                    as_tuple=False,
                ).flatten()
                if kick_ids.numel() > 0:
                    root_velocity = torch.cat(
                        (
                            robot.data.root_lin_vel_w[kick_ids].clone(),
                            robot.data.root_ang_vel_w[kick_ids].clone(),
                        ),
                        dim=-1,
                    )
                    root_velocity[:, 0] += args_cli.lateral_velocity_kick
                    robot.write_root_velocity_to_sim(root_velocity, env_ids=kick_ids)
                    perturbation_applied[kick_ids] = True

            with torch.inference_mode():
                action = policy(obs)
                if isinstance(action, (tuple, list)):
                    action = action[0]
                obs, reward, done, infos = env.step(action)

            episode_return += reward
            episode_length += 1
            forces = _latest_forces(contact.data.net_forces_w)
            force_norm = torch.linalg.vector_norm(forces, dim=-1)
            collision_seen |= (force_norm[:, collision_bodies] > 1.0).any(dim=1)
            foot_contact = force_norm[:, feet] > 1.0
            foot_speed = torch.linalg.vector_norm(
                robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1
            )
            foot_contact_steps += foot_contact.sum(dim=1)
            foot_slip_steps += (foot_contact & (foot_speed > 0.1)).sum(dim=1)

            if args_cli.recovery_eval:
                command = raw_env.command_manager.get_command("base_velocity")[:, :2]
                tracking_error = torch.linalg.vector_norm(
                    robot.data.root_lin_vel_b[:, :2] - command, dim=1
                )
                recovering = perturbation_applied & (recovery_time < 0.0)
                stable = recovering & (tracking_error <= args_cli.recovery_error_threshold)
                recovery_stable_steps = torch.where(stable, recovery_stable_steps + 1, torch.zeros_like(recovery_stable_steps))
                recovered = stable & (recovery_stable_steps >= recovery_hold_steps)
                recovery_time[recovered] = (
                    (episode_length[recovered] - args_cli.perturbation_step) * control_dt
                )

            done_ids = torch.nonzero(done, as_tuple=False).flatten()
            if done_ids.numel() == 0:
                continue
            timeouts = infos.get("time_outs")
            if timeouts is None:
                timeouts = torch.zeros_like(done, dtype=torch.bool)
            else:
                timeouts = torch.as_tensor(timeouts, device=device).bool()
            remaining = args_cli.episodes - len(records)
            for env_id in done_ids[:remaining].tolist():
                contact_steps = float(foot_contact_steps[env_id].item())
                records.append({
                    "return": float(episode_return[env_id].item()),
                    "length_steps": int(episode_length[env_id].item()),
                    "success_timeout": bool(timeouts[env_id].item()),
                    "fall_non_timeout": not bool(timeouts[env_id].item()),
                    "nonfoot_collision": bool(collision_seen[env_id].item()),
                    "foot_slip_fraction": (
                        float(foot_slip_steps[env_id].item() / contact_steps)
                        if contact_steps > 0 else 0.0
                    ),
                    "perturbation_applied": bool(perturbation_applied[env_id].item()),
                    # A threshold crossing before a later fall is not recovery.
                    # Require the perturbed episode to finish by timeout.
                    "recovery_success": bool(
                        recovery_time[env_id].item() >= 0.0 and timeouts[env_id].item()
                    ),
                    "recovery_time_seconds": (
                        float(recovery_time[env_id].item()) if recovery_time[env_id].item() >= 0.0 else None
                    ),
                })
            reset_ids = done_ids
            episode_return[reset_ids] = 0.0
            episode_length[reset_ids] = 0
            collision_seen[reset_ids] = False
            foot_contact_steps[reset_ids] = 0.0
            foot_slip_steps[reset_ids] = 0.0
            perturbation_applied[reset_ids] = False
            recovery_stable_steps[reset_ids] = 0
            recovery_time[reset_ids] = -1.0

        summary = {
            "episodes": len(records),
            "return": _summary([float(row["return"]) for row in records]),
            "length_steps": _summary([float(row["length_steps"]) for row in records]),
            "success_rate": float(np.mean([row["success_timeout"] for row in records])),
            "fall_rate": float(np.mean([row["fall_non_timeout"] for row in records])),
            "collision_rate": float(np.mean([row["nonfoot_collision"] for row in records])),
            "foot_slip_fraction": _summary([float(row["foot_slip_fraction"]) for row in records]),
        }
        if args_cli.recovery_eval:
            perturbed_records = [row for row in records if row["perturbation_applied"]]
            recovered_records = [row for row in perturbed_records if row["recovery_success"]]
            summary["recovery_episodes"] = len(perturbed_records)
            summary["recovery_rate"] = (
                len(recovered_records) / len(perturbed_records) if perturbed_records else 0.0
            )
            summary["recovery_time_seconds"] = _summary(
                [float(row["recovery_time_seconds"]) for row in recovered_records]
            ) if recovered_records else {"mean": None, "std": None}
        report = {
            "task": args_cli.task,
            "group": args_cli.group or args_cli.task,
            "checkpoint": str(Path(args_cli.checkpoint).resolve()),
            "seed": args_cli.seed,
            "num_envs": args_cli.num_envs,
            "definitions": {
                "success": "episode ended by the configured timeout term",
                "fall": "episode ended by a non-timeout termination",
                "collision": "any non-foot/non-shank body force above 1 N",
                "foot_slip": "fraction of foot-contact observations with horizontal foot speed above 0.1 m/s",
            },
            "summary": summary,
            "episodes_detail": records,
        }
        if args_cli.recovery_eval:
            report["recovery_protocol"] = {
                "perturbation": "add fixed world-frame +x root linear velocity at the configured episode step",
                "perturbation_step": args_cli.perturbation_step,
                "lateral_velocity_kick_mps": args_cli.lateral_velocity_kick,
                "tracking_error_threshold_mps": args_cli.recovery_error_threshold,
                "continuous_hold_seconds": args_cli.recovery_hold_seconds,
                "control_dt_seconds": control_dt,
                "recovery_success_definition": "planar body-frame base-velocity command error remains below threshold for the hold duration",
            }
        output = Path(args_cli.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print_dict(summary, nesting=2)
        print(f"Saved evaluation report: {output}")
    finally:
        env.close()


try:
    main()
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    simulation_app.close()
