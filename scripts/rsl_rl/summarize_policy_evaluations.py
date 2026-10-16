"""Aggregate fixed-episode evaluations with the training seed as the unit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "return",
    "length_steps",
    "success_rate",
    "fall_rate",
    "collision_rate",
    "foot_slip_fraction",
    "recovery_rate",
    "recovery_time_seconds",
)


def seed_metric(report: dict, metric: str) -> float:
    value = report["summary"][metric]
    return float(value["mean"]) if isinstance(value, dict) else float(value)


def summarize(values: list[float], bootstrap_samples: int, rng: np.random.Generator) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 1:
        interval = [float(array[0]), float(array[0])]
    else:
        indices = rng.integers(0, array.size, size=(bootstrap_samples, array.size))
        means = array[indices].mean(axis=1)
        interval = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    return {
        "mean_across_seeds": float(array.mean()),
        "std_across_seeds": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "seed_bootstrap_95ci": interval,
        "seed_values": [float(value) for value in array],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="Per-run JSON files from evaluate_policy.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap_samples must be positive")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in args.reports:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        grouped[report.get("group", report["task"])].append(report)

    result = {"source_reports": args.reports, "groups": {}}
    rng = np.random.default_rng(args.seed)
    for group, reports in sorted(grouped.items()):
        seeds = [int(report["seed"]) for report in reports]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate evaluation seed in group {group}: {seeds}")
        group_result = {
            "seed_count": len(reports),
            "seeds": seeds,
            "metrics": {},
        }
        for metric in METRICS:
            if not all(metric in report["summary"] for report in reports):
                continue
            if metric == "recovery_time_seconds" and any(
                report["summary"][metric].get("mean") is None for report in reports
            ):
                continue
            values = [seed_metric(report, metric) for report in reports]
            group_result["metrics"][metric] = summarize(values, args.bootstrap_samples, rng)
        result["groups"][group] = group_result

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["groups"], indent=2))
    print(f"Saved seed-level summary: {output}")


if __name__ == "__main__":
    main()
