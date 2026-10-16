#!/usr/bin/env python3
"""Aggregate Appendix-A per-training-seed evaluation summaries."""

from __future__ import annotations

import argparse
import json
import re
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
TRAIN_SEED_RE = re.compile(r"_train(?P<seed>\d+)(?:_recovery)?_seed_summary\.json$")


def scalar_metric(summary: dict, metric: str) -> float | None:
    value = summary.get("metrics", {}).get(metric)
    if value is None:
        return None
    mean = value.get("mean_across_seeds")
    return None if mean is None else float(mean)


def summarize(values: list[float], samples: int, rng: np.random.Generator) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 1:
        interval = [float(array[0]), float(array[0])]
    else:
        indices = rng.integers(0, array.size, size=(samples, array.size))
        means = array[indices].mean(axis=1)
        interval = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    return {
        "mean_across_training_seeds": float(array.mean()),
        "std_across_training_seeds": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "training_seed_bootstrap_95ci": interval,
        "training_seed_values": [float(value) for value in array],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap_samples must be positive")

    # Older queue processes used a broad shell glob that also included
    # recovery summaries in the ordinary matrix call. Select the requested
    # family from the output name so those processes remain safe to finish.
    output_is_recovery = "recovery" in args.output.name
    summaries = [
        path for path in args.summaries
        if ("_recovery_seed_summary" in path.name) == output_is_recovery
    ]
    if not summaries:
        raise ValueError(f"no {'recovery' if output_is_recovery else 'ordinary'} summaries supplied")

    records = []
    for path in summaries:
        data = json.loads(path.read_text(encoding="utf-8"))
        match = TRAIN_SEED_RE.search(path.name)
        if match is None:
            raise ValueError(f"summary filename does not contain _train<seed>_: {path}")
        groups = data.get("groups", {})
        if len(groups) != 1:
            raise ValueError(f"expected one group in {path}, found {list(groups)}")
        group, summary = next(iter(groups.items()))
        records.append({
            "path": str(path),
            "group": group.rsplit("_train", 1)[0],
            "training_seed": int(match.group("seed")),
            "metrics": summary.get("metrics", {}),
        })

    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["group"], []).append(record)
    rng = np.random.default_rng(args.seed)
    result = {"source_summaries": [str(path) for path in summaries], "groups": {}}
    for group, group_records in sorted(groups.items()):
        seeds = [int(record["training_seed"]) for record in group_records]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate training seed in {group}: {seeds}")
        metrics = {}
        for metric in METRICS:
            values = [scalar_metric(record, metric) for record in group_records]
            if all(value is not None for value in values):
                metrics[metric] = summarize([float(value) for value in values], args.bootstrap_samples, rng)
        result["groups"][group] = {
            "training_seed_count": len(seeds),
            "training_seeds": sorted(seeds),
            "metrics": metrics,
            "source_summaries": [record["path"] for record in group_records],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["groups"], indent=2))
    print(f"Saved Appendix-A matrix summary: {args.output}")


if __name__ == "__main__":
    main()
