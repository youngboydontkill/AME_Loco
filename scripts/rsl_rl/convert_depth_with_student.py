"""Enhance saved depth .npy frames with the pretrained S54 Student model.

Example:
    python scripts/rsl_rl/convert_depth_with_student.py depth_data/pyramid_stairs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "pretrained/student_depth_edge_20261016.pt"


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.depthwise(x))
        return self.activation(self.pointwise(x) + residual)


class ProbabilityHead(nn.Module):
    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.output = nn.Conv2d(hidden, 1, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.output(self.features(x)))


class StudentNetwork(nn.Module):
    """Inference network matching student_depth_edge_20261016.pt."""

    def __init__(self, channels: int, num_blocks: int, input_channels: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.stem = nn.Sequential(nn.Conv2d(input_channels, channels, 3, padding=1), nn.SiLU(inplace=True))
        self.backbone = nn.Sequential(*(ResidualBlock(channels) for _ in range(num_blocks)))
        self.residual_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
        )
        hidden = max(channels // 2, 8)
        self.validity_head = ProbabilityHead(channels, hidden)
        self.confidence_head = ProbabilityHead(channels, hidden)
        self.edge_head = ProbabilityHead(channels, hidden)

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        depth = model_input[:, :1]
        features = self.backbone(self.stem(model_input))
        residual = self.residual_head(features)
        validity = self.validity_head(features)
        return depth + validity * residual


def load_model(checkpoint_path: Path, device: torch.device) -> StudentNetwork:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Student checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    architecture = checkpoint.get("architecture", {})
    channels = int(architecture.get("channels", state_dict["stem.0.weight"].shape[0]))
    input_channels = int(architecture.get("input_channels", state_dict["stem.0.weight"].shape[1]))
    block_ids = {
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("backbone.") and key.split(".")[1].isdigit()
    }
    num_blocks = int(architecture.get("residual_blocks", max(block_ids, default=-1) + 1))
    model = StudentNetwork(channels, num_blocks, input_channels).to(device)
    # Training checkpoints also save the alias mask_head for validity_head.
    state_dict = {key: value for key, value in state_dict.items() if not key.startswith("mask_head.")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Invalid Student checkpoint: missing={missing}, unexpected={unexpected}")
    if input_channels not in (1, 2):
        raise ValueError(f"Student expects 1 or 2 input channels, got {input_channels}")
    return model.eval().requires_grad_(False)


def enhance_frame(
    frame: np.ndarray,
    model: StudentNetwork,
    device: torch.device,
    model_size: tuple[int, int],
    min_depth: float,
    max_depth: float,
    sensor_valid_max: float,
) -> np.ndarray:
    if frame.ndim != 2:
        raise ValueError(f"Expected a 2-D depth frame, got shape {frame.shape}")
    depth = torch.as_tensor(frame, dtype=torch.float32, device=device)[None, None]
    valid = torch.isfinite(depth) & (depth >= min_depth) & (depth <= sensor_valid_max)
    depth = torch.nan_to_num(depth, nan=sensor_valid_max, posinf=sensor_valid_max, neginf=min_depth)
    depth = torch.where(valid, depth, torch.full_like(depth, sensor_valid_max))
    depth = depth.clamp(min=min_depth, max=sensor_valid_max)

    # The policy Student was trained on 42x42 camera frames.
    depth = F.interpolate(depth, size=model_size, mode="bilinear", align_corners=False)
    valid_small = F.interpolate(valid.float(), size=model_size, mode="nearest")
    model_input = depth if model.input_channels == 1 else torch.cat((depth, valid_small), dim=1)
    with torch.inference_mode():
        enhanced = model(model_input).clamp(min=min_depth, max=max_depth)
    enhanced = F.interpolate(enhanced, size=frame.shape, mode="bilinear", align_corners=False)
    return enhanced[0, 0].cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=REPO_ROOT / "depth_data",
        help="Directory to scan recursively for *_depth.npy files (default: depth_data)",
    )
    parser.add_argument("--output-dir", type=Path, help="Default: sibling directory named <input>_student")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-size", type=int, nargs=2, default=(42, 42), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument(
        "--noise-std", type=float, default=0.02, help="Gaussian depth noise standard deviation in meters"
    )
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--sensor-valid-max", type=float, default=4.9)
    parser.add_argument("--seed", type=int, help="Optional random seed for repeatable noise")
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")
    if args.noise_std < 0:
        parser.error("--noise-std must be non-negative")
    if args.min_depth >= args.max_depth:
        parser.error("--min-depth must be less than --max-depth")
    if args.sensor_valid_max < args.min_depth:
        parser.error("--sensor-valid-max must be at least --min-depth")
    if min(args.model_size) < 1:
        parser.error("--model-size values must be positive")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (
        input_dir.parent / f"{input_dir.name}_student"
    )
    if output_dir == input_dir or input_dir in output_dir.parents:
        parser.error("Output directory must be outside the input directory")
    frame_paths = sorted(input_dir.rglob("*_depth.npy"))
    if not frame_paths:
        parser.error(f"No *_depth.npy files found under {input_dir}")

    device = torch.device(args.device)
    model = load_model(args.checkpoint.expanduser().resolve(), device)
    rng = np.random.default_rng(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, frame_path in enumerate(frame_paths, start=1):
        relative_path = frame_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame = np.load(frame_path, allow_pickle=False)
        enhanced = enhance_frame(
            frame,
            model,
            device,
            tuple(args.model_size),
            args.min_depth,
            args.max_depth,
            args.sensor_valid_max,
        )
        if args.noise_std > 0:
            enhanced += rng.normal(0.0, args.noise_std, size=enhanced.shape).astype(np.float32)
        enhanced = np.clip(enhanced, args.min_depth, args.max_depth).astype(np.float32)
        np.save(output_path, enhanced)
        if index == 1 or index % 50 == 0 or index == len(frame_paths):
            print(f"[{index}/{len(frame_paths)}] {relative_path}")

    print(f"Saved {len(frame_paths)} enhanced depth frames under {output_dir}")
    print(f"Student: {args.checkpoint}; noise std: {args.noise_std:g} m; device: {device}")


if __name__ == "__main__":
    main()
