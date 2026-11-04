"""Create a depth contact sheet, frame statistics, and animation from saved .npy files.

Example:
    python scripts/rsl_rl/visualize_depth_data.py depth_data/pyramid_stairs
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw


MIN_DEPTH = 0.05
MAX_DEPTH = 5.0  # S54 camera clipping range; collection stores out-of-range depth at this maximum.


def load_frame(path: Path, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    frame = np.load(path, allow_pickle=False)
    if frame.ndim != 2 or (expected_shape is not None and frame.shape != expected_shape):
        raise ValueError(f"Unexpected frame shape in {path}: {frame.shape}")
    return frame


def sample_indices(count: int, wanted: int) -> np.ndarray:
    return np.unique(np.linspace(0, count - 1, min(count, wanted), dtype=int))


def add_edge_holes(
    frame: np.ndarray,
    frame_index: int,
    enabled: bool,
    band_ratio: float,
    probability: float,
    seed: int,
) -> np.ndarray:
    """Overlay random max-range pixels along the frame border for display only."""
    if not enabled or band_ratio == 0.0 or probability == 0.0:
        return frame
    height, width = frame.shape
    band = max(1, round(min(height, width) * band_ratio))
    edge_region = np.zeros((height, width), dtype=bool)
    edge_region[:band, :] = True
    edge_region[-band:, :] = True
    edge_region[:, :band] = True
    edge_region[:, -band:] = True
    rng = np.random.default_rng(seed + frame_index)
    holes = (rng.random((height, width)) < probability) & edge_region
    display_frame = frame.copy()
    display_frame[holes] = MAX_DEPTH
    return display_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing *_depth.npy frames")
    parser.add_argument("--output-dir", type=Path, help="Default: INPUT_DIR/visualization")
    parser.add_argument("--animation-frames", type=int, default=76)
    parser.add_argument(
        "--edge-holes",
        action="store_true",
        help="Overlay random max-range holes near image borders in visualizations; source .npy files stay unchanged.",
    )
    parser.add_argument(
        "--edge-band-ratio",
        type=float,
        default=0.08,
        help="Border-band width as a fraction of the shorter image side (default: 0.08)",
    )
    parser.add_argument(
        "--edge-hole-probability",
        type=float,
        default=0.1,
        help="Per-pixel hole probability inside the border band (default: 0.1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used for repeatable edge-hole overlays")
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*_depth.npy"))
    if not paths:
        parser.error(f"No *_depth.npy files found in {args.input_dir}")
    if args.animation_frames < 1:
        parser.error("--animation-frames must be positive")
    if not 0.0 <= args.edge_band_ratio < 0.5:
        parser.error("--edge-band-ratio must be in [0, 0.5)")
    if not 0.0 <= args.edge_hole_probability <= 1.0:
        parser.error("--edge-hole-probability must be in [0, 1]")

    output_dir = args.output_dir or args.input_dir / "visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = load_frame(paths[0]).shape
    # Match Isaac Lab's native camera demo (scripts/demos/sensors/cameras.py),
    # which plots distance_to_image_plane with cmap="turbo".
    cmap = plt.colormaps["turbo"].copy()
    norm = Normalize(vmin=MIN_DEPTH, vmax=MAX_DEPTH, clip=True)

    # A common distance scale makes the frames directly comparable.
    overview_indices = sample_indices(len(paths), 12)
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), layout="constrained")
    for ax, index in zip(axes.flat, overview_indices):
        frame = load_frame(paths[index], shape)
        frame = add_edge_holes(
            frame,
            int(index),
            args.edge_holes,
            args.edge_band_ratio,
            args.edge_hole_probability,
            args.seed,
        )
        ax.imshow(np.ma.masked_invalid(frame), cmap=cmap, norm=norm)
        ax.set_title(f"Frame {index + 1:03d}")
        ax.axis("off")
    for ax in axes.flat[len(overview_indices) :]:
        ax.axis("off")
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.72)
    colorbar.set_label("Distance to image plane (m)")
    fig.suptitle(f"{args.input_dir.name}: Isaac Lab turbo depth view (5 m = camera range limit)", fontsize=16)
    fig.savefig(output_dir / "depth_overview.png", dpi=140)
    plt.close(fig)

    medians = np.full(len(paths), np.nan)
    clipped_fractions = np.zeros(len(paths))
    for index, path in enumerate(paths):
        frame = load_frame(path, shape)
        frame = add_edge_holes(
            frame,
            index,
            args.edge_holes,
            args.edge_band_ratio,
            args.edge_hole_probability,
            args.seed,
        )
        valid = np.isfinite(frame) & (frame < MAX_DEPTH)
        clipped_fractions[index] = 1 - np.count_nonzero(valid) / frame.size
        if np.any(valid):
            medians[index] = np.median(frame[valid])

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, layout="constrained")
    frame_numbers = np.arange(1, len(paths) + 1)
    axes[0].plot(frame_numbers, medians, color="#3b528b", linewidth=1.6)
    axes[0].set_ylabel("Median visible depth (m)")
    axes[0].grid(alpha=0.2)
    axes[1].plot(frame_numbers, clipped_fractions * 100, color="#b75c24", linewidth=1.6)
    axes[1].set_ylabel("Clipped pixels (%)")
    axes[1].set_xlabel("Saved frame")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.2)
    fig.suptitle(f"{args.input_dir.name}: depth over time")
    fig.savefig(output_dir / "depth_timeline.png", dpi=160)
    plt.close(fig)

    animation = []
    for index in sample_indices(len(paths), args.animation_frames):
        frame = load_frame(paths[index], shape)
        frame = add_edge_holes(
            frame,
            int(index),
            args.edge_holes,
            args.edge_band_ratio,
            args.edge_hole_probability,
            args.seed,
        )
        rgba = cmap(norm(np.nan_to_num(frame, nan=MAX_DEPTH, posinf=MAX_DEPTH, neginf=MIN_DEPTH)), bytes=True)
        rgb = np.asarray(rgba[..., :3]).copy()
        image = Image.fromarray(rgb).resize((640, 480), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 140, 27), fill=(20, 25, 35))
        draw.text((8, 7), f"Frame {index + 1:03d}/{len(paths):03d}", fill="white")
        animation.append(image)
    animation[0].save(
        output_dir / "depth_animation.gif",
        save_all=True,
        append_images=animation[1:],
        duration=100,
        loop=0,
        optimize=True,
    )

    print(f"Saved {len(paths)}-frame visualization to {output_dir}")
    print(
        f"Frame size: {shape[1]}x{shape[0]}; pixels at the {MAX_DEPTH:g} m range limit: "
        f"{clipped_fractions.min():.1%}–{clipped_fractions.max():.1%}"
    )


if __name__ == "__main__":
    main()
