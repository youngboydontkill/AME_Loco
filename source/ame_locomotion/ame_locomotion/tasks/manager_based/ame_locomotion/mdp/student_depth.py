"""Frozen Student depth enhancement for S54 playback."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCamera
from torch import nn


_REPO_ROOT = Path(__file__).resolve().parents[7]
DEFAULT_STUDENT_CHECKPOINT = str(
    Path(
        os.environ.get(
            "AME_STUDENT_CHECKPOINT",
            _REPO_ROOT / "pretrained/student_depth_edge_20261016.pt",
        )
    )
)
EDGE_STUDENT_CHECKPOINT = str(_REPO_ROOT / "pretrained/student_depth_edge_20261016.pt")


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.depthwise(x))
        return self.activation(self.pointwise(x) + residual)


class _ProbabilityHead(nn.Module):
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
    """Architecture used by ``student_depth_edge_20261016.pt``.

    Only the enhanced depth path is consumed by the observation function. The
    auxiliary edge, confidence, and validity heads remain loaded solely because
    they are part of the checkpoint's shared network state.
    """

    def __init__(self, channels: int, num_blocks: int, input_channels: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.stem = nn.Sequential(nn.Conv2d(input_channels, channels, 3, padding=1), nn.SiLU(inplace=True))
        self.backbone = nn.Sequential(*(_ResidualBlock(channels) for _ in range(num_blocks)))
        self.residual_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
        )
        hidden = max(channels // 2, 8)
        self.validity_head = _ProbabilityHead(channels, hidden)
        self.confidence_head = _ProbabilityHead(channels, hidden)
        self.edge_head = _ProbabilityHead(channels, hidden)

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        depth = model_input[:, :1]
        features = self.backbone(self.stem(model_input))
        residual = self.residual_head(features)
        validity = self.validity_head(features)
        return depth + validity * residual


def _load_student(env: ManagerBasedEnv, checkpoint_path: str) -> StudentNetwork:
    model = getattr(env, "_ame_student_depth_model", None)
    if model is not None and getattr(env, "_ame_student_depth_checkpoint", None) == checkpoint_path:
        return model

    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Student checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=env.device, weights_only=True)
    except TypeError:  # PyTorch < 2.6
        checkpoint = torch.load(path, map_location=env.device)
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
    model = StudentNetwork(channels, num_blocks, input_channels).to(env.device)
    # ``mask_head`` aliases ``validity_head`` in the training model, so the
    # checkpoint contains duplicate keys that are not part of this inference model.
    state_dict = {key: value for key, value in state_dict.items() if not key.startswith("mask_head.")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Invalid Student checkpoint {path}: missing={missing}, unexpected={unexpected}")
    model.eval().requires_grad_(False)
    env._ame_student_depth_model = model
    env._ame_student_depth_checkpoint = checkpoint_path
    return model


def student_depth_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    checkpoint_path: str = DEFAULT_STUDENT_CHECKPOINT,
    min_depth: float = 0.05,
    max_depth: float = 5.0,
    sensor_valid_max: float = 4.9,
    noise_std: float = 0.02,
    dropout_probability: float = 0.02,
) -> torch.Tensor:
    """Return noisy, hole-corrupted Student depth as a flattened ``[-1, 1]`` map."""
    if max_depth <= min_depth:
        raise ValueError(f"max_depth must be greater than min_depth, got {min_depth} and {max_depth}")
    if noise_std < 0.0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")
    if not 0.0 <= dropout_probability <= 1.0:
        raise ValueError(f"dropout_probability must be in [0, 1], got {dropout_probability}")

    camera: TiledCamera = env.scene.sensors[sensor_cfg.name]
    cache_key = (
        getattr(env, "common_step_counter", None),
        sensor_cfg.name,
        checkpoint_path,
        min_depth,
        max_depth,
        sensor_valid_max,
        noise_std,
        dropout_probability,
    )
    cache = getattr(env, "_ame_student_depth_cache", None)
    if cache is not None and cache["key"] == cache_key:
        return cache["value"]

    depth = camera.data.output["distance_to_image_plane"]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3 or tuple(depth.shape[-2:]) != (42, 42):
        raise ValueError(f"Student expects camera depth [N, 42, 42], got {tuple(depth.shape)}")

    valid = torch.isfinite(depth) & (depth >= min_depth) & (depth <= sensor_valid_max)
    sim_depth = torch.nan_to_num(depth, nan=sensor_valid_max, posinf=sensor_valid_max, neginf=min_depth)
    sim_depth = torch.where(valid, sim_depth, torch.full_like(sim_depth, sensor_valid_max))
    sim_depth = sim_depth.clamp(min=min_depth, max=sensor_valid_max).unsqueeze(1)
    model_input = torch.cat((sim_depth, valid.unsqueeze(1).to(sim_depth.dtype)), dim=1)
    with torch.inference_mode():
        enhanced = _load_student(env, checkpoint_path)(model_input)[:, 0]
    enhanced = enhanced.clamp(min=min_depth, max=max_depth)

    if noise_std > 0.0:
        enhanced = (enhanced + torch.randn_like(enhanced) * noise_std).clamp(min_depth, max_depth)
    if dropout_probability > 0.0:
        holes = torch.rand_like(enhanced) < dropout_probability
        enhanced = torch.where(holes, torch.full_like(enhanced, max_depth), enhanced)

    normalized = (2.0 * enhanced / max_depth - 1.0).reshape(depth.shape[0], -1)
    env._ame_student_depth_cache = {"key": cache_key, "value": normalized}
    return normalized


def mixed_student_depth_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    checkpoint_path: str = EDGE_STUDENT_CHECKPOINT,
    enhance_probability: float = 0.7,
    min_depth: float = 0.05,
    max_depth: float = 5.0,
    sensor_valid_max: float = 4.9,
    noise_std: float = 0.02,
    dropout_probability: float = 0.02,
) -> torch.Tensor:
    """Mix Student-enhanced and raw depth across environments for training.

    The branch assignment is sampled independently per environment at every
    simulation step. Both branches receive the same metric-depth noise and
    synthetic hole corruption after the raw/Student depth has been selected.
    """
    if max_depth <= min_depth:
        raise ValueError(f"max_depth must be greater than min_depth, got {min_depth} and {max_depth}")
    if not 0.0 <= enhance_probability <= 1.0:
        raise ValueError(f"enhance_probability must be in [0, 1], got {enhance_probability}")
    if noise_std < 0.0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")
    if not 0.0 <= dropout_probability <= 1.0:
        raise ValueError(f"dropout_probability must be in [0, 1], got {dropout_probability}")

    camera: TiledCamera = env.scene.sensors[sensor_cfg.name]
    cache_key = (
        getattr(env, "common_step_counter", None),
        sensor_cfg.name,
        checkpoint_path,
        enhance_probability,
        min_depth,
        max_depth,
        sensor_valid_max,
        noise_std,
        dropout_probability,
    )
    cache = getattr(env, "_ame_mixed_student_depth_cache", None)
    if cache is not None and cache["key"] == cache_key:
        return cache["value"]

    depth = camera.data.output["distance_to_image_plane"]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3 or tuple(depth.shape[-2:]) != (42, 42):
        raise ValueError(f"Student expects camera depth [N, 42, 42], got {tuple(depth.shape)}")

    valid = torch.isfinite(depth) & (depth >= min_depth) & (depth <= sensor_valid_max)
    raw_depth = torch.nan_to_num(depth, nan=sensor_valid_max, posinf=sensor_valid_max, neginf=min_depth)
    raw_depth = torch.where(valid, raw_depth, torch.full_like(raw_depth, sensor_valid_max))
    raw_depth = raw_depth.clamp(min=min_depth, max=sensor_valid_max)

    use_enhanced = torch.rand(depth.shape[0], device=depth.device) < enhance_probability
    mixed_depth = raw_depth.clone()
    if use_enhanced.any():
        sim_depth = raw_depth[use_enhanced].unsqueeze(1)
        model = _load_student(env, checkpoint_path)
        model_input = sim_depth
        if model.input_channels == 2:
            model_input = torch.cat((sim_depth, valid[use_enhanced].unsqueeze(1).to(sim_depth.dtype)), dim=1)
        elif model.input_channels != 1:
            raise ValueError(f"Unsupported Student input channel count: {model.input_channels}")
        with torch.inference_mode():
            enhanced = model(model_input)[:, 0]
        mixed_depth[use_enhanced] = enhanced.clamp(min=min_depth, max=max_depth)

    if noise_std > 0.0:
        mixed_depth = (mixed_depth + torch.randn_like(mixed_depth) * noise_std).clamp(min_depth, max_depth)
    if dropout_probability > 0.0:
        holes = torch.rand_like(mixed_depth) < dropout_probability
        mixed_depth = torch.where(holes, torch.full_like(mixed_depth, max_depth), mixed_depth)

    normalized = (2.0 * mixed_depth / max_depth - 1.0).reshape(depth.shape[0], -1)
    env._ame_mixed_student_depth_cache = {
        "key": cache_key,
        "value": normalized,
        "use_enhanced": use_enhanced,
    }
    return normalized
