"""Frozen Stable-Sim2Real Student preprocessing for Isaac Lab depth observations."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab.sensors import TiledCamera


_REPO_ROOT = Path(__file__).resolve().parents[7]
DEFAULT_STUDENT_CHECKPOINT = str(
    Path(os.environ.get("AME_STUDENT_CHECKPOINT", _REPO_ROOT / "pretrained/student_depth_latest.pt"))
)


class _DepthwiseSeparableResidualBlock(nn.Module):
    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.depthwise(x))
        x = self.pointwise(x)
        return self.activation(x + residual)


class _ProbabilityHead(nn.Module):
    def __init__(self, channels: int = 16, hidden: int = 8) -> None:
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
    """The 16-channel, four-block Student used by stable-sim2real."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.SiLU(inplace=True))
        self.backbone = nn.Sequential(*(_DepthwiseSeparableResidualBlock(16) for _ in range(4)))
        self.residual_head = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )
        self.validity_head = _ProbabilityHead()
        self.confidence_head = _ProbabilityHead()

    def forward(self, sim_depth_m: torch.Tensor) -> torch.Tensor:
        features = self.backbone(self.stem(sim_depth_m))
        residual = self.residual_head(features)
        validity = self.validity_head(features)
        return sim_depth_m + validity * residual


def _load_student(env: ManagerBasedEnv, checkpoint_path: str) -> StudentNetwork:
    """Load once per environment and keep the frozen model on the simulation device."""
    model = getattr(env, "_ame_student_depth_model", None)
    loaded_path = getattr(env, "_ame_student_depth_checkpoint", None)
    if model is not None and loaded_path == checkpoint_path:
        return model

    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Student checkpoint not found: {path}. Set AME_STUDENT_CHECKPOINT or pass "
            "checkpoint_path in the enhancement observation config."
        )

    model = StudentNetwork().to(env.device)
    try:
        checkpoint = torch.load(path, map_location=env.device, weights_only=True)
    except TypeError:  # PyTorch < 2.6
        checkpoint = torch.load(path, map_location=env.device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # The source model exposes mask_head as an alias; it is not needed for inference.
    state_dict = {key: value for key, value in state_dict.items() if not key.startswith("mask_head.")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Invalid Student checkpoint {path}: missing={missing}, unexpected={unexpected}"
        )
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
) -> torch.Tensor:
    """Enhance camera depth with the frozen Student and return a flat ``[-1, 1]`` map.

    The Student consumes metres at 42x42. Isaac Lab's camera output is already
    42x42 for the S54 task; the shape check catches accidental camera changes.
    Invalid pixels follow the training convention and are represented by the
    maximum sensor depth before enhancement.
    """
    camera: TiledCamera = env.scene.sensors[sensor_cfg.name]
    depth = camera.data.output["distance_to_image_plane"]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3 or tuple(depth.shape[-2:]) != (42, 42):
        raise ValueError(f"Student expects camera depth [N, 42, 42], got {tuple(depth.shape)}")

    valid = torch.isfinite(depth) & (depth >= min_depth) & (depth <= sensor_valid_max)
    sim_depth = torch.nan_to_num(depth, nan=sensor_valid_max, posinf=sensor_valid_max, neginf=min_depth)
    sim_depth = torch.where(valid, sim_depth, torch.full_like(sim_depth, sensor_valid_max))
    sim_depth = sim_depth.clamp(min=min_depth, max=sensor_valid_max).unsqueeze(1)

    with torch.inference_mode():
        enhanced = _load_student(env, checkpoint_path)(sim_depth)
    enhanced = enhanced[:, 0].clamp(min=min_depth, max=max_depth)
    normalized = 2.0 * (enhanced - 0.0) / max_depth - 1.0
    return normalized.reshape(normalized.shape[0], -1)
