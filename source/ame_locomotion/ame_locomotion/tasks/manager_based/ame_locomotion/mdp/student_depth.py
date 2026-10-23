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
    Path(
        os.environ.get(
            "AME_STUDENT_CHECKPOINT",
            _REPO_ROOT / "pretrained/student_depth_gemini_combined_20261005.pt",
        )
    )
)
EDGE_STUDENT_CHECKPOINT = str(_REPO_ROOT / "pretrained/student_depth_edge_20261016.pt")


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
    def __init__(self, channels: int = 16, hidden: int = 8, output_bias: float = 0.0) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.output = nn.Conv2d(hidden, 1, 1)
        self.activation = nn.Sigmoid()
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, output_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.output(self.features(x)))


class StudentNetwork(nn.Module):
    """Inference-compatible multi-task Student architecture."""

    def __init__(
        self,
        channels: int = 16,
        num_blocks: int = 4,
        input_channels: int = 1,
        foothold_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.foothold_enabled = foothold_enabled
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, padding=1), nn.SiLU(inplace=True)
        )
        self.backbone = nn.Sequential(
            *(_DepthwiseSeparableResidualBlock(channels) for _ in range(num_blocks))
        )
        self.residual_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 1, 1),
        )
        hidden = max(channels // 2, 8)
        self.validity_head = _ProbabilityHead(channels, hidden)
        self.confidence_head = _ProbabilityHead(channels, hidden)
        self.edge_head = _ProbabilityHead(channels, hidden, output_bias=-2.0)
        if foothold_enabled:
            self.foothold_head = _ProbabilityHead(channels, hidden)

    def forward_all(self, model_input: torch.Tensor) -> dict[str, torch.Tensor]:
        if model_input.shape[1] != self.input_channels:
            raise ValueError(
                f"Student expects {self.input_channels} input channels, got {model_input.shape[1]}"
            )
        sim_depth_m = model_input[:, :1]
        features = self.backbone(self.stem(model_input))
        residual = self.residual_head(features)
        validity = self.validity_head(features)
        output = {
            "depth": sim_depth_m + validity * residual,
            "validity": validity,
            "confidence": self.confidence_head(features),
            "edge": self.edge_head(features),
        }
        if self.foothold_enabled:
            output["foothold"] = self.foothold_head(features)
        return output

    def forward(self, sim_depth_m: torch.Tensor) -> torch.Tensor:
        return self.forward_all(sim_depth_m)["depth"]


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
    num_blocks = int(architecture.get("residual_blocks", max(block_ids, default=3) + 1))
    foothold_enabled = bool(architecture.get("foothold_enabled", True))
    model = StudentNetwork(
        channels, num_blocks, input_channels, foothold_enabled=foothold_enabled
    ).to(env.device)
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


def student_depth_confidence_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    checkpoint_path: str = DEFAULT_STUDENT_CHECKPOINT,
    min_depth: float = 0.05,
    max_depth: float = 5.0,
    sensor_valid_max: float = 4.9,
    raw_depth_probability: float = 0.2,
) -> torch.Tensor:
    """Return interleaved normalized enhanced depth and confidence maps.

    The Student consumes metres at 42x42. Isaac Lab's camera output is already
    42x42 for the S54 task; the shape check catches accidental camera changes.
    Invalid pixels follow the training convention and are represented by the
    maximum sensor depth before enhancement.
    """
    camera: TiledCamera = env.scene.sensors[sensor_cfg.name]

    # The observation manager evaluates the policy and critic groups one after
    # another. They consume the same rendered frame, so reuse the frozen
    # Student result for both groups instead of running the CNN twice. The
    # manager's common step counter is a Python integer and avoids synchronizing
    # a CUDA frame counter just to decide whether a cache entry is current.
    cache_key = (
        getattr(env, "common_step_counter", None),
        sensor_cfg.name,
        checkpoint_path,
        min_depth,
        max_depth,
        sensor_valid_max,
        raw_depth_probability,
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

    model = _load_student(env, checkpoint_path)
    model_input = sim_depth
    if model.input_channels == 2:
        model_input = torch.cat((sim_depth, valid.unsqueeze(1).to(sim_depth.dtype)), dim=1)
    with torch.inference_mode():
        output = model.forward_all(model_input)
    enhanced = output["depth"][:, 0].clamp(min=min_depth, max=max_depth)
    normalized_depth = 2.0 * enhanced / max_depth - 1.0
    # The combined training masks are all valid, so its confidence head cannot
    # learn invalid-pixel behavior. Preserve known camera invalidity at runtime.
    confidence = torch.minimum(
        output["confidence"][:, 0].clamp(0.0, 1.0), valid.to(sim_depth.dtype)
    )
    if not 0.0 <= raw_depth_probability <= 1.0:
        raise ValueError("raw_depth_probability must be in [0, 1]")
    raw_depth = 2.0 * (sim_depth[:, 0] - min_depth) / (max_depth - min_depth) - 1.0
    use_raw = torch.rand((depth.shape[0], 1, 1), device=depth.device) < raw_depth_probability
    normalized_depth = torch.where(use_raw, raw_depth, normalized_depth)
    confidence = torch.where(use_raw, torch.zeros_like(confidence), confidence)
    # ActorCriticEncoder reshapes the final map values as H x W x C. Interleave
    # channels per pixel before flattening so depth and confidence stay aligned.
    normalized = torch.stack((normalized_depth, confidence), dim=-1).reshape(
        normalized_depth.shape[0], -1
    )
    env._ame_student_depth_cache = {"key": cache_key, "value": normalized}
    return normalized


def student_depth_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    checkpoint_path: str = DEFAULT_STUDENT_CHECKPOINT,
    min_depth: float = 0.05,
    max_depth: float = 5.0,
    sensor_valid_max: float = 4.9,
) -> torch.Tensor:
    """Return the enhanced depth channel using the legacy flat observation shape."""
    paired = student_depth_confidence_image(
        env,
        sensor_cfg,
        checkpoint_path=checkpoint_path,
        min_depth=min_depth,
        max_depth=max_depth,
        sensor_valid_max=sensor_valid_max,
    )
    return paired[:, 0::2]


def student_depth_geometry_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    checkpoint_path: str = DEFAULT_STUDENT_CHECKPOINT,
    min_depth: float = 0.05,
    max_depth: float = 5.0,
    sensor_valid_max: float = 4.9,
    raw_depth_probability: float = 0.2,
) -> torch.Tensor:
    """Return interleaved enhanced depth, confidence, edge and foothold maps."""
    camera: TiledCamera = env.scene.sensors[sensor_cfg.name]
    cache_key = (
        "geometry", getattr(env, "common_step_counter", None), sensor_cfg.name,
        checkpoint_path, min_depth, max_depth, sensor_valid_max,
    )
    cache = getattr(env, "_ame_student_geometry_cache", None)
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
    model = _load_student(env, checkpoint_path)
    model_input = sim_depth
    if model.input_channels == 2:
        model_input = torch.cat((sim_depth, valid.unsqueeze(1).to(sim_depth.dtype)), dim=1)
    with torch.inference_mode():
        output = model.forward_all(model_input)
    enhanced = output["depth"][:, 0].clamp(min=min_depth, max=max_depth)
    normalized_depth = 2.0 * enhanced / max_depth - 1.0
    confidence = torch.minimum(output["confidence"][:, 0].clamp(0.0, 1.0), valid.to(sim_depth.dtype))
    edge = output["edge"][:, 0].clamp(0.0, 1.0)
    foothold = (
        output["foothold"][:, 0].clamp(0.0, 1.0)
        if "foothold" in output else torch.zeros_like(edge)
    )
    if not 0.0 <= raw_depth_probability <= 1.0:
        raise ValueError("raw_depth_probability must be in [0, 1]")
    raw_depth = 2.0 * (sim_depth[:, 0] - min_depth) / (max_depth - min_depth) - 1.0
    use_raw = torch.rand((depth.shape[0], 1, 1), device=depth.device) < raw_depth_probability
    normalized_depth = torch.where(use_raw, raw_depth, normalized_depth)
    confidence = torch.where(use_raw, torch.zeros_like(confidence), confidence)
    edge = torch.where(use_raw, torch.zeros_like(edge), edge)
    foothold = torch.where(use_raw, torch.zeros_like(foothold), foothold)
    normalized = torch.stack((normalized_depth, confidence, edge, foothold), dim=-1).reshape(
        normalized_depth.shape[0], -1
    )
    env._ame_student_geometry_cache = {"key": cache_key, "value": normalized}
    return normalized


def student_depth_prior_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    checkpoint_path: str = DEFAULT_STUDENT_CHECKPOINT,
    min_depth: float = 0.05,
    max_depth: float = 5.0,
    sensor_valid_max: float = 4.9,
    raw_depth_probability: float = 0.2,
) -> torch.Tensor:
    """Return depth plus all frozen Student auxiliary maps for the privileged Critic."""
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
    model = _load_student(env, checkpoint_path)
    model_input = sim_depth
    if model.input_channels == 2:
        model_input = torch.cat((sim_depth, valid.unsqueeze(1).to(sim_depth.dtype)), dim=1)
    with torch.inference_mode():
        output = model.forward_all(model_input)
    enhanced = output["depth"][:, 0].clamp(min=min_depth, max=max_depth)
    normalized_depth = 2.0 * enhanced / max_depth - 1.0
    confidence = torch.minimum(output["confidence"][:, 0].clamp(0.0, 1.0), valid.to(sim_depth.dtype))
    normalized = torch.stack(
        (
            normalized_depth,
            output["validity"][:, 0].clamp(0.0, 1.0),
            confidence,
            output["edge"][:, 0].clamp(0.0, 1.0),
            output.get("foothold", torch.zeros_like(output["edge"]))[:, 0].clamp(0.0, 1.0),
        ),
        dim=-1,
    )
    if not 0.0 <= raw_depth_probability <= 1.0:
        raise ValueError("raw_depth_probability must be in [0, 1]")
    raw_depth = 2.0 * (sim_depth[:, 0] - min_depth) / (max_depth - min_depth) - 1.0
    use_raw = torch.rand((depth.shape[0], 1, 1, 1), device=depth.device) < raw_depth_probability
    normalized[..., 0] = torch.where(use_raw, raw_depth, normalized[..., 0])
    normalized[..., 1:] = torch.where(
        use_raw.unsqueeze(-1), torch.zeros_like(normalized[..., 1:]), normalized[..., 1:]
    )
    return normalized.reshape(normalized.shape[0], -1)


def _randomized_depth_like(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    min_depth: float,
    max_depth: float,
    noise_std: float,
    dropout_probability: float,
) -> torch.Tensor:
    camera: TiledCamera = env.scene.sensors[sensor_cfg.name]
    depth = camera.data.output["distance_to_image_plane"]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=min_depth)
    depth = depth + torch.randn_like(depth) * noise_std
    dropout = torch.rand_like(depth) < dropout_probability
    depth = torch.where(dropout, torch.full_like(depth, max_depth), depth)
    depth = depth.clamp(min=min_depth, max=max_depth)
    return (2.0 * (depth - min_depth) / (max_depth - min_depth) - 1.0).reshape(depth.shape[0], -1)


def student_depth_mixed_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    enhance_probability: float = 0.8,
    **kwargs,
) -> torch.Tensor:
    """Use Student depth for most environments and randomized raw depth for the rest."""
    if not 0.0 <= enhance_probability <= 1.0:
        raise ValueError("enhance_probability must be in [0, 1]")
    enhanced = student_depth_image(env, sensor_cfg, raw_depth_probability=0.0, **kwargs)
    raw = _randomized_depth_like(env, sensor_cfg, kwargs.get("min_depth", 0.05), kwargs.get("max_depth", 5.0), 0.02, 0.02)
    use_enhance = torch.rand((enhanced.shape[0], 1), device=enhanced.device) < enhance_probability
    return torch.where(use_enhance, enhanced, raw)


def student_depth_prior_mixed_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    enhance_probability: float = 0.8,
    **kwargs,
) -> torch.Tensor:
    """Mixed five-channel map for a privileged Critic."""
    if not 0.0 <= enhance_probability <= 1.0:
        raise ValueError("enhance_probability must be in [0, 1]")
    enhanced = student_depth_prior_image(env, sensor_cfg, raw_depth_probability=0.0, **kwargs)
    raw_depth = _randomized_depth_like(env, sensor_cfg, kwargs.get("min_depth", 0.05), kwargs.get("max_depth", 5.0), 0.02, 0.02)
    raw = torch.zeros_like(enhanced)
    raw[:, 0::5] = raw_depth
    use_enhance = torch.rand((enhanced.shape[0], 1), device=enhanced.device) < enhance_probability
    return torch.where(use_enhance, enhanced, raw)


def _student_depth_edge_mixed_pair(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    checkpoint_path: str,
    enhance_probability: float,
    min_depth: float,
    max_depth: float,
    sensor_valid_max: float,
) -> torch.Tensor:
    """Return aligned mixed depth and edge maps, cached for both observation groups."""
    if not 0.0 <= enhance_probability <= 1.0:
        raise ValueError("enhance_probability must be in [0, 1]")
    cache_key = (
        getattr(env, "common_step_counter", None), sensor_cfg.name, checkpoint_path,
        enhance_probability, min_depth, max_depth, sensor_valid_max,
    )
    cache = getattr(env, "_ame_student_depth_edge_mixed_cache", None)
    if cache is not None and cache["key"] == cache_key:
        return cache["value"]

    camera: TiledCamera = env.scene.sensors[sensor_cfg.name]
    depth = camera.data.output["distance_to_image_plane"]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3 or tuple(depth.shape[-2:]) != (42, 42):
        raise ValueError(f"Student expects camera depth [N, 42, 42], got {tuple(depth.shape)}")

    valid = torch.isfinite(depth) & (depth >= min_depth) & (depth <= sensor_valid_max)
    sim_depth = torch.nan_to_num(
        depth, nan=sensor_valid_max, posinf=sensor_valid_max, neginf=min_depth
    )
    sim_depth = torch.where(valid, sim_depth, torch.full_like(sim_depth, sensor_valid_max))
    sim_depth = sim_depth.clamp(min=min_depth, max=sensor_valid_max).unsqueeze(1)
    model = _load_student(env, checkpoint_path)
    model_input = sim_depth
    if model.input_channels == 2:
        model_input = torch.cat((sim_depth, valid.unsqueeze(1).to(sim_depth.dtype)), dim=1)
    with torch.inference_mode():
        output = model.forward_all(model_input)

    enhanced_depth = (
        2.0 * output["depth"][:, 0].clamp(min=min_depth, max=max_depth) / max_depth - 1.0
    ).reshape(depth.shape[0], -1)
    enhanced_edge = output["edge"][:, 0].clamp(0.0, 1.0).reshape(depth.shape[0], -1)
    raw_depth = _randomized_depth_like(
        env, sensor_cfg, min_depth, max_depth, noise_std=0.02, dropout_probability=0.02
    )
    use_enhance = torch.rand((depth.shape[0], 1), device=depth.device) < enhance_probability
    mixed_depth = torch.where(use_enhance, enhanced_depth, raw_depth)
    mixed_edge = torch.where(use_enhance, enhanced_edge, torch.zeros_like(raw_depth))
    pair = torch.stack((mixed_depth, mixed_edge), dim=-1).reshape(depth.shape[0], -1)
    env._ame_student_depth_edge_mixed_cache = {"key": cache_key, "value": pair}
    return pair


def student_depth_edge_mixed_actor_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    checkpoint_path: str = EDGE_STUDENT_CHECKPOINT,
    enhance_probability: float = 0.8,
    min_depth: float = 0.05,
    max_depth: float = 5.0,
    sensor_valid_max: float = 4.9,
) -> torch.Tensor:
    """Single-channel enhanced/raw mix for the v2 Actor."""
    pair = _student_depth_edge_mixed_pair(
        env, sensor_cfg, checkpoint_path, enhance_probability,
        min_depth, max_depth, sensor_valid_max,
    )
    return pair[:, 0::2]


def student_depth_edge_mixed_critic_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    checkpoint_path: str = EDGE_STUDENT_CHECKPOINT,
    enhance_probability: float = 0.8,
    min_depth: float = 0.05,
    max_depth: float = 5.0,
    sensor_valid_max: float = 4.9,
) -> torch.Tensor:
    """Two-channel mixed depth and privileged edge map for the v2 Critic."""
    return _student_depth_edge_mixed_pair(
        env, sensor_cfg, checkpoint_path, enhance_probability,
        min_depth, max_depth, sensor_valid_max,
    )
