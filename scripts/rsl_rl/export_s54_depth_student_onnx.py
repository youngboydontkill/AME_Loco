"""Export the S54 CNN/props student actor without launching Isaac Sim.

The inputs are the flattened 42x42 depth observation and the five-frame
proprioceptive observation produced by the MLP-5props task. Depth enhancement,
history management, and action scaling remain the caller's responsibility.

Example:
python scripts/rsl_rl/export_s54_depth_student_onnx.py \
  --checkpoint logs/rsl_rl/kuavo_s54_depth_enhance_dagger_cnn_5props/model_20000.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "rsl_rl"))

from rsl_rl.networks import MLP


DEPTH_SIZE = 42
DEPTH_DIM = DEPTH_SIZE * DEPTH_SIZE
PROPS_HISTORY_LENGTH = 5
DEPTH_FEATURE_DIM = 128
PROPS_FEATURE_DIM = 128
ACTION_DIM = 27


class S54DepthStudentOnnx(nn.Module):
    """Student actor path, exposing depth and proprioception as separate inputs."""

    def __init__(self, state_dict: dict[str, torch.Tensor], props_dim: int):
        super().__init__()
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, DEPTH_FEATURE_DIM),
            nn.ELU(),
        )
        self.props_encoder = MLP(PROPS_HISTORY_LENGTH * props_dim, PROPS_FEATURE_DIM, [512, 256], "elu")
        self.actor = MLP(DEPTH_FEATURE_DIM + PROPS_FEATURE_DIM, ACTION_DIM, [512, 256, 128], "elu")

        self.depth_encoder.load_state_dict(_submodule_state(state_dict, "depth_encoder."), strict=True)
        self.props_encoder.load_state_dict(_submodule_state(state_dict, "props_encoder."), strict=True)
        self.actor.load_state_dict(_submodule_state(state_dict, "actor."), strict=True)
        self.eval()

    def forward(self, depth: torch.Tensor, props: torch.Tensor) -> torch.Tensor:
        batch_size = depth.shape[0]
        depth_features = self.depth_encoder(depth.reshape(batch_size, 1, DEPTH_SIZE, DEPTH_SIZE))
        props_features = self.props_encoder(props.reshape(batch_size, -1))
        return self.actor(torch.cat((depth_features, props_features), dim=-1))


def _submodule_state(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    result = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
    if not result:
        raise ValueError(f"Checkpoint does not contain {prefix[:-1]} weights")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="DAgger or PPO student checkpoint")
    parser.add_argument("--output", default=None, help="ONNX path; defaults to <checkpoint-dir>/exported/policy.onnx")
    parser.add_argument("--props-dim", type=int, default=None, help="Values per proprioception frame; inferred by default")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint must contain a model_state_dict mapping")
    props_encoder_state = _submodule_state(state_dict, "props_encoder.")
    first_weight = props_encoder_state.get("0.weight")
    if first_weight is None:
        raise ValueError("Cannot infer proprioception width from props_encoder.0.weight")
    inferred_props_dim = first_weight.shape[1] // PROPS_HISTORY_LENGTH
    props_dim = args.props_dim or inferred_props_dim
    if props_dim != inferred_props_dim:
        raise ValueError(f"Checkpoint expects props_dim={inferred_props_dim}, got {props_dim}")

    policy = S54DepthStudentOnnx(state_dict, props_dim)
    output_path = args.output or os.path.join(os.path.dirname(checkpoint), "exported", "policy.onnx")
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    example_depth = torch.zeros(1, DEPTH_DIM, dtype=torch.float32)
    example_props = torch.zeros(1, PROPS_HISTORY_LENGTH, props_dim, dtype=torch.float32)
    torch.onnx.export(
        policy,
        (example_depth, example_props),
        output_path,
        export_params=True,
        opset_version=args.opset,
        input_names=["depth", "props"],
        output_names=["action"],
        dynamic_axes={
            "depth": {0: "batch"},
            "props": {0: "batch"},
            "action": {0: "batch"},
        },
    )

    print(f"Exported: {output_path}")
    print(f"Inputs: depth [batch, {DEPTH_DIM}], props [batch, {PROPS_HISTORY_LENGTH}, {props_dim}]")
    print(f"Output: action [batch, {ACTION_DIM}], raw policy action in Isaac joint order")
    print("Depth must use the task's mixed_student_depth_image preprocessing.")


if __name__ == "__main__":
    main()
