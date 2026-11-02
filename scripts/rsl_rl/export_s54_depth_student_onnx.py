"""Export the S54 CNN/props student actor without launching Isaac Sim.

Inputs match the S54 deployment interface with batch size 1: ``obs`` [1, 450]
contains five chronological 90-value proprioception frames, and ``actor_depth``
[1, 1, 42, 42] contains the current processed depth image. The caller owns depth
preprocessing, history management, and action scaling. Proprioception is
supplied in deployment joint order; the exported model converts it to Isaac
training order and returns actions in deployment order.

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
PROPS_HISTORY_LENGTH = 5
DEPTH_FEATURE_DIM = 128
PROPS_FEATURE_DIM = 128
ACTION_DIM = 27
PROPRIO_DIM = 90
DEPLOYMENT_JOINT_NAMES = (
    "leg_l1_joint",
    "leg_l2_joint",
    "leg_l3_joint",
    "leg_l4_joint",
    "leg_l5_joint",
    "leg_l6_joint",
    "leg_r1_joint",
    "leg_r2_joint",
    "leg_r3_joint",
    "leg_r4_joint",
    "leg_r5_joint",
    "leg_r6_joint",
    "waist_yaw_joint",
    "zarm_l1_joint",
    "zarm_l2_joint",
    "zarm_l3_joint",
    "zarm_l4_joint",
    "zarm_l5_joint",
    "zarm_l6_joint",
    "zarm_l7_joint",
    "zarm_r1_joint",
    "zarm_r2_joint",
    "zarm_r3_joint",
    "zarm_r4_joint",
    "zarm_r5_joint",
    "zarm_r6_joint",
    "zarm_r7_joint",
)
ISAAC_JOINT_NAMES = (
    "leg_l1_joint",
    "leg_r1_joint",
    "waist_yaw_joint",
    "leg_l2_joint",
    "leg_r2_joint",
    "zarm_l1_joint",
    "zarm_r1_joint",
    "leg_l3_joint",
    "leg_r3_joint",
    "zarm_l2_joint",
    "zarm_r2_joint",
    "leg_l4_joint",
    "leg_r4_joint",
    "zarm_l3_joint",
    "zarm_r3_joint",
    "leg_l5_joint",
    "leg_r5_joint",
    "zarm_l4_joint",
    "zarm_r4_joint",
    "leg_l6_joint",
    "leg_r6_joint",
    "zarm_l5_joint",
    "zarm_r5_joint",
    "zarm_l6_joint",
    "zarm_r6_joint",
    "zarm_l7_joint",
    "zarm_r7_joint",
)
ISAAC_TO_DEPLOYMENT = tuple(DEPLOYMENT_JOINT_NAMES.index(name) for name in ISAAC_JOINT_NAMES)
DEPLOYMENT_TO_ISAAC = tuple(ISAAC_JOINT_NAMES.index(name) for name in DEPLOYMENT_JOINT_NAMES)


class S54DepthStudentOnnx(nn.Module):
    """Student actor adapted to the S54 deployment input and joint order."""

    def __init__(self, state_dict: dict[str, torch.Tensor], props_dim: int):
        super().__init__()
        if props_dim != PROPRIO_DIM:
            raise ValueError(f"S54 props must contain {PROPRIO_DIM} values per frame, got {props_dim}.")
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
        self.register_buffer("isaac_to_deployment", torch.tensor(ISAAC_TO_DEPLOYMENT, dtype=torch.long))
        self.register_buffer("deployment_to_isaac", torch.tensor(DEPLOYMENT_TO_ISAAC, dtype=torch.long))
        self.eval()

    def forward(self, obs: torch.Tensor, actor_depth: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        depth_features = self.depth_encoder(actor_depth.reshape(batch_size, 1, DEPTH_SIZE, DEPTH_SIZE))
        props = obs.reshape(batch_size, PROPS_HISTORY_LENGTH, PROPRIO_DIM)
        joint_pos = props[..., 9:36].index_select(-1, self.isaac_to_deployment)
        joint_vel = props[..., 36:63].index_select(-1, self.isaac_to_deployment)
        last_action = props[..., 63:90].index_select(-1, self.isaac_to_deployment)
        training_props = torch.cat((props[..., :9], joint_pos, joint_vel, last_action), dim=-1)
        props_features = self.props_encoder(training_props.reshape(batch_size, -1))
        action_isaac = self.actor(torch.cat((depth_features, props_features), dim=-1))
        return action_isaac.index_select(1, self.deployment_to_isaac)


def _submodule_state(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    result = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
    if not result:
        raise ValueError(f"Checkpoint does not contain {prefix[:-1]} weights")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="DAgger or PPO student checkpoint")
    parser.add_argument("--output", default=None, help="ONNX path; defaults to <checkpoint-dir>/exported/policy.onnx")
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
    if inferred_props_dim != PROPRIO_DIM:
        raise ValueError(f"Checkpoint expects props_dim={inferred_props_dim}; S54 deployment requires {PROPRIO_DIM}")

    policy = S54DepthStudentOnnx(state_dict, inferred_props_dim)
    output_path = args.output or os.path.join(os.path.dirname(checkpoint), "exported", "policy.onnx")
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    example_obs = torch.zeros(1, PROPS_HISTORY_LENGTH * PROPRIO_DIM, dtype=torch.float32)
    example_depth = torch.zeros(1, 1, DEPTH_SIZE, DEPTH_SIZE, dtype=torch.float32)
    torch.onnx.export(
        policy,
        (example_obs, example_depth),
        output_path,
        export_params=True,
        opset_version=args.opset,
        input_names=["obs", "actor_depth"],
        output_names=["action"],
    )

    print(f"Exported: {output_path}")
    print(
        f"Inputs: obs [1, {PROPS_HISTORY_LENGTH * PROPRIO_DIM}], "
        f"actor_depth [1, 1, {DEPTH_SIZE}, {DEPTH_SIZE}]"
    )
    print(f"Output: action [1, {ACTION_DIM}], raw policy action in deployment joint order")
    print("Deployment joint order: " + ", ".join(DEPLOYMENT_JOINT_NAMES))
    print("Depth must use the task's mixed_student_depth_image preprocessing.")


if __name__ == "__main__":
    main()
