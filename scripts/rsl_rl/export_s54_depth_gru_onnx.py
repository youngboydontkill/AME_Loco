"""Export the S54 depth GRU policy without launching Isaac Sim.

Inputs match the multi-frame DepthWalkControllerV2 interface: ``obs`` [1, 450]
contains five chronological 90-value proprio frames, and ``actor_depth``
[1, 5, 1, 42, 42] contains the corresponding depth frames in the same order.
The caller owns history initialization and rolling. The adapter only converts
joint order between deployment and training. Output [1, 27] is the raw policy
action in deployment joint order.

Example:
python scripts/rsl_rl/export_s54_depth_gru_onnx.py \
  --checkpoint logs/rsl_rl/kuavo_s54_depth_enhance_gru_amp/model_40000.pt \
  --output logs/rsl_rl/kuavo_s54_depth_enhance_gru_amp/exported/policy.onnx
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

from rsl_rl.modules import ActorCriticGRU


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
HISTORY_LENGTH = 5
PROPRIO_DIM = 90
DEPTH_SIZE = 42
DEPTH_DIM = DEPTH_SIZE * DEPTH_SIZE


class S54DepthGRUOnnx(nn.Module):
    """Adapt chronological deployment inputs to the trained GRU actor."""

    def __init__(self, policy: ActorCriticGRU):
        super().__init__()
        if policy.actor_proprio_dim != PROPRIO_DIM:
            raise ValueError(
                f"Checkpoint actor expects {policy.actor_proprio_dim} proprio values per frame; "
                f"expected {PROPRIO_DIM}."
            )
        if policy.map_size != DEPTH_DIM:
            raise ValueError(f"Checkpoint depth map has {policy.map_size} pixels, expected {DEPTH_DIM}.")
        if policy.gru.input_size != 128 or policy.gru.hidden_size != 256:
            raise ValueError("Checkpoint GRU dimensions do not match the S54 depth GRU export interface.")

        self.map_cnn = policy.map_cnn.cpu().eval()
        self.actor_proprio_embedding = policy.actor_proprio_embedding.cpu().eval()
        self.actor_obs_normalizer = policy.actor_obs_normalizer.cpu().eval()
        self.gru = policy.gru.cpu().eval()
        self.actor = policy.actor.cpu().eval()
        self.register_buffer("isaac_to_deployment", torch.tensor(ISAAC_TO_DEPLOYMENT, dtype=torch.long))
        self.register_buffer("deployment_to_isaac", torch.tensor(DEPLOYMENT_TO_ISAAC, dtype=torch.long))

    def forward(self, obs: torch.Tensor, actor_depth: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        proprio = obs.reshape(batch_size, HISTORY_LENGTH, PROPRIO_DIM)

        joint_pos = proprio[..., 9:36].index_select(-1, self.isaac_to_deployment)
        joint_vel = proprio[..., 36:63].index_select(-1, self.isaac_to_deployment)
        last_action = proprio[..., 63:90].index_select(-1, self.isaac_to_deployment)
        training_proprio = torch.cat((proprio[..., :9], joint_pos, joint_vel, last_action), dim=-1)
        training_proprio = self.actor_obs_normalizer(training_proprio.reshape(-1, PROPRIO_DIM))

        depth = actor_depth.reshape(batch_size * HISTORY_LENGTH, 1, DEPTH_SIZE, DEPTH_SIZE)
        map_features = self.map_cnn(depth).mean(dim=(2, 3))
        proprio_features = self.actor_proprio_embedding(training_proprio)
        sequence = torch.cat(
            (
                map_features.reshape(batch_size, HISTORY_LENGTH, -1),
                proprio_features.reshape(batch_size, HISTORY_LENGTH, -1),
            ),
            dim=-1,
        )
        encoded, _ = self.gru(sequence)
        action_isaac = self.actor(encoded[:, -1])
        return action_isaac.index_select(1, self.deployment_to_isaac)


def make_policy() -> ActorCriticGRU:
    actor_obs = torch.zeros(1, HISTORY_LENGTH, PROPRIO_DIM + DEPTH_DIM)
    critic_obs = torch.zeros(1, PROPRIO_DIM + 3 + DEPTH_DIM)
    obs = {"policy": actor_obs, "critic": critic_obs}
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    return ActorCriticGRU(
        obs,
        obs_groups,
        num_actions=len(ISAAC_JOINT_NAMES),
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        map_scan_dim=(DEPTH_SIZE, DEPTH_SIZE, 1),
        cnn_output_dim=64,
        gru_hidden_size=256,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="RSL-RL checkpoint, such as model_40000.pt")
    parser.add_argument(
        "--output", default=None, help="ONNX path; defaults to <checkpoint-dir>/exported/policy.onnx"
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    checkpoint = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy = make_policy()
    policy.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
    exporter = S54DepthGRUOnnx(policy).eval()

    output_path = args.output or os.path.join(os.path.dirname(checkpoint), "exported", "policy.onnx")
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    example_obs = torch.zeros(1, HISTORY_LENGTH * PROPRIO_DIM, dtype=torch.float32)
    example_depth = torch.zeros(1, HISTORY_LENGTH, 1, DEPTH_SIZE, DEPTH_SIZE, dtype=torch.float32)
    torch.onnx.export(
        exporter,
        (example_obs, example_depth),
        output_path,
        export_params=True,
        opset_version=args.opset,
        input_names=["obs", "actor_depth"],
        output_names=["action"],
    )

    print(f"Exported: {output_path}")
    print(f"Inputs: obs [1, {HISTORY_LENGTH * PROPRIO_DIM}], actor_depth [1, {HISTORY_LENGTH}, 1, 42, 42]")
    print(f"Output: action [1, {len(DEPLOYMENT_JOINT_NAMES)}], raw policy action")
    print("Frame order: oldest to newest; output uses the newest GRU state")
    print("Deployment joint order: " + ", ".join(DEPLOYMENT_JOINT_NAMES))


if __name__ == "__main__":
    main()
