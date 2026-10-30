"""Export the AME Kuavo S54 depth policy without launching Isaac Sim.

Inputs match the DepthWalkControllerV2 interface: ``obs`` [1, 90] and
``actor_depth`` [1, 1, 42, 42]. Configure the deployment .info parameters to
match training scales, reference pose, action scale, and depth normalization.
The adapter only converts joint order between deployment and training. Output
[1, 27] is the raw policy action in deployment joint order.

指令：
python scripts/rsl_rl/export_s54_depth_onnx.py \
  --checkpoint logs/rsl_rl/kuavo_s54_depth_ame/test/model_66000.pt \
  --output logs/rsl_rl/kuavo_s54_depth_ame/test/policy_attention_66_newer.onnx

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn

# Prefer this repository's custom RSL-RL fork over any globally installed copy.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "rsl_rl"))

from rsl_rl.modules import ActorCriticEncoder


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
PROPRIO_DIM = 90
DEPTH_SIZE = 42
DEPTH_DIM = DEPTH_SIZE * DEPTH_SIZE


class S54DepthOnnx(nn.Module):
    """Adapt DepthWalkControllerV2 inputs/outputs to the trained actor."""

    def __init__(self, policy: ActorCriticEncoder):
        super().__init__()
        if policy.is_recurrent:
            raise ValueError("The S54 depth actor must be feed-forward.")
        if policy.actor_proprio_dim != PROPRIO_DIM:
            raise ValueError(
                f"Checkpoint actor expects {policy.actor_proprio_dim} proprio values; expected {PROPRIO_DIM}."
            )
        if tuple(policy.map_scan_dim) != (DEPTH_SIZE, DEPTH_SIZE, 1):
            raise ValueError(f"Checkpoint map shape is {policy.map_scan_dim}, expected (42, 42, 1).")

        self.actor = policy.actor.cpu().eval()
        self.map_cnn = policy.map_cnn.cpu().eval()
        self.mha = policy.mha.cpu().eval()
        self.actor_proprio_embedding = policy.actor_proprio_embedding.cpu().eval()
        self.actor_obs_normalizer = policy.actor_obs_normalizer.cpu().eval()
        self.register_buffer("isaac_to_deployment", torch.tensor(ISAAC_TO_DEPLOYMENT, dtype=torch.long))
        self.register_buffer("deployment_to_isaac", torch.tensor(DEPLOYMENT_TO_ISAAC, dtype=torch.long))

    def forward(self, obs: torch.Tensor, actor_depth: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        # The deployment .info applies training observation scales and uses the
        # training reference pose before assembling obs. Only joint ordering
        # differs: the imported training articulation follows URDF tree order.
        joint_pos = obs[:, 9:36].index_select(1, self.isaac_to_deployment)
        joint_vel = obs[:, 36:63].index_select(1, self.isaac_to_deployment)
        last_action = obs[:, 63:90].index_select(1, self.isaac_to_deployment)
        training_obs = torch.cat((obs[:, :9], joint_pos, joint_vel, last_action), dim=1)
        proprio = self.actor_obs_normalizer(training_obs)
        map_image = actor_depth.reshape(batch_size, 1, DEPTH_SIZE, DEPTH_SIZE)

        cnn_features = self.map_cnn(map_image)
        local_features = cnn_features.permute(0, 2, 3, 1).reshape(
            batch_size, -1, cnn_features.shape[1]
        )
        query = self.actor_proprio_embedding(proprio).unsqueeze(1)
        mha_output = self.mha(query=query, key=local_features, value=local_features, need_weights=False)[0]
        encoded_obs = torch.cat((mha_output.squeeze(1), proprio), dim=-1)
        action_isaac = self.actor(encoded_obs)
        return action_isaac.index_select(1, self.deployment_to_isaac)


def make_policy() -> ActorCriticEncoder:
    actor_obs = torch.zeros(1, PROPRIO_DIM + DEPTH_DIM)
    critic_obs = torch.zeros(1, PROPRIO_DIM + 3 + DEPTH_DIM)
    obs = {"policy": actor_obs, "critic": critic_obs}
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    return ActorCriticEncoder(
        obs,
        obs_groups,
        num_actions=len(ISAAC_JOINT_NAMES),
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        map_scan_dim=(DEPTH_SIZE, DEPTH_SIZE, 1),
        mha_dim=64,
        num_heads=16,
        cnn_downsample=True,
        attach_global=False,
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
    exporter = S54DepthOnnx(policy).eval()

    output_path = args.output or os.path.join(os.path.dirname(checkpoint), "exported", "policy.onnx")
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    example_obs = torch.zeros(1, PROPRIO_DIM, dtype=torch.float32)
    example_depth = torch.zeros(1, 1, DEPTH_SIZE, DEPTH_SIZE, dtype=torch.float32)
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
    print(f"Inputs: obs [1, {PROPRIO_DIM}], actor_depth [1, 1, {DEPTH_SIZE}, {DEPTH_SIZE}]")
    print(f"Output: action [1, {len(DEPLOYMENT_JOINT_NAMES)}], raw policy action")
    print("Deployment joint order: " + ", ".join(DEPLOYMENT_JOINT_NAMES))


if __name__ == "__main__":
    main()
