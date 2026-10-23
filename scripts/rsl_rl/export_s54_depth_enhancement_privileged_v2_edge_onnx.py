"""Export the v2 edge-privileged S54 depth Actor to ONNX.

The exported policy takes the Actor's mixed depth image and proprioception.
The edge map is a training-only Critic input and is not part of deployment.
"""

import argparse
import copy
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export the S54 v2 edge-privileged Actor to ONNX.")
parser.add_argument(
    "--task",
    type=str,
    default="AME-Kuavo-S54-Depth-enhancement-privileged-v2-edge-Play",
    help="Registered playback task for the v2 edge-privileged policy.",
)
parser.add_argument("--checkpoint", type=str, required=True, help="RSL-RL policy checkpoint.")
parser.add_argument("--output", type=str, default=None, help="Output ONNX path (default: beside checkpoint).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import gymnasium as gym

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import ame_locomotion.tasks  # noqa: F401


IMAGE_HEIGHT = 42
IMAGE_WIDTH = 42
NUM_JOINTS = 27
PROPRIO_DIM = 3 + 3 + 3 + NUM_JOINTS * 3
NUM_ACTIONS = NUM_JOINTS


class S54DepthPrivilegedV2EdgeOnnx(nn.Module):
    """Training-order proprioception and Actor mixed-depth in, actions out."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        if policy.is_recurrent:
            raise ValueError("The S54 depth exporter only supports feed-forward policies.")
        if policy.actor_proprio_dim != PROPRIO_DIM:
            raise ValueError(f"Expected {PROPRIO_DIM} proprio features, got {policy.actor_proprio_dim}.")
        if tuple(policy.actor_map_scan_dim) != (IMAGE_HEIGHT, IMAGE_WIDTH, 1):
            raise ValueError(
                "Expected the v2-edge Actor map shape "
                f"({IMAGE_HEIGHT}, {IMAGE_WIDTH}, 1), got {policy.actor_map_scan_dim}."
            )

        self.actor = copy.deepcopy(policy.actor).cpu().eval()
        self.map_cnn = copy.deepcopy(policy.actor_map_cnn).cpu().eval()
        self.mha = copy.deepcopy(policy.mha).cpu().eval()
        self.actor_proprio_embedding = copy.deepcopy(policy.actor_proprio_embedding).cpu().eval()
        self.actor_obs_normalizer = copy.deepcopy(policy.actor_obs_normalizer).cpu().eval()
        self.mha_dim = policy.mha_dim
        self.attach_global = policy.attach_global
        if self.attach_global:
            self.global_encoder = copy.deepcopy(policy.global_encoder).cpu().eval()
            self.query_projector = copy.deepcopy(policy.query_projector).cpu().eval()

    def forward(self, actor_depth: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        proprio = obs
        map_scan = actor_depth.reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
        # Match ActorCriticEncoder's flattened map reshape and CNN feature order.
        map_image = map_scan.permute(0, 3, 1, 2)
        cnn_features = self.map_cnn(map_image)
        cnn_height, cnn_width = cnn_features.shape[-2:]
        cnn_features = cnn_features.permute(0, 2, 3, 1).reshape(
            -1, cnn_height * cnn_width, self.mha_dim
        )

        y = torch.linspace(-1.0, 1.0, cnn_height, device=cnn_features.device, dtype=cnn_features.dtype)
        x = torch.linspace(-1.0, 1.0, cnn_width, device=cnn_features.device, dtype=cnn_features.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        channels = torch.arange(self.mha_dim, device=cnn_features.device, dtype=cnn_features.dtype)
        frequencies = torch.pow(10000.0, -2.0 * torch.floor(channels / 2.0) / self.mha_dim)
        position = torch.empty(
            (cnn_height * cnn_width, self.mha_dim), device=cnn_features.device, dtype=cnn_features.dtype
        )
        position[:, 0::2] = torch.sin(coords[:, 0:1] * frequencies[0::2])
        position[:, 1::2] = torch.cos(coords[:, 1:2] * frequencies[1::2])
        local_features = cnn_features + position.unsqueeze(0)

        proprio = self.actor_obs_normalizer(proprio)
        proprio_embedding = self.actor_proprio_embedding(proprio)
        if self.attach_global:
            global_features = self.global_encoder(local_features)
            global_features_max = torch.max(global_features, dim=1).values
            query_input = torch.cat((global_features_max, proprio_embedding), dim=-1)
            proprio_embedding = self.query_projector(query_input)

        mha_output = self.mha(
            query=proprio_embedding.unsqueeze(1),
            key=local_features,
            value=local_features,
            need_weights=False,
        )[0].squeeze(1)
        encoded_obs = torch.cat((mha_output, proprio), dim=-1)
        if self.attach_global:
            encoded_obs = torch.cat((global_features_max, encoded_obs), dim=-1)

        return self.actor(encoded_obs)


def main(env_cfg, agent_cfg):
    checkpoint = retrieve_file_path(args_cli.checkpoint)
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    agent_cfg.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint)
        policy = runner.alg.policy
        actor_output = next(layer for layer in reversed(policy.actor) if isinstance(layer, nn.Linear))
        if actor_output.out_features != NUM_ACTIONS:
            raise ValueError(f"Expected {NUM_ACTIONS} actor actions, got {actor_output.out_features}.")

        export_model = S54DepthPrivilegedV2EdgeOnnx(policy).eval()
        output_path = args_cli.output or os.path.join(os.path.dirname(checkpoint), "exported", "policy.onnx")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        actor_depth = torch.zeros(1, IMAGE_HEIGHT, IMAGE_WIDTH, dtype=torch.float32)
        obs = torch.zeros(1, PROPRIO_DIM, dtype=torch.float32)
        torch.onnx.export(
            export_model,
            (actor_depth, obs),
            output_path,
            export_params=True,
            opset_version=17,
            input_names=["actor_depth", "obs"],
            output_names=["actions"],
            dynamic_axes={"actor_depth": {0: "batch"}, "obs": {0: "batch"}, "actions": {0: "batch"}},
        )
        print(f"[INFO] Exported ONNX policy to: {output_path}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        main(env_cfg, agent_cfg)
    finally:
        simulation_app.close()
