"""Export the S54 depth-enhancement actor with separate depth and proprio inputs."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export an AME S54 depth-enhancement policy to ONNX.")
parser.add_argument("--task", type=str, default="AME-Kuavo-S54-Depth-enhancement-Play")
parser.add_argument("--checkpoint", type=str, required=True, help="RSL-RL policy checkpoint.")
parser.add_argument("--output", type=str, default=None, help="Output ONNX path (default: beside checkpoint).")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import copy
import torch
import torch.nn as nn
import gymnasium as gym

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import ame_locomotion.tasks  # noqa: F401


LAB_TO_GYM = (0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 24, 3, 7, 11, 15, 19, 23, 25)
GYM_TO_LAB = tuple(LAB_TO_GYM.index(i) for i in range(26))
PROPRIO_DIM = 3 + 3 + 3 + 26 + 26 + 26


class S54DepthEnhancementOnnx(nn.Module):
    """ONNX wrapper: Gym-order proprio and normalized depth in, Gym-order actions out."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        if policy.is_recurrent:
            raise ValueError("The S54 depth-enhancement exporter only supports feed-forward policies.")
        if policy.actor_proprio_dim != PROPRIO_DIM:
            raise ValueError(f"Expected {PROPRIO_DIM} proprio features, got {policy.actor_proprio_dim}.")
        if tuple(policy.actor_map_scan_dim) != (42, 42, 2):
            raise ValueError(f"Expected actor map shape (42, 42, 2), got {policy.actor_map_scan_dim}.")

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

        self.register_buffer("gym_to_lab", torch.tensor(GYM_TO_LAB, dtype=torch.long))
        self.register_buffer("lab_to_gym", torch.tensor(LAB_TO_GYM, dtype=torch.long))

    def _gym_obs_to_lab(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                obs[:, :9],
                obs[:, 9:35][:, self.gym_to_lab],
                obs[:, 35:61][:, self.gym_to_lab],
                obs[:, 61:87][:, self.gym_to_lab],
            ),
            dim=1,
        )

    def forward(self, actor_depth: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        proprio = self._gym_obs_to_lab(obs)
        map_depth = actor_depth.reshape(-1, 42, 42, 1)
        # The two-input deployment interface has no confidence input; treat pixels as valid.
        confidence = torch.ones_like(map_depth)
        map_scan = torch.cat((map_depth, confidence), dim=-1)

        # Match ActorCriticEncoder's stored [L, W, C] reshape and CNN path.
        map_scan = map_scan.reshape(-1, 42, 42, 2).permute(0, 3, 1, 2)
        cnn_features = self.map_cnn(map_scan)
        cnn_height, cnn_width = cnn_features.shape[-2:]
        cnn_features = cnn_features.permute(0, 2, 3, 1).reshape(-1, cnn_height * cnn_width, self.mha_dim)

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

        actions_lab = self.actor(encoded_obs)
        return actions_lab[:, self.lab_to_gym]


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    checkpoint = retrieve_file_path(args_cli.checkpoint)
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    agent_cfg.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.alg.policy

    export_model = S54DepthEnhancementOnnx(policy).eval()
    output_path = args_cli.output or os.path.join(os.path.dirname(checkpoint), "exported", "policy.onnx")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    depth = torch.zeros(1, 42, 42, dtype=torch.float32)
    obs = torch.zeros(1, PROPRIO_DIM, dtype=torch.float32)
    torch.onnx.export(
        export_model,
        (depth, obs),
        output_path,
        export_params=True,
        opset_version=17,
        input_names=["actor_depth", "obs"],
        output_names=["actions"],
        dynamic_axes={"actor_depth": {0: "batch"}, "obs": {0: "batch"}, "actions": {0: "batch"}},
    )
    print(f"[INFO] Exported ONNX policy to: {output_path}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
