from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP


class ActorCriticGRU(nn.Module):
    """Actor-critic that encodes five-frame depth and proprioception with a GRU."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        map_scan_dim=(42, 42, 1),
        cnn_output_dim=64,
        gru_hidden_size=256,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(f"ActorCriticGRU ignored unexpected arguments: {list(kwargs)}")

        self.obs_groups = obs_groups
        self.L, self.W, self.coord_dim = map_scan_dim
        self.map_size = self.L * self.W * self.coord_dim
        actor_dim = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        critic_dim = sum(obs[group].shape[-1] for group in obs_groups["critic"])
        self.actor_proprio_dim = actor_dim - self.map_size
        self.critic_proprio_dim = critic_dim - self.map_size
        if min(self.actor_proprio_dim, self.critic_proprio_dim) <= 0:
            raise ValueError("Policy and critic observations must contain proprioception followed by a depth image.")

        self.map_cnn = nn.Sequential(
            nn.Conv2d(self.coord_dim, 16, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, cnn_output_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cnn_output_dim),
        )
        self.actor_proprio_embedding = nn.Sequential(nn.Linear(self.actor_proprio_dim, cnn_output_dim), nn.ELU())
        self.critic_proprio_embedding = nn.Sequential(nn.Linear(self.critic_proprio_dim, cnn_output_dim), nn.ELU())
        self.gru = nn.GRU(cnn_output_dim * 2, gru_hidden_size, batch_first=True)
        self.actor = MLP(gru_hidden_size, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(gru_hidden_size, 1, critic_hidden_dims, activation)
        self.velocity_head = MLP(cnn_output_dim, 3, [128, 64], activation)

        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(self.actor_proprio_dim) if actor_obs_normalization else nn.Identity()
        )
        self.critic_obs_normalizer = (
            EmpiricalNormalization(self.critic_proprio_dim) if critic_obs_normalization else nn.Identity()
        )
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}")
        self.distribution = None
        Normal.set_default_validate_args(False)

    def _get_obs(self, obs, groups):
        return torch.cat([obs[group] for group in groups], dim=-1)

    def _encode(self, obs, branch):
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)
        if obs.ndim != 3:
            raise ValueError(f"Expected observations shaped [batch, history, features], got {tuple(obs.shape)}")

        proprio_dim = self.actor_proprio_dim if branch == "actor" else self.critic_proprio_dim
        normalizer = self.actor_obs_normalizer if branch == "actor" else self.critic_obs_normalizer
        proprio = normalizer(obs[..., :-self.map_size])
        depth = obs[..., -self.map_size :].reshape(-1, self.W, self.L, self.coord_dim).permute(0, 3, 1, 2)
        map_features = self.map_cnn(depth).mean(dim=(2, 3))
        map_features = map_features.reshape(obs.shape[0], obs.shape[1], -1)
        embedding_layer = self.actor_proprio_embedding if branch == "actor" else self.critic_proprio_embedding
        proprio_embedding = embedding_layer(proprio.reshape(-1, proprio_dim))
        proprio_embedding = proprio_embedding.reshape(obs.shape[0], obs.shape[1], -1)
        sequence = torch.cat((map_features, proprio_embedding), dim=-1)
        encoded, _ = self.gru(sequence)
        return encoded[:, -1], proprio_embedding[:, -1]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        encoded, _ = self._encode(obs, "actor")
        mean = self.actor(encoded)
        std = (self.std if self.noise_std_type == "scalar" else torch.exp(self.log_std)).expand_as(mean)
        self.distribution = Normal(mean, std)

    def get_actor_obs(self, obs):
        return self._get_obs(obs, self.obs_groups["policy"])

    def get_critic_obs(self, obs):
        return self._get_obs(obs, self.obs_groups["critic"])

    def act(self, obs, **kwargs):
        self.update_distribution(self.get_actor_obs(obs))
        return self.distribution.sample()

    def act_inference(self, obs):
        encoded, _ = self._encode(self.get_actor_obs(obs), "actor")
        return self.actor(encoded)

    def evaluate(self, obs, **kwargs):
        encoded, _ = self._encode(self.get_critic_obs(obs), "critic")
        return self.critic(encoded)

    def compute_auxiliary_losses(self, obs):
        policy_obs = self.get_actor_obs(obs)
        _, proprio_embedding = self._encode(policy_obs, "actor")
        critic_obs = self.get_critic_obs(obs)
        target_velocity = critic_obs[..., :3]
        if target_velocity.ndim == 3:
            target_velocity = target_velocity[:, -1]
        velocity_prediction = self.velocity_head(proprio_embedding)
        return {"velocity": torch.nn.functional.mse_loss(velocity_prediction, target_velocity.detach())}

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)[..., :-self.map_size]
            self.actor_obs_normalizer.update(actor_obs.reshape(-1, self.actor_proprio_dim))
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)[..., :-self.map_size]
            self.critic_obs_normalizer.update(critic_obs.reshape(-1, self.critic_proprio_dim))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
