from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP
from rsl_rl.modules.actor_critic_encoder import ActorCriticEncoder


class ActorCriticCNNProps(nn.Module):
    """PPO actor-critic reusing the DAgger depth and proprioceptive student encoders."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        student_depth_group="depth",
        student_props_group="props",
        critic_group="teacher_critic",
        teacher_group="teacher",
        teacher_map_scan_dim=(33, 21, 3),
        depth_feature_dim=128,
        props_hidden_dims=(512, 256),
        props_feature_dim=128,
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=0.1,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            raise TypeError(f"Unexpected ActorCriticCNNProps arguments: {sorted(kwargs)}")
        self.obs_groups = obs_groups
        self.student_depth_group = student_depth_group
        self.student_props_group = student_props_group
        self.critic_group = critic_group
        self.teacher_group = teacher_group
        if actor_obs_normalization or critic_obs_normalization:
            raise ValueError("ActorCriticCNNProps uses the DAgger student inputs without observation normalization")

        depth_shape = obs[student_depth_group].shape[1:]
        props_shape = obs[student_props_group].shape[1:]
        if len(depth_shape) != 1 or depth_shape[0] != 42 * 42:
            raise ValueError(f"Expected flat 42x42 depth input, got {tuple(depth_shape)}")
        if len(props_shape) != 2 or props_shape[0] != 5:
            raise ValueError(f"Expected props with shape (batch, 5, features), got {tuple(props_shape)}")

        self.depth_height = 42
        self.depth_width = 42
        self.props_history_length, props_dim = props_shape
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, depth_feature_dim),
            nn.ELU(),
        )
        self.props_encoder = MLP(
            self.props_history_length * props_dim, props_feature_dim, list(props_hidden_dims), activation
        )
        self.actor = MLP(depth_feature_dim + props_feature_dim, num_actions, list(actor_hidden_dims), activation)
        critic_dim = sum(obs[group].shape[-1] for group in obs_groups["critic"])
        self.critic = MLP(critic_dim, 1, list(critic_hidden_dims), activation)

        if noise_std_type != "scalar":
            raise ValueError(f"Unsupported noise_std_type: {noise_std_type}; expected 'scalar'")
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        self.teacher = ActorCriticEncoder(
            obs,
            {"policy": [teacher_group], "critic": [critic_group]},
            num_actions,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            init_noise_std=1.0,
            map_scan_dim=teacher_map_scan_dim,
            cnn_downsample=True,
        )
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()
        self.loaded_teacher = False
        Normal.set_default_validate_args(False)

    def _encode_actor(self, obs):
        depth = obs[self.student_depth_group].reshape(-1, 1, self.depth_height, self.depth_width)
        props = obs[self.student_props_group]
        if props.shape[1] != self.props_history_length:
            raise ValueError(f"Expected {self.props_history_length} proprioception frames, got {props.shape[1]}")
        depth_features = self.depth_encoder(depth)
        props_features = self.props_encoder(props.flatten(start_dim=1))
        return torch.cat((depth_features, props_features), dim=-1)

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
        mean = self.actor(self._encode_actor(obs))
        self.distribution = Normal(mean, self.std.expand_as(mean))

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        return self.actor(self._encode_actor(obs))

    def evaluate(self, obs, **kwargs):
        critic_obs = torch.cat([obs[group] for group in self.obs_groups["critic"]], dim=-1)
        return self.critic(critic_obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        pass

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def compute_bc_loss(self, obs):
        if not self.loaded_teacher:
            raise RuntimeError("BC loss is enabled, but no teacher weights were loaded from a checkpoint")
        student_actions = self.act_inference(obs)
        with torch.no_grad(), torch.amp.autocast(device_type=student_actions.device.type, enabled=False):
            teacher_actions, _ = self.teacher.act_inference({self.teacher_group: obs[self.teacher_group]})
        return torch.nn.functional.mse_loss(student_actions, teacher_actions)

    def load_student_checkpoint(self, path, map_location="cpu"):
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise TypeError("Student checkpoint must contain a model_state_dict mapping")

        current_state = self.state_dict()
        student_prefixes = ("depth_encoder.", "props_encoder.", "actor.", "critic.", "std")
        student_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(student_prefixes) and key in current_state and current_state[key].shape == value.shape
        }
        required_prefixes = ("depth_encoder.", "props_encoder.", "actor.")
        missing = [prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in student_state)]
        if missing:
            raise ValueError(
                f"Checkpoint does not contain compatible student weights for {missing}: {path}"
            )
        teacher_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith("teacher.") and key in current_state and current_state[key].shape == value.shape
        }
        required_teacher_keys = {key for key in current_state if key.startswith("teacher.")}
        missing_teacher_keys = required_teacher_keys - teacher_state.keys()
        if missing_teacher_keys:
            preview = sorted(missing_teacher_keys)[:5]
            raise ValueError(f"Checkpoint is missing compatible teacher weights {preview}: {path}")
        student_state.update(teacher_state)
        self.load_state_dict(student_state, strict=False)
        self.loaded_teacher = True
        self.teacher.eval()
        return sorted(student_state)

    def load_state_dict(self, state_dict, strict=True):
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        result = super().load_state_dict(state_dict, strict=strict)
        self.loaded_teacher = any(key.startswith("teacher.") for key in state_dict)
        self.teacher.eval()
        return result
