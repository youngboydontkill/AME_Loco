from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP
from rsl_rl.modules.actor_critic_encoder import ActorCriticEncoder


class StudentTeacherCNNMLP(nn.Module):
    """Depth CNN and five-frame proprioceptive MLP student with encoder teacher."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        student_depth_group="depth",
        student_props_group="props",
        teacher_map_scan_dim=(33, 21, 3),
        depth_feature_dim=128,
        props_hidden_dims=(512, 256),
        props_feature_dim=128,
        actor_hidden_dims=(512, 256, 128),
        velocity_hidden_dims=(128,),
        activation="elu",
        init_noise_std=0.1,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            raise TypeError(f"Unexpected StudentTeacherCNNMLP arguments: {sorted(kwargs)}")

        self.obs_groups = obs_groups
        self.student_depth_group = student_depth_group
        self.student_props_group = student_props_group
        depth_shape = obs[student_depth_group].shape[1:]
        props_shape = obs[student_props_group].shape[1:]
        if len(depth_shape) != 1 or depth_shape[0] != 42 * 42:
            raise ValueError(f"Expected flat 42x42 depth input, got {tuple(depth_shape)}")
        if len(props_shape) != 2:
            raise ValueError(f"Expected props with shape (batch, history, features), got {tuple(props_shape)}")
        if props_shape[0] != 5:
            raise ValueError(f"Expected five proprioception frames, got {props_shape[0]}")

        self.depth_height = 42
        self.depth_width = 42
        self.props_history_length = props_shape[0]
        props_dim = props_shape[1]
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
        self.velocity_head = MLP(props_feature_dim, 3, list(velocity_hidden_dims), activation)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None

        self.teacher = ActorCriticEncoder(
            obs,
            {"policy": ["teacher"], "critic": ["teacher_critic"]},
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

    def _encode_student(self, obs):
        depth = obs[self.student_depth_group].reshape(-1, 1, self.depth_height, self.depth_width)
        props = obs[self.student_props_group]
        if props.shape[1] != self.props_history_length:
            raise ValueError(
                f"Expected {self.props_history_length} proprioception frames, got {props.shape[1]}"
            )
        depth_features = self.depth_encoder(depth)
        props_features = self.props_encoder(props.flatten(start_dim=1))
        return depth_features, props_features

    def act_inference(self, obs):
        depth_features, props_features = self._encode_student(obs)
        return self.actor(torch.cat((depth_features, props_features), dim=-1))

    def act(self, obs):
        mean = self.act_inference(obs)
        self.distribution = Normal(mean, self.std.expand_as(mean))
        return self.distribution.sample()

    def evaluate(self, obs):
        with torch.no_grad():
            actions, _ = self.teacher.act_inference({"teacher": obs["teacher"]})
        return actions

    def compute_auxiliary_losses(self, obs):
        _, props_features = self._encode_student(obs)
        predicted_velocity = self.velocity_head(props_features)
        target_velocity = obs["teacher_critic"][..., :3].detach()
        return {"velocity": torch.nn.functional.mse_loss(predicted_velocity, target_velocity)}

    @property
    def action_std(self):
        return self.std

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None, hidden_states=None):
        pass

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass

    def update_normalization(self, obs):
        pass

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def load_state_dict(self, state_dict, strict=True):
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        if any(key.startswith("teacher.") for key in state_dict):
            nn.Module.load_state_dict(self, state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            return True
        self.teacher.load_state_dict(state_dict, strict=strict)
        self.loaded_teacher = True
        self.teacher.eval()
        return False
