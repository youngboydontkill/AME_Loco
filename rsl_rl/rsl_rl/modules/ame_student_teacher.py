"""AME teacher-student policy used by the S54 height-scan to depth task."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic_encoder import ActorCriticEncoder


class _AMEActor(nn.Module):
    """Actor-only wrapper around the AME encoder used for distillation."""

    is_recurrent = False

    def __init__(self, obs, obs_groups, num_actions, map_scan_dim, hidden_dims, activation, **kwargs):
        super().__init__()
        self.obs_groups = obs_groups
        self._encoder = ActorCriticEncoder(
            obs,
            {"policy": obs_groups["policy"], "critic": obs_groups["policy"]},
            num_actions,
            actor_hidden_dims=hidden_dims,
            critic_hidden_dims=hidden_dims,
            activation=activation,
            init_noise_std=1.0,
            map_scan_dim=map_scan_dim,
            **kwargs,
        )

    def get_obs(self, obs):
        if not torch.is_tensor(obs):
            return torch.cat([obs[group] for group in self.obs_groups["policy"]], dim=-1)
        return obs

    def forward(self, obs):
        obs = self.get_obs(obs)
        obs = self._encoder.actor_obs_normalizer(obs)
        encoded, _ = self._encoder._encode_terrain(obs)
        return self._encoder.actor(encoded)

    @property
    def obs_normalizer(self):
        return self._encoder.actor_obs_normalizer


class AMEStudentTeacher(nn.Module):
    """Distill a height-scan AME policy into a depth AME policy."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        student_map_scan_dim=(42, 42, 1),
        teacher_map_scan_dim=(33, 21, 3),
        student_hidden_dims=(512, 256, 128),
        teacher_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_cnn_downsample=True,
        teacher_cnn_downsample=True,
        student_mha_dim=64,
        teacher_mha_dim=64,
        student_num_heads=16,
        teacher_num_heads=16,
        student_attach_global=False,
        teacher_attach_global=False,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(f"AMEStudentTeacher ignored arguments: {sorted(kwargs)}")
        self.obs_groups = obs_groups
        self.loaded_teacher = False
        self.student = _AMEActor(
            obs,
            {"policy": obs_groups["policy"]},
            num_actions,
            student_map_scan_dim,
            student_hidden_dims,
            activation,
            cnn_downsample=student_cnn_downsample,
            mha_dim=student_mha_dim,
            num_heads=student_num_heads,
            attach_global=student_attach_global,
        )
        self.teacher = _AMEActor(
            obs,
            {"policy": obs_groups["teacher"]},
            num_actions,
            teacher_map_scan_dim,
            teacher_hidden_dims,
            activation,
            cnn_downsample=teacher_cnn_downsample,
            mha_dim=teacher_mha_dim,
            num_heads=teacher_num_heads,
            attach_global=teacher_attach_global,
        )
        self.student_obs_normalization = student_obs_normalization
        self.teacher_obs_normalization = teacher_obs_normalization
        self.student_obs_normalizer = self.student.obs_normalizer if student_obs_normalization else nn.Identity()
        self.teacher_obs_normalizer = self.teacher.obs_normalizer if teacher_obs_normalization else nn.Identity()
        self.student._encoder.actor_obs_normalizer = self.student_obs_normalizer
        self.teacher._encoder.actor_obs_normalizer = self.teacher_obs_normalizer
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}")
        self.distribution = None
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None, hidden_states=None):
        pass

    def get_student_obs(self, obs):
        return self.student.get_obs(obs)

    def get_teacher_obs(self, obs):
        return self.teacher.get_obs(obs)

    def update_distribution(self, obs):
        mean = self.student(obs)
        std = self.std.expand_as(mean) if self.noise_std_type == "scalar" else torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        return self.student(obs)

    def evaluate(self, obs):
        with torch.no_grad():
            return self.teacher(obs)

    def update_normalization(self, obs):
        if self.student_obs_normalization:
            self.student_obs_normalizer.update(self.get_student_obs(obs))

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def load_state_dict(self, state_dict, strict=True):
        """Load either a PPO actor checkpoint or a complete distillation checkpoint."""
        if any(key.startswith("student.") for key in state_dict):
            result = super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
        else:
            teacher_state = {}
            for key, value in state_dict.items():
                if key.startswith("actor."):
                    teacher_state[key] = value
                elif key.startswith(
                    ("map_cnn.", "mha.", "actor_proprio_embedding.", "global_encoder.", "query_projector.")
                ):
                    teacher_state[key] = value
                elif key.startswith("actor_obs_normalizer."):
                    teacher_state[key] = value
            self.teacher._encoder.load_state_dict(teacher_state, strict=False)
            # PPO checkpoints also contain critic-only parameters.  They are
            # intentionally ignored because distillation only needs the actor.
            actor_keys = [key for key in teacher_state if key.startswith("actor.")]
            if not actor_keys:
                raise RuntimeError("Teacher checkpoint does not contain AME actor parameters")
            result = False
            self.loaded_teacher = True
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        return result
