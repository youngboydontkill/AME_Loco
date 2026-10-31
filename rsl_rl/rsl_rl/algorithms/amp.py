"""Adversarial motion-prior components used by AMP-style PPO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ExpertMotionDataset:
    """Loads retargeted motion features and samples within-clip transitions."""

    def __init__(self, path: str, device: str | torch.device):
        dataset_path = Path(path).expanduser()
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"AMP motion dataset not found: {dataset_path}. "
                "Prepare a retargeted .npz with states [frames, features] and clip_ids [frames]."
            )
        with np.load(dataset_path) as data:
            if "states" not in data:
                raise ValueError(f"AMP dataset {dataset_path} must contain a 'states' array.")
            states = np.asarray(data["states"], dtype=np.float32)
            clip_ids = np.asarray(data["clip_ids"]) if "clip_ids" in data else np.zeros(len(states), dtype=np.int64)

        if states.ndim != 2 or states.shape[0] < 2:
            raise ValueError(f"AMP states must have shape [frames >= 2, features], got {states.shape}.")
        if clip_ids.shape != (states.shape[0],):
            raise ValueError(f"AMP clip_ids must have shape ({states.shape[0]},), got {clip_ids.shape}.")
        if not np.isfinite(states).all():
            raise ValueError("AMP states contain NaN or infinite values.")

        valid = np.flatnonzero(clip_ids[:-1] == clip_ids[1:])
        if valid.size == 0:
            raise ValueError("AMP dataset must contain at least one pair of consecutive frames in the same clip.")

        self.device = torch.device(device)
        self.states = torch.as_tensor(states, device=self.device)
        self.transition_indices = torch.as_tensor(valid, dtype=torch.long, device=self.device)
        self.mean = self.states.mean(dim=0)
        self.std = self.states.std(dim=0, unbiased=False).clamp_min(1e-3)
        self.feature_dim = states.shape[1]
        self.path = str(dataset_path)

    def normalize(self, states: torch.Tensor) -> torch.Tensor:
        return (states - self.mean) / self.std

    def sample_transitions(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self.transition_indices[
            torch.randint(self.transition_indices.numel(), (batch_size,), device=self.device)
        ]
        return self.normalize(self.states[indices]), self.normalize(self.states[indices + 1])


class AMPReplayBuffer:
    """Bounded replay buffer of policy state transitions."""

    def __init__(self, capacity: int, feature_dim: int, device: str | torch.device):
        self.capacity = capacity
        self.device = torch.device(device)
        self.states = torch.empty(capacity, feature_dim, device=self.device)
        self.next_states = torch.empty_like(self.states)
        self.position = 0
        self.size = 0

    def add(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
        states = states.detach().to(self.device)
        next_states = next_states.detach().to(self.device)
        count = states.shape[0]
        if count >= self.capacity:
            states, next_states = states[-self.capacity :], next_states[-self.capacity :]
            count = self.capacity

        indices = (torch.arange(count, device=self.device) + self.position) % self.capacity
        self.states[indices] = states
        self.next_states[indices] = next_states
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size: int, normalize) -> tuple[torch.Tensor, torch.Tensor]:
        if self.size < batch_size:
            raise RuntimeError(f"AMP replay buffer has {self.size} transitions; {batch_size} are required.")
        indices = torch.randint(self.size, (batch_size,), device=self.device)
        return normalize(self.states[indices]), normalize(self.next_states[indices])


class AMPDiscriminator(nn.Module):
    """Classifies expert and policy motion state transitions."""

    def __init__(self, feature_dim: int, hidden_dims: tuple[int, ...] = (1024, 512, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = feature_dim * 2
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ELU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((states, next_states), dim=-1)).squeeze(-1)


class AMPReward:
    """Owns the AMP discriminator, expert sampler, and discriminator optimizer."""

    def __init__(
        self,
        cfg: dict,
        device: str | torch.device,
        multi_gpu_cfg: dict | None = None,
    ):
        if multi_gpu_cfg is not None:
            raise ValueError("AMP discriminator training currently supports single-process training only.")
        self.cfg = cfg
        self.expert = ExpertMotionDataset(cfg["motion_file"], device)
        if self.expert.feature_dim != 87:
            raise ValueError(
                f"Kuavo S54 AMP expects 87 motion features, but {self.expert.path} contains "
                f"{self.expert.feature_dim}. Run scripts/rsl_rl/prepare_amp_motion.py on a GMR archive."
            )
        self.discriminator = AMPDiscriminator(self.expert.feature_dim).to(device)
        self.optimizer = optim.Adam(self.discriminator.parameters(), lr=cfg.get("learning_rate", 1e-4))
        self.replay = AMPReplayBuffer(cfg.get("replay_size", 100_000), self.expert.feature_dim, device)
        self.batch_size = cfg.get("batch_size", 4096)
        self.num_updates = cfg.get("num_updates", 2)
        self.gradient_penalty_coef = cfg.get("gradient_penalty_coef", 5.0)
        self.reward_coef = cfg.get("reward_coef", 0.5)

    def reward(self, states: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            states = self.expert.normalize(states)
            next_states = self.expert.normalize(next_states)
            logits = self.discriminator(states, next_states)
            return torch.clamp(1.0 - 0.25 * torch.square(logits - 1.0), min=0.0)

    def update(self) -> dict[str, float]:
        if self.replay.size < self.batch_size:
            return {"amp_discriminator": 0.0, "amp_expert_accuracy": 0.0, "amp_policy_accuracy": 0.0}

        losses = []
        expert_accuracies = []
        policy_accuracies = []
        for _ in range(self.num_updates):
            expert_states, expert_next_states = self.expert.sample_transitions(self.batch_size)
            policy_states, policy_next_states = self.replay.sample(self.batch_size, self.expert.normalize)
            expert_logits = self.discriminator(expert_states, expert_next_states)
            policy_logits = self.discriminator(policy_states, policy_next_states)

            expert_loss = nn.functional.binary_cross_entropy_with_logits(
                expert_logits, torch.ones_like(expert_logits)
            )
            policy_loss = nn.functional.binary_cross_entropy_with_logits(
                policy_logits, torch.zeros_like(policy_logits)
            )

            alpha = torch.rand(self.batch_size, 1, device=expert_states.device)
            interpolated_states = (
                alpha * expert_states + (1.0 - alpha) * policy_states
            ).requires_grad_(True)
            interpolated_next_states = (
                alpha * expert_next_states + (1.0 - alpha) * policy_next_states
            ).requires_grad_(True)
            interpolated_logits = self.discriminator(interpolated_states, interpolated_next_states)
            gradients = torch.autograd.grad(
                outputs=interpolated_logits.sum(),
                inputs=(interpolated_states, interpolated_next_states),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            gradient_norm = torch.cat([grad.flatten(start_dim=1) for grad in gradients], dim=1).norm(2, dim=1)
            gradient_penalty = torch.square(gradient_norm - 1.0).mean()
            loss = expert_loss + policy_loss + self.gradient_penalty_coef * gradient_penalty

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10.0)
            self.optimizer.step()

            losses.append(loss.detach())
            expert_accuracies.append((expert_logits.detach() > 0).float().mean())
            policy_accuracies.append((policy_logits.detach() < 0).float().mean())

        return {
            "amp_discriminator": torch.stack(losses).mean().item(),
            "amp_expert_accuracy": torch.stack(expert_accuracies).mean().item(),
            "amp_policy_accuracy": torch.stack(policy_accuracies).mean().item(),
        }
