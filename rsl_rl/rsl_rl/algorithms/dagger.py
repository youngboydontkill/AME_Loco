"""Dataset Aggregation (DAgger) for teacher-student policy distillation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rsl_rl.utils import resolve_optimizer

from .distillation import Distillation


class DAgger(Distillation):
    """Collect states from the student-controlled rollout and label them online with the teacher."""

    def __init__(
        self,
        *args,
        velocity_loss_coef=0.1,
        gamma=0.99,
        lam=0.95,
        value_learning_rate=1e-3,
        value_num_learning_epochs=1,
        **kwargs,
    ):
        optimizer_name = kwargs.get("optimizer", "adam")
        learning_rate = kwargs.get("learning_rate", 1e-3)
        super().__init__(*args, **kwargs)
        self.velocity_loss_coef = velocity_loss_coef
        self.gamma = gamma
        self.lam = lam
        self.value_num_learning_epochs = value_num_learning_epochs
        if not hasattr(self.policy, "critic") or not hasattr(self.policy, "evaluate_value"):
            raise TypeError("DAgger value training requires a policy with critic and evaluate_value(obs)")

        self.actor_parameters = [
            parameter
            for name, parameter in self.policy.named_parameters()
            if not name.startswith("critic.")
        ]
        self.optimizer = resolve_optimizer(optimizer_name)(self.actor_parameters, lr=learning_rate)
        self.value_optimizer = resolve_optimizer(optimizer_name)(
            self.policy.critic.parameters(), lr=value_learning_rate
        )

    def act(self, obs):
        # DAgger executes the current student policy, then queries the frozen
        # expert at the exact state reached by that student-controlled policy.
        self.transition.actions = self.policy.act_inference(obs).detach()
        self.transition.privileged_actions = self.policy.evaluate(obs).detach()
        self.transition.observations = obs
        return self.transition.actions

    def update(self, last_obs=None):
        self.num_updates += 1
        behavior_total = 0.0
        velocity_total = 0.0
        update_count = 0
        accumulated_loss = 0.0

        returns = self._compute_gae_returns(last_obs)

        for _ in range(self.num_learning_epochs):
            for obs, _, teacher_actions, dones in self.storage.generator():
                # Keep master parameters in FP32; autocast only forward and loss operations.
                with torch.amp.autocast(
                    device_type=torch.device(self.device).type,
                    dtype=torch.bfloat16,
                    enabled=self.use_mixed_precision,
                ):
                    student_actions = self.policy.act_inference(obs)
                    behavior_loss = self.loss_fn(student_actions, teacher_actions)
                    loss = behavior_loss
                    behavior_total += behavior_loss.item()

                    if self.velocity_loss_coef > 0 and hasattr(self.policy, "compute_auxiliary_losses"):
                        aux_losses = self.policy.compute_auxiliary_losses(obs)
                        velocity_loss = aux_losses.get("velocity")
                        if velocity_loss is not None:
                            loss = loss + self.velocity_loss_coef * velocity_loss
                            velocity_total += velocity_loss.item()

                accumulated_loss = accumulated_loss + loss
                update_count += 1
                if update_count % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    accumulated_loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.actor_parameters, self.max_grad_norm)
                    self.optimizer.step()
                    accumulated_loss = 0.0
                self.policy.reset(dones.view(-1))

        if update_count % self.gradient_length:
            self.optimizer.zero_grad()
            accumulated_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.actor_parameters, self.max_grad_norm)
            self.optimizer.step()

        value_total = 0.0
        for _ in range(self.value_num_learning_epochs):
            for step in range(returns.shape[0]):
                obs = self.storage.observations[step]
                target = returns[step]
                # Keep master parameters in FP32; autocast only forward and loss operations.
                with torch.amp.autocast(
                    device_type=torch.device(self.device).type,
                    dtype=torch.bfloat16,
                    enabled=self.use_mixed_precision,
                ):
                    predicted_value = self.policy.evaluate_value(obs)
                    value_loss = F.mse_loss(predicted_value, target)
                self.value_optimizer.zero_grad()
                value_loss.backward()
                if self.is_multi_gpu:
                    self.reduce_value_parameters()
                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                value_total += value_loss.item()
        for parameter in self.policy.critic.parameters():
            parameter.grad = None

        self.storage.clear()
        losses = {"behavior": behavior_total / max(update_count, 1)}
        losses["value"] = value_total / max(self.value_num_learning_epochs * returns.shape[0], 1)
        if self.velocity_loss_coef > 0 and hasattr(self.policy, "compute_auxiliary_losses"):
            losses["velocity"] = velocity_total / max(update_count, 1)
        return losses

    @torch.no_grad()
    def _compute_gae_returns(self, last_obs=None):
        values = torch.stack(
            [self.policy.evaluate_value(self.storage.observations[step]) for step in range(self.storage.step)]
        )
        last_value = torch.zeros_like(values[-1]) if last_obs is None else self.policy.evaluate_value(last_obs)
        advantages = torch.zeros_like(self.storage.rewards[: self.storage.step])
        advantage = torch.zeros_like(last_value)
        next_value = last_value
        for step in reversed(range(self.storage.step)):
            not_done = 1.0 - self.storage.dones[step].float()
            delta = self.storage.rewards[step] + self.gamma * next_value * not_done - values[step]
            advantage = delta + self.gamma * self.lam * not_done * advantage
            advantages[step] = advantage
            next_value = values[step]
        return advantages + values

    def reduce_value_parameters(self):
        gradients = [
            parameter.grad.reshape(-1)
            for parameter in self.policy.critic.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            return
        all_gradients = torch.cat(gradients)
        torch.distributed.all_reduce(all_gradients, op=torch.distributed.ReduceOp.SUM)
        all_gradients /= self.gpu_world_size
        offset = 0
        for parameter in self.policy.critic.parameters():
            if parameter.grad is not None:
                count = parameter.numel()
                parameter.grad.copy_(all_gradients[offset : offset + count].view_as(parameter.grad))
                offset += count
