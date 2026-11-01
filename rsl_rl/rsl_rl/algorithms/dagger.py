"""Dataset Aggregation (DAgger) for teacher-student policy distillation."""

from __future__ import annotations

import torch

from .distillation import Distillation


class DAgger(Distillation):
    """Collect states from the student-controlled rollout and label them online with the teacher."""

    def __init__(self, *args, velocity_loss_coef=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.velocity_loss_coef = velocity_loss_coef

    def act(self, obs):
        # DAgger executes the current student policy, then queries the frozen
        # expert at the exact state reached by that student-controlled policy.
        self.transition.actions = self.policy.act_inference(obs).detach()
        self.transition.privileged_actions = self.policy.evaluate(obs).detach()
        self.transition.observations = obs
        return self.transition.actions

    def update(self):
        self.num_updates += 1
        behavior_total = 0.0
        velocity_total = 0.0
        update_count = 0
        accumulated_loss = 0.0

        for _ in range(self.num_learning_epochs):
            for obs, _, teacher_actions, dones in self.storage.generator():
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
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    accumulated_loss = 0.0
                self.policy.reset(dones.view(-1))

        if update_count % self.gradient_length:
            self.optimizer.zero_grad()
            accumulated_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            if self.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.storage.clear()
        losses = {"behavior": behavior_total / max(update_count, 1)}
        if self.velocity_loss_coef > 0 and hasattr(self.policy, "compute_auxiliary_losses"):
            losses["velocity"] = velocity_total / max(update_count, 1)
        return losses
