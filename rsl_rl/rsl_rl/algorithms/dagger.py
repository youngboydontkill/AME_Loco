"""Dataset Aggregation (DAgger) for teacher-student policy distillation."""

from __future__ import annotations

import torch

from .distillation import Distillation


class DAgger(Distillation):
    """Collect states from the student-controlled rollout and label them online with the teacher."""

    def act(self, obs):
        # DAgger executes the current student policy, then queries the frozen
        # expert at the exact state reached by that student-controlled policy.
        self.transition.actions = self.policy.act_inference(obs).detach()
        self.transition.privileged_actions = self.policy.evaluate(obs).detach()
        self.transition.observations = obs
        return self.transition.actions
