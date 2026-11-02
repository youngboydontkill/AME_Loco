"""RobanS17 actuator model ported from LejuLab-Train."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class RobanS17DelayedPDActuator(DelayedPDActuator):
    """Delayed PD actuator with S17's velocity-dependent torque saturation."""

    cfg: RobanS17DelayedPDActuatorCfg

    def __init__(self, cfg: RobanS17DelayedPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.friction_static = self._parse_joint_parameter(cfg.friction_static, 0.0)
        self.friction_activation_vel = self._parse_joint_parameter(cfg.friction_activation_vel, torch.inf)
        self.friction_dynamic = self._parse_joint_parameter(cfg.friction_dynamic, 0.0)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._zeros_effort = torch.zeros_like(self.computed_effort)
        self._saturation_effort = self._build_effort_limit_tensor()

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        self._joint_vel[env_ids] = 0.0

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)

        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        self.computed_effort = (
            self.stiffness * error_pos
            + self.damping * error_vel
            + control_action.joint_efforts
            - (self.friction_static * torch.tanh(joint_vel / self.friction_activation_vel)
               + self.friction_dynamic * joint_vel)
        )
        control_action.joint_efforts = self._clip_effort(self.computed_effort)
        self.applied_effort = control_action.joint_efforts
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def _build_effort_limit_tensor(self) -> torch.Tensor:
        limits = []
        for name in self.joint_names:
            for pattern, value in self.cfg.effort_limit_sim.items():
                if re.fullmatch(pattern.replace("[l,r]", "[lr]"), name):
                    limits.append(value)
                    break
            else:
                raise ValueError(f"No effort limit for joint: {name}")
        return torch.tensor(
            [limits], device=self.computed_effort.device, dtype=self.computed_effort.dtype
        )

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        safe_velocity = torch.where(
            self.velocity_limit > 1e-6,
            self._joint_vel / self.velocity_limit,
            torch.zeros_like(self._joint_vel),
        )
        max_effort = torch.clamp(
            self._saturation_effort * (1.0 - safe_velocity), min=self._zeros_effort, max=self.effort_limit
        )
        min_effort = torch.clamp(
            self._saturation_effort * (-1.0 - safe_velocity), min=-self.effort_limit, max=self._zeros_effort
        )
        return torch.clamp(effort, min=min_effort, max=max_effort)


@configclass
class RobanS17DelayedPDActuatorCfg(DelayedPDActuatorCfg):
    """Delayed PD actuator configuration used by the RobanS17 model."""

    class_type: type = RobanS17DelayedPDActuator
    effort_limit_rated: float | dict[str, float] = MISSING
    friction_static: float | dict[str, float] = 0.0
    friction_activation_vel: float = torch.inf
    friction_dynamic: float | dict[str, float] = 0.0
