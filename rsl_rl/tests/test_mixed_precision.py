"""Real-model update smoke tests, independent of Isaac Sim (CPU and optional CUDA)."""

import math
import unittest

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import DAgger, Distillation, PPO
from rsl_rl.modules import ActorCriticCNNProps, ActorCriticEncoder, ActorCriticGRU, StudentTeacher, StudentTeacherCNNMLP


class MixedPrecisionTest(unittest.TestCase):
    def devices(self):
        devices = ["cpu"]
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False):
            devices.append("cuda:0")
        return devices

    def collect(self, alg, obs, training_type):
        alg.init_storage(training_type, 4, 2, obs, [3])
        with torch.inference_mode():
            for _ in range(2):
                actions = alg.act(obs)
                self.assertEqual(actions.dtype, torch.float32)
                alg.process_env_step(obs, torch.randn(4, device=alg.device), torch.zeros(4, device=alg.device), {})
            if training_type == "rl":
                alg.compute_returns(obs)

    def check_update(self, alg, obs, enabled, training_type):
        self.collect(alg, obs, training_type)
        outputs = []
        teachers = []
        handles = []
        for name, module in alg.policy.named_modules():
            if isinstance(module, torch.nn.Linear):
                destination = teachers if name.startswith("teacher.") else outputs
                handles.append(module.register_forward_hook(lambda m, args, out, dest=destination: dest.append(out.dtype)))
        before = {name: p.detach().clone() for name, p in alg.policy.named_parameters()}
        try:
            losses = alg.update()
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(all(math.isfinite(value) for value in losses.values()), losses)
        self.assertIn(torch.bfloat16 if enabled else torch.float32, outputs)
        if not enabled:
            self.assertNotIn(torch.bfloat16, outputs)
        self.assertNotIn(torch.bfloat16, teachers)
        self.assertTrue(any(not torch.equal(before[name], p) for name, p in alg.policy.named_parameters()))
        for name, parameter in alg.policy.named_parameters():
            self.assertEqual(parameter.dtype, torch.float32)
            if name.startswith("teacher."):
                self.assertTrue(torch.equal(before[name], parameter))
            if parameter.grad is not None:
                self.assertEqual(parameter.grad.dtype, torch.float32)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        for state in alg.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value) and value.is_floating_point():
                    self.assertEqual(value.dtype, torch.float32)
        self.assertFalse(torch.is_autocast_enabled(torch.device(alg.device).type))

    def test_ppo_attention_and_gru(self):
        for device in self.devices():
            for enabled in (False, True):
                for model in (ActorCriticEncoder, ActorCriticGRU):
                    with self.subTest(device=device, enabled=enabled, model=model.__name__):
                        torch.manual_seed(3)
                        obs = TensorDict({"policy": torch.randn(4, 8 + 9 * 9, device=device)}, batch_size=[4])
                        policy = model(obs, {"policy": ["policy"], "critic": ["policy"]}, 3,
                                       map_scan_dim=(9, 9, 1), actor_hidden_dims=[16], critic_hidden_dims=[16])
                        alg = PPO(policy, device=device, use_mixed_precision=enabled,
                                  num_learning_epochs=1, num_mini_batches=1)
                        self.check_update(alg, obs, enabled, "rl")

    def test_distillation(self):
        for device in self.devices():
            for enabled in (False, True):
                with self.subTest(device=device, enabled=enabled):
                    obs = TensorDict({"policy": torch.randn(4, 8, device=device)}, batch_size=[4])
                    policy = StudentTeacher(obs, {"policy": ["policy"], "teacher": ["policy"]}, 3,
                                            student_hidden_dims=[16], teacher_hidden_dims=[16])
                    alg = Distillation(policy, device=device, use_mixed_precision=enabled, gradient_length=1)
                    self.check_update(alg, obs, enabled, "distillation")

    def test_dagger(self):
        for device in self.devices():
            for enabled in (False, True):
                with self.subTest(device=device, enabled=enabled):
                    obs = TensorDict({
                        "depth": torch.randn(4, 42 * 42, device=device),
                        "props": torch.randn(4, 5, 8, device=device),
                        "teacher": torch.randn(4, 8 + 9 * 9, device=device),
                        "teacher_critic": torch.randn(4, 10 + 9 * 9, device=device),
                    }, batch_size=[4])
                    policy = StudentTeacherCNNMLP(obs, {"policy": ["depth", "props"], "teacher": ["teacher"]}, 3,
                                                  teacher_map_scan_dim=(9, 9, 1), actor_hidden_dims=[16],
                                                  critic_hidden_dims=[16], props_hidden_dims=[16])
                    alg = DAgger(policy, device=device, use_mixed_precision=enabled, gradient_length=1)
                    self.check_update(alg, obs, enabled, "distillation")

    def test_ppo_bc_teacher_stays_fp32(self):
        for device in self.devices():
            for enabled in (False, True):
                with self.subTest(device=device, enabled=enabled):
                    obs = TensorDict({
                        "depth": torch.randn(4, 42 * 42, device=device),
                        "props": torch.randn(4, 5, 8, device=device),
                        "teacher": torch.randn(4, 8 + 9 * 9, device=device),
                        "teacher_critic": torch.randn(4, 10 + 9 * 9, device=device),
                    }, batch_size=[4])
                    policy = ActorCriticCNNProps(obs, {"policy": ["depth", "props"], "critic": ["teacher_critic"]}, 3,
                                                  teacher_map_scan_dim=(9, 9, 1), actor_hidden_dims=[16],
                                                  critic_hidden_dims=[16], props_hidden_dims=[16])
                    policy.loaded_teacher = True
                    alg = PPO(policy, device=device, use_mixed_precision=enabled, bc_loss_coef=0.2,
                              num_learning_epochs=1, num_mini_batches=1)
                    self.check_update(alg, obs, enabled, "rl")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
