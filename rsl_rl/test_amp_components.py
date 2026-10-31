import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from rsl_rl.algorithms.amp import AMPReward, ExpertMotionDataset
from scripts.rsl_rl.prepare_amp_motion import build_amp_states


class TestAMPMotionData(unittest.TestCase):
    def test_build_amp_states_derives_features(self):
        frame_count = 4
        root_pos = np.zeros((frame_count, 3), dtype=np.float32)
        root_pos[:, 0] = np.arange(frame_count) / 50.0
        root_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frame_count, 1))
        joint_pos = np.zeros((frame_count, 27), dtype=np.float32)
        end_effectors = np.zeros((frame_count, 4, 3), dtype=np.float32)
        end_effectors[:, :, 2] = 0.1
        states = build_amp_states(
            root_pos,
            root_quat,
            joint_pos,
            end_effectors,
            np.zeros(frame_count, dtype=np.int64),
            fps=50.0,
        )

        self.assertEqual(states.shape, (frame_count, 87))
        np.testing.assert_allclose(states[:, 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(states[:, 8], -1.0, atol=1e-5)
        np.testing.assert_allclose(states[:, 63:75].reshape(frame_count, 4, 3)[:, :, 2], 0.1, atol=1e-5)

    def test_dataset_samples_only_within_clip_transitions(self):
        states = np.arange(20 * 87, dtype=np.float32).reshape(20, 87)
        clip_ids = np.repeat(np.arange(2), 10)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "motions.npz"
            np.savez(path, states=states, clip_ids=clip_ids)
            dataset = ExpertMotionDataset(str(path), "cpu")
            self.assertEqual(dataset.transition_indices.numel(), 18)
            sampled, sampled_next = dataset.sample_transitions(8)
            self.assertEqual(sampled.shape, (8, 87))
            self.assertEqual(sampled_next.shape, (8, 87))

    def test_discriminator_reward_and_update(self):
        states = np.random.default_rng(0).normal(size=(32, 87)).astype(np.float32)
        clip_ids = np.zeros(32, dtype=np.int64)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "motions.npz"
            np.savez(path, states=states, clip_ids=clip_ids)
            amp = AMPReward(
                {
                    "motion_file": str(path),
                    "batch_size": 4,
                    "num_updates": 1,
                    "replay_size": 32,
                },
                "cpu",
            )
            policy_states = torch.randn(8, 87)
            policy_next_states = torch.randn(8, 87)
            amp.replay.add(policy_states, policy_next_states)
            rewards = amp.reward(policy_states, policy_next_states)
            self.assertTrue(torch.isfinite(rewards).all())
            self.assertTrue((rewards >= 0).all())
            metrics = amp.update()
            self.assertTrue(np.isfinite(metrics["amp_discriminator"]))


if __name__ == "__main__":
    unittest.main()
