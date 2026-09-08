from __future__ import annotations

import unittest

from fastwam.inference.config import InferenceConfig
from fastwam.inference.kwargs import filter_infer_action_kwargs, supports_cfg_mix
from fastwam.inference.obs import load_text_embedding_file
from fastwam.inference.rollout import rollout_episode


def _fastwam_infer_action(
    prompt,
    input_image,
    action_horizon,
    proprio=None,
    context=None,
    context_mask=None,
    negative_prompt=None,
    negative_context=None,
    negative_context_mask=None,
    failure_prompt=None,
    failure_context=None,
    failure_context_mask=None,
    text_cfg_scale=1.0,
    cfg_gate_mode=None,
    cfg_exec_horizon=24,
    cfg_value_prev=None,
    num_inference_steps=10,
    seed=None,
):
    del prompt, input_image, action_horizon, proprio, context, context_mask
    del negative_prompt, negative_context, negative_context_mask
    del failure_prompt, failure_context, failure_context_mask
    del text_cfg_scale, cfg_gate_mode, cfg_exec_horizon, cfg_value_prev
    del num_inference_steps, seed


def _joint_infer_action(
    prompt,
    input_image,
    action_horizon,
    num_video_frames,
    proprio=None,
    context=None,
    context_mask=None,
    negative_prompt=None,
    text_cfg_scale=1.0,
    num_inference_steps=20,
    seed=None,
):
    del prompt, input_image, action_horizon, num_video_frames, proprio
    del context, context_mask, negative_prompt, text_cfg_scale
    del num_inference_steps, seed


class InferenceKwargsTest(unittest.TestCase):
    def test_fastwam_supports_cfg_mix_and_keeps_value_options(self) -> None:
        self.assertTrue(supports_cfg_mix(_fastwam_infer_action))
        filtered = filter_infer_action_kwargs(
            _fastwam_infer_action,
            {
                "prompt": "task",
                "input_image": None,
                "action_horizon": 32,
                "cfg_gate_mode": "value_growth",
                "cfg_exec_horizon": 8,
                "cfg_value_prev": 0.4,
                "num_video_frames": 9,
            },
        )
        self.assertEqual(filtered["cfg_gate_mode"], "value_growth")
        self.assertEqual(filtered["cfg_exec_horizon"], 8)
        self.assertNotIn("num_video_frames", filtered)

    def test_joint_and_idm_drop_cfg_kwargs_for_baseline_infer(self) -> None:
        self.assertFalse(supports_cfg_mix(_joint_infer_action))
        filtered = filter_infer_action_kwargs(
            _joint_infer_action,
            {
                "prompt": "task",
                "input_image": None,
                "action_horizon": 32,
                "num_video_frames": 9,
                "cfg_gate_mode": "value",
                "negative_context": None,
                "text_cfg_scale": 1.0,
            },
        )
        self.assertIn("num_video_frames", filtered)
        self.assertEqual(filtered["num_video_frames"], 9)
        self.assertNotIn("cfg_gate_mode", filtered)
        self.assertNotIn("negative_context", filtered)

    def test_cfg_mix_on_joint_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not implement DEWO CFG mix"):
            filter_infer_action_kwargs(
                _joint_infer_action,
                {"prompt": "task", "text_cfg_scale": 1.1},
                require_cfg_mix=True,
            )

    def test_value_replan_defaults_to_action_replan(self) -> None:
        cfg = InferenceConfig(action_horizon=32, replan_steps=24)
        self.assertEqual(cfg.resolved_value_replan_steps, 24)
        cfg = InferenceConfig(action_horizon=32, replan_steps=24, value_replan_steps=8)
        self.assertEqual(cfg.resolved_value_replan_steps, 8)

    def test_wants_cfg_mix_follows_formula_weight(self) -> None:
        self.assertFalse(InferenceConfig().wants_cfg_mix())
        self.assertFalse(InferenceConfig(text_cfg_scale=0.0).wants_cfg_mix())
        self.assertTrue(InferenceConfig(text_cfg_scale=1.0).wants_cfg_mix())
        self.assertTrue(InferenceConfig(text_cfg_scale=2.0).wants_cfg_mix())
        self.assertTrue(InferenceConfig(text_cfg_scale=0.0, cfg_gate_mode="value").wants_cfg_mix())


class IndependentValueReplanTest(unittest.TestCase):
    def test_value_is_queried_between_action_replans(self) -> None:
        class FakeEnv:
            def __init__(self) -> None:
                self.steps = 0

            def step(self, _action):
                self.steps += 1
                obs = {
                    "front": __import__("numpy").zeros((8, 8, 3), dtype="uint8"),
                    "wrist": __import__("numpy").zeros((8, 8, 3), dtype="uint8"),
                    "state": __import__("numpy").zeros(23, dtype="float32"),
                }
                done = self.steps >= 6
                return obs, 0.0, done, False, {"succeed": done}

        class FakePolicy:
            replan_steps = 4
            value_replan_steps = 2
            action_calls = 0
            value_calls = 0

            def infer_with_extras(self, obs, noise_seed=None, options=None):
                del obs, noise_seed, options
                self.action_calls += 1
                import numpy as np

                return {
                    "action": np.zeros((32, 22), dtype=np.float32),
                    "cfg_value": np.float32(0.5),
                }

            def infer_value(self, obs, options=None):
                del obs, options
                self.value_calls += 1
                import numpy as np

                return {"cfg_value": np.float32(0.25)}

        policy = FakePolicy()
        env = FakeEnv()
        obs = {
            "front": __import__("numpy").zeros((8, 8, 3), dtype="uint8"),
            "wrist": __import__("numpy").zeros((8, 8, 3), dtype="uint8"),
            "state": __import__("numpy").zeros(23, dtype="float32"),
        }
        row, _frames = rollout_episode(
            env,
            obs,
            policy,
            seed=0,
            repeat=0,
            max_steps=6,
            capture_frames=False,
        )
        self.assertEqual(policy.action_calls, 2)
        self.assertGreaterEqual(policy.value_calls, 1)
        self.assertEqual(row["value_replan_steps"], 2)
        self.assertGreaterEqual(row["value_calls"], 2)

    def test_record_trajectory_stores_lerobot_payload(self) -> None:
        class FakeEnv:
            def __init__(self) -> None:
                self.steps = 0

            def step(self, _action):
                self.steps += 1
                obs = {
                    "front": __import__("numpy").full((8, 8, 3), 3, dtype="uint8"),
                    "wrist": __import__("numpy").full((8, 8, 3), 7, dtype="uint8"),
                    "state": __import__("numpy").arange(38, dtype="float32"),
                }
                done = self.steps >= 3
                return obs, 0.0, done, False, {"succeed": done}

        class FakePolicy:
            replan_steps = 4
            value_replan_steps = 4

            def infer_with_extras(self, obs, noise_seed=None, options=None):
                del obs, noise_seed, options
                import numpy as np

                return {"action": np.ones((32, 22), dtype=np.float32)}

        import numpy as np

        obs = {
            "front": np.full((8, 8, 3), 1, dtype=np.uint8),
            "wrist": np.full((8, 8, 3), 2, dtype=np.uint8),
            "state": np.arange(38, dtype=np.float32),
        }
        row, frames = rollout_episode(
            FakeEnv(),
            obs,
            FakePolicy(),
            seed=0,
            repeat=0,
            max_steps=3,
            capture_frames=False,
            record_trajectory=True,
        )
        self.assertEqual(frames, [])
        self.assertEqual(row["actions"].shape, (3, 22))
        self.assertEqual(row["states"].shape, (3, 23))
        self.assertEqual(len(row["frames"]["observation.images.front"]), 3)
        np.testing.assert_allclose(row["states"][0], np.arange(23, dtype=np.float32))
        self.assertTrue(bool(row["success"]))


class TextEmbeddingLoadTest(unittest.TestCase):
    def test_bfloat16_pt_cache_converts_to_float32(self) -> None:
        import tempfile
        from pathlib import Path

        import numpy as np
        import torch

        payload = {
            "context": torch.randn(4, 8, dtype=torch.bfloat16),
            "mask": torch.tensor([True, True, False, False]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.pt"
            torch.save(payload, path)
            context, mask = load_text_embedding_file(path)
        self.assertEqual(context.dtype, np.float32)
        self.assertEqual(context.shape, (4, 8))
        self.assertEqual(mask.dtype, bool)
        self.assertTrue(np.all(mask))


if __name__ == "__main__":
    unittest.main()
