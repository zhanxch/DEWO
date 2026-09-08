"""Joint DexJoCo eval must match FastWAM-infer-in-DexJoco joint-infer-eval."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

from fastwam.inference.contract import to_inference_tensors
from fastwam.inference.loader import resolve_num_video_frames_from_cfg
from fastwam.inference.obs import (
    DexJoCoFastWAMAdapter,
    fastwam_action_to_dexjoco,
    load_dexjoco_eval_settings,
    resize_rgb_area,
)


ROOT = Path(__file__).resolve().parents[1]
JOINT_RUN_DIR = ROOT / "configs/eval/dexjoco/mixed_5task_fastwam_joint"
VANILLA_RUN_DIR = ROOT / "configs/eval/dexjoco/mixed_5task_fastwam"


class JointEvalConfigTest(unittest.TestCase):
    def test_joint_yaml_matches_fastwam_infer_in_dexjoco(self) -> None:
        cfg = yaml.safe_load((JOINT_RUN_DIR / "config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(cfg["model"]["_target_"], "fastwam.runtime.create_fastwam_joint")
        self.assertFalse(cfg["model"]["mot_checkpoint_mixed_attn"])
        self.assertFalse(cfg["model"]["load_text_encoder"])
        self.assertTrue(cfg["model"]["skip_dit_load_from_pretrain"])
        self.assertIsNone(cfg["model"]["action_dit_pretrained_path"])
        self.assertEqual(cfg["num_video_frames"], 9)
        self.assertEqual(cfg["EVALUATION"]["num_video_frames"], 9)
        self.assertEqual(cfg["EVALUATION"]["image_resize"], "area")
        processor = cfg["data"]["train"]["processor"]
        self.assertIsNone(processor["train_transforms"])
        self.assertIsNone(processor["val_transforms"])
        self.assertEqual(
            resolve_num_video_frames_from_cfg(cfg, fallback=int(cfg["data"]["train"]["num_frames"])),
            9,
        )

    def test_vanilla_yaml_is_unchanged_create_fastwam(self) -> None:
        cfg = yaml.safe_load((VANILLA_RUN_DIR / "config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(cfg["model"]["_target_"], "fastwam.runtime.create_fastwam")
        self.assertTrue(cfg["model"]["mot_checkpoint_mixed_attn"])
        self.assertEqual(
            resolve_num_video_frames_from_cfg(cfg, fallback=int(cfg["data"]["train"]["num_frames"])),
            33,
        )

    def test_eval_settings_select_joint_image_path(self) -> None:
        settings = load_dexjoco_eval_settings(JOINT_RUN_DIR)
        self.assertEqual(settings["image_resize"], "area")
        self.assertTrue(settings["keep_uint8_image"])
        self.assertEqual(settings["concat_multi_camera"], "horizontal")

        vanilla = load_dexjoco_eval_settings(VANILLA_RUN_DIR)
        self.assertEqual(vanilla["image_resize"], "bilinear")
        self.assertFalse(vanilla["keep_uint8_image"])


class JointImageAndActionTest(unittest.TestCase):
    def test_zero_rotvec_becomes_identity_wxyz(self) -> None:
        action = np.arange(22, dtype=np.float64)
        action[3:6] = 0.0
        converted = fastwam_action_to_dexjoco(action)
        self.assertEqual(converted.shape, (23,))
        np.testing.assert_allclose(converted[:3], action[:3])
        np.testing.assert_allclose(converted[3:7], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(converted[7:], action[6:])

    def test_joint_cameras_are_front_then_wrist_area_resize(self) -> None:
        settings = load_dexjoco_eval_settings(JOINT_RUN_DIR)
        adapter = DexJoCoFastWAMAdapter(settings)
        front = np.full((40, 50, 3), 10, dtype=np.uint8)
        wrist = np.full((30, 35, 3), 20, dtype=np.uint8)
        obs = adapter.env_obs_to_policy_obs(
            {"front": front, "wrist": wrist, "state": np.zeros(23, dtype=np.float32)},
            camera_key="front",
            camera_mapping={"base": "front", "front": "front", "wrist": "wrist"},
            task_prompt="fold the glasses",
        )
        image = obs["input_image"]
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(image.shape, (1, 3, 224, 448))
        expected = np.concatenate(
            [resize_rgb_area(front, (224, 224)), resize_rgb_area(wrist, (224, 224))],
            axis=1,
        )
        np.testing.assert_array_equal(image[0].transpose(1, 2, 0), expected)

    def test_uint8_image_scales_after_dtype_cast(self) -> None:
        import torch

        image = np.full((1, 3, 4, 4), 255, dtype=np.uint8)
        tensors = to_inference_tensors(
            {
                "input_image": image,
                "prompt": "task",
            },
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        scaled = tensors["input_image"]
        self.assertEqual(scaled.dtype, torch.bfloat16)
        self.assertTrue(torch.allclose(scaled, torch.ones_like(scaled), atol=1e-2))


if __name__ == "__main__":
    unittest.main()
