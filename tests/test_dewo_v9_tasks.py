"""Tests for DEWO v9 task registry and CFG env overrides."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dewo_v2.tasks import (  # noqa: E402
    COLLECT_EXPERT_SLACK_FRAMES,
    CfgRecipe,
    eval_task_yaml,
    expert_episode_lengths,
    get_task,
    is_env_success,
    is_train_success,
    parse_cfg_recipe,
    resolve_collect_max_steps,
    t5_cache_name,
    train_success_cap,
)


class DewoV9TaskTests(unittest.TestCase):
    def test_water_plant_prompt_and_t5_hash(self) -> None:
        task = get_task("water_plant")
        self.assertEqual(
            task.success_prompt,
            "Grasp the watering can and apply water to the plant.",
        )
        self.assertEqual(
            t5_cache_name(task.success_prompt),
            "f742556deff61d95d9f67eb3522f56d6f6c69ff9833ffa5b4beb83dc0d6a40df.t5_len128.wan22ti2v5b.pt",
        )

    def test_v9_cfg_defaults(self) -> None:
        cfg = parse_cfg_recipe({})
        self.assertEqual(cfg.primary, (0.9, 0.0, 0.1))
        self.assertEqual(cfg.aux_success, (1.0, 0.0, 0.0))
        self.assertEqual(cfg.aux_fail, (1.0, 0.0, 0.0))
        self.assertEqual(cfg.success_suffix, " Successful execution.")
        self.assertEqual(cfg.failure_suffix, " Failed execution.")
        self.assertEqual(cfg.recipe_name, "v9")
        self.assertTrue(cfg.fast_fail_closed)
        self.assertEqual(cfg.dropout, 0.0)

    def test_compact_cfg_override(self) -> None:
        cfg = parse_cfg_recipe(
            {
                "CFG_PRIMARY": "0.6,0.0,0.4",
                "CFG_AUX_SUCCESS": "0.3,0.3,0.4",
                "CFG_AUX_FAIL": "0.0,0.5,0.5",
                "CFG_FAILURE_SUFFIX": " Failed execution.",
                "CFG_FAST_FAIL_CLOSED": "0",
            }
        )
        self.assertEqual(cfg.primary, (0.6, 0.0, 0.4))
        self.assertEqual(cfg.aux_success, (0.3, 0.3, 0.4))
        self.assertEqual(cfg.aux_fail, (0.0, 0.5, 0.5))
        self.assertEqual(cfg.failure_suffix, " Failed execution.")
        self.assertFalse(cfg.fast_fail_closed)

    def test_primary_fast_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_cfg_recipe({"CFG_PRIMARY": "0.4,0.2,0.4"})

    def test_eval_yaml_uses_success_suffix(self) -> None:
        task = get_task("water_plant")
        text = eval_task_yaml(task, CfgRecipe())
        self.assertIn(
            "Grasp the watering can and apply water to the plant. Successful execution.",
            text,
        )
        self.assertIn("cfg_base_prompt:", text)
        self.assertNotIn("cfg_failure_prompt:", text)

    def test_eval_yaml_never_includes_failure_prompt(self) -> None:
        task = get_task("water_plant")
        text = eval_task_yaml(
            task,
            CfgRecipe(failure_suffix=" Failed execution."),
        )
        self.assertNotIn("cfg_failure_prompt:", text)
        self.assertNotIn("Failed execution.", text)

    def test_unknown_task(self) -> None:
        with self.assertRaises(KeyError):
            get_task("not_a_task")

    def test_resolve_ckpt_falls_back_to_mixed_joint(self) -> None:
        from dewo_v2.tasks import resolve_ckpt

        fold = get_task("fold_glasses")
        ckpt = resolve_ckpt(fold)
        self.assertTrue(ckpt.is_file())
        self.assertEqual(ckpt.name, "step_055000.pt")
        self.assertIn("mixed_5task_fastwam_joint", str(ckpt))

    def test_hammer_nail_collect_caps_at_expert_max_plus_slack(self) -> None:
        hammer = get_task("hammer_nail")
        lengths = expert_episode_lengths(hammer)
        self.assertGreater(max(lengths), 0)
        self.assertEqual(
            resolve_collect_max_steps(hammer),
            max(lengths) + COLLECT_EXPERT_SLACK_FRAMES,
        )
        self.assertEqual(hammer.max_steps, 1000)
        fold = get_task("fold_glasses")
        self.assertEqual(resolve_collect_max_steps(fold), fold.max_steps)

    def test_hammer_nail_train_success_cap_excludes_long_env_wins(self) -> None:
        hammer = get_task("hammer_nail")
        cap = train_success_cap("hammer_nail")
        self.assertEqual(cap, resolve_collect_max_steps(hammer))
        self.assertIsNone(train_success_cap("fold_glasses"))
        long_success = {"success": True, "outcome": "success"}
        self.assertTrue(is_env_success(long_success))
        self.assertFalse(is_train_success(long_success, cap + 1, task_name="hammer_nail"))
        self.assertTrue(is_train_success(long_success, cap, task_name="hammer_nail"))
        self.assertTrue(is_train_success(long_success, cap + 1, task_name="fold_glasses"))
        self.assertFalse(is_train_success({"success": False}, 100, task_name="hammer_nail"))


if __name__ == "__main__":
    unittest.main()
