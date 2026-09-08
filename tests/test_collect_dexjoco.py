from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CollectDexJocoEntryTest(unittest.TestCase):
    def test_parser_defaults_match_collect_4x50(self) -> None:
        import collect_dexjoco as collect

        parser = collect.build_parser()
        args = parser.parse_args(
            [
                "--task-name",
                "fold_glasses",
                "--run-dir",
                str(ROOT / "configs/eval/dexjoco/mixed_5task_fastwam_joint"),
                "--checkpoint-dir",
                "/tmp/weights",
                "--checkpoint-steps",
                "55000",
                "--dataset-stats",
                "/tmp/stats.json",
            ]
        )
        self.assertEqual(args.seed_start, 10086)
        self.assertEqual(args.seed_end, 10135)
        self.assertEqual(args.repeats, 4)
        self.assertEqual(args.action_horizon, 32)
        self.assertEqual(args.replan_steps, 24)
        self.assertEqual(args.num_inference_steps, 10)
        self.assertEqual(args.max_steps, 1200)
        self.assertEqual(args.text_cfg_scale, 0.0)
        self.assertEqual(args.gpus, [1])

    def test_resolve_args_fills_task_prompt_and_source_dataset(self) -> None:
        import collect_dexjoco as collect

        args = collect.build_parser().parse_args(
            [
                "--task-name",
                "fold_glasses",
                "--run-dir",
                str(ROOT / "configs/eval/dexjoco/mixed_5task_fastwam_joint"),
                "--checkpoint-dir",
                "/tmp/weights",
                "--checkpoint-steps",
                "55000",
                "--dataset-stats",
                "/tmp/stats.json",
            ]
        )
        resolved = collect._resolve_args(args)
        self.assertEqual(
            resolved.success_prompt,
            "Fold the glasses and place them into the case.",
        )
        self.assertEqual(
            resolved.source_dataset,
            ROOT / "data/dexjoco/dexjoco_lerobot_datasets/fold_glasses",
        )
        self.assertTrue(str(resolved.output_dir).endswith("collect_results/dexjoco/fold_glasses") or "collect_results/dexjoco/fold_glasses" in str(resolved.output_dir))

    def test_done_pairs_require_committed_repeat(self) -> None:
        import collect_dexjoco as collect

        attempts = [
            {"seed": 10086, "repeat": 0, "success": True, "saved_episode_index": 0},
            {"seed": 10086, "repeat": 1, "success": False, "saved_episode_index": None},
            {"seed": 10087, "success": True, "saved_episode_index": 1},
        ]
        self.assertEqual(collect._done_pairs(attempts), {(10086, 0)})


class CollectDexJocoCliParityTest(unittest.TestCase):
    def test_collect_keeps_eval_inference_flags(self) -> None:
        import collect_dexjoco as collect
        import eval_dexjoco as evaluate

        collect_flags = {action.dest for action in collect.build_parser()._actions}
        eval_flags = {action.dest for action in evaluate.build_parser()._actions}
        shared = {
            "task_name",
            "run_dir",
            "checkpoint_dir",
            "checkpoint_steps",
            "dataset_stats",
            "text_embedding",
            "text_cfg_scale",
            "action_horizon",
            "replan_steps",
            "num_inference_steps",
            "max_steps",
            "gpus",
        }
        self.assertTrue(shared.issubset(collect_flags))
        self.assertTrue(shared.issubset(eval_flags))
        self.assertNotIn("video_samples_per_result", collect_flags)


class CollectDexJocoValidateTest(unittest.TestCase):
    def test_validate_args_skips_eval_only_video_samples(self) -> None:
        import argparse
        from unittest.mock import patch

        import collect_dexjoco as collect
        import eval_dexjoco as evaluate

        args = argparse.Namespace(
            run_dir=ROOT,
            dataset_stats=ROOT / "artifacts/mixed_5task/dataset_stats.json",
            checkpoint_dir=ROOT,
            text_embedding=None,
            text_embedding_base=None,
            text_embedding_failure=None,
            text_cfg_scale=0.0,
            cfg_gate_mode="off",
            load_text_encoder=False,
            checkpoint_steps=[55000],
            gpus=[1],
            seed_start=10086,
            seed_end=10135,
            repeats=4,
            max_steps=1200,
            source_dataset=ROOT / "data/dexjoco/dexjoco_lerobot_datasets/fold_glasses",
            success_prompt="Fold the glasses and place them into the case.",
        )
        self.assertFalse(hasattr(args, "video_samples_per_result"))
        with (
            patch.object(evaluate, "_checkpoint_path", return_value=ROOT / "step.pt"),
            patch("eval_dexjoco.torch.cuda.is_available", return_value=True),
            patch("eval_dexjoco.torch.cuda.device_count", return_value=8),
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "is_file", return_value=True),
        ):
            collect._validate_args(args)


class CollectDexJocoFinalizeTest(unittest.TestCase):
    def test_attempt_rows_fill_inference_seconds_for_eval_summary(self) -> None:
        import tempfile

        import collect_dexjoco as collect
        import eval_dexjoco as evaluate

        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "shards" / "gpu_1"
            shard.mkdir(parents=True)
            (shard / "collection_summary.json").write_text(
                json.dumps(
                    {
                        "attempt_log": [
                            {
                                "seed": 10086,
                                "repeat": 0,
                                "success": True,
                                "steps": 400,
                                "elapsed_s": 30.0,
                            },
                            {
                                "seed": 10086,
                                "repeat": 1,
                                "success": False,
                                "steps": 1200,
                                "elapsed_s": 80.0,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = collect._attempt_rows_from_shards(Path(tmp))
            summary = evaluate._summarize_episodes(
                rows, step=55000, video_counts={"success": 0, "failure": 0}
            )
        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["successes"], 1)
        self.assertGreater(summary["total_inference_seconds_across_workers"], 0.0)


if __name__ == "__main__":
    unittest.main()
