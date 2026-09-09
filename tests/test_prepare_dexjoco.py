from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _touch_rollout(raw: Path) -> None:
    meta = raw / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text("{}\n", encoding="utf-8")
    (raw / "collection_summary.json").write_text(
        json.dumps({"status": "complete", "attempt_log": []}),
        encoding="utf-8",
    )


class PrepareDexJocoLayoutTest(unittest.TestCase):
    def test_resolves_collect_dexjoco_stamp(self) -> None:
        import prepare_dexjoco as prepare

        with tempfile.TemporaryDirectory() as tmp:
            collect = Path(tmp) / "collect_results" / "dexjoco" / "fold_glasses" / "20260907_141408"
            raw = collect / "step_055000" / "rollout_raw"
            _touch_rollout(raw)
            (collect / "collect_config.json").write_text(
                json.dumps(
                    {
                        "task_name": "fold_glasses",
                        "checkpoint_steps": [55000],
                        "run_dir": str(ROOT / "configs/eval/dexjoco/mixed_5task_fastwam_joint"),
                        "max_steps": 1200,
                    }
                ),
                encoding="utf-8",
            )
            layout = prepare.resolve_collect_layout(collect)
            self.assertEqual(layout["layout"], "collect_dexjoco")
            self.assertEqual(layout["steps"], [55000])
            self.assertEqual(layout["raw_by_step"][55000], raw.resolve())

    def test_resolves_legacy_rollout_raw_200(self) -> None:
        import prepare_dexjoco as prepare

        with tempfile.TemporaryDirectory() as tmp:
            collect = Path(tmp) / "data" / "fold_glasses_mixed_s0_collect_stamp"
            raw = collect / "rollout_raw_200"
            _touch_rollout(raw)
            layout = prepare.resolve_collect_layout(collect)
            self.assertEqual(layout["layout"], "legacy_collect")
            self.assertEqual(layout["raw_by_step"][55000], raw.resolve())

    def test_scan_complete_requires_pairs(self) -> None:
        import prepare_dexjoco as prepare

        with tempfile.TemporaryDirectory() as tmp:
            scan = Path(tmp)
            self.assertFalse(prepare.scan_is_complete(scan))
            (scan / "summary.json").write_text(
                json.dumps({"status": "complete", "num_complete_event_pairs": 0}),
                encoding="utf-8",
            )
            self.assertFalse(prepare.scan_is_complete(scan))
            (scan / "summary.json").write_text(
                json.dumps({"status": "complete", "num_complete_event_pairs": 3}),
                encoding="utf-8",
            )
            self.assertTrue(prepare.scan_is_complete(scan))

    def test_scan_d0_complete_uses_prefix_count(self) -> None:
        import prepare_dexjoco as prepare

        with tempfile.TemporaryDirectory() as tmp:
            scan = Path(tmp)
            self.assertFalse(prepare.scan_d0_is_complete(scan))
            (scan / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "scan_mode": "d0_collect",
                        "num_prefix_results": 0,
                        "num_complete_event_pairs": 0,
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(prepare.scan_d0_is_complete(scan))
            self.assertFalse(prepare.scan_is_complete(scan))
            (scan / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "scan_mode": "d0_collect",
                        "num_prefix_results": 12,
                        "num_complete_event_pairs": 0,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(prepare.scan_d0_is_complete(scan))
            self.assertFalse(prepare.scan_is_complete(scan))


class PrepareDexJocoCliTest(unittest.TestCase):
    def test_parser_defaults_match_collect_style(self) -> None:
        import prepare_dexjoco as prepare

        parser = prepare.build_parser()
        args = parser.parse_args(
            [
                "--task-name",
                "fold_glasses",
                "--collect-dir",
                "/tmp/collect",
            ]
        )
        self.assertEqual(args.gpus, [1])
        self.assertEqual(args.phases, list(prepare.ALL_PHASES))
        self.assertEqual(args.dewo_version, "v9.1")
        self.assertTrue(args.use_vae)
        self.assertFalse(args.require_existing_scan)
        self.assertEqual(args.pass_m, 10)

    def test_phases_scan_d0_is_allowed(self) -> None:
        import prepare_dexjoco as prepare

        parser = prepare.build_parser()
        args = parser.parse_args(
            [
                "--task-name",
                "fold_glasses",
                "--collect-dir",
                "/tmp/collect",
                "--phases",
                "scan_d0",
            ]
        )
        self.assertEqual(args.phases, ["scan_d0"])
        self.assertIn("scan_d0", prepare.ALL_PHASES)

    def test_resolve_args_fills_from_collect_config(self) -> None:
        import prepare_dexjoco as prepare

        with tempfile.TemporaryDirectory() as tmp:
            collect = Path(tmp) / "collect_results" / "dexjoco" / "fold_glasses" / "stamp"
            raw = collect / "step_055000" / "rollout_raw"
            _touch_rollout(raw)
            (collect / "collect_config.json").write_text(
                json.dumps(
                    {
                        "task_name": "fold_glasses",
                        "checkpoint_dir": str(
                            ROOT / "checkpoints/dexjoco/mixed_5task_fastwam_joint/weights"
                        ),
                        "checkpoint_steps": [55000],
                        "run_dir": str(ROOT / "configs/eval/dexjoco/mixed_5task_fastwam_joint"),
                        "dataset_stats": str(ROOT / "artifacts/mixed_5task/dataset_stats.json"),
                        "max_steps": 1200,
                        "action_horizon": 32,
                        "replan_steps": 24,
                        "num_inference_steps": 10,
                        "success_prompt": "Fold the glasses and place them into the case.",
                    }
                ),
                encoding="utf-8",
            )
            args = prepare.build_parser().parse_args(
                [
                    "--task-name",
                    "fold_glasses",
                    "--collect-dir",
                    str(collect),
                    "--gpus",
                    "1,2,3",
                    "--output-dir",
                    str(Path(tmp) / "prepare_out"),
                ]
            )
            resolved = prepare._resolve_args(args)
            self.assertEqual(resolved.checkpoint_steps, [55000])
            self.assertEqual(resolved.max_steps, 1200)
            self.assertEqual(resolved.gpus, [1, 2, 3])
            self.assertEqual(
                resolved.success_prompt,
                "Fold the glasses and place them into the case.",
            )
            self.assertTrue(str(resolved.output_dir).endswith("prepare_out"))
            self.assertEqual(
                resolved.model_config,
                (ROOT / "configs/eval/dexjoco/mixed_5task_fastwam_joint/config.yaml").resolve(),
            )

    def test_v91_helpers(self) -> None:
        import prepare_dexjoco as prepare

        self.assertTrue(prepare.is_v91("v9.1"))
        self.assertFalse(prepare.is_v91("v9"))
        self.assertEqual(prepare.prepare_hydra_task("v9.1"), prepare.HYDRA_SCRATCH)
        self.assertEqual(
            prepare.pool_index_path(Path("/tmp/pair"), "v9.1").name,
            "pool_index.json",
        )
        self.assertEqual(
            prepare.pool_index_path(Path("/tmp/pair"), "v9").name,
            "pair_index.json",
        )

    def test_shared_flags_with_collect(self) -> None:
        import collect_dexjoco as collect
        import prepare_dexjoco as prepare

        collect_flags = {action.dest for action in collect.build_parser()._actions}
        prepare_flags = {action.dest for action in prepare.build_parser()._actions}
        shared = {"task_name", "gpus", "checkpoint_steps", "dataset_stats"}
        self.assertTrue(shared.issubset(collect_flags))
        self.assertTrue(shared.issubset(prepare_flags))
        self.assertIn("collect_dir", prepare_flags)
        self.assertNotIn("video_samples_per_result", prepare_flags)


class PrepareDexJocoProtocolTest(unittest.TestCase):
    def test_protocol_env_is_paths_only(self) -> None:
        import argparse

        import prepare_dexjoco as prepare

        with tempfile.TemporaryDirectory() as tmp:
            step_dir = Path(tmp)
            eve_root = step_dir / "eve_v02"
            env_file = eve_root / "protocol" / "offline_v1_b1_jump_fast.env"
            args = argparse.Namespace(
                task_name="fold_glasses",
                checkpoint_dir=ROOT / "checkpoints/dexjoco/mixed_5task_fastwam_joint/weights",
                dataset_stats=ROOT / "artifacts/mixed_5task/dataset_stats.json",
                model_config=ROOT / "configs/eval/dexjoco/mixed_5task_fastwam_joint/config.yaml",
                success_prompt="Fold the glasses and place them into the case.",
                use_vae=True,
            )
            prepare._write_protocol_files(
                args=args,
                step=55000,
                step_dir=step_dir,
                raw=step_dir / "rollout_raw",
                pair_out=step_dir / "pair_lerobot",
                eve_root=eve_root,
                text_cache=step_dir / "text_embeds_cache",
                vae_cache=step_dir / "vae_latent_cache",
                pair_manifest=eve_root / "manifests" / "offline_b1_jump_fast_pair.json",
                val_manifest=eve_root / "manifests" / "offline_selection_primary_success.json",
                env_file=env_file,
            )
            text = env_file.read_text(encoding="utf-8")
            self.assertIn("export EVE_MANIFEST_PATH=", text)
            self.assertIn("export TEXT_EMBEDDING_CACHE_DIR=", text)
            self.assertNotIn("CFG_PRIMARY=", text)
            protocol = json.loads(
                (eve_root / "protocol" / "offline_v1_b1_jump_fast.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(protocol["cfg"]["success_suffix"], " Successful execution.")
            self.assertEqual(protocol["primary_kind"], "all_success_seeds")


if __name__ == "__main__":
    unittest.main()
