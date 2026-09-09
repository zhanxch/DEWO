from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dewo_v2.v91_pool import (  # noqa: E402
    adjacent_cfg_events,
    build_v91_pool_specs,
    crop_span,
    fail_cliff_span,
    is_cfg_event,
    pool_role_of,
)


class V91PoolGeometryTest(unittest.TestCase):
    def test_cfg_event_needs_drop_of_three_and_alive(self) -> None:
        self.assertTrue(is_cfg_event(10, 7))
        self.assertTrue(is_cfg_event(7, 4))
        self.assertFalse(is_cfg_event(4, 5))
        self.assertFalse(is_cfg_event(10, 8))
        self.assertFalse(is_cfg_event(3, 0))
        self.assertFalse(is_cfg_event(10, 0))

    def test_crop_span_is_33_or_none(self) -> None:
        self.assertEqual(crop_span(48, 200), (48, 81))
        self.assertIsNone(crop_span(180, 200))

    def test_fail_cliff_starts_at_m(self) -> None:
        lo, hi = fail_cliff_span(72, 96, 198, min_len=33, post=24)
        self.assertEqual(lo, 96)
        self.assertEqual(hi, 129)

    def test_adjacent_events_skip_t48_and_noise(self) -> None:
        rows = [
            {"prefix_frame": 48, "success_count": 10},
            {"prefix_frame": 72, "success_count": 10},
            {"prefix_frame": 96, "success_count": 7},
            {"prefix_frame": 120, "success_count": 6},
            {"prefix_frame": 144, "success_count": 0},
        ]
        events = adjacent_cfg_events(rows)
        self.assertEqual([e["prefix_frame"] for e in events], [96])

    def test_pool_role_of(self) -> None:
        self.assertEqual(pool_role_of({"pool_role": "d_scan"}), "d_scan")
        self.assertIsNone(pool_role_of({"pool_role": "primary"}))
        self.assertIsNone(pool_role_of({}))


class V91PoolBuildTest(unittest.TestCase):
    def test_build_specs_scan_and_dplus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scan = Path(tmp)
            ep = 3
            t = 96
            rep = scan / "prefixes" / f"ep{ep:06d}_f{t:04d}" / "replicate_00"
            rep.mkdir(parents=True)
            (rep / "trajectory.npz").write_bytes(b"npz")
            (rep / "continuation_front.mp4").write_bytes(b"mp4")
            (rep / "continuation_wrist.mp4").write_bytes(b"mp4")
            specs = build_v91_pool_specs(
                scan_root=scan,
                episodes={ep: {"length": 200}},
                prefix_labels=[
                    {
                        "source_failure_episode_index": ep,
                        "prefix_frame": 48,
                        "success_count": 10,
                        "pass_m": 10,
                        "successful_replicate_indices": [0],
                    },
                    {
                        "source_failure_episode_index": ep,
                        "prefix_frame": 72,
                        "success_count": 10,
                        "pass_m": 10,
                        "successful_replicate_indices": [0],
                    },
                    {
                        "source_failure_episode_index": ep,
                        "prefix_frame": 96,
                        "success_count": 4,
                        "pass_m": 10,
                        "successful_replicate_indices": [0],
                    },
                    {
                        "source_failure_episode_index": ep,
                        "prefix_frame": 120,
                        "success_count": 0,
                        "pass_m": 10,
                        "successful_replicate_indices": [],
                    },
                ],
            )
            self.assertEqual(specs["counts"]["d_scan"], 4)
            self.assertEqual(specs["counts"]["d_fail"], 1)
            self.assertEqual(specs["counts"]["dplus"], 1)
            cliff = [row for row in specs["scan_windows"] if row["is_cliff"]]
            self.assertEqual(cliff[0]["prefix_frame"], 120)
            self.assertEqual(cliff[0]["value_target"], 0.0)
            self.assertEqual(specs["dplus"][0]["prefix_frame"], 96)
            self.assertEqual(specs["dplus"][0]["prev_success_count"], 10)


class V91ManifestBuildTest(unittest.TestCase):
    def test_merges_d0_and_pool_units(self) -> None:
        from fastwam.everobot_schema import with_manifest_hash

        hashes = {
            "round_meta_sha256": "1" * 64,
            "episode_meta_sha256": "2" * 64,
            "event_meta_sha256": "3" * 64,
        }
        collect = with_manifest_hash(
            {
                "format": "EveRobotTrainManifest",
                "schema_version": "0.2",
                "manifest_name": "offline_primary_success",
                "frame_interval": "half_open",
                "selection": {"include_outcomes": ["success"]},
                "dataset_roots": {"collect": "/tmp/collect"},
                "source_round_ids": ["collect:round:0"],
                "source_hashes": hashes,
                "samples": [
                    {
                        "sample_type": "episode",
                        "sample_id": "collect_ep000000",
                        "dataset_id": "collect",
                        "dataset_root": "/tmp/collect",
                        "episode_id": "collect:episode:000000",
                        "episode_index": 0,
                        "round_id": "collect:round:0",
                        "collection_round": 0,
                        "start_frame": 0,
                        "end_frame": 80,
                        "sample_stride": 1,
                        "action_loss": "enabled",
                        "episode_outcome": "success",
                        "batch_role": "primary",
                        "split": "train",
                    }
                ],
            }
        )
        expert = with_manifest_hash(
            {
                "format": "EveRobotTrainManifest",
                "schema_version": "0.2",
                "manifest_name": "offline_expert_success",
                "frame_interval": "half_open",
                "selection": {"include_outcomes": ["success"]},
                "dataset_roots": {"fold_glasses_expert_success": "/tmp/expert"},
                "source_round_ids": ["fold_glasses_expert_success:round:-1"],
                "source_hashes": hashes,
                "samples": [
                    {
                        "sample_type": "episode",
                        "sample_id": "expert_ep000000",
                        "dataset_id": "fold_glasses_expert_success",
                        "dataset_root": "/tmp/expert",
                        "episode_id": "expert:episode:000000",
                        "episode_index": 0,
                        "round_id": "fold_glasses_expert_success:round:-1",
                        "collection_round": -1,
                        "start_frame": 0,
                        "end_frame": 90,
                        "sample_stride": 1,
                        "action_loss": "enabled",
                        "episode_outcome": "success",
                        "split": "train",
                    }
                ],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collect_path = root / "collect.json"
            expert_path = root / "expert.json"
            pool_index = root / "pool_index.json"
            out = root / "train.json"
            collect_path.write_text(__import__("json").dumps(collect), encoding="utf-8")
            expert_path.write_text(__import__("json").dumps(expert), encoding="utf-8")
            pool_index.write_text(
                __import__("json").dumps(
                    {
                        "units": [
                            {
                                "kind": "d_scan",
                                "episode_index": 0,
                                "length": 33,
                                "source_failure_episode_index": 7,
                                "prefix_frame": 48,
                                "success_count": 10,
                                "pass_m": 10,
                                "value_target": 1.0,
                                "seed": 1,
                            },
                            {
                                "kind": "d_fail",
                                "episode_index": 1,
                                "length": 33,
                                "source_failure_episode_index": 7,
                                "prefix_frame": 96,
                                "success_count": 0,
                                "pass_m": 10,
                                "value_target": 0.0,
                                "seed": 1,
                            },
                            {
                                "kind": "dplus",
                                "episode_index": 2,
                                "length": 33,
                                "source_failure_episode_index": 7,
                                "prefix_frame": 72,
                                "success_count": 4,
                                "pass_m": 10,
                                "replicate": 0,
                                "seed": 1,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            sys.path.insert(0, str(ROOT / "scripts"))
            from dewo_v2.build_v91_manifest import main as build_main

            self.assertEqual(
                build_main(
                    [
                        "--collect-manifest",
                        str(collect_path),
                        "--expert-manifest",
                        str(expert_path),
                        "--pool-index",
                        str(pool_index),
                        "--pool-dataset",
                        str(root),
                        "--pool-dataset-id",
                        "fold_glasses_pair_events",
                        "--prompt",
                        "Fold the glasses.",
                        "--recipe",
                        "fold_glasses_dewo_v91_scratch_pool",
                        "--output",
                        str(out),
                    ]
                ),
                0,
            )
            payload = __import__("json").loads(out.read_text(encoding="utf-8"))
            roles = [row["pool_role"] for row in payload["samples"]]
            self.assertEqual(roles.count("d0"), 2)
            self.assertEqual(roles.count("d_scan"), 1)
            self.assertEqual(roles.count("d_fail"), 1)
            self.assertEqual(roles.count("dplus"), 1)
            d0 = next(
                row
                for row in payload["samples"]
                if row["pool_role"] == "d0" and "expert" in row["dataset_id"]
            )
            self.assertEqual(d0["value_loss_weight"], 0.0)
            scan = next(row for row in payload["samples"] if row["pool_role"] == "d_scan")
            self.assertEqual(scan["value_target"], 1.0)
            self.assertEqual(scan["action_loss"], "disabled")
            fail = next(row for row in payload["samples"] if row["pool_role"] == "d_fail")
            self.assertEqual(fail["action_loss"], "disabled")
            dplus = next(row for row in payload["samples"] if row["pool_role"] == "dplus")
            self.assertEqual(dplus["action_loss"], "enabled")
            self.assertEqual(dplus["value_loss_weight"], 0.0)

            index_path = root / "d0_collect_value_index.json"
            index_path.write_text(
                __import__("json").dumps(
                    {
                        "format": "D0CollectValueIndex",
                        "episodes": {"0": {"48": 1.0, "72": 0.8}},
                    }
                ),
                encoding="utf-8",
            )
            out2 = root / "train_d0v.json"
            self.assertEqual(
                build_main(
                    [
                        "--collect-manifest",
                        str(collect_path),
                        "--expert-manifest",
                        str(expert_path),
                        "--pool-index",
                        str(pool_index),
                        "--pool-dataset",
                        str(root),
                        "--pool-dataset-id",
                        "fold_glasses_pair_events",
                        "--prompt",
                        "Fold the glasses.",
                        "--recipe",
                        "fold_glasses_dewo_v91_scratch_pool",
                        "--d0-value-index",
                        str(index_path),
                        "--output",
                        str(out2),
                    ]
                ),
                0,
            )
            payload2 = __import__("json").loads(out2.read_text(encoding="utf-8"))
            collect_d0 = next(
                row
                for row in payload2["samples"]
                if row["pool_role"] == "d0" and "expert" not in row["dataset_id"]
            )
            expert_d0 = next(
                row
                for row in payload2["samples"]
                if row["pool_role"] == "d0" and "expert" in row["dataset_id"]
            )
            self.assertEqual(collect_d0["scan_value_by_frame"]["48"], 1.0)
            self.assertNotIn("scan_value_by_frame", expert_d0)


if __name__ == "__main__":
    unittest.main()
