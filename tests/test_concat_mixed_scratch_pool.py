"""CPU tests for mixed DEWO scratch pool concatenation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dewo_v2.concat_mixed_scratch_pool import (  # noqa: E402
    concat_manifests,
    parse_queue_file,
)
from fastwam.everobot_schema import with_manifest_hash  # noqa: E402


def _unit(task: str, dataset_id: str, root: str, episode: int) -> dict:
    return {
        "sample_type": "episode",
        "sample_id": f"{dataset_id}_ep{episode:06d}",
        "dataset_id": dataset_id,
        "dataset_root": root,
        "episode_id": f"{dataset_id}:episode:{episode:06d}",
        "episode_index": episode,
        "round_id": f"{dataset_id}:round:0",
        "collection_round": 0,
        "start_frame": 0,
        "end_frame": 40,
        "sample_stride": 1,
        "action_loss": "enabled",
        "task": task,
        "split": "train",
        "episode_outcome": "success",
        "batch_role": "primary",
    }


def _manifest(task: str, tmp: Path) -> dict:
    root = str(tmp / task)
    Path(root).mkdir(parents=True, exist_ok=True)
    dataset_id = f"{task}_s0_success_rollouts"
    payload = {
        "schema_version": "0.2",
        "format": "EveRobotTrainManifest",
        "manifest_name": "offline_b1_jump_fast_pair",
        "frame_interval": "half_open",
        "dataset_roots": {dataset_id: root},
        "source_round_ids": [f"{dataset_id}:round:0"],
        "source_hashes": {
            "round_meta_sha256": "1" * 64,
            "episode_meta_sha256": "2" * 64,
            "event_meta_sha256": "3" * 64,
        },
        "samples": [_unit(task, dataset_id, root, 0)],
        "selection": {"recipe": f"{task}_pairs"},
    }
    return with_manifest_hash(payload)


class ConcatMixedScratchPoolTests(unittest.TestCase):
    def test_concat_keeps_task_prefixed_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fold = _manifest("fold_glasses", tmp_path)
            hammer = _manifest("hammer_nail", tmp_path)
            mixed = concat_manifests(
                [fold, hammer],
                manifest_name="mixed",
                recipe="mixed_5task",
                eve_root=tmp_path / "eve",
            )
            self.assertEqual(len(mixed["samples"]), 2)
            ids = {row["dataset_id"] for row in mixed["samples"]}
            self.assertEqual(
                ids,
                {"fold_glasses_s0_success_rollouts", "hammer_nail_s0_success_rollouts"},
            )
            self.assertEqual(len(mixed["dataset_roots"]), 2)

    def test_parse_queue_file_groups_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queue.env"
            path.write_text(
                "TASK=fold_glasses\n"
                "ENV_FILE=/tmp/fold.env\n"
                "TASK=hammer_nail\n"
                "ENV_FILE=/tmp/hammer.env\n"
            )
            blocks = parse_queue_file(path)
            self.assertEqual([row["TASK"] for row in blocks], ["fold_glasses", "hammer_nail"])
            self.assertEqual(blocks[1]["ENV_FILE"], "/tmp/hammer.env")


if __name__ == "__main__":
    unittest.main()
