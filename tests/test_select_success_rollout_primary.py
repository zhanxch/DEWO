"""Tests for selecting S0 success rollouts as DEWO v2 primary."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dewo_v2.select_success_rollout_primary import (  # noqa: E402
    MIN_COMPLETE_LENGTH,
    build_split_rows,
    is_complete_success,
    select_primary,
)


class SelectSuccessRolloutPrimaryTests(unittest.TestCase):
    def test_rejects_pair_length_success(self) -> None:
        self.assertFalse(is_complete_success({"success": True}, MIN_COMPLETE_LENGTH))
        self.assertTrue(is_complete_success({"success": True}, MIN_COMPLETE_LENGTH + 1))

    def test_samples_n_and_splits_cover_all(self) -> None:
        outcomes = []
        lengths = {}
        for i in range(20):
            outcomes.append({"episode_index": i, "success": i < 18, "outcome": "success" if i < 18 else "failure"})
            lengths[i] = 200 if i < 18 else 1000
        primary, leftover, failures = select_primary(
            outcomes=outcomes, lengths=lengths, n=15, seed=20260820
        )
        self.assertEqual(len(primary), 15)
        self.assertEqual(len(leftover), 3)
        self.assertEqual(len(failures), 2)
        splits = build_split_rows(
            dataset_id="water_plant_s0_success_rollouts",
            primary=primary,
            leftover=leftover,
            failures=failures,
            seed=20260820,
        )
        self.assertEqual(len(splits), 20)
        self.assertEqual(sum(r["split"] == "train" for r in splits), 15)
        self.assertTrue(all(r["split"] == "train" for r in splits if r["episode_index"] in {int(x["episode_index"]) for x in primary}))

    def test_too_few_successes(self) -> None:
        outcomes = [{"episode_index": 0, "success": True, "outcome": "success"}]
        with self.assertRaises(SystemExit):
            select_primary(outcomes=outcomes, lengths={0: 200}, n=15, seed=1)

    def test_hammer_nail_over_cap_stays_success_but_not_train(self) -> None:
        outcomes = []
        lengths = {}
        for i in range(16):
            over = i == 0
            outcomes.append(
                {
                    "episode_index": i,
                    "success": True,
                    "outcome": "success",
                }
            )
            lengths[i] = 800 if over else 200
        primary, leftover, failures = select_primary(
            outcomes=outcomes,
            lengths=lengths,
            n=15,
            seed=1,
            task_name="hammer_nail",
        )
        self.assertFalse(is_complete_success(outcomes[0], 800, task_name="hammer_nail"))
        self.assertTrue(is_complete_success(outcomes[1], 200, task_name="hammer_nail"))
        self.assertEqual(len(primary), 15)
        self.assertEqual(len(leftover), 0)
        self.assertEqual(len(failures), 1)
        self.assertEqual(int(failures[0]["episode_index"]), 0)
        splits = build_split_rows(
            dataset_id="hammer_nail_s0_success_rollouts",
            primary=primary,
            leftover=leftover,
            failures=failures,
            seed=1,
        )
        over = next(row for row in splits if row["episode_index"] == 0)
        self.assertEqual(over["episode_outcome"], "success")
        self.assertEqual(over["split"], "test")


if __name__ == "__main__":
    unittest.main()
