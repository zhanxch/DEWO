from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dewo_v2.d0_collect_value import (  # noqa: E402
    build_d0_value_index,
    load_d0_value_index,
    lookup_scan_value,
)


class D0CollectValueIndexTest(unittest.TestCase):
    def test_builds_sparse_pass_at_m_table(self) -> None:
        payload = build_d0_value_index(
            [
                {
                    "source_episode_index": 136,
                    "prefix_frame": 48,
                    "success_count": 10,
                    "pass_m": 10,
                    "replicates_evaluated": 10,
                },
                {
                    "source_failure_episode_index": 136,
                    "prefix_frame": 72,
                    "success_count": 9,
                    "pass_m": 10,
                    "replicates_evaluated": 10,
                },
                {
                    "source_episode_index": 8,
                    "prefix_frame": 48,
                    "success_count": 3,
                    "pass_m": 10,
                    "replicates_evaluated": 4,
                },
            ]
        )
        self.assertEqual(payload["num_episodes"], 1)
        self.assertEqual(payload["num_labeled_frames"], 2)
        self.assertEqual(payload["num_incomplete_prefixes"], 1)
        self.assertEqual(payload["episodes"]["136"]["48"], 1.0)
        self.assertEqual(payload["episodes"]["136"]["72"], 0.9)
        self.assertNotIn("8", payload["episodes"])

    def test_lookup_is_exact_node_only(self) -> None:
        table = {"48": 1.0, "72": 0.9}
        self.assertEqual(lookup_scan_value(table, 48), 1.0)
        self.assertEqual(lookup_scan_value(table, 72), 0.9)
        self.assertIsNone(lookup_scan_value(table, 49))
        self.assertIsNone(lookup_scan_value(table, 60))
        self.assertIsNone(lookup_scan_value({}, 48))

    def test_roundtrip_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d0_collect_value_index.json"
            payload = build_d0_value_index(
                [
                    {
                        "source_episode_index": 12,
                        "prefix_frame": 96,
                        "success_count": 5,
                        "pass_m": 10,
                        "replicates_evaluated": 10,
                    }
                ]
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_d0_value_index(path)
            self.assertEqual(loaded[12][96], 0.5)


if __name__ == "__main__":
    unittest.main()
