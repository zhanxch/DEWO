from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "fold_glasses"))

import compose_failure_recoverability_videos as viz  # noqa: E402


class CfgEventRuleTest(unittest.TestCase):
    def test_drop_of_three_with_alive_k_is_an_event(self) -> None:
        nodes = [(48, 10), (72, 10), (96, 7), (120, 7), (144, 0)]
        events = viz.cfg_events(nodes, pass_m=10)
        self.assertEqual([event.t for event in events], [96])
        self.assertEqual(events[0].delta_k, 3)
        self.assertEqual(events[0].k, 7)

    def test_small_drops_and_cliff_are_not_events(self) -> None:
        nodes = [(48, 10), (72, 9), (96, 8), (120, 0)]
        self.assertEqual(viz.cfg_events(nodes), [])

    def test_first_scan_node_cannot_be_an_event(self) -> None:
        self.assertEqual(viz.cfg_events([(48, 4), (72, 4)]), [])

    def test_drop_to_zero_is_cliff_not_cfg(self) -> None:
        nodes = [(48, 10), (72, 0)]
        self.assertEqual(viz.cfg_events(nodes), [])

    def test_rebound_is_not_an_event(self) -> None:
        self.assertEqual(viz.cfg_events([(48, 4), (72, 7)]), [])


class HeldVSeriesTest(unittest.TestCase):
    def test_holds_between_nodes_without_interpolation(self) -> None:
        series = viz.held_v_series([(48, 10), (72, 4)], n_frames=90, pass_m=10)
        self.assertTrue(np.all(np.isnan(series[:48])))
        self.assertTrue(np.allclose(series[48:72], 1.0))
        self.assertTrue(np.allclose(series[72:], 0.4))
        self.assertNotAlmostEqual(float(series[60]), 0.7)

    def test_holds_last_node_through_the_tail(self) -> None:
        series = viz.held_v_series([(48, 0)], n_frames=80, pass_m=10)
        self.assertTrue(np.allclose(series[48:], 0.0))


class OverlayWindowTest(unittest.TestCase):
    def test_badge_covers_event_span_from_t(self) -> None:
        event = viz.CfgEvent(t=96, k=7, k_prev=10, delta_k=3, pass_m=10)
        self.assertIsNone(viz.active_cfg_event([event], 95))
        self.assertEqual(viz.active_cfg_event([event], 96), event)
        self.assertEqual(viz.active_cfg_event([event], 128), event)
        self.assertIsNone(viz.active_cfg_event([event], 129))


class ComposeSmokeTest(unittest.TestCase):
    def test_writes_stacked_mp4_with_cfg_badge_logic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "fail.mp4"
            writer = viz.Mp4Writer(src, 64, 64, 10)
            for i in range(8):
                frame = np.full((64, 64, 3), i * 20, dtype=np.uint8)
                writer.write(frame)
            writer.close()

            episode = viz.EpisodeScan(
                ep=1,
                seed=10086,
                cls="mixed",
                pass_m=10,
                nodes=[(0, 10), (4, 6), (7, 0)],
                cliff_m=7,
                dataset=None,
            )
            events = viz.cfg_events(episode.nodes, replan_steps=4, pass_m=10)
            self.assertEqual([event.t for event in events], [4])
            out = tmp_path / "out.mp4"
            row = viz.compose_episode(
                video_path=src,
                episode=episode,
                output_path=out,
                events=events,
            )
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 0)
            self.assertEqual(row["n_frames"], 8)
            self.assertEqual(row["cfg_events"][0]["t"], 4)


if __name__ == "__main__":
    unittest.main()
