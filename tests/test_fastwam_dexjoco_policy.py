"""CPU tests for in-repo FastWAMDexJocoPolicy path resolution."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fastwam_dexjoco.policy import _resolve_run_dir  # noqa: E402


class FastWAMDexJocoPolicyPathTests(unittest.TestCase):
    def test_resolve_run_dir_from_config_yaml(self) -> None:
        cfg = ROOT / "configs/eval/dexjoco/mixed_5task_fastwam_joint/config.yaml"
        self.assertTrue(cfg.is_file())
        self.assertEqual(_resolve_run_dir(cfg), cfg.parent.resolve())

    def test_resolve_run_dir_from_directory(self) -> None:
        run_dir = ROOT / "configs/eval/dexjoco/mixed_5task_fastwam_joint"
        self.assertEqual(_resolve_run_dir(run_dir), run_dir.resolve())

    def test_reject_bare_model_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "fastwam_joint.yaml"
            yaml_path.write_text("model: {}\n")
            with self.assertRaises(ValueError):
                _resolve_run_dir(yaml_path)


if __name__ == "__main__":
    unittest.main()
