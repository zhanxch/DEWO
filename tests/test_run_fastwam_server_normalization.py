from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastwam.inference.loader import resolve_normalization_binding


ROOT = Path(__file__).resolve().parents[1]


def _write_meta(meta_dir: Path) -> None:
    meta_dir.mkdir(parents=True)
    (meta_dir / "stats.json").write_text("{}\n", encoding="utf-8")
    (meta_dir / "modality.json").write_text("{}\n", encoding="utf-8")


class RunFastWamServerNormalizationTest(unittest.TestCase):
    def test_explicit_meta_dir_overrides_stale_config_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            meta_dir = Path(tmp) / "relocated-meta"
            _write_meta(meta_dir)
            processor_cfg = {
                "norm_stats_source": "meta",
                "norm_stats_meta_dir": "stale/meta",
            }

            kind, path = resolve_normalization_binding(
                processor_cfg,
                run_dir=run_dir,
                dataset_stats_path=None,
                norm_stats_meta_dir=str(meta_dir),
            )

            self.assertEqual(kind, "meta")
            self.assertEqual(path, meta_dir.resolve())
            self.assertEqual(processor_cfg["norm_stats_meta_dir"], str(meta_dir.resolve()))

    def test_meta_config_rejects_dataset_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            stats = run_dir / "dataset_stats.json"
            stats.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "norm_stats_source=meta"):
                resolve_normalization_binding(
                    {"norm_stats_source": "meta"},
                    run_dir=run_dir,
                    dataset_stats_path=str(stats),
                    norm_stats_meta_dir=None,
                )

    def test_compute_config_rejects_meta_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            meta_dir = run_dir / "meta"
            _write_meta(meta_dir)
            with self.assertRaisesRegex(ValueError, "not allowed"):
                resolve_normalization_binding(
                    {"norm_stats_source": "compute"},
                    run_dir=run_dir,
                    dataset_stats_path=None,
                    norm_stats_meta_dir=str(meta_dir),
                )

    def test_meta_files_must_be_nonempty(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            meta_dir = run_dir / "meta"
            _write_meta(meta_dir)
            (meta_dir / "stats.json").write_bytes(b"")
            with self.assertRaisesRegex(FileNotFoundError, "non-empty"):
                resolve_normalization_binding(
                    {"norm_stats_source": "meta"},
                    run_dir=run_dir,
                    dataset_stats_path=None,
                    norm_stats_meta_dir=str(meta_dir),
                )

    def test_legacy_meta_config_path_still_works_outside_strict_s0_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            configured_meta = run_dir / "legacy-meta"
            _write_meta(configured_meta)
            kind, path = resolve_normalization_binding(
                {
                    "norm_stats_source": "meta",
                    "norm_stats_meta_dir": "legacy-meta",
                },
                run_dir=run_dir,
                dataset_stats_path=None,
                norm_stats_meta_dir=None,
            )
            self.assertEqual(kind, "meta")
            self.assertEqual(path, configured_meta.resolve())


if __name__ == "__main__":
    unittest.main()
