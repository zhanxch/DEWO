from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _write_prepare_stamp(root: Path, task: str = "fold_glasses", step: int = 55000) -> Path:
    stamp = root / "prepare_results" / "dexjoco" / task / "20260908_090014"
    step_dir = stamp / f"step_{step:06d}"
    eve = step_dir / "eve_v02"
    protocol = eve / "protocol"
    manifests = eve / "manifests"
    text_cache = step_dir / "text_embeds_cache"
    vae_cache = step_dir / "vae_latent_cache"
    for path in (protocol, manifests, text_cache, vae_cache):
        path.mkdir(parents=True)
    (text_cache / "dummy.pt").write_bytes(b"pt")
    (vae_cache / "lat_0.pt").write_bytes(b"vae")
    pair_manifest = manifests / "offline_b1_jump_fast_pair.json"
    val_manifest = manifests / "offline_selection_primary_success.json"
    pair_manifest.write_text("{}\n", encoding="utf-8")
    val_manifest.write_text("{}\n", encoding="utf-8")
    ckpt = root / "step_055000.pt"
    ckpt.write_bytes(b"ckpt")
    stats = root / "dataset_stats.json"
    stats.write_text("{}\n", encoding="utf-8")
    env_file = protocol / "offline_v1_b1_jump_fast.env"
    env_file.write_text(
        "\n".join(
            [
                f"export EVE_MANIFEST_PATH={pair_manifest}",
                f"export EVE_VAL_MANIFEST_PATH={val_manifest}",
                f"export TEXT_EMBEDDING_CACHE_DIR={text_cache}",
                f"export VAE_LATENT_CACHE_DIR={vae_cache}",
                f"export INIT_WEIGHTS={ckpt}",
                f"export CKPT={ckpt}",
                f"export PRETRAINED_NORM_STATS={stats}",
                "export CFG_PRIMARY=0.5,0.5,0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (stamp / "prepare_config.json").write_text(
        json.dumps({"task_name": task, "checkpoint_steps": [step]}),
        encoding="utf-8",
    )
    return stamp


class TrainDexJocoLayoutTest(unittest.TestCase):
    def test_resolves_prepare_dexjoco_stamp(self) -> None:
        import train_dexjoco as train

        with tempfile.TemporaryDirectory() as tmp:
            stamp = _write_prepare_stamp(Path(tmp))
            layout = train.resolve_prepare_layout(stamp)
            self.assertEqual(layout["layout"], "prepare_dexjoco")
            self.assertEqual(layout["steps"], [55000])
            self.assertTrue(layout["env_file"].is_file())
            env_file = train.pick_env_file(layout, None)
            self.assertEqual(env_file, layout["env_by_step"][55000])

    def test_resolves_legacy_eve_root(self) -> None:
        import train_dexjoco as train

        with tempfile.TemporaryDirectory() as tmp:
            stamp = _write_prepare_stamp(Path(tmp))
            eve = stamp / "step_055000" / "eve_v02"
            layout = train.resolve_prepare_layout(eve)
            self.assertEqual(layout["layout"], "eve_root")
            self.assertTrue(layout["env_file"].is_file())


class TrainDexJocoRecipeTest(unittest.TestCase):
    def test_s0_and_scratch_recipes(self) -> None:
        import train_dexjoco as train

        s0 = train.build_recipe(task_name="fold_glasses", init="s0")
        self.assertEqual(s0.hydra_task, train.HYDRA_S0)
        self.assertIn("eval_every=0", s0.hydra_base)
        self.assertNotIn("resume=null", s0.hydra_base)
        scratch = train.build_recipe(task_name="fold_glasses", init="scratch")
        self.assertEqual(scratch.hydra_task, train.HYDRA_SCRATCH)
        self.assertIn("resume=null", scratch.hydra_base)

    def test_lora_rejected(self) -> None:
        import train_dexjoco as train

        with self.assertRaises(ValueError):
            train.build_recipe(task_name="fold_glasses", init="lora")

    def test_cfg_mix_is_owned_by_train_entry(self) -> None:
        import train_dexjoco as train

        stripped = train.strip_cfg_mix({"CFG_PRIMARY": "1,0,0", "EVE_MANIFEST_PATH": "/x"})
        self.assertNotIn("CFG_PRIMARY", stripped)
        env = train.cfg_env()
        self.assertEqual(env["CFG_PRIMARY_OUTCOME"], "0.9")
        self.assertEqual(env["CFG_PRIMARY_FAST"], "0.0")
        self.assertEqual(env["CFG_PRIMARY_BASE"], "0.1")
        self.assertEqual(env["CFG_SUCCESS_SUFFIX"], " Successful execution.")
        self.assertEqual(env["CFG_FAILURE_SUFFIX"], " Failed execution.")


class TrainDexJocoCliTest(unittest.TestCase):
    def test_parser_defaults_match_prepare_style(self) -> None:
        import train_dexjoco as train

        args = train.build_parser().parse_args(
            ["--task-name", "fold_glasses", "--env-file", "/tmp/offline_v1_b1_jump_fast.env"]
        )
        self.assertEqual(args.init, "scratch")
        self.assertEqual(args.gpus, [0, 1, 2, 3])
        self.assertEqual(args.dewo_version, "v9.1")
        self.assertTrue(args.use_vae)
        self.assertFalse(args.tmux)
        self.assertFalse(args.inline)

    def test_shared_flags_with_prepare(self) -> None:
        import prepare_dexjoco as prepare
        import train_dexjoco as train

        train_flags = {action.dest for action in train.build_parser()._actions}
        prepare_flags = {action.dest for action in prepare.build_parser()._actions}
        shared = {"task_name", "gpus", "checkpoint_steps"}
        self.assertTrue(shared.issubset(train_flags))
        self.assertTrue(shared.issubset(prepare_flags))
        self.assertIn("init", train_flags)
        self.assertIn("prepare_dir", train_flags)
        self.assertNotIn("video_samples_per_result", train_flags)

    def test_resolve_s0_from_prepare_stamp(self) -> None:
        import train_dexjoco as train

        with tempfile.TemporaryDirectory() as tmp:
            stamp = _write_prepare_stamp(Path(tmp))
            args = train.build_parser().parse_args(
                [
                    "--task-name",
                    "fold_glasses",
                    "--init",
                    "s0",
                    "--prepare-dir",
                    str(stamp),
                    "--gpus",
                    "1,2,3",
                    "--dry-run",
                ]
            )
            resolved = train._resolve_args(args)
            self.assertEqual(resolved._recipe.init, "s0")
            self.assertEqual(resolved._recipe.hydra_task, train.HYDRA_S0)
            self.assertTrue(resolved._run_inline)
            self.assertNotIn("CFG_PRIMARY", resolved._exports)
            self.assertEqual(resolved._vae_policy["SKIP_VAE_PREENCODE"], "1")
            self.assertIn("eval_every=0", resolved._hydra)
            launch = train.build_launch_env(resolved)
            self.assertEqual(launch["DEWO_INIT"], "s0")
            self.assertEqual(launch["DEWO_TASK"], train.HYDRA_S0)
            self.assertEqual(launch["CFG_PRIMARY_OUTCOME"], "0.9")
            self.assertEqual(launch["GPUS"], "1,2,3")
            self.assertEqual(launch["RUN_INLINE"], "1")
            self.assertEqual(launch["DEWO_VERSION"], "v9.1")
            train._validate_args(resolved)

    def test_resolve_scratch_uses_action_dit(self) -> None:
        import train_dexjoco as train

        with tempfile.TemporaryDirectory() as tmp:
            stamp = _write_prepare_stamp(Path(tmp))
            args = train.build_parser().parse_args(
                [
                    "--task-name",
                    "fold_glasses",
                    "--init",
                    "scratch",
                    "--prepare-dir",
                    str(stamp),
                    "--gpus",
                    "1,2,3,4",
                    "--max-steps",
                    "300000",
                    "--batch-size",
                    "32",
                ]
            )
            resolved = train._resolve_args(args)
            self.assertEqual(resolved._recipe.init, "scratch")
            self.assertEqual(resolved._init_weights, train.ACTION_DIT.resolve())
            self.assertIn("resume=null", resolved._hydra)
            self.assertIn("max_steps=300000", resolved._hydra)
            self.assertIn("batch_size=32", resolved._hydra)
            launch = train.build_launch_env(resolved)
            self.assertEqual(launch["DEWO_TASK"], train.HYDRA_SCRATCH)
            self.assertEqual(launch["INIT_WEIGHTS"], str(train.ACTION_DIT.resolve()))

    def test_tmux_flag_disables_inline(self) -> None:
        import train_dexjoco as train

        with tempfile.TemporaryDirectory() as tmp:
            stamp = _write_prepare_stamp(Path(tmp))
            args = train.build_parser().parse_args(
                [
                    "--task-name",
                    "fold_glasses",
                    "--init",
                    "s0",
                    "--prepare-dir",
                    str(stamp),
                    "--tmux",
                ]
            )
            resolved = train._resolve_args(args)
            self.assertFalse(resolved._run_inline)
            self.assertEqual(train.build_launch_env(resolved)["RUN_INLINE"], "0")

    def test_dry_run_does_not_exec(self) -> None:
        import train_dexjoco as train

        with tempfile.TemporaryDirectory() as tmp:
            stamp = _write_prepare_stamp(Path(tmp))
            argv = [
                "train_dexjoco.py",
                "--task-name",
                "fold_glasses",
                "--init",
                "s0",
                "--prepare-dir",
                str(stamp),
                "--gpus",
                "1",
                "--dry-run",
            ]
            with (
                patch.object(train.sys, "argv", argv),
                patch.object(train.os, "execvpe") as execvpe,
            ):
                train.main()
            execvpe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
