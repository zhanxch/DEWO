#!/usr/bin/env python3
"""DEWO v9 / v9.1 data prep after ``scripts/collect_dexjoco.py``.

In-process CLI, same house style as ``eval_dexjoco.py`` / ``collect_dexjoco.py``.
Reads a collect stamp (or a legacy ``rollout_raw_200`` root) and writes:

  scan → scan_d0 → critic index → pool LeRobot → Eve manifests + text/VAE

v9.1 (default): D0 = expert ∪ collect successes, D_scan / D_fail / D+ crops,
no stitch. ``scan_d0`` labels D0 collect successes with real Pass@M V
(sparse, no interpolation; expert D0 stays V-off). v9 keeps the old
full-horizon pair stitch.

hammer_nail prepare D0 only includes env successes whose length is at most
expert_max + 10. Longer successes stay successes; they are not train/val D0.
Eval and collect are unchanged.

Example (Joint collect stamp)::

    python scripts/prepare_dexjoco.py \\
      --task-name fold_glasses \\
      --collect-dir collect_results/dexjoco/fold_glasses/20260907_141408 \\
      --gpus 0,1,2,3

Incremental D0 collect V on an existing stamp (does not redo failure scan)::

    python scripts/prepare_dexjoco.py \\
      --task-name fold_glasses \\
      --collect-dir collect_results/dexjoco/fold_glasses/20260907_141408 \\
      --output-dir prepare_results/dexjoco/fold_glasses/20260908_090014 \\
      --phases scan_d0 \\
      --gpus 0,2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (
    REPO_ROOT / "src",
    REPO_ROOT / "scripts",
    REPO_ROOT / "third_party" / "dexjoco" / "dexjoco",
):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import eval_dexjoco as eval_entry
from dewo_v2.tasks import CfgRecipe, eval_task_yaml, get_task, resolve_collect_max_steps, resolve_expert

ALL_PHASES = ("scan", "scan_d0", "critic", "materialize", "eve")
HYDRA_S0 = "dexjoco/dexjoco_dewo_v9_offline_b1_jump_fast_uncond"
HYDRA_SCRATCH = "dexjoco/dexjoco_dewo_scratch_joint"
HYDRA_TASK = HYDRA_S0
PRIMARY_KIND = "all_success_seeds"
PRIMARY_SEED = 20260820
STEP_DIR_RE = re.compile(r"^step_(\d+)$")
CFG_ENV = {
    "CFG_SUCCESS_SUFFIX": " Successful execution.",
    "CFG_FAILURE_SUFFIX": " Failed execution.",
    "CFG_PRIMARY_OUTCOME": "0.9",
    "CFG_PRIMARY_FAST": "0.0",
    "CFG_PRIMARY_BASE": "0.1",
    "CFG_AUX_SUCCESS_OUTCOME": "1.0",
    "CFG_AUX_SUCCESS_FAST": "0.0",
    "CFG_AUX_SUCCESS_BASE": "0.0",
    "CFG_AUX_FAIL_OUTCOME": "1.0",
    "CFG_AUX_FAIL_FAST": "0.0",
    "CFG_AUX_FAIL_BASE": "0.0",
}


def is_v91(version: str | None) -> bool:
    return str(version or "").strip() in {"v9.1", "v91"}


def prepare_hydra_task(version: str | None) -> str:
    return HYDRA_SCRATCH if is_v91(version) else HYDRA_S0


def pool_index_path(pair_out: Path, version: str | None) -> Path:
    return pair_out / ("pool_index.json" if is_v91(version) else "pair_index.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_phases(value: str) -> list[str]:
    text = str(value).strip()
    if not text or text == "all":
        return list(ALL_PHASES)
    phases = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [item for item in phases if item not in ALL_PHASES]
    if not phases:
        raise argparse.ArgumentTypeError("expected a non-empty phase list")
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown phases {unknown}; allowed: {', '.join(ALL_PHASES)}"
        )
    return phases


def _gpu_csv(gpus: list[int]) -> str:
    return ",".join(str(item) for item in gpus)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _shell_export(key: str, value: Any) -> str:
    if value is None:
        return f"export {key}="
    if isinstance(value, bool):
        return f"export {key}={'true' if value else 'false'}"
    if isinstance(value, (int, float)):
        return f"export {key}={value}"
    return f"export {key}={shlex.quote(str(value))}"


def _resolve_model_config(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    if run_dir.is_file() and run_dir.suffix in {".yaml", ".yml"}:
        return run_dir.resolve()
    candidate = run_dir / "config.yaml"
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"No config.yaml under {run_dir}")


def resolve_collect_layout(
    collect_dir: Path,
    checkpoint_steps: list[int] | None = None,
) -> dict[str, Any]:
    """Map a collect stamp or legacy root onto ``step -> rollout_raw``."""

    collect_dir = Path(collect_dir).expanduser().resolve()
    if not collect_dir.exists():
        raise FileNotFoundError(f"Missing collect dir: {collect_dir}")

    config: dict[str, Any] | None = None
    config_path = collect_dir / "collect_config.json"
    if config_path.is_file():
        config = _read_json(config_path)

    raw_by_step: dict[int, Path] = {}
    for step_dir in sorted(collect_dir.glob("step_*")):
        match = STEP_DIR_RE.match(step_dir.name)
        if match is None:
            continue
        raw = step_dir / "rollout_raw"
        if (raw / "meta" / "info.json").is_file():
            raw_by_step[int(match.group(1))] = raw.resolve()

    if raw_by_step:
        if checkpoint_steps:
            steps = [int(step) for step in checkpoint_steps]
        elif config and config.get("checkpoint_steps"):
            steps = [int(step) for step in config["checkpoint_steps"]]
        else:
            steps = sorted(raw_by_step)
        missing = [step for step in steps if step not in raw_by_step]
        if missing:
            raise FileNotFoundError(
                f"{collect_dir} is missing rollout_raw for steps {missing}"
            )
        return {
            "layout": "collect_dexjoco",
            "collect_dir": collect_dir,
            "config": config,
            "steps": steps,
            "raw_by_step": {step: raw_by_step[step] for step in steps},
        }

    for name in ("rollout_raw_200", "rollout_raw"):
        raw = collect_dir / name
        if (raw / "meta" / "info.json").is_file():
            if checkpoint_steps:
                steps = [int(step) for step in checkpoint_steps]
            elif config and config.get("checkpoint_steps"):
                steps = [int(step) for step in config["checkpoint_steps"]]
            else:
                steps = [55000]
            return {
                "layout": "legacy_collect",
                "collect_dir": collect_dir,
                "config": config,
                "steps": steps,
                "raw_by_step": {step: raw.resolve() for step in steps},
            }

    raise FileNotFoundError(
        f"No rollout_raw under {collect_dir}. "
        "Expected collect_results/.../step_XXXXX/rollout_raw or rollout_raw_200."
    )


def scan_is_complete(scan_root: Path) -> bool:
    summary_path = Path(scan_root) / "summary.json"
    if not summary_path.is_file():
        return False
    summary = _read_json(summary_path)
    n_pairs = int(summary.get("num_complete_event_pairs") or 0)
    return summary.get("status") == "complete" and n_pairs > 0


def scan_d0_is_complete(scan_root: Path) -> bool:
    summary_path = Path(scan_root) / "summary.json"
    if not summary_path.is_file():
        return False
    summary = _read_json(summary_path)
    n_prefix = int(summary.get("num_prefix_results") or 0)
    return summary.get("status") == "complete" and n_prefix > 0


def _find_existing_critic(collect_dir: Path, step_dir: Path) -> Path | None:
    for path in (
        step_dir / "v9_critic_index.json",
        collect_dir / "v9_critic_index.json",
        collect_dir / "v8_critic_index.json",
    ):
        if path.is_file():
            return path.resolve()
    return None


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    extra = [
        str(REPO_ROOT / "src"),
        str(REPO_ROOT / "scripts"),
        str(REPO_ROOT / "third_party" / "dexjoco" / "dexjoco"),
    ]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(extra + ([existing] if existing else []))
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    env.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(REPO_ROOT / "checkpoints"))
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    return env


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[prepare] {printable}", flush=True)
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env or _base_env())
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({eval_entry._format_exitcode(completed.returncode)}): {printable}"
        )


def _python() -> str:
    return sys.executable


def run_scan(
    *,
    raw: Path,
    scan_root: Path,
    args: argparse.Namespace,
    step: int,
) -> None:
    if scan_is_complete(scan_root) and not args.overwrite:
        print(f"[step {step}] scan already complete; skipping {scan_root}", flush=True)
        return
    if args.require_existing_scan:
        raise RuntimeError(f"Scan incomplete at {scan_root} and --require-existing-scan was set")
    cmd = [
        _python(),
        str(REPO_ROOT / "scripts" / "fold_glasses" / "run_recoverability_pair_scan.py"),
        "--gpus",
        _gpu_csv(args.gpus),
        "--dataset",
        str(raw),
        "--output",
        str(scan_root),
        "--checkpoint",
        str(eval_entry._checkpoint_path(args.checkpoint_dir, step)),
        "--model-config",
        str(args.model_config),
        "--dataset-stats",
        str(args.dataset_stats),
        "--task-name",
        str(args.task_name),
        "--max-steps",
        str(args.max_steps),
        "--action-horizon",
        str(args.action_horizon),
        "--replan-steps",
        str(args.replan_steps),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--pass-m",
        str(args.pass_m),
        "--skip-pin-check",
    ]
    if args.text_embedding is not None:
        cmd.extend(["--text-embedding", str(args.text_embedding)])
    if args.overwrite:
        cmd.append("--overwrite")
    _run(cmd)
    if not scan_is_complete(scan_root):
        raise RuntimeError(f"Scan did not complete at {scan_root}")


def write_d0_value_index(scan_root: Path, dest: Path) -> Path:
    from dewo_v2.d0_collect_value import (
        build_d0_value_index,
        load_jsonl,
        write_d0_value_index as _write,
    )

    prefixes = load_jsonl(scan_root / "prefix_results.jsonl")
    payload = build_d0_value_index(prefixes, scan_root=scan_root)
    _write(dest, payload)
    print(
        f"[prepare] D0 value index {dest} "
        f"episodes={payload['num_episodes']} "
        f"frames={payload['num_labeled_frames']}",
        flush=True,
    )
    return dest


def run_scan_d0(
    *,
    raw: Path,
    scan_root: Path,
    value_index: Path,
    args: argparse.Namespace,
    step: int,
) -> None:
    if scan_d0_is_complete(scan_root) and value_index.is_file() and not args.overwrite:
        print(f"[step {step}] D0 collect scan already complete; skipping {scan_root}", flush=True)
        return
    cmd = [
        _python(),
        str(REPO_ROOT / "scripts" / "fold_glasses" / "run_recoverability_pair_scan.py"),
        "--gpus",
        _gpu_csv(args.gpus),
        "--dataset",
        str(raw),
        "--output",
        str(scan_root),
        "--checkpoint",
        str(eval_entry._checkpoint_path(args.checkpoint_dir, step)),
        "--model-config",
        str(args.model_config),
        "--dataset-stats",
        str(args.dataset_stats),
        "--task-name",
        str(args.task_name),
        "--max-steps",
        str(args.max_steps),
        "--action-horizon",
        str(args.action_horizon),
        "--replan-steps",
        str(args.replan_steps),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--pass-m",
        str(args.pass_m),
        "--selection",
        "d0_collect",
        "--skip-pin-check",
    ]
    if args.text_embedding is not None:
        cmd.extend(["--text-embedding", str(args.text_embedding)])
    if args.overwrite:
        cmd.append("--overwrite")
    _run(cmd)
    if not scan_d0_is_complete(scan_root):
        raise RuntimeError(f"D0 collect scan did not complete at {scan_root}")
    write_d0_value_index(scan_root, value_index)


def run_result_videos(
    *,
    scan_root: Path,
    raw: Path,
    output_dir: Path,
    overwrite: bool,
) -> None:
    cmd = [
        _python(),
        str(REPO_ROOT / "scripts" / "fold_glasses" / "compose_failure_recoverability_videos.py"),
        "--scan-root",
        str(scan_root),
        "--raw-dataset",
        str(raw),
        "--output-dir",
        str(output_dir),
    ]
    if overwrite:
        cmd.append("--overwrite")
    cmd.extend(["--jobs", "2"])
    _run(cmd)


def run_critic(
    *,
    collect_dir: Path,
    raw: Path,
    scan_root: Path,
    critic_path: Path,
    overwrite: bool,
    step: int,
    task_name: str,
) -> Path:
    existing = critic_path if critic_path.is_file() else _find_existing_critic(collect_dir, critic_path.parent)
    if existing is not None and not overwrite:
        print(f"[step {step}] reuse critic index {existing}", flush=True)
        return existing
    cmd = [
        _python(),
        str(REPO_ROOT / "scripts" / "dewo_v2" / "build_v9_critic_index.py"),
        "--collect-root",
        str(collect_dir),
        "--scan-root",
        str(scan_root),
        "--raw-dataset",
        str(raw),
        "--output",
        str(critic_path),
        "--task-name",
        str(task_name),
    ]
    _run(cmd)
    if not critic_path.is_file():
        raise RuntimeError(f"Missing critic index after build: {critic_path}")
    return critic_path


def run_materialize(
    *,
    critic_path: Path,
    raw: Path,
    pair_out: Path,
    success_prompt: str,
    overwrite: bool,
    step: int,
    dewo_version: str,
) -> Path:
    index_path = pool_index_path(pair_out, dewo_version)
    if index_path.is_file() and not overwrite:
        print(f"[step {step}] reuse pool LeRobot {pair_out}", flush=True)
        return pair_out
    script = (
        "materialize_v91_pool_lerobot.py"
        if is_v91(dewo_version)
        else "materialize_v9_full_pair_lerobot.py"
    )
    cmd = [
        _python(),
        str(REPO_ROOT / "scripts" / "dewo_v2" / script),
        "--critic-index",
        str(critic_path),
        "--source-dataset",
        str(raw),
        "--output-dataset",
        str(pair_out),
        "--success-prompt",
        success_prompt,
        "--overwrite",
    ]
    _run(cmd)
    if not index_path.is_file():
        raise RuntimeError(f"Missing {index_path.name} after materialize: {index_path}")
    return pair_out


def _write_protocol_files(
    *,
    args: argparse.Namespace,
    step: int,
    step_dir: Path,
    raw: Path,
    pair_out: Path,
    eve_root: Path,
    text_cache: Path,
    vae_cache: Path,
    pair_manifest: Path,
    val_manifest: Path,
    env_file: Path,
    expert_root: Path | None = None,
) -> None:
    spec = get_task(args.task_name)
    recipe = CfgRecipe()
    protocol = eve_root / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    cfg_json = protocol / "cfg_recipe.json"
    eval_entry._write_json(
        cfg_json,
        {"task": asdict(spec), "cfg": recipe.as_json()},
    )
    eval_dir = step_dir / "eval_task_cfg"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_yaml = eval_dir / f"{args.task_name}.yaml"
    eval_yaml.write_text(eval_task_yaml(spec, recipe), encoding="utf-8")

    ckpt = eval_entry._checkpoint_path(args.checkpoint_dir, step)
    bundle = protocol / "opensource_bundle_manifest.txt"
    bundle.write_text(
        "\n".join(
            [
                f"checkpoint={ckpt}",
                f"model_config={args.model_config}",
                f"dataset_stats={args.dataset_stats}",
                "stack=opensource_FastWAMDexJocoPolicy",
                "image_size=224",
                "norm=z-score",
                f"recipe=recoverability_pairs_{PRIMARY_KIND}",
                f"task={args.task_name}",
                f"cfg_recipe={cfg_json}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    src_cfg_sha = _sha256_file(args.model_config)
    norm_sha = _sha256_file(args.dataset_stats)
    hydra_task = prepare_hydra_task(getattr(args, "dewo_version", None))
    v91 = is_v91(getattr(args, "dewo_version", None))
    values = {
        "FITWAM_ENV_PREFIX": str(Path(_python()).parent),
        "TASK": args.task_name,
        "DEWO_TASK_NAME": args.task_name,
        "DEWO_VERSION": "v9.1" if v91 else "v9",
        "BASE_DATASET": str(raw),
        "PAIR_DATASET": str(pair_out),
        "ROLLOUT_RAW": str(raw),
        "EXPERT_DATASET": str(expert_root) if expert_root is not None else "",
        "PRIMARY_KIND": PRIMARY_KIND,
        "PRIMARY_N": 15,
        "CKPT": str(ckpt),
        "INIT_WEIGHTS": str(ckpt),
        "SOURCE_CHECKPOINT": str(ckpt),
        "STATS": str(args.dataset_stats),
        "FASTWAM_SOURCE_CONFIG": str(args.model_config),
        "FASTWAM_SOURCE_CONFIG_SHA256": src_cfg_sha,
        "SOURCE_BUNDLE_MANIFEST": str(bundle),
        "B1_MANIFEST_PATH": str(pair_manifest),
        "EVE_MANIFEST_PATH": str(pair_manifest),
        "EVE_VAL_MANIFEST_PATH": str(val_manifest),
        "PROTOCOL_BUNDLE_PATH": str(protocol / "offline_v1_b1_jump_fast.json"),
        "PRETRAINED_NORM_STATS": str(args.dataset_stats),
        "NORM_STATS_SOURCE": "compute",
        "NORM_STATS_META_DIR": "",
        "NORM_STATS_BUNDLE_SHA256": norm_sha,
        "TEXT_EMBEDDING_CACHE_DIR": str(text_cache),
        "CFG_TASK_CONFIG_DIR": str(eval_dir),
        "B1_VIDEO_EXPERIMENT_ROOT": str(step_dir),
        "USE_VAE_LATENT_CACHE": 1 if args.use_vae else 0,
        "VAE_LATENT_CACHE_DIR": str(vae_cache),
        "REQUIRE_VAE_LATENT_CACHE": 1 if args.use_vae else 0,
        "DEWO_TASK": hydra_task,
        "DEWO_VARIANT": (
            "DEWO-scratch-joint" if v91 else "B1-jump-fast-v9-uncond-adapter"
        ),
        "DEWO_PROTOCOL": (
            f"{args.task_name}_dewo_scratch_joint"
            if v91
            else f"{args.task_name}_dewo_v9_uncond_adapter_isolated"
        ),
        "DEWO_OUTPUT_DIR": (
            f"./runs/dexjoco_{args.task_name}_dewo_scratch_joint"
            if v91
            else f"./runs/dexjoco_{args.task_name}_dewo_v9"
        ),
        "FITWAM_WANDB_GROUP": (
            f"{args.task_name}_dewo_scratch_joint"
            if v91
            else f"{args.task_name}_dewo_v9_opensource"
        ),
        "SUCCESS_PROMPT": args.success_prompt,
    }
    lines = [
        "# Generated by scripts/prepare_dexjoco.py (opensource 224 / z-score)",
        "# Paths / VAE / text cache only. CFG mixing is owned by scripts/train_dexjoco.py",
        "# (Successful / Failed execution., D+ 0.9/0/0.1, D_fail 1.0/0/0, no FAST).",
    ]
    lines.extend(_shell_export(key, value) for key, value in values.items())
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    eval_entry._write_json(
        protocol / "offline_v1_b1_jump_fast.json",
        {
            "protocol": (
                f"{args.task_name}_dewo_v91_scratch_pool"
                if v91
                else f"{args.task_name}_dewo_v9_recoverability_pairs"
            ),
            "variant": (
                "DEWO-scratch-joint" if v91 else "B1-jump-fast-v9-uncond-adapter"
            ),
            "stack": "opensource_224_zscore",
            "manifest": str(pair_manifest),
            "val_manifest": str(val_manifest),
            "pair_dataset": str(pair_out),
            "expert_dataset": str(expert_root) if expert_root is not None else None,
            "pretrained_norm_stats": str(args.dataset_stats),
            "source_config": str(args.model_config),
            "checkpoint": str(ckpt),
            "include_s0_success_rollouts": True,
            "include_expert_success": bool(v91),
            "primary_kind": PRIMARY_KIND,
            "cfg": {
                "success_suffix": " Successful execution.",
                "failure_suffix": " Failed execution.",
                "primary": "0.9,0.0,0.1",
                "aux_success": "1.0,0.0,0.0",
                "aux_fail": "1.0,0.0,0.0",
                "note": "Owned by scripts/train_dexjoco.py; not mixed from this file.",
            },
        },
    )


def _hydra_env(env_file_values: dict[str, str]) -> dict[str, str]:
    env = _base_env()
    env.update(CFG_ENV)
    env.update(env_file_values)
    for compact in ("CFG_PRIMARY", "CFG_AUX_SUCCESS", "CFG_AUX_FAIL"):
        env.pop(compact, None)
    return env


def _load_exports(env_file: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("export "):
            continue
        body = stripped[len("export ") :]
        if "=" not in body:
            continue
        key, raw = body.split("=", 1)
        parts = shlex.split(raw) if raw.strip() else []
        values[key] = parts[0] if parts else ""
    return values


def run_eve(
    *,
    args: argparse.Namespace,
    step_dir: Path,
    raw: Path,
    pair_out: Path,
    step: int,
) -> Path:
    eve_root = step_dir / "eve_v02"
    text_cache = args.text_cache or (step_dir / "text_embeds_cache")
    vae_cache = args.vae_cache or (step_dir / "vae_latent_cache")
    env_file = eve_root / "protocol" / "offline_v1_b1_jump_fast.env"
    pair_manifest = eve_root / "manifests" / "offline_b1_jump_fast_pair.json"
    val_manifest = eve_root / "manifests" / "offline_selection_primary_success.json"
    primary_id = f"{args.task_name}_s0_success_rollouts"
    pair_id = f"{args.task_name}_pair_events"
    expert_id = f"{args.task_name}_expert_success"
    v91 = is_v91(getattr(args, "dewo_version", None))
    hydra_task = prepare_hydra_task(getattr(args, "dewo_version", None))

    if (
        env_file.is_file()
        and pair_manifest.is_file()
        and (not args.use_vae or any(vae_cache.glob("*.pt")))
        and not args.overwrite
    ):
        print(f"[step {step}] reuse Eve protocol {env_file}", flush=True)
        return env_file

    expert_root = resolve_expert(get_task(args.task_name)) if v91 else None
    expert_eve = step_dir / "eve_expert" if v91 else None

    if args.overwrite and eve_root.exists():
        shutil.rmtree(eve_root)
    if args.overwrite and expert_eve is not None and expert_eve.exists():
        shutil.rmtree(expert_eve)

    eve_root.mkdir(parents=True, exist_ok=True)
    (eve_root / "protocol").mkdir(parents=True, exist_ok=True)

    splits = eve_root / "splits" / "episode_splits.jsonl"
    if not splits.is_file() or args.overwrite:
        print(f"[step {step}] select D0 (one complete success per 4/4 seed)", flush=True)
        _run(
            [
                _python(),
                str(REPO_ROOT / "scripts" / "dewo_v2" / "select_success_rollout_primary.py"),
                "--dataset",
                str(raw),
                "--dataset-id",
                primary_id,
                "--mode",
                "one_per_all_success_seed",
                "--seed",
                str(PRIMARY_SEED),
                "--task-name",
                str(args.task_name),
                "--output-json",
                str(eve_root / "protocol" / "primary_success_episodes.json"),
                "--output-splits",
                str(splits),
            ]
        )

    if not (eve_root / "episode_meta.jsonl").is_file() or args.overwrite:
        print(f"[step {step}] init Eve sidecar on S0 success rollouts", flush=True)
        _run(
            [
                _python(),
                str(REPO_ROOT / "scripts" / "everobot" / "build_eve_sidecar.py"),
                "init-base",
                "--dataset-root",
                str(raw),
                "--dataset-id",
                primary_id,
                "--eve-root",
                str(eve_root),
                "--task-name",
                str(args.task_name),
                "--source-type",
                "policy_rollout",
                "--source-policy",
                "s0_success_rollout",
                "--collection-round",
                "0",
                "--force-success",
                "--split-map",
                str(splits),
                "--config-path",
                str(args.model_config),
                "--code-commit",
                _git_head(REPO_ROOT),
            ]
        )

    primary_manifest = eve_root / "manifests" / "offline_primary_success.json"
    expert_manifest = (
        expert_eve / "manifests" / "offline_expert_success.json" if expert_eve is not None else None
    )
    print(f"[step {step}] build primary + pool manifests", flush=True)
    _run(
        [
            _python(),
            str(REPO_ROOT / "scripts" / "everobot" / "build_eve_sidecar.py"),
            "build-manifest",
            "--eve-root",
            str(eve_root),
            "--manifest-name",
            "offline_primary_success",
            "--include-outcomes",
            "success",
            "--success-dataset-ids",
            primary_id,
            "--success-sample-mode",
            "episode_only",
            "--splits",
            "train",
        ]
    )
    if v91:
        if expert_root is None or expert_eve is None:
            raise RuntimeError("v9.1 prepare requires the expert LeRobot dataset")
        if not (expert_eve / "episode_meta.jsonl").is_file() or args.overwrite:
            print(f"[step {step}] init Eve sidecar on expert successes {expert_root}", flush=True)
            _run(
                [
                    _python(),
                    str(REPO_ROOT / "scripts" / "everobot" / "build_eve_sidecar.py"),
                    "init-base",
                    "--dataset-root",
                    str(expert_root),
                    "--dataset-id",
                    expert_id,
                    "--eve-root",
                    str(expert_eve),
                    "--task-name",
                    str(args.task_name),
                    "--source-type",
                    "expert_success",
                    "--source-policy",
                    "human_or_expert",
                    "--collection-round",
                    "-1",
                    "--force-success",
                    "--split",
                    "train",
                    "--config-path",
                    str(args.model_config),
                    "--code-commit",
                    _git_head(REPO_ROOT),
                ]
            )
        _run(
            [
                _python(),
                str(REPO_ROOT / "scripts" / "everobot" / "build_eve_sidecar.py"),
                "build-manifest",
                "--eve-root",
                str(expert_eve),
                "--manifest-name",
                "offline_expert_success",
                "--include-outcomes",
                "success",
                "--success-dataset-ids",
                expert_id,
                "--success-sample-mode",
                "episode_only",
                "--splits",
                "train",
            ]
        )
        v91_cmd = [
                _python(),
                str(REPO_ROOT / "scripts" / "dewo_v2" / "build_v91_manifest.py"),
                "--collect-manifest",
                str(primary_manifest),
                "--expert-manifest",
                str(expert_manifest),
                "--pool-index",
                str(pair_out / "pool_index.json"),
                "--pool-dataset",
                str(pair_out),
                "--pool-dataset-id",
                pair_id,
                "--prompt",
                str(args.success_prompt),
                "--recipe",
                f"{args.task_name}_dewo_v91_scratch_pool",
                "--output",
                str(pair_manifest),
        ]
        d0_value_index = step_dir / "d0_collect_value_index.json"
        if d0_value_index.is_file():
            v91_cmd.extend(["--d0-value-index", str(d0_value_index)])
        _run(v91_cmd)
    else:
        _run(
            [
                _python(),
                str(REPO_ROOT / "scripts" / "dewo_v2" / "build_pair_manifest.py"),
                "--expert-manifest",
                str(primary_manifest),
                "--pair-dataset",
                str(pair_out),
                "--pair-dataset-id",
                pair_id,
                "--prompt",
                str(args.success_prompt),
                "--recipe",
                f"{args.task_name}_dewo_v9_recoverability_pairs",
                "--output",
                str(pair_manifest),
                "--primary-source",
                PRIMARY_KIND,
                "--horizon",
                "full",
                "--skip-aux-success",
            ]
        )
    _run(
        [
            _python(),
            str(REPO_ROOT / "scripts" / "everobot" / "build_eve_sidecar.py"),
            "build-manifest",
            "--eve-root",
            str(eve_root),
            "--manifest-name",
            "offline_selection_primary_success",
            "--include-outcomes",
            "success",
            "--success-dataset-ids",
            primary_id,
            "--success-sample-mode",
            "episode_only",
            "--splits",
            "val",
        ]
    )

    text_cache.mkdir(parents=True, exist_ok=True)
    vae_cache.mkdir(parents=True, exist_ok=True)
    _write_protocol_files(
        args=args,
        step=step,
        step_dir=step_dir,
        raw=raw,
        pair_out=pair_out,
        eve_root=eve_root,
        text_cache=text_cache,
        vae_cache=vae_cache,
        pair_manifest=pair_manifest,
        val_manifest=val_manifest,
        env_file=env_file,
        expert_root=expert_root,
    )

    hydra_env = _hydra_env(_load_exports(env_file))
    print(f"[step {step}] precompute base + outcome text embeds", flush=True)
    _run(
        [
            _python(),
            str(REPO_ROOT / "scripts" / "precompute_text_embeds.py"),
            f"task={hydra_task}",
        ],
        env=hydra_env,
    )
    _run(
        [
            _python(),
            str(REPO_ROOT / "scripts" / "export_text_embed_cache_npz.py"),
            "--cache-dir",
            str(text_cache),
        ],
        env=hydra_env,
    )
    text_sha = hashlib.sha256()
    for path in sorted(text_cache.glob("*.pt")):
        text_sha.update(path.name.encode())
        text_sha.update(str(path.stat().st_size).encode())
    with env_file.open("a", encoding="utf-8") as handle:
        handle.write(f"export TEXT_EMBEDDING_CACHE_SHA256={text_sha.hexdigest()}\n")

    if not args.use_vae:
        print(f"[step {step}] skip VAE pre-encode (--no-vae)", flush=True)
        return env_file

    print(f"[step {step}] VAE pre-encode on GPUs {_gpu_csv(args.gpus)}", flush=True)
    world = len(args.gpus)
    workers: list[subprocess.Popen] = []
    for rank, gpu_id in enumerate(args.gpus):
        shard_env = dict(hydra_env)
        shard_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        shard_env["WORLD_SIZE"] = "1"
        for key in ("RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK", "LOCAL_WORLD_SIZE"):
            shard_env.pop(key, None)
        cmd = [
            _python(),
            str(REPO_ROOT / "scripts" / "precompute_vae_latents.py"),
            f"task={hydra_task}",
            f"+vae_latent_cache_dir={vae_cache}",
            f"+encode_val={str(args.vae_encode_val).lower()}",
        ]
        if world > 1:
            cmd.extend(
                [
                    f"+vae_shard_rank={rank}",
                    f"+vae_shard_world={world}",
                ]
            )
        print(f"[prepare] gpu={gpu_id} {' '.join(shlex.quote(part) for part in cmd)}", flush=True)
        workers.append(
            subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=shard_env)
        )
    failures = []
    for rank, proc in enumerate(workers):
        code = proc.wait()
        if code != 0:
            failures.append((rank, eval_entry._format_exitcode(code)))
    if failures:
        raise RuntimeError(f"VAE shard failures: {failures}")
    n_vae = len(list(vae_cache.glob("*.pt")))
    print(f"[step {step}] VAE pre-encode done files={n_vae} cache={vae_cache}", flush=True)
    if n_vae < 1:
        raise RuntimeError(f"VAE cache is empty: {vae_cache}")
    return env_file


def _append_queue(queue_file: Path, payload: dict[str, str]) -> None:
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    block = "".join(f"{key}={value}\n" for key, value in payload.items())
    with queue_file.open("a", encoding="utf-8") as handle:
        handle.write(block)
        if not block.endswith("\n"):
            handle.write("\n")


def _write_results_md(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    task = config.get("task_name", "")
    lines = [
        f"# {task} DEWO v9 prepare",
        "",
        f"Collect: `{config.get('collect_dir', '')}`",
        f"Output: `{output_dir}`",
        f"Failure + V(s) videos: `{output_dir / 'result'}`",
        "",
        "| Ckpt | Scan pairs | Pair episodes | Env |",
        "|---|---:|---:|---|",
    ]
    for row in summaries:
        env = row.get("env_file", "")
        lines.append(
            f"| `{row.get('checkpoint_step')}` | {row.get('num_complete_event_pairs', 0)} "
            f"| {row.get('num_pair_episodes', 0)} | `{env}` |"
        )
    if summaries:
        env_file = summaries[-1].get("env_file")
        if env_file:
            lines.extend(
                [
                    "",
                    "Next:",
                    "",
                    "```bash",
                    f"python scripts/train_dexjoco.py \\",
                    f"  --task-name {task} --init scratch --dewo-version v9.1 \\",
                    f"  --prepare-dir {output_dir} --gpus 0,1,2,3",
                    "```",
                    "",
                ]
            )
    (output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True)
    parser.add_argument(
        "--collect-dir",
        type=Path,
        required=True,
        help="collect_dexjoco stamp dir, or legacy collect root with rollout_raw_200.",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-steps", type=eval_entry._parse_int_list, default=None)
    parser.add_argument("--dataset-stats", type=Path, default=None)
    parser.add_argument("--text-embedding", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scan-dir", type=Path, default=None)
    parser.add_argument("--pair-dataset", type=Path, default=None)
    parser.add_argument("--text-cache", type=Path, default=None)
    parser.add_argument("--vae-cache", type=Path, default=None)
    parser.add_argument("--queue-file", type=Path, default=None)
    parser.add_argument("--gpus", type=eval_entry._parse_int_list, default=[1])
    parser.add_argument("--phases", type=_parse_phases, default=list(ALL_PHASES))
    parser.add_argument("--success-prompt", type=str, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--replan-steps", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--pass-m",
        type=int,
        default=10,
        help="Recoverability scan Pass@M trials per prefix. Stops at the first 0/M cliff.",
    )
    parser.add_argument(
        "--require-existing-scan",
        action="store_true",
        help="Do not launch recoverability scan; fail if summary.json is incomplete.",
    )
    parser.add_argument("--no-vae", dest="use_vae", action="store_false")
    parser.set_defaults(use_vae=True)
    parser.add_argument("--vae-encode-val", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dewo-version",
        default="v9.1",
        choices=("v9", "v9.1"),
        help="v9.1 = D0 expert+collect, scan crops, no stitch. v9 = full-horizon stitch.",
    )
    return parser


def _fill_from_collect_config(args: argparse.Namespace, config: dict[str, Any] | None) -> None:
    config = config or {}

    def _take(name: str, key: str, cast: Any = None) -> None:
        if getattr(args, name) is not None:
            return
        if key not in config or config[key] in (None, ""):
            return
        value = config[key]
        setattr(args, name, cast(value) if cast is not None else value)

    _take("run_dir", "run_dir", Path)
    _take("checkpoint_dir", "checkpoint_dir", Path)
    if args.checkpoint_steps is None and config.get("checkpoint_steps"):
        args.checkpoint_steps = [int(step) for step in config["checkpoint_steps"]]
    _take("dataset_stats", "dataset_stats", Path)
    _take("text_embedding", "text_embedding", Path)
    _take("success_prompt", "success_prompt", str)
    _take("action_horizon", "action_horizon", int)
    _take("replan_steps", "replan_steps", int)
    _take("num_inference_steps", "num_inference_steps", int)
    _take("max_steps", "max_steps", int)


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    spec = get_task(args.task_name)
    layout = resolve_collect_layout(args.collect_dir, args.checkpoint_steps)
    args.collect_dir = layout["collect_dir"]
    args._layout = layout
    _fill_from_collect_config(args, layout.get("config"))
    args.success_prompt = args.success_prompt or spec.success_prompt
    args.action_horizon = int(args.action_horizon or spec.action_horizon)
    args.replan_steps = int(args.replan_steps or spec.replan_steps)
    args.num_inference_steps = int(args.num_inference_steps or spec.nfe)
    if args.max_steps is None:
        args.max_steps = int(resolve_collect_max_steps(spec))
    else:
        args.max_steps = int(args.max_steps)
    args.checkpoint_steps = [int(step) for step in layout["steps"]]
    if args.run_dir is None:
        args.run_dir = REPO_ROOT / "configs/eval/dexjoco/mixed_5task_fastwam_joint"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = (
            REPO_ROOT / "checkpoints/dexjoco/mixed_5task_fastwam_joint/weights"
        )
    if args.dataset_stats is None:
        args.dataset_stats = REPO_ROOT / "artifacts/mixed_5task/dataset_stats.json"
    args.run_dir = Path(args.run_dir).expanduser().resolve()
    args.checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    args.dataset_stats = Path(args.dataset_stats).expanduser().resolve()
    if args.text_embedding is not None:
        args.text_embedding = Path(args.text_embedding).expanduser().resolve()
    args.model_config = _resolve_model_config(args.run_dir)
    args.output_dir = args.output_dir or (
        REPO_ROOT
        / "prepare_results"
        / "dexjoco"
        / args.task_name
        / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    if args.scan_dir is not None:
        args.scan_dir = Path(args.scan_dir).expanduser().resolve()
    if args.pair_dataset is not None:
        args.pair_dataset = Path(args.pair_dataset).expanduser().resolve()
    if args.text_cache is not None:
        args.text_cache = Path(args.text_cache).expanduser().resolve()
    if args.vae_cache is not None:
        args.vae_cache = Path(args.vae_cache).expanduser().resolve()
    if args.queue_file is not None:
        args.queue_file = Path(args.queue_file).expanduser().resolve()
    if len(args.checkpoint_steps) > 1 and (args.scan_dir or args.pair_dataset):
        raise ValueError("--scan-dir / --pair-dataset only apply when preparing one checkpoint")
    return args


def _validate_args(args: argparse.Namespace) -> None:
    for label, path in {
        "collect dir": args.collect_dir,
        "run dir": args.run_dir,
        "dataset stats": args.dataset_stats,
        "checkpoint directory": args.checkpoint_dir,
        "model config": args.model_config,
    }.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if args.text_embedding is not None and not Path(args.text_embedding).exists():
        raise FileNotFoundError(f"Missing text embedding: {args.text_embedding}")
    if not str(args.success_prompt).strip():
        raise ValueError("--success-prompt must not be empty")
    collect_task = (args._layout.get("config") or {}).get("task_name")
    if collect_task and str(collect_task) != str(args.task_name):
        raise ValueError(
            f"--task-name {args.task_name} does not match collect_config.task_name={collect_task}"
        )
    if args.task_name not in str(args.collect_dir):
        print(
            f"[prepare] warning: collect-dir {args.collect_dir} does not contain task {args.task_name}",
            flush=True,
        )
    for step in args.checkpoint_steps:
        ckpt = eval_entry._checkpoint_path(args.checkpoint_dir, step)
        if not ckpt.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")


def _step_paths(args: argparse.Namespace, step: int, raw: Path) -> dict[str, Path]:
    step_dir = eval_entry._step_output_dir(args.output_dir, step)
    layout = args._layout
    if args.scan_dir is not None:
        scan_root = args.scan_dir
    elif layout["layout"] == "legacy_collect":
        scan_root = layout["collect_dir"] / "recoverability_pairs_v2"
    else:
        scan_root = step_dir / "recoverability_pairs"
    d0_scan_root = step_dir / "d0_collect_recoverability"
    pair_out = args.pair_dataset or (step_dir / "pair_lerobot")
    return {
        "step_dir": step_dir,
        "raw": raw,
        "scan_root": scan_root,
        "d0_scan_root": d0_scan_root,
        "d0_value_index": step_dir / "d0_collect_value_index.json",
        "pair_out": pair_out,
        "critic": step_dir / "v9_critic_index.json",
    }


def prepare_step(args: argparse.Namespace, step: int, raw: Path) -> dict[str, Any]:
    paths = _step_paths(args, step, raw)
    step_dir = paths["step_dir"]
    step_dir.mkdir(parents=True, exist_ok=True)
    phases = set(args.phases)
    started = time.perf_counter()
    env_file: Path | None = None

    if "scan" in phases:
        run_scan(raw=paths["raw"], scan_root=paths["scan_root"], args=args, step=step)
        run_result_videos(
            scan_root=paths["scan_root"],
            raw=paths["raw"],
            output_dir=args.output_dir / "result",
            overwrite=args.overwrite,
        )
    elif bool(phases & {"critic", "materialize", "eve"}) and not scan_is_complete(
        paths["scan_root"]
    ):
        raise RuntimeError(
            f"Phase list skipped scan but {paths['scan_root']} is incomplete"
        )

    if "scan_d0" in phases:
        run_scan_d0(
            raw=paths["raw"],
            scan_root=paths["d0_scan_root"],
            value_index=paths["d0_value_index"],
            args=args,
            step=step,
        )
    elif (
        bool(phases & {"eve"})
        and paths["d0_value_index"].is_file() is False
        and scan_d0_is_complete(paths["d0_scan_root"])
    ):
        write_d0_value_index(paths["d0_scan_root"], paths["d0_value_index"])

    critic_path = paths["critic"]
    later_needs_pool = bool(phases & {"materialize", "eve"})
    if "critic" in phases:
        critic_path = run_critic(
            collect_dir=args.collect_dir,
            raw=paths["raw"],
            scan_root=paths["scan_root"],
            critic_path=paths["critic"],
            overwrite=args.overwrite,
            step=step,
            task_name=str(args.task_name),
        )
    elif later_needs_pool and not critic_path.is_file():
        existing = _find_existing_critic(args.collect_dir, step_dir)
        if existing is None:
            raise RuntimeError(f"Phase list skipped critic but no index at {critic_path}")
        critic_path = existing

    if "materialize" in phases:
        run_materialize(
            critic_path=critic_path,
            raw=paths["raw"],
            pair_out=paths["pair_out"],
            success_prompt=args.success_prompt,
            overwrite=args.overwrite,
            step=step,
            dewo_version=str(args.dewo_version),
        )
    elif "eve" in phases and not pool_index_path(paths["pair_out"], args.dewo_version).is_file():
        raise RuntimeError(f"Phase list skipped materialize but missing {paths['pair_out']}")

    if "eve" in phases:
        env_file = run_eve(
            args=args,
            step_dir=step_dir,
            raw=paths["raw"],
            pair_out=paths["pair_out"],
            step=step,
        )
    else:
        candidate = step_dir / "eve_v02" / "protocol" / "offline_v1_b1_jump_fast.env"
        env_file = candidate if candidate.is_file() else None

    scan_summary = {}
    if (paths["scan_root"] / "summary.json").is_file():
        scan_summary = _read_json(paths["scan_root"] / "summary.json")
    d0_scan_summary = {}
    if (paths["d0_scan_root"] / "summary.json").is_file():
        d0_scan_summary = _read_json(paths["d0_scan_root"] / "summary.json")
    pair_index = {}
    pair_index_path = pool_index_path(paths["pair_out"], args.dewo_version)
    if pair_index_path.is_file():
        pair_index = _read_json(pair_index_path)
    pool_counts = pair_index.get("counts") or {}
    summary = {
        "checkpoint_step": int(step),
        "dewo_version": str(args.dewo_version),
        "rollout_raw": str(paths["raw"]),
        "scan_root": str(paths["scan_root"]),
        "d0_scan_root": str(paths["d0_scan_root"]),
        "d0_value_index": str(paths["d0_value_index"])
        if paths["d0_value_index"].is_file()
        else None,
        "critic_index": str(critic_path) if Path(critic_path).is_file() else None,
        "pair_dataset": str(paths["pair_out"]),
        "env_file": str(env_file) if env_file else None,
        "num_complete_event_pairs": int(scan_summary.get("num_complete_event_pairs") or 0),
        "num_d0_prefix_results": int(d0_scan_summary.get("num_prefix_results") or 0),
        "num_d0_success_episodes": int(
            d0_scan_summary.get("num_selected_success_episodes") or 0
        ),
        "num_pair_episodes": int(pair_index.get("num_pairs") or 0) * 2,
        "num_pairs": int(pair_index.get("num_pairs") or 0),
        "v91_d_scan": int(pool_counts.get("d_scan") or 0),
        "v91_d_fail": int(pool_counts.get("d_fail") or 0),
        "v91_dplus": int(pool_counts.get("dplus") or 0),
        "checkpoint_wall_seconds": time.perf_counter() - started,
        "completed_at": _utc_now(),
    }
    scan_d0_only = list(args.phases) == ["scan_d0"]
    if scan_d0_only:
        eval_entry._write_json(step_dir / "d0_collect_scan_summary.json", summary)
    else:
        eval_entry._write_json(step_dir / "summary.json", summary)
    return summary


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    args = _resolve_args(build_parser().parse_args())
    _validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scan_d0_only = list(args.phases) == ["scan_d0"]
    existing_config = args.output_dir / "prepare_config.json"
    log_name = "scan_d0_orchestrator.log" if scan_d0_only else "orchestrator.log"
    eval_entry._install_file_logging(args.output_dir / "logs" / log_name)

    layout = args._layout
    config_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if not key.startswith("_")
    }
    config_payload.update(
        {
            "layout": layout["layout"],
            "started_at": _utc_now(),
            "hydra_task": prepare_hydra_task(args.dewo_version),
            "primary_kind": PRIMARY_KIND,
            "model_config": str(args.model_config),
        }
    )
    config_path = (
        args.output_dir / "scan_d0_config.json"
        if scan_d0_only and existing_config.is_file()
        else args.output_dir / "prepare_config.json"
    )
    eval_entry._write_json(config_path, config_payload)
    print(f"Prepare output: {args.output_dir}", flush=True)
    print(f"  collect: {args.collect_dir} ({layout['layout']})", flush=True)
    print(f"  logs:    {args.output_dir / 'logs'}", flush=True)

    summaries: list[dict[str, Any]] = []
    for step in args.checkpoint_steps:
        raw = layout["raw_by_step"][step]
        print(f"[step {step}] rollout_raw={raw}", flush=True)
        summary = prepare_step(args, step, raw)
        summaries.append(summary)
        if scan_d0_only:
            eval_entry._write_json(args.output_dir / "scan_d0_summary.json", summaries)
        else:
            eval_entry._write_json(args.output_dir / "summary.json", summaries)
            _write_results_md(args.output_dir, summaries, config_payload)
        print(
            f"[step {step}] pairs={summary['num_pairs']} env={summary['env_file']}"
            + (
                f" d0_prefixes={summary.get('num_d0_prefix_results')}"
                if scan_d0_only
                else ""
            ),
            flush=True,
        )
        if args.queue_file is not None and summary.get("env_file"):
            _append_queue(
                args.queue_file,
                {
                    "TASK": args.task_name,
                    "ENV_FILE": str(summary["env_file"]),
                    "EXP_ROOT": str(eval_entry._step_output_dir(args.output_dir, step)),
                    "PAIR_OUT": str(summary["pair_dataset"]),
                    "COLLECT_ROOT": str(args.collect_dir),
                    "PREPARED_AT": _utc_now(),
                },
            )

    if scan_d0_only:
        print(f"Complete: {args.output_dir / 'scan_d0_summary.json'}", flush=True)
    else:
        print(f"Complete: {args.output_dir / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
