#!/usr/bin/env python3
"""Closed-loop DexJoCo eval for FastWAM / FastWAMJoint / FastWAMIDM / DEWOv9.

In-process (no ZMQ). ``--text-cfg-scale`` is the mix weight ``w``::

    ε_cfg = ε_base + w (ε_posi − ε_base)

  0 = 本体 (base prompt, adapter off)
  1 = 纯优势 (success prompt, adapter on)
  >1 = CFG guide (e.g. 2)

Rebuild tables from an existing output dir::

    python scripts/eval_dexjoco.py --summarize-dir evaluate_results/dexjoco/<task>/<stamp>
"""

from __future__ import annotations

import argparse
import csv
import faulthandler
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (
    REPO_ROOT / "src",
    REPO_ROOT / "scripts",
    REPO_ROOT / "third_party" / "dexjoco" / "dexjoco",
):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from fastwam.inference.config import InferenceConfig
from fastwam.inference.dexjoco import FastWAMDexJocoPolicy
from fastwam.inference.rollout import rollout_episode, save_rollout_mp4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated integer list")
    return values


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _optional_path(value: Any) -> str | None:
    if value in (None, "", "None"):
        return None
    return str(value)


class _Tee:
    def __init__(self, *files: Any) -> None:
        self.files = files

    def write(self, data: Any) -> None:
        if isinstance(data, bytes):
            data = data.decode("utf-8", "replace")
        for handle in self.files:
            handle.write(data)
            handle.flush()

    def flush(self) -> None:
        for handle in self.files:
            handle.flush()

    def fileno(self) -> int:
        return self.files[0].fileno()


def _format_exitcode(code: int | None) -> str:
    if code is None:
        return "running"
    if code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"signal_{-code}"
        return f"{code} ({name})"
    return str(code)


def _install_file_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    faulthandler.enable(file=log_file, all_threads=True)


def _set_worker_render_env(gpu_id: int) -> None:
    """Pin MuJoCo EGL to the same physical GPU the policy will use.

    ``CUDA_VISIBLE_DEVICES`` is intentionally not changed here: this module
    already imported torch, so the worker still addresses ``cuda:{gpu_id}``.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _create_task_env(task_name: str, seed: int) -> Any:
    from dexjoco.tasks import CONFIG_MAPPING

    return CONFIG_MAPPING[task_name]().get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=False,
        seed=seed,
        randomize_dynamics=False,
    )


def _checkpoint_path(checkpoint_dir: Path, step: int) -> Path:
    return checkpoint_dir / f"step_{step:06d}.pt"


def _step_output_dir(run_dir: Path, step: int) -> Path:
    return run_dir / f"step_{step:06d}"


def _candidate_priority(step: int, result: str, seed: int, repeat: int) -> int:
    key = f"fastwam-dexjoco:{step}:{result}:{seed}:{repeat}".encode()
    return int.from_bytes(hashlib.sha256(key).digest(), byteorder="big")


def _candidate_identity(path: Path) -> tuple[int, int]:
    parts = path.stem.split("_")
    if len(parts) != 4 or parts[0] != "seed" or parts[2] != "repeat":
        raise ValueError(f"Unexpected candidate filename: {path.name}")
    return int(parts[1]), int(parts[3])


def _would_retain_candidate(
    step_dir: Path,
    gpu_id: int,
    step: int,
    result: str,
    seed: int,
    repeat: int,
    capacity: int,
) -> bool:
    candidate_dir = step_dir / "videos" / "candidates" / result / f"gpu_{gpu_id}"
    candidates = list(candidate_dir.glob("*.mp4"))
    if len(candidates) < capacity:
        return True
    new_priority = _candidate_priority(step, result, seed, repeat)
    worst_priority = max(
        _candidate_priority(step, result, *_candidate_identity(path))
        for path in candidates
    )
    return new_priority < worst_priority


def _retain_candidate(
    temporary_video: Path,
    step_dir: Path,
    gpu_id: int,
    step: int,
    result: str,
    seed: int,
    repeat: int,
    capacity: int,
) -> None:
    candidate_dir = step_dir / "videos" / "candidates" / result / f"gpu_{gpu_id}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate = candidate_dir / f"seed_{seed:03d}_repeat_{repeat}.mp4"
    temporary_video.replace(candidate)
    candidates = list(candidate_dir.glob("*.mp4"))
    candidates.sort(
        key=lambda path: _candidate_priority(
            step, result, *_candidate_identity(path)
        )
    )
    for discarded in candidates[capacity:]:
        discarded.unlink(missing_ok=True)


def _finalize_videos(step_dir: Path, step: int, capacity: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    candidates_root = step_dir / "videos" / "candidates"
    for result in ("success", "failure"):
        candidates = list((candidates_root / result).glob("gpu_*/*.mp4"))
        candidates.sort(
            key=lambda path: _candidate_priority(
                step, result, *_candidate_identity(path)
            )
        )
        selected = candidates[:capacity]
        output_dir = step_dir / "videos" / result
        output_dir.mkdir(parents=True, exist_ok=True)
        for rank, candidate in enumerate(selected, start=1):
            seed, repeat = _candidate_identity(candidate)
            destination = output_dir / (
                f"trajectory_{rank:02d}_seed_{seed:03d}_repeat_{repeat}_{result}.mp4"
            )
            candidate.replace(destination)
        counts[result] = len(selected)
    if candidates_root.exists():
        shutil.rmtree(candidates_root)
    return counts


def _worker_main(
    gpu_id: int,
    assigned_seeds: list[int],
    args_dict: dict[str, Any],
    step: int,
) -> None:
    try:
        _run_worker(gpu_id, assigned_seeds, args_dict, step)
    except Exception:
        traceback.print_exc()
        raise


def _run_worker(
    gpu_id: int,
    assigned_seeds: list[int],
    args_dict: dict[str, Any],
    step: int,
) -> None:
    os.chdir(REPO_ROOT)
    torch.set_num_threads(1)
    _set_worker_render_env(gpu_id)

    args = argparse.Namespace(**args_dict)
    run_dir = Path(args.output_dir)
    step_dir = _step_output_dir(run_dir, step)
    _install_file_logging(run_dir / "logs" / f"step_{step:06d}_gpu_{gpu_id}.log")
    worker_results = step_dir / "workers" / f"gpu_{gpu_id}.jsonl"
    existing_rows = _read_jsonl(worker_results)
    completed = {(int(row["seed"]), int(row["repeat"])) for row in existing_rows}
    expected = {(seed, repeat) for seed in assigned_seeds for repeat in range(args.repeats)}
    remaining_seeds = [
        seed
        for seed in assigned_seeds
        if not all((seed, repeat) in completed for repeat in range(args.repeats))
    ]
    if expected and not remaining_seeds:
        print(f"[step {step} gpu {gpu_id}] shard already complete", flush=True)
        return

    # Create the EGL renderer before the policy fills the GPU. Vanilla FastWAM
    # loads a VAE; initializing MuJoCo afterwards is a common SIGABRT.
    first_seed = remaining_seeds[0]
    print(
        f"[step {step} gpu {gpu_id}] creating env seed={first_seed} "
        f"(MUJOCO_GL={os.environ.get('MUJOCO_GL')} "
        f"MUJOCO_EGL_DEVICE_ID={os.environ.get('MUJOCO_EGL_DEVICE_ID')})",
        flush=True,
    )
    env = _create_task_env(args.task_name, first_seed)
    print(f"[step {step} gpu {gpu_id}] env ready", flush=True)

    torch.cuda.set_device(gpu_id)
    checkpoint = _checkpoint_path(Path(args.checkpoint_dir), step)
    load_start = time.perf_counter()
    infer_cfg = InferenceConfig(
        action_horizon=int(args.action_horizon),
        replan_steps=int(args.replan_steps),
        value_replan_steps=(
            None if args.value_replan_steps is None else int(args.value_replan_steps)
        ),
        num_inference_steps=int(args.num_inference_steps),
        text_cfg_scale=float(args.text_cfg_scale),
        cfg_exec_horizon=int(args.cfg_exec_horizon),
        adaptive_cfg_tau=args.adaptive_cfg_tau,
        cfg_epsilon_l=args.cfg_epsilon_l,
        cfg_residual_clip_mode=str(args.cfg_residual_clip_mode),
        cfg_gate_mode=str(args.cfg_gate_mode),
        cfg_v_high=args.cfg_v_high,
        cfg_drop_delta=float(args.cfg_drop_delta),
        cfg_growth_tau=float(args.cfg_growth_tau),
        cfg_growth_start_replan=int(args.cfg_growth_start_replan),
        cfg_growth_stop_replan=args.cfg_growth_stop_replan,
        cfg_low_value_threshold=float(args.cfg_low_value_threshold),
        cfg_growth_delta=float(args.cfg_growth_delta),
        cfg_growth_once=bool(args.cfg_growth_once),
    )
    policy = FastWAMDexJocoPolicy(
        model_config=args.run_dir,
        checkpoint=checkpoint,
        dataset_stats=args.dataset_stats,
        text_embedding=_optional_path(args.text_embedding),
        text_embedding_base=_optional_path(args.text_embedding_base),
        text_embedding_failure=_optional_path(args.text_embedding_failure),
        device=f"cuda:{gpu_id}",
        action_horizon=int(args.action_horizon),
        replan_steps=int(args.replan_steps),
        value_replan_steps=args.value_replan_steps,
        num_inference_steps=int(args.num_inference_steps),
        task_name=args.task_name,
        load_text_encoder=args.load_text_encoder,
        inference_config=infer_cfg,
    )
    print(
        f"[step {step} gpu {gpu_id}] loaded in {time.perf_counter() - load_start:.1f}s; "
        f"seeds={assigned_seeds}",
        flush=True,
    )
    inner = policy.inner
    print(
        f"[step {step} gpu {gpu_id}] infer align: "
        f"num_video_frames={inner.num_video_frames} "
        f"action_horizon={inner.action_horizon} "
        f"num_inference_steps={inner.num_inference_steps} "
        f"image_resize={policy.adapter.image_resize} "
        f"keep_uint8_image={policy.adapter.keep_uint8_image} "
        f"mot_target={getattr(inner.model, '__class__', type(inner.model)).__name__}",
        flush=True,
    )

    current_seed = first_seed
    try:
        for seed in remaining_seeds:
            if seed != current_seed:
                env.close()
                env = _create_task_env(args.task_name, seed)
                current_seed = seed
            for repeat in range(args.repeats):
                obs, _ = env.reset()
                if (seed, repeat) in completed:
                    continue
                row, frames = rollout_episode(
                    env=env,
                    initial_obs=obs,
                    policy=policy,
                    seed=seed,
                    repeat=repeat,
                    max_steps=args.max_steps,
                )
                row.pop("cfg_values", None)
                row.pop("cfg_value_rels", None)
                row.update({"checkpoint_step": step, "gpu_id": gpu_id})

                if _would_retain_candidate(
                    step_dir,
                    gpu_id,
                    step,
                    row["result"],
                    seed,
                    repeat,
                    args.video_samples_per_result,
                ):
                    temporary_video = (
                        step_dir
                        / "videos"
                        / "temp"
                        / f"gpu_{gpu_id}_seed_{seed:03d}_repeat_{repeat}.mp4"
                    )
                    temporary_video.unlink(missing_ok=True)
                    encode_start = time.perf_counter()
                    save_rollout_mp4(frames, temporary_video, fps=args.video_fps)
                    row["video_encode_seconds"] = time.perf_counter() - encode_start
                    row["wall_seconds"] += row["video_encode_seconds"]
                    _retain_candidate(
                        temporary_video,
                        step_dir,
                        gpu_id,
                        step,
                        row["result"],
                        seed,
                        repeat,
                        args.video_samples_per_result,
                    )
                    print(
                        f"[step {step} gpu {gpu_id}] saved candidate video "
                        f"{row['result']} seed={seed} repeat={repeat}",
                        flush=True,
                    )
                else:
                    row["video_encode_seconds"] = 0.0

                _append_jsonl(worker_results, row)
                completed.add((seed, repeat))
                print(
                    f"[step {step} gpu {gpu_id}] seed={seed} repeat={repeat} "
                    f"result={row['result']} steps={row['episode_steps']} "
                    f"wall={row['wall_seconds']:.1f}s",
                    flush=True,
                )
    finally:
        env.close()


def _episode_success(row: dict[str, Any]) -> bool:
    value = row.get("success")
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _population_mean_std(values: list[float]) -> tuple[float, float, float]:
    """Population std (divide by n), matching the official DexJoCo 4×50 tables."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0, 0.0
    var = float(np.mean((np.asarray(values, dtype=np.float64) - mean) ** 2))
    return mean, var, float(np.sqrt(var))


def _load_step_rows(step_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for worker_path in sorted((step_dir / "workers").glob("gpu_*.jsonl")):
        rows.extend(_read_jsonl(worker_path))
    if not rows:
        csv_path = step_dir / "episodes.csv"
        if csv_path.is_file():
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
    unique = {(int(row["seed"]), int(row["repeat"])): row for row in rows}
    return sorted(unique.values(), key=lambda row: (int(row["seed"]), int(row["repeat"])))


def _protocol_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_repeat: dict[int, list[dict[str, Any]]] = {}
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_repeat.setdefault(int(row["repeat"]), []).append(row)
        by_seed.setdefault(int(row["seed"]), []).append(row)

    runs: list[dict[str, Any]] = []
    for repeat in sorted(by_repeat):
        group = sorted(by_repeat[repeat], key=lambda item: int(item["seed"]))
        successes = sum(_episode_success(item) for item in group)
        failed_seeds = [
            int(item["seed"]) for item in group if not _episode_success(item)
        ]
        runs.append(
            {
                "run": repeat + 1,
                "repeat": repeat,
                "episodes": len(group),
                "successes": successes,
                "failures": len(group) - successes,
                "success_rate": successes / len(group),
                "failed_seeds": failed_seeds,
            }
        )

    rates = [float(run["success_rate"]) for run in runs]
    mean_rate, var_rate, std_rate = _population_mean_std(rates)

    per_seed: list[dict[str, Any]] = []
    for seed in sorted(by_seed):
        group = sorted(by_seed[seed], key=lambda item: int(item["repeat"]))
        flags = [_episode_success(item) for item in group]
        per_seed.append(
            {
                "seed": seed,
                "repeats": len(group),
                "successes": int(sum(flags)),
                "success_rate": sum(flags) / len(flags),
                "results": [
                    "success" if flag else "failure" for flag in flags
                ],
                "episode_steps": [int(item["episode_steps"]) for item in group],
            }
        )

    return {
        "protocol": "official_4x50_seeds" if len(by_seed) == 50 and len(by_repeat) == 4 else "repeats_x_seeds",
        "n_runs": len(runs),
        "n_seeds": len(by_seed),
        "mean_success_rate": mean_rate,
        "std_success_rate": std_rate,
        "var_success_rate": var_rate,
        "runs_summary": ", ".join(
            f"r{run['run']}={run['successes']}/{run['episodes']}" for run in runs
        ),
        "runs": runs,
        "per_seed": per_seed,
    }


def _summarize_episodes(
    rows: list[dict[str, Any]],
    *,
    step: int,
    video_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    video_counts = video_counts or {"success": 0, "failure": 0}
    successes = sum(_episode_success(row) for row in rows)
    summary: dict[str, Any] = {
        "checkpoint_step": step,
        "episodes": len(rows),
        "successes": successes,
        "failures": len(rows) - successes,
        "success_rate": successes / len(rows) if rows else float("nan"),
        "mean_episode_steps": float(np.mean([int(row["episode_steps"]) for row in rows])),
        "mean_episode_wall_seconds": float(
            np.mean([float(row.get("wall_seconds", 0.0)) for row in rows])
        ),
        "total_inference_seconds_across_workers": float(
            sum(float(row.get("inference_seconds", 0.0)) for row in rows)
        ),
        "saved_success_videos": int(video_counts.get("success", 0)),
        "saved_failure_videos": int(video_counts.get("failure", 0)),
        "completed_at": _utc_now(),
    }
    summary.update(_protocol_stats(rows))
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_summary_for_csv(summary: dict[str, Any]) -> dict[str, Any]:
    skip = {"runs", "per_seed"}
    row = {key: value for key, value in summary.items() if key not in skip}
    if summary.get("runs"):
        row["runs_summary"] = summary.get("runs_summary") or ", ".join(
            f"r{run['run']}={run['successes']}/{run['episodes']}"
            for run in summary["runs"]
        )
    return row


def _write_protocol_tables(step_dir: Path, summary: dict[str, Any]) -> None:
    _write_json(
        step_dir / "official_4x50.json",
        {
            "checkpoint_step": summary["checkpoint_step"],
            "protocol": summary.get("protocol"),
            "pooled": {
                "successes": summary["successes"],
                "episodes": summary["episodes"],
                "success_rate": summary["success_rate"],
            },
            "mean_success_rate": summary.get("mean_success_rate"),
            "std_success_rate": summary.get("std_success_rate"),
            "var_success_rate": summary.get("var_success_rate"),
            "n_runs": summary.get("n_runs"),
            "n_seeds": summary.get("n_seeds"),
            "runs": summary.get("runs", []),
            "per_seed": summary.get("per_seed", []),
        },
    )
    run_rows = []
    for run in summary.get("runs", []):
        run_rows.append(
            {
                "run": run["run"],
                "repeat": run["repeat"],
                "successes": run["successes"],
                "failures": run["failures"],
                "episodes": run["episodes"],
                "success_rate": run["success_rate"],
                "failed_seeds": " ".join(str(seed) for seed in run["failed_seeds"]),
            }
        )
    _write_csv(step_dir / "runs.csv", run_rows)

    seed_rows = []
    for item in summary.get("per_seed", []):
        row = {
            "seed": item["seed"],
            "successes": item["successes"],
            "repeats": item["repeats"],
            "success_rate": item["success_rate"],
        }
        for index, result in enumerate(item["results"]):
            row[f"r{index + 1}"] = result
        for index, steps in enumerate(item["episode_steps"]):
            row[f"r{index + 1}_steps"] = steps
        seed_rows.append(row)
    _write_csv(step_dir / "per_seed.csv", seed_rows)


def _write_results_md(
    run_dir: Path,
    summaries: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> None:
    config = config or {}
    task = config.get("task_name", run_dir.parent.name)
    lines = [
        f"# {task} official 4×50",
        "",
        f"Root: `{run_dir}`",
        "",
        "| Ckpt | Pooled | Mean±Std (4 runs) | Runs |",
        "|---|---:|---:|---|",
    ]
    for summary in summaries:
        mean = summary.get("mean_success_rate")
        std = summary.get("std_success_rate")
        mean_std = (
            "n/a"
            if mean is None or std is None or not np.isfinite(mean)
            else f"{100 * float(mean):.1f}%±{100 * float(std):.1f}%"
        )
        lines.append(
            f"| `{summary['checkpoint_step']}` | "
            f"{summary['successes']}/{summary['episodes']} "
            f"({100 * float(summary['success_rate']):.1f}%) | "
            f"{mean_std} | {summary.get('runs_summary', '')} |"
        )

    for summary in summaries:
        lines.extend(["", f"## step_{int(summary['checkpoint_step']):06d} per-run", ""])
        for run in summary.get("runs", []):
            failed = ", ".join(str(seed) for seed in run["failed_seeds"]) or "none"
            lines.extend(
                [
                    f"### Run {run['run']} (repeat {run['repeat']}): "
                    f"{run['successes']}/{run['episodes']} "
                    f"({100 * float(run['success_rate']):.1f}%)",
                    "",
                    f"Failed seeds ({run['failures']}): {failed}",
                    "",
                ]
            )
        lines.extend(
            [
                f"## step_{int(summary['checkpoint_step']):06d} per-seed (50 env seeds × 4 repeats)",
                "",
                "| seed | r1 | r2 | r3 | r4 | successes |",
                "|---:|---|---|---|---|---:|",
            ]
        )
        for item in summary.get("per_seed", []):
            cells = item["results"] + ["-"] * max(0, 4 - len(item["results"]))
            lines.append(
                f"| {item['seed']} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | "
                f"{item['successes']}/{item['repeats']} |"
            )

    (run_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_step_outputs(step_dir: Path, summary: dict[str, Any]) -> None:
    _write_json(step_dir / "summary.json", summary)
    _write_protocol_tables(step_dir, summary)


def _collect_step(
    run_dir: Path,
    step: int,
    expected_episodes: int,
    video_capacity: int,
) -> dict[str, Any]:
    step_dir = _step_output_dir(run_dir, step)
    rows = _load_step_rows(step_dir)
    if len(rows) != expected_episodes:
        raise RuntimeError(
            f"step {step}: expected {expected_episodes} episodes, found {len(rows)}"
        )

    fieldnames = sorted({key for row in rows for key in row})
    with (step_dir / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    video_counts = _finalize_videos(step_dir, step, video_capacity)
    summary = _summarize_episodes(rows, step=step, video_counts=video_counts)
    _write_step_outputs(step_dir, summary)
    temporary_dir = step_dir / "videos" / "temp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    return summary


def _write_overall_summary(
    run_dir: Path,
    summaries: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> None:
    _write_json(run_dir / "summary.json", summaries)
    _write_results_md(run_dir, summaries, config)
    if not summaries:
        return
    flat = [_flatten_summary_for_csv(summary) for summary in summaries]
    fieldnames = sorted({key for row in flat for key in row})
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)


def _summarize_existing_output(output_dir: Path) -> None:
    config_path = output_dir / "eval_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing eval_config.json under {output_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = int(config["expected_episodes_per_checkpoint"])
    summaries: list[dict[str, Any]] = []
    for step in config["checkpoint_steps"]:
        step_dir = _step_output_dir(output_dir, int(step))
        rows = _load_step_rows(step_dir)
        if len(rows) != expected:
            raise RuntimeError(
                f"step {step}: expected {expected} episodes, found {len(rows)}"
            )
        existing: dict[str, Any] = {}
        summary_path = step_dir / "summary.json"
        if summary_path.is_file():
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
        video_counts = {
            "success": int(existing.get("saved_success_videos", 0)),
            "failure": int(existing.get("saved_failure_videos", 0)),
        }
        summary = _summarize_episodes(
            rows, step=int(step), video_counts=video_counts
        )
        if "checkpoint_wall_seconds" in existing:
            summary["checkpoint_wall_seconds"] = existing["checkpoint_wall_seconds"]
        if "completed_at" in existing:
            summary["completed_at"] = existing["completed_at"]
        _write_step_outputs(step_dir, summary)
        summaries.append(summary)
        print(_format_step_line(summary), flush=True)
    _write_overall_summary(output_dir, summaries, config)
    print(f"Wrote {output_dir / 'RESULTS.md'}", flush=True)


def _format_step_line(summary: dict[str, Any]) -> str:
    mean = summary.get("mean_success_rate")
    std = summary.get("std_success_rate")
    mean_std = (
        "n/a"
        if mean is None or std is None or not np.isfinite(float(mean))
        else f"{100 * float(mean):.2f}%±{100 * float(std):.2f}%"
    )
    return (
        f"[step {summary['checkpoint_step']}] "
        f"pooled {summary['successes']}/{summary['episodes']} "
        f"({100 * float(summary['success_rate']):.2f}%); "
        f"mean±std {mean_std}; {summary.get('runs_summary', '')}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-steps", type=_parse_int_list, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument(
        "--text-embedding",
        type=Path,
        default=None,
        help="Cached T5 for the success / task prompt. With --text-cfg-scale 0 this is the only text used.",
    )
    parser.add_argument(
        "--text-embedding-base",
        type=Path,
        default=None,
        help="Cached T5 for the base prompt (ε_base). Required when --text-cfg-scale != 0 unless the text encoder is loaded.",
    )
    parser.add_argument(
        "--text-embedding-failure",
        type=Path,
        default=None,
        help="Cached T5 for the failure prompt. Optional for v9 (mix subtracts ε_base).",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpus", type=_parse_int_list, default=[0])
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=49)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument(
        "--value-replan-steps",
        type=int,
        default=None,
        help="Value-head query stride. Default: same as --replan-steps.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument(
        "--text-cfg-scale",
        type=float,
        default=0.0,
        help="Mix weight w: 0=本体, 1=纯优势, >1=CFG guide.",
    )
    parser.add_argument("--cfg-exec-horizon", type=int, default=24)
    parser.add_argument("--adaptive-cfg-tau", type=float, default=None)
    parser.add_argument("--cfg-epsilon-l", type=float, default=None)
    parser.add_argument("--cfg-residual-clip-mode", default="rms")
    parser.add_argument("--cfg-gate-mode", default="off")
    parser.add_argument("--cfg-v-high", type=float, default=None)
    parser.add_argument("--cfg-drop-delta", type=float, default=0.15)
    parser.add_argument("--cfg-growth-tau", type=float, default=0.05)
    parser.add_argument("--cfg-growth-start-replan", type=int, default=2)
    parser.add_argument("--cfg-growth-stop-replan", type=int, default=None)
    parser.add_argument("--cfg-low-value-threshold", type=float, default=0.10)
    parser.add_argument("--cfg-growth-delta", type=float, default=0.01)
    parser.add_argument("--cfg-growth-once", action="store_true")
    parser.add_argument(
        "--load-text-encoder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default: load encoder only when no --text-embedding is given.",
    )
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-samples-per-result", type=int, default=5)
    return parser


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.output_dir = args.output_dir or (
        REPO_ROOT
        / "evaluate_results"
        / "dexjoco"
        / args.task_name
        / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    return args


def _validate_args(args: argparse.Namespace) -> None:
    for label, path in {
        "run dir": args.run_dir,
        "dataset stats": args.dataset_stats,
        "checkpoint directory": args.checkpoint_dir,
    }.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    for label, path in {
        "text embedding": args.text_embedding,
        "text embedding base": args.text_embedding_base,
        "text embedding failure": args.text_embedding_failure,
    }.items():
        if path is not None and not Path(path).exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    wants_cfg = float(args.text_cfg_scale) != 0.0 or str(args.cfg_gate_mode) not in {"", "off"}
    if wants_cfg and args.text_embedding is not None and args.text_embedding_base is None:
        if args.load_text_encoder is False:
            raise ValueError(
                "CFG mix (text_cfg_scale != 0 or a CFG gate) needs --text-embedding-base "
                "or a loaded text encoder."
            )
    for step in args.checkpoint_steps:
        path = _checkpoint_path(Path(args.checkpoint_dir), step)
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for gpu_id in args.gpus:
        if not 0 <= gpu_id < torch.cuda.device_count():
            raise ValueError(
                f"GPU {gpu_id} unavailable; visible count={torch.cuda.device_count()}"
            )
    if args.seed_end < args.seed_start:
        raise ValueError("seed_end must be >= seed_start")
    for name in ("repeats", "max_steps"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    video_samples = getattr(args, "video_samples_per_result", None)
    if video_samples is not None and int(video_samples) <= 0:
        raise ValueError("video_samples_per_result must be positive")


def main() -> None:
    if "--summarize-dir" in sys.argv:
        parser = argparse.ArgumentParser(
            description="Rebuild DexJoCo eval summaries from an existing output dir."
        )
        parser.add_argument("--summarize-dir", type=Path, required=True)
        args, _unknown = parser.parse_known_args()
        _summarize_existing_output(Path(args.summarize_dir).resolve())
        return

    os.environ.setdefault("MUJOCO_GL", "egl")
    args = _resolve_args(build_parser().parse_args())
    _validate_args(args)
    args.output_dir = Path(args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _install_file_logging(args.output_dir / "logs" / "orchestrator.log")

    seeds = list(range(args.seed_start, args.seed_end + 1))
    expected_episodes = len(seeds) * args.repeats
    config_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config_payload.update(
        {
            "expected_episodes_per_checkpoint": expected_episodes,
            "started_at": _utc_now(),
        }
    )
    _write_json(args.output_dir / "eval_config.json", config_payload)

    args_dict = config_payload.copy()
    summaries: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    print(f"Evaluation output: {args.output_dir}", flush=True)
    print(f"  logs:    {args.output_dir / 'logs'}", flush=True)
    print(
        f"  videos:  {args.output_dir / 'step_<ckpt>' / 'videos'}/{{success,failure}} "
        f"(up to {args.video_samples_per_result} each)",
        flush=True,
    )
    print(
        f"  rows:    {args.output_dir / 'step_<ckpt>' / 'workers'}/gpu_*.jsonl",
        flush=True,
    )

    for step in args.checkpoint_steps:
        step_dir = _step_output_dir(args.output_dir, step)
        completed_summary_path = step_dir / "summary.json"
        if completed_summary_path.exists():
            completed_summary = json.loads(completed_summary_path.read_text(encoding="utf-8"))
            if int(completed_summary.get("episodes", -1)) == expected_episodes:
                if "mean_success_rate" not in completed_summary:
                    rows = _load_step_rows(step_dir)
                    completed_summary = _summarize_episodes(
                        rows,
                        step=step,
                        video_counts={
                            "success": int(completed_summary.get("saved_success_videos", 0)),
                            "failure": int(completed_summary.get("saved_failure_videos", 0)),
                        },
                    )
                    _write_step_outputs(step_dir, completed_summary)
                print(f"[step {step}] already complete; skipping", flush=True)
                print(_format_step_line(completed_summary), flush=True)
                summaries.append(completed_summary)
                _write_overall_summary(args.output_dir, summaries, config_payload)
                continue

        checkpoint_start = time.perf_counter()
        processes: list[mp.Process] = []
        for worker_index, gpu_id in enumerate(args.gpus):
            assigned_seeds = seeds[worker_index :: len(args.gpus)]
            if not assigned_seeds:
                continue
            process = context.Process(
                target=_worker_main,
                args=(gpu_id, assigned_seeds, args_dict, step),
                name=f"dexjoco-step{step}-gpu{gpu_id}",
            )
            process.start()
            processes.append(process)

        failures = []
        for process in processes:
            process.join()
            if process.exitcode != 0:
                failures.append((process.name, _format_exitcode(process.exitcode)))
        if failures:
            raise RuntimeError(
                f"Worker failures for step {step}: {failures}. "
                f"See {args.output_dir / 'logs'}"
            )

        summary = _collect_step(
            args.output_dir,
            step,
            expected_episodes,
            args.video_samples_per_result,
        )
        summary["checkpoint_wall_seconds"] = time.perf_counter() - checkpoint_start
        _write_json(step_dir / "summary.json", summary)
        summaries.append(summary)
        _write_overall_summary(args.output_dir, summaries, config_payload)
        print(_format_step_line(summary), flush=True)

    print(f"Complete: {args.output_dir / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
