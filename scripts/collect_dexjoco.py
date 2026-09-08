#!/usr/bin/env python3
"""Closed-loop DexJoCo collect for FastWAM / FastWAMJoint / FastWAMIDM / DEWOv9.

In-process (no ZMQ), same inference stack as ``scripts/eval_dexjoco.py``.
Each GPU writes a LeRobot shard; the parent merges to ``rollout_raw``.

Default protocol is collect 4×50 (seeds 10086–10135 × 4 = 200), so it does
not overlap eval seeds 0–49.

Example (Joint, avoid GPU 0)::

    python scripts/collect_dexjoco.py \\
      --task-name fold_glasses \\
      --run-dir configs/eval/dexjoco/mixed_5task_fastwam_joint \\
      --checkpoint-dir checkpoints/dexjoco/mixed_5task_fastwam_joint/weights \\
      --checkpoint-steps 55000 \\
      --dataset-stats artifacts/mixed_5task/dataset_stats.json \\
      --text-embedding <t5.pt> --no-load-text-encoder \\
      --gpus 1,2,3,4,5,6,7 --text-cfg-scale 0
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
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

import eval_dexjoco as eval_entry
from build_rollout_datasets import merge_shards, validate_outcome_dataset
from collect_dexjoco_rollouts import (
    OUTCOME_LEDGER_NAME,
    aggregate_stats,
    append_jsonl,
    make_outcome_row,
    prepare_dataset,
    save_lerobot_episode,
    serialize_dict,
    update_info,
    write_json,
)
from dewo_v2.tasks import get_task
from fastwam.inference.config import InferenceConfig
from fastwam.inference.dexjoco import FastWAMDexJocoPolicy
from fastwam.inference.rollout import rollout_episode

FAILURE_PHRASE = "Failed to finish the whole process."


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _shard_dir(step_dir: Path, gpu_id: int) -> Path:
    return step_dir / "shards" / f"gpu_{gpu_id}"


def _done_pairs(attempts: list[dict[str, Any]]) -> set[tuple[int, int]]:
    done: set[tuple[int, int]] = set()
    for row in attempts:
        if row.get("saved_episode_index") is None or row.get("success") is None:
            continue
        if "repeat" not in row:
            continue
        done.add((int(row["seed"]), int(row["repeat"])))
    return done


def _write_shard_summary(
    *,
    shard_dir: Path,
    args: argparse.Namespace,
    gpu_id: int,
    seeds: list[int],
    attempts: list[dict[str, Any]],
    n_ep: int,
    global_i: int,
    status: str,
    step: int,
) -> None:
    write_json(
        shard_dir / "collection_summary.json",
        {
            "status": status,
            "mode": "save_all",
            "outcome_task_mode": "clean",
            "checkpoint_step": int(step),
            "target_episodes": len(seeds) * int(args.repeats),
            "attempts": len(attempts),
            "episodes": n_ep,
            "frames": global_i,
            "failures": sum(1 for item in attempts if not item["success"]),
            "successes_saved": sum(1 for item in attempts if item["success"]),
            "attempt_log": attempts,
            "inference_stack": "eval_dexjoco.FastWAMDexJocoPolicy",
            "base_seed": int(args.seed_start),
            "seed_end": int(args.seed_end),
            "repeats": int(args.repeats),
            "gpu_id": int(gpu_id),
        },
    )


def _build_policy(args: argparse.Namespace, gpu_id: int, checkpoint: Path) -> FastWAMDexJocoPolicy:
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
    return FastWAMDexJocoPolicy(
        model_config=args.run_dir,
        checkpoint=checkpoint,
        dataset_stats=args.dataset_stats,
        text_embedding=eval_entry._optional_path(args.text_embedding),
        text_embedding_base=eval_entry._optional_path(args.text_embedding_base),
        text_embedding_failure=eval_entry._optional_path(args.text_embedding_failure),
        device=f"cuda:{gpu_id}",
        action_horizon=int(args.action_horizon),
        replan_steps=int(args.replan_steps),
        value_replan_steps=args.value_replan_steps,
        num_inference_steps=int(args.num_inference_steps),
        task_name=args.task_name,
        prompt=args.success_prompt,
        load_text_encoder=args.load_text_encoder,
        inference_config=infer_cfg,
    )


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
    eval_entry._set_worker_render_env(gpu_id)

    args = argparse.Namespace(**args_dict)
    output_dir = Path(args.output_dir)
    step_dir = eval_entry._step_output_dir(output_dir, step)
    shard_dir = _shard_dir(step_dir, gpu_id)
    eval_entry._install_file_logging(
        output_dir / "logs" / f"step_{step:06d}_gpu_{gpu_id}.log"
    )

    expected = {(seed, repeat) for seed in assigned_seeds for repeat in range(args.repeats)}
    summary_path = shard_dir / "collection_summary.json"
    info_path = shard_dir / "meta" / "info.json"
    overwrite = bool(args.overwrite)
    resume = (not overwrite) and (summary_path.is_file() or info_path.is_file())
    if resume and not summary_path.is_file():
        try:
            saved = int(json.loads(info_path.read_text(encoding="utf-8")).get("total_episodes") or 0)
        except Exception:
            saved = 0
        if saved > 0:
            raise RuntimeError(
                f"Shard has {saved} episodes but no collection_summary.json: {shard_dir}"
            )
        import shutil

        shutil.rmtree(shard_dir)
        resume = False

    info, n_ep, global_i, stats_list, attempts = prepare_dataset(
        Path(args.source_dataset),
        shard_dir,
        args.success_prompt,
        FAILURE_PHRASE,
        overwrite=overwrite,
        resume=resume,
        save_all_trajectories=True,
        outcome_task_mode="clean",
        base_seed=int(args.seed_start),
    )
    completed = _done_pairs(attempts)
    remaining_seeds = [
        seed
        for seed in assigned_seeds
        if not all((seed, repeat) in completed for repeat in range(args.repeats))
    ]
    if expected and not remaining_seeds:
        print(f"[step {step} gpu {gpu_id}] shard already complete", flush=True)
        _write_shard_summary(
            shard_dir=shard_dir,
            args=args,
            gpu_id=gpu_id,
            seeds=assigned_seeds,
            attempts=attempts,
            n_ep=n_ep,
            global_i=global_i,
            status="complete",
            step=step,
        )
        return

    first_seed = remaining_seeds[0]
    print(
        f"[step {step} gpu {gpu_id}] creating env seed={first_seed} "
        f"(MUJOCO_GL={os.environ.get('MUJOCO_GL')} "
        f"MUJOCO_EGL_DEVICE_ID={os.environ.get('MUJOCO_EGL_DEVICE_ID')})",
        flush=True,
    )
    env = eval_entry._create_task_env(args.task_name, first_seed)
    print(f"[step {step} gpu {gpu_id}] env ready", flush=True)

    torch.cuda.set_device(gpu_id)
    checkpoint = eval_entry._checkpoint_path(Path(args.checkpoint_dir), step)
    load_start = time.perf_counter()
    policy = _build_policy(args, gpu_id, checkpoint)
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
                env = eval_entry._create_task_env(args.task_name, seed)
                current_seed = seed
            for repeat in range(args.repeats):
                if (seed, repeat) in completed:
                    continue
                obs, _ = env.reset()
                row, _preview = rollout_episode(
                    env=env,
                    initial_obs=obs,
                    policy=policy,
                    seed=seed,
                    repeat=repeat,
                    max_steps=args.max_steps,
                    capture_frames=False,
                    record_trajectory=True,
                )
                length = save_lerobot_episode(
                    shard_dir,
                    info,
                    stats_list,
                    episode_index=n_ep,
                    global_start_index=global_i,
                    episode=row,
                    task_text=args.success_prompt,
                    task_index=0,
                    fps=int(args.video_fps),
                )
                append_jsonl(
                    shard_dir / "meta" / OUTCOME_LEDGER_NAME,
                    make_outcome_row(
                        episode_index=n_ep,
                        success=bool(row["success"]),
                        attempt_index=len(attempts),
                        seed=int(seed),
                    ),
                )
                attempts.append(
                    {
                        "attempt_index": len(attempts),
                        "seed": int(seed),
                        "repeat": int(repeat),
                        "success": bool(row["success"]),
                        "done": True,
                        "steps": int(length),
                        "elapsed_s": float(row["wall_seconds"]),
                        "inference_seconds": float(row.get("inference_seconds", 0.0)),
                        "saved_failure_index": None,
                        "saved_episode_index": int(n_ep),
                    }
                )
                global_i += length
                n_ep += 1
                completed.add((seed, repeat))
                update_info(
                    shard_dir, info, num_episodes=n_ep, total_frames=global_i, total_tasks=1
                )
                write_json(
                    shard_dir / "meta" / "stats.json",
                    serialize_dict(aggregate_stats(stats_list)),
                )
                _write_shard_summary(
                    shard_dir=shard_dir,
                    args=args,
                    gpu_id=gpu_id,
                    seeds=assigned_seeds,
                    attempts=attempts,
                    n_ep=n_ep,
                    global_i=global_i,
                    status="in_progress",
                    step=step,
                )
                print(
                    f"[step {step} gpu {gpu_id}] seed={seed} repeat={repeat} "
                    f"result={row['result']} steps={length} "
                    f"wall={row['wall_seconds']:.1f}s ep={n_ep - 1}",
                    flush=True,
                )
                del row
    finally:
        env.close()

    if len(completed) != len(expected):
        raise RuntimeError(
            f"gpu {gpu_id} incomplete: {len(completed)}/{len(expected)} episodes"
        )
    _write_shard_summary(
        shard_dir=shard_dir,
        args=args,
        gpu_id=gpu_id,
        seeds=assigned_seeds,
        attempts=attempts,
        n_ep=n_ep,
        global_i=global_i,
        status="complete",
        step=step,
    )
    print(f"[step {step} gpu {gpu_id}] DONE episodes={n_ep} frames={global_i}", flush=True)


def _attempt_rows_from_shards(step_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted((step_dir / "shards").glob("gpu_*/collection_summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for item in payload.get("attempt_log", []):
            rows.append(
                {
                    "seed": int(item["seed"]),
                    "repeat": int(item["repeat"]),
                    "success": bool(item["success"]),
                    "result": "success" if item["success"] else "failure",
                    "episode_steps": int(item.get("steps", 0)),
                    "wall_seconds": float(item.get("elapsed_s", 0.0)),
                    "inference_seconds": float(
                        item.get("inference_seconds", item.get("elapsed_s", 0.0))
                    ),
                }
            )
    unique = {(int(row["seed"]), int(row["repeat"])): row for row in rows}
    return sorted(unique.values(), key=lambda row: (int(row["seed"]), int(row["repeat"])))


def _finalize_step(
    output_dir: Path,
    step: int,
    expected_episodes: int,
    failure_phrase: str,
) -> dict[str, Any]:
    step_dir = eval_entry._step_output_dir(output_dir, step)
    rows = _attempt_rows_from_shards(step_dir)
    if len(rows) != expected_episodes:
        raise RuntimeError(
            f"step {step}: expected {expected_episodes} episodes, found {len(rows)}"
        )
    shard_dirs = sorted(
        path.parent
        for path in (step_dir / "shards").glob("gpu_*/collection_summary.json")
    )
    raw_out = step_dir / "rollout_raw"
    merge_shards(shard_dirs, raw_out, overwrite=True, failure_phrase=failure_phrase)
    report = validate_outcome_dataset(
        raw_out,
        failure_phrase=failure_phrase,
        expected_episodes=expected_episodes,
        check_media=False,
    )
    eval_entry._write_json(step_dir / "rollout_outcome_validation.json", report)
    summary = eval_entry._summarize_episodes(
        rows, step=step, video_counts={"success": 0, "failure": 0}
    )
    summary["rollout_raw"] = str(raw_out)
    summary["saved_success_videos"] = 0
    summary["saved_failure_videos"] = 0
    eval_entry._write_json(step_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-steps", type=eval_entry._parse_int_list, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--text-embedding", type=Path, default=None)
    parser.add_argument("--text-embedding-base", type=Path, default=None)
    parser.add_argument("--text-embedding-failure", type=Path, default=None)
    parser.add_argument("--source-dataset", type=Path, default=None)
    parser.add_argument("--success-prompt", type=str, default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpus", type=eval_entry._parse_int_list, default=[1])
    parser.add_argument("--seed-start", type=int, default=10086)
    parser.add_argument("--seed-end", type=int, default=10135)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--value-replan-steps", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--text-cfg-scale", type=float, default=0.0)
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
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    spec = get_task(args.task_name)
    args.success_prompt = args.success_prompt or spec.success_prompt
    args.source_dataset = args.source_dataset or (REPO_ROOT / spec.expert_rel)
    args.output_dir = args.output_dir or (
        REPO_ROOT
        / "collect_results"
        / "dexjoco"
        / args.task_name
        / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    )
    return args


def _validate_args(args: argparse.Namespace) -> None:
    eval_entry._validate_args(args)
    if not Path(args.source_dataset).exists():
        raise FileNotFoundError(f"Missing source dataset: {args.source_dataset}")
    if not str(args.success_prompt).strip():
        raise ValueError("--success-prompt must not be empty")


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    args = _resolve_args(build_parser().parse_args())
    _validate_args(args)
    args.output_dir = Path(args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_entry._install_file_logging(args.output_dir / "logs" / "orchestrator.log")

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
            "inference_stack": "eval_dexjoco.FastWAMDexJocoPolicy",
        }
    )
    eval_entry._write_json(args.output_dir / "collect_config.json", config_payload)

    args_dict = config_payload.copy()
    summaries: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    print(f"Collection output: {args.output_dir}", flush=True)
    print(f"  logs:    {args.output_dir / 'logs'}", flush=True)
    print(
        f"  shards:  {args.output_dir / 'step_<ckpt>' / 'shards'}/gpu_*",
        flush=True,
    )
    print(
        f"  merged:  {args.output_dir / 'step_<ckpt>' / 'rollout_raw'}",
        flush=True,
    )

    for step in args.checkpoint_steps:
        step_dir = eval_entry._step_output_dir(args.output_dir, step)
        completed_summary_path = step_dir / "summary.json"
        raw_out = step_dir / "rollout_raw"
        if completed_summary_path.exists() and (raw_out / "meta" / "info.json").is_file():
            completed_summary = json.loads(completed_summary_path.read_text(encoding="utf-8"))
            if int(completed_summary.get("episodes", -1)) == expected_episodes:
                print(f"[step {step}] already complete; skipping", flush=True)
                print(eval_entry._format_step_line(completed_summary), flush=True)
                summaries.append(completed_summary)
                eval_entry._write_overall_summary(args.output_dir, summaries, config_payload)
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
                name=f"dexjoco-collect-step{step}-gpu{gpu_id}",
            )
            process.start()
            processes.append(process)

        failures = []
        for process in processes:
            process.join()
            if process.exitcode != 0:
                failures.append((process.name, eval_entry._format_exitcode(process.exitcode)))
        if failures:
            raise RuntimeError(
                f"Worker failures for step {step}: {failures}. "
                f"See {args.output_dir / 'logs'}"
            )

        summary = _finalize_step(
            args.output_dir, step, expected_episodes, FAILURE_PHRASE
        )
        summary["checkpoint_wall_seconds"] = time.perf_counter() - checkpoint_start
        eval_entry._write_json(step_dir / "summary.json", summary)
        summaries.append(summary)
        eval_entry._write_overall_summary(args.output_dir, summaries, config_payload)
        print(eval_entry._format_step_line(summary), flush=True)

    print(f"Complete: {args.output_dir / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
