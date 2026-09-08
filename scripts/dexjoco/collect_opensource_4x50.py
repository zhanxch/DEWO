#!/usr/bin/env python3
"""Opensource-aligned DexJoCo 4×50 rollout collection → LeRobot shards.

Uses FastWAM-infer-in-DexJoco FastWAMDexJocoPolicy (224 / z-score / replan=24),
not the local async server / s0_bundle path.

The CUDA policy runs in a persistent FastWAM server process, while each
episode runs in a fresh torch-free DexJoCo/EGL client process.  Keeping the
policy and MuJoCo renderer in separate processes avoids the native CUDA/EGL
interaction that can abort in ``mjr_readPixels``.  The client process is also
cheap to restart after a native abort, without reloading the model.

Prefer the shell wrapper:
  TASK=fold_glasses GPUS=4,5,6,7 bash scripts/dewo_v2/collect_opensource_4x50.sh

Protocol: seeds [seed_start, seed_end] × repeats (default 10086..10135 × 4 = 200).
Shards seeds across GPUs; each GPU writes its own LeRobot shard; parent merges.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import shutil
import socket
import sys
import time
from collections import deque
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OPEN = Path(
    os.environ.get(
        "OPEN_REPO",
        os.environ.get("FASTWAM_OPEN_REPO", str(ROOT.parent / "FastWAM-infer-in-DexJoco")),
    )
)
FASTWAM_PIN = Path(os.environ.get("FASTWAM_PIN", str(ROOT / "third_party/FastWAM_pin_45d8e14")))
DEXJOCO = ROOT / "third_party" / "dexjoco" / "dexjoco"
EXPECTED_PIN = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"

DEFAULT_CFG = ROOT / "configs/eval/dexjoco/mixed_5task_fastwam_joint/config.yaml"


def _setup_paths() -> None:
    paths = [
        str(ROOT / "scripts"),
        str(ROOT / "src"),
        str(DEXJOCO),
    ]
    open_src = OPEN / "src"
    if open_src.is_dir():
        paths.append(str(open_src))
    pin_src = FASTWAM_PIN / "src"
    if pin_src.is_dir():
        paths.append(str(pin_src))
    for p in reversed(paths):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def _parse_gpus(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", type=str, required=True)
    p.add_argument("--seed-start", type=int, default=10086)
    p.add_argument("--seed-end", type=int, default=10135)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--action-horizon", type=int, default=32)
    p.add_argument("--replan-steps", type=int, default=24)
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model-config", type=Path, default=None)
    p.add_argument("--dataset-stats", type=Path, required=True)
    p.add_argument("--text-embedding", type=Path, default=None)
    p.add_argument("--source-dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--task-name", required=True)
    p.add_argument("--success-prompt", required=True)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--skip-pin-check", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--max-episode-retries",
        type=int,
        default=5,
        help="Process-level retries after a native crash or failed episode commit.",
    )
    p.add_argument(
        "--server-base-port",
        type=int,
        default=43000,
        help="Base localhost port for per-GPU FastWAM policy servers.",
    )
    return p.parse_args()


def assert_pin() -> None:
    import subprocess

    head = subprocess.check_output(
        ["git", "-C", str(FASTWAM_PIN), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != EXPECTED_PIN:
        raise SystemExit(f"FastWAM pin mismatch: {head}")


def _safe_rgb(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 2:
        x = np.stack([x] * 3, axis=-1)
    if x.shape[-1] == 1:
        x = np.concatenate([x] * 3, axis=-1)
    return np.ascontiguousarray(x[..., :3])


def rollout_episode(
    env: Any,
    policy: Any,
    *,
    seed: int,
    repeat: int,
    max_steps: int,
    log: Any | None = None,
) -> dict[str, Any]:
    from fastwam_dexjoco.policy import fastwam_action_to_dexjoco

    obs, _ = env.reset()
    pending: deque[np.ndarray] = deque()
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    fronts: list[np.ndarray] = []
    wrists: list[np.ndarray] = []
    replan_index = 0
    success = False
    t0 = time.perf_counter()

    for _step in range(max_steps):
        if not pending:
            if log is not None:
                log(f"infer seed={seed} repeat={repeat} step={_step} replan={replan_index}")
            noise_seed = seed * 100_000 + repeat * 1_000 + replan_index
            chunk = policy.infer(obs, noise_seed=noise_seed)
            pending.extend(np.asarray(a, dtype=np.float32) for a in chunk[: policy.replan_steps])
            replan_index += 1

        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if state.shape[0] > 23:
            state = state[:23]
        action = pending.popleft()
        actions.append(np.asarray(action, dtype=np.float32))
        states.append(state.astype(np.float32))
        fronts.append(_safe_rgb(obs["front"]))
        wrists.append(_safe_rgb(obs["wrist"]))

        obs, _, terminated, truncated, info = env.step(fastwam_action_to_dexjoco(action))
        success = bool(info.get("succeed", False))
        if terminated or truncated:
            break

    return {
        "actions": np.stack(actions, axis=0),
        "states": np.stack(states, axis=0),
        "frames": {
            "observation.images.front": fronts,
            "observation.images.wrist": wrists,
        },
        "success": bool(success),
        "seed": int(seed),
        "repeat": int(repeat),
        "steps": len(actions),
        "elapsed_s": float(time.perf_counter() - t0),
    }


class _Tee:
    def __init__(self, *files: Any) -> None:
        self.files = files

    def write(self, data: Any) -> None:
        if isinstance(data, bytes):
            data = data.decode("utf-8", "replace")
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self) -> None:
        for f in self.files:
            f.flush()


def _set_worker_environment(gpu_id: int) -> None:
    """Pin both CUDA and MuJoCo EGL to one physical GPU before imports."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    os.environ["PYOPENGL_PLATFORM"] = "egl"


def _open_worker_log(gpu_id: int, output_dir: Path) -> Any:
    log_path = output_dir / "logs" / f"gpu_{gpu_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    import faulthandler

    faulthandler.enable(file=log_file, all_threads=True)
    return log_file


def _summary_done_pairs(shard_dir: Path) -> tuple[dict[str, Any], set[tuple[int, int]]]:
    summary_path = shard_dir / "collection_summary.json"
    if not summary_path.is_file():
        return {}, set()
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {summary_path}: {exc}") from exc
    attempts = summary.get("attempt_log", [])
    if not isinstance(attempts, list):
        raise RuntimeError(f"Invalid attempt_log in {summary_path}")
    done = {
        (int(row["seed"]), int(row.get("repeat", -1)))
        for row in attempts
        if isinstance(row, dict)
        and row.get("saved_episode_index") is not None
        and row.get("success") is not None
    }
    return summary, done


def _write_progress_summary(
    *,
    shard_dir: Path,
    args: argparse.Namespace,
    gpu_id: int,
    seeds: list[int],
    attempts: list[dict[str, Any]],
    n_ep: int,
    global_i: int,
    status: str,
) -> None:
    from collect_dexjoco_rollouts import write_json

    write_json(
        shard_dir / "collection_summary.json",
        {
            "status": status,
            "mode": "save_all",
            "outcome_task_mode": "clean",
            "target_episodes": len(seeds) * int(args.repeats),
            "attempts": len(attempts),
            "episodes": n_ep,
            "frames": global_i,
            "failures": sum(1 for item in attempts if not item["success"]),
            "successes_saved": sum(1 for item in attempts if item["success"]),
            "attempt_log": attempts,
            "inference_stack": "opensource_FastWAMDexJocoPolicy",
            "base_seed": int(args.seed_start),
            "seed_end": int(args.seed_end),
            "repeats": int(args.repeats),
            "gpu_id": int(gpu_id),
        },
    )


def _episode_entry(
    gpu_id: int,
    seed: int,
    repeat: int,
    args_dict: dict[str, Any],
    seeds: list[int],
    force_overwrite: bool,
) -> None:
    """Spawn entry for exactly one rollout; native aborts die with this process."""
    _set_worker_environment(gpu_id)
    _episode_main(gpu_id, seed, repeat, args_dict, seeds, force_overwrite)


def _episode_main(
    gpu_id: int,
    seed: int,
    repeat: int,
    args_dict: dict[str, Any],
    seeds: list[int],
    force_overwrite: bool,
) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))
    _setup_paths()

    import torch
    from dexjoco.tasks import CONFIG_MAPPING
    from fastwam_dexjoco.policy import FastWAMDexJocoPolicy
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

    args = argparse.Namespace(**args_dict)
    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "shards" / f"gpu_{gpu_id}"
    log_file = _open_worker_log(gpu_id, output_dir)

    def log(msg: str) -> None:
        print(f"[collect gpu{gpu_id} {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    summary_path = shard_dir / "collection_summary.json"
    info_path = shard_dir / "meta" / "info.json"
    if force_overwrite:
        overwrite = True
        resume = False
    elif summary_path.is_file() or info_path.is_file():
        overwrite = False
        resume = True
    else:
        overwrite = False
        resume = False

    # A child that aborts before its first commit can leave only an empty
    # initialized directory. It is safe to recreate that directory; a shard
    # with committed episodes must have a summary and is never deleted here.
    if resume and not summary_path.is_file():
        try:
            saved = int(json.loads(info_path.read_text()).get("total_episodes") or 0)
        except Exception:
            saved = 0
        if saved > 0:
            raise RuntimeError(
                f"Shard has {saved} episodes but no collection_summary.json: {shard_dir}"
            )
        shutil.rmtree(shard_dir)
        resume = False

    info, n_ep, global_i, stats_list, attempts = prepare_dataset(
        Path(args.source_dataset),
        shard_dir,
        args.success_prompt,
        "Failed to finish the whole process.",
        overwrite=overwrite,
        resume=resume,
        save_all_trajectories=True,
        outcome_task_mode="clean",
    )
    log(f"episode process ready seed={seed} repeat={repeat} resume={resume} episodes={n_ep}")

    torch.cuda.set_device(0)
    device = "cuda:0"
    log(
        "cuda map "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER')} "
        f"MUJOCO_EGL_DEVICE_ID={os.environ.get('MUJOCO_EGL_DEVICE_ID')} "
        f"name={torch.cuda.get_device_name(0)}"
    )
    t_load = time.perf_counter()
    policy = FastWAMDexJocoPolicy(
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        text_embedding=args.text_embedding,
        device=device,
        action_horizon=args.action_horizon,
        replan_steps=args.replan_steps,
        num_inference_steps=args.num_inference_steps,
        prompt=args.success_prompt,
        task_name=args.task_name,
    )
    log(f"policy ready in {time.perf_counter() - t_load:.1f}s")

    env = None
    try:
        env = CONFIG_MAPPING[args.task_name]().get_environment(
            policy_mode=True,
            render_mode="rgb_array",
            randomize=False,
            seed=int(seed),
            randomize_dynamics=False,
        )
        log(f"start seed={seed} repeat={repeat}")
        ep = rollout_episode(
            env,
            policy,
            seed=int(seed),
            repeat=int(repeat),
            max_steps=int(args.max_steps),
            log=log,
        )
    finally:
        if env is not None:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    attempt_idx = len(attempts)
    length = save_lerobot_episode(
        shard_dir,
        info,
        stats_list,
        episode_index=n_ep,
        global_start_index=global_i,
        episode=ep,
        task_text=args.success_prompt,
        task_index=0,
        fps=int(args.fps),
    )
    append_jsonl(
        shard_dir / "meta" / OUTCOME_LEDGER_NAME,
        make_outcome_row(
            episode_index=n_ep,
            success=bool(ep["success"]),
            attempt_index=attempt_idx,
            seed=int(seed),
        ),
    )
    attempts.append(
        {
            "attempt_index": attempt_idx,
            "seed": int(seed),
            "repeat": int(repeat),
            "success": bool(ep["success"]),
            "done": True,
            "steps": int(length),
            "elapsed_s": float(ep["elapsed_s"]),
            "saved_failure_index": None,
            "saved_episode_index": int(n_ep),
        }
    )
    global_i += length
    n_ep += 1
    update_info(shard_dir, info, num_episodes=n_ep, total_frames=global_i, total_tasks=1)
    write_json(shard_dir / "meta" / "stats.json", serialize_dict(aggregate_stats(stats_list)))
    _write_progress_summary(
        shard_dir=shard_dir,
        args=args,
        gpu_id=gpu_id,
        seeds=seeds,
        attempts=attempts,
        n_ep=n_ep,
        global_i=global_i,
        status="in_progress",
    )
    log(f"saved seed={seed} repeat={repeat} success={ep['success']} steps={length} ep={n_ep - 1}")


def _worker_entry(gpu_id: int, seeds: list[int], args_dict: dict[str, Any]) -> None:
    """Top-level GPU worker; each episode runs in a disposable child process."""
    _set_worker_environment(gpu_id)
    worker_main(gpu_id, seeds, args_dict)


def worker_main(gpu_id: int, seeds: list[int], args_dict: dict[str, Any]) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))
    _setup_paths()

    from collect_dexjoco_rollouts import write_json

    args = argparse.Namespace(**args_dict)
    output_dir = Path(args.output_dir)
    shard_dir = output_dir / "shards" / f"gpu_{gpu_id}"
    log_file = _open_worker_log(gpu_id, output_dir)

    def log(msg: str) -> None:
        print(f"[collect gpu{gpu_id} {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    if not seeds:
        log("no seeds assigned; exit")
        return
    target = len(seeds) * int(args.repeats)
    max_retries = max(1, int(args.max_episode_retries))
    ctx = mp.get_context("spawn")
    first_launch = True

    for seed in seeds:
        for repeat in range(int(args.repeats)):
            summary, done_pairs = _summary_done_pairs(shard_dir)
            if (seed, repeat) in done_pairs:
                log(f"skip seed={seed} repeat={repeat} (already saved)")
                continue
            if str(summary.get("status")) == "complete" and int(summary.get("episodes", 0)) >= target:
                log(f"shard already complete episodes={summary.get('episodes')} target={target}; skip")
                return

            committed = False
            for retry in range(1, max_retries + 1):
                log(f"launch seed={seed} repeat={repeat} process_attempt={retry}/{max_retries}")
                proc = ctx.Process(
                    target=_episode_entry,
                    args=(gpu_id, int(seed), int(repeat), args_dict, seeds, first_launch and bool(args.overwrite)),
                    name=f"collect-gpu{gpu_id}-seed{seed}-repeat{repeat}-try{retry}",
                )
                proc.start()
                first_launch = False
                proc.join()
                exitcode = proc.exitcode
                proc.close()
                summary, done_pairs = _summary_done_pairs(shard_dir)
                # The committed summary is authoritative.  A process can be
                # terminated after writing the episode but before returning
                # cleanly; never rerun a pair that is already committed.
                if (seed, repeat) in done_pairs:
                    committed = True
                    if exitcode != 0:
                        log(
                            f"episode process exited {exitcode} after committing "
                            f"seed={seed} repeat={repeat}; continuing"
                        )
                    break
                log(
                    f"episode process failed seed={seed} repeat={repeat} "
                    f"exitcode={exitcode}; retrying" if retry < max_retries else
                    f"episode process failed seed={seed} repeat={repeat} exitcode={exitcode}"
                )
            if not committed:
                log(f"giving up seed={seed} repeat={repeat} after {max_retries} attempts")
                raise RuntimeError(f"gpu{gpu_id} could not commit seed={seed} repeat={repeat}")

    summary, done_pairs = _summary_done_pairs(shard_dir)
    if len(done_pairs) != target:
        log(f"incomplete shard committed_pairs={len(done_pairs)} target={target}")
        raise RuntimeError(f"gpu{gpu_id} shard incomplete: {len(done_pairs)}/{target}")
    summary["status"] = "complete"
    write_json(shard_dir / "collection_summary.json", summary)
    log(f"DONE episodes={summary.get('episodes', target)} frames={summary.get('frames', 0)}")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(ROOT / "checkpoints"))
    _setup_paths()
    pin_exists = FASTWAM_PIN.is_dir()
    if not args.skip_pin_check and pin_exists:
        assert_pin()
    if args.model_config is None:
        local_joint = ROOT / "configs/eval/dexjoco/mixed_5task_fastwam_joint/config.yaml"
        args.model_config = local_joint if local_joint.is_file() else DEFAULT_CFG

    gpus = _parse_gpus(args.gpus)
    seeds = list(range(int(args.seed_start), int(args.seed_end) + 1))
    if not seeds:
        raise SystemExit("empty seed range")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(
            {
                **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "n_seeds": len(seeds),
                "n_episodes": len(seeds) * int(args.repeats),
                "stack": "opensource_FastWAMDexJocoPolicy",
            },
            indent=2,
        )
        + "\n"
    )

    # Shard seeds round-robin across GPUs.
    assignments: dict[int, list[int]] = {g: [] for g in gpus}
    for i, seed in enumerate(seeds):
        assignments[gpus[i % len(gpus)]].append(seed)

    print(f"[collect-orch] gpus={gpus} seeds={seeds[0]}..{seeds[-1]} ×{args.repeats}", flush=True)
    for g, ss in assignments.items():
        print(f"  gpu{g}: {len(ss)} seeds -> {len(ss) * args.repeats} eps", flush=True)

    args_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    # spawn: safe with CUDA (fork + cuda init in parent is unsafe).
    ctx = mp.get_context("spawn")
    procs: list[mp.Process] = []
    for g in gpus:
        seed_list = list(assignments[g])
        proc = ctx.Process(
            target=_worker_entry,
            args=(g, seed_list, args_dict),
            name=f"collect-gpu{g}",
        )
        proc.start()
        procs.append(proc)
        print(f"[collect-orch] launched gpu{g} pid={proc.pid}", flush=True)

    rc = 0
    for proc in procs:
        proc.join()
        if proc.exitcode not in (0, None):
            print(f"[collect-orch] {proc.name} FAILED rc={proc.exitcode}", flush=True)
            rc = proc.exitcode or 1

    if rc != 0:
        return int(rc)

    # Merge shards via CLI (avoid brittle symbol imports).
    shard_dirs = [
        out / "shards" / f"gpu_{g}"
        for g in gpus
        if (out / "shards" / f"gpu_{g}" / "collection_summary.json").exists()
    ]
    raw_out = out / "rollout_raw_200"
    print(f"[collect-orch] merging {len(shard_dirs)} shards -> {raw_out}", flush=True)
    import subprocess

    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_rollout_datasets.py"),
            "merge-shards",
            "--shard-datasets",
            *[str(p) for p in shard_dirs],
            "--output-dataset",
            str(raw_out),
            "--overwrite",
        ],
        cwd=str(ROOT),
    )
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_rollout_datasets.py"),
            "validate-outcomes",
            "--dataset",
            str(raw_out),
            "--expected-episodes",
            str(len(seeds) * int(args.repeats)),
            "--report",
            str(out / "rollout_outcome_validation.json"),
        ],
        cwd=str(ROOT),
    )

    # Summary rates across attempt_logs.
    pooled_s = sum(
        int(json.loads((out / "shards" / f"gpu_{g}" / "collection_summary.json").read_text())["successes_saved"])
        for g in gpus
    )
    pooled_n = len(seeds) * int(args.repeats)
    agg = {
        "protocol": f"collect_4x50_seeds_{args.seed_start}_{args.seed_end}",
        "inference_stack": "opensource_FastWAMDexJocoPolicy",
        "pooled_successes": pooled_s,
        "pooled_episodes": pooled_n,
        "pooled_success_rate": pooled_s / pooled_n if pooled_n else None,
        "rollout_raw": str(raw_out),
    }
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
    print(json.dumps(agg, indent=2), flush=True)
    print(f"[collect-orch] DONE raw={raw_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
