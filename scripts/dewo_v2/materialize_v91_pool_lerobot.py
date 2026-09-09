#!/usr/bin/env python3
"""Materialize DEWO v9.1 pool LeRobot: D_scan / D_fail crops + D+ τ[t, t+33).

No stitch. Original-fail 33-frame windows at every scanned prefix. CFG events
get a separate 33-frame crop from the first successful RGB continuation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.collect_dexjoco_rollouts import (  # noqa: E402
    aggregate_stats,
    append_jsonl,
    prepare_dataset,
    read_json,
    serialize_dict,
    update_info,
    write_json,
)
from materialize_v9_full_pair_lerobot import (  # noqa: E402
    FAILURE_PHRASE,
    load_episode_arrays,
    load_rgb_frames,
    save_variable_episode,
    video_path,
)
from v91_pool import MIN_EVENT_FRAMES, crop_span  # noqa: E402


def _slice_fail(
    *,
    actions: np.ndarray,
    states: np.ndarray,
    front: list[np.ndarray],
    wrist: list[np.ndarray],
    lo: int,
    hi: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    return actions[lo:hi], states[lo:hi], front[lo:hi], wrist[lo:hi]


def _tau_crop(npz_path: Path, front_path: Path, wrist_path: Path) -> (
    tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]] | None
):
    cont = np.load(npz_path)
    actions = np.asarray(cont["actions"])
    states = np.asarray(cont["states"])
    if actions.shape[0] < MIN_EVENT_FRAMES or states.shape[0] < MIN_EVENT_FRAMES:
        return None
    front = load_rgb_frames(front_path)
    wrist = load_rgb_frames(wrist_path)
    n = min(len(front), len(wrist), int(actions.shape[0]), int(states.shape[0]))
    if n < MIN_EVENT_FRAMES:
        return None
    return (
        actions[:MIN_EVENT_FRAMES],
        states[:MIN_EVENT_FRAMES],
        front[:MIN_EVENT_FRAMES],
        wrist[:MIN_EVENT_FRAMES],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic-index", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--success-prompt", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    success_prompt = str(args.success_prompt)

    critic = read_json(args.critic_index.expanduser().resolve())
    pool = critic.get("v91_pool") or {}
    scan_windows = list(pool.get("scan_windows") or [])
    dplus = list(pool.get("dplus") or [])
    if not scan_windows:
        raise SystemExit(f"No v91_pool.scan_windows in {args.critic_index}")

    source = args.source_dataset.expanduser().resolve()
    source_info = read_json(source / "meta" / "info.json")
    output = args.output_dataset.expanduser().resolve()
    info, n_ep, global_i, stats_list, _attempts = prepare_dataset(
        source,
        output,
        success_prompt,
        f"{success_prompt} {FAILURE_PHRASE}",
        overwrite=bool(args.overwrite),
        resume=False,
        save_all_trajectories=True,
        outcome_task_mode="clean",
    )
    fps = int(info.get("fps", 30))
    fail_cache: dict[int, tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]] = {}
    units: list[dict[str, Any]] = []

    def cached_fail(ep: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
        if ep not in fail_cache:
            actions, states = load_episode_arrays(source, source_info, ep)
            front = load_rgb_frames(
                video_path(source, source_info, ep, "observation.images.front")
            )
            wrist = load_rgb_frames(
                video_path(source, source_info, ep, "observation.images.wrist")
            )
            n_vid = min(len(front), len(wrist), int(actions.shape[0]))
            fail_cache[ep] = (
                actions[:n_vid],
                states[:n_vid],
                front[:n_vid],
                wrist[:n_vid],
            )
        return fail_cache[ep]

    def write_clip(
        *,
        actions: np.ndarray,
        states: np.ndarray,
        front: list[np.ndarray],
        wrist: list[np.ndarray],
        prompt: str,
    ) -> tuple[int, int]:
        nonlocal n_ep, global_i
        ep_idx = n_ep
        n_written = save_variable_episode(
            output_dataset=output,
            info=info,
            stats_list=stats_list,
            episode_index=ep_idx,
            global_start_index=global_i,
            actions=actions,
            states=states,
            front=front,
            wrist=wrist,
            task_text=prompt,
            fps=fps,
        )
        n_ep += 1
        global_i += n_written
        return ep_idx, n_written

    for row in scan_windows:
        ep = int(row["source_failure_episode_index"])
        lo, hi = (int(row["fail_span"][0]), int(row["fail_span"][1]))
        actions, states, front, wrist = cached_fail(ep)
        if hi > int(actions.shape[0]):
            span = crop_span(int(row["prefix_frame"]), int(actions.shape[0]))
            if span is None:
                continue
            lo, hi = span
        fa, fs, ff, fw = _slice_fail(
            actions=actions, states=states, front=front, wrist=wrist, lo=lo, hi=hi
        )
        if fa.shape[0] < MIN_EVENT_FRAMES:
            continue
        kind = "d_fail" if bool(row.get("is_cliff")) else "d_scan"
        prompt = success_prompt
        ep_idx, n_written = write_clip(
            actions=fa, states=fs, front=ff, wrist=fw, prompt=prompt
        )
        append_jsonl(
            output / "meta" / "episode_outcomes.jsonl",
            {
                "episode_index": ep_idx,
                "success": False,
                "outcome": "failure",
                "event_role": kind,
                "seed": row.get("seed"),
                "source_failure_episode_index": ep,
                "prefix_frame": int(row["prefix_frame"]),
            },
        )
        units.append(
            {
                "kind": kind,
                "episode_index": ep_idx,
                "length": n_written,
                "source_failure_episode_index": ep,
                "prefix_frame": int(row["prefix_frame"]),
                "success_count": int(row["success_count"]),
                "pass_m": int(row["pass_m"]),
                "value_target": float(row["value_target"]),
                "is_cliff": bool(row.get("is_cliff")),
                "seed": row.get("seed"),
            }
        )

    for row in dplus:
        cropped = _tau_crop(Path(row["npz"]), Path(row["front"]), Path(row["wrist"]))
        if cropped is None:
            continue
        ta, ts, tf, tw = cropped
        ep_idx, n_written = write_clip(
            actions=ta, states=ts, front=tf, wrist=tw, prompt=success_prompt
        )
        append_jsonl(
            output / "meta" / "episode_outcomes.jsonl",
            {
                "episode_index": ep_idx,
                "success": True,
                "outcome": "success",
                "event_role": "dplus",
                "seed": row.get("seed"),
                "source_failure_episode_index": int(row["source_failure_episode_index"]),
                "prefix_frame": int(row["prefix_frame"]),
                "replicate": int(row["replicate"]),
            },
        )
        units.append(
            {
                "kind": "dplus",
                "episode_index": ep_idx,
                "length": n_written,
                "source_failure_episode_index": int(row["source_failure_episode_index"]),
                "prefix_frame": int(row["prefix_frame"]),
                "success_count": int(row["success_count"]),
                "pass_m": int(row.get("pass_m") or 10),
                "replicate": int(row["replicate"]),
                "seed": row.get("seed"),
            }
        )

    write_json(output / "meta" / "stats.json", serialize_dict(aggregate_stats(stats_list)))
    update_info(
        output,
        info,
        num_episodes=n_ep,
        total_frames=global_i,
        total_tasks=1,
    )
    write_json(
        output / "pool_index.json",
        {
            "units": units,
            "counts": {
                "d_scan": sum(1 for u in units if u["kind"] == "d_scan"),
                "d_fail": sum(1 for u in units if u["kind"] == "d_fail"),
                "dplus": sum(1 for u in units if u["kind"] == "dplus"),
            },
            "horizon": "crop33",
            "source_window_rule": "v91_scan_and_tau_crop",
            "critic_index": str(args.critic_index.expanduser().resolve()),
        },
    )
    print(
        f"wrote {n_ep} v9.1 episodes to {output} "
        f"(scan={sum(1 for u in units if u['kind']=='d_scan')} "
        f"fail={sum(1 for u in units if u['kind']=='d_fail')} "
        f"dplus={sum(1 for u in units if u['kind']=='dplus')})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
