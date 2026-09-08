#!/usr/bin/env python3
"""Build a cleaned copy of grasp_anything_joint_absolute without touching the source.

Mismatch correction is NOT a global action<->state swap. Arms already agree, so
they are the pose/clock anchor. Each hand-joint mismatch run is classified from
onset jumps and whether the two streams rejoin at the action plateau or the
state plateau:

  - tracking lag (dominant): command leads, measured catches up later.
    True pose is state; rewrite action on that joint. Never copy action onto
    state (that would invent a close the camera did not see).
  - action glitch: command jumps away and snaps back to state. Rewrite action.
  - state glitch (rare): measured jumps away and returns to a smooth command
    while the arm still agrees. Rewrite state on that joint.

Still-frame rules and video re-encode are unchanged. See meta/clean_manifest.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 30.0
ACTION_DIM = 54
TAU_ACTION = 0.008  # rad L2; ~p5 of original action steps
TAU_STATE_INTERNAL = 0.016  # rad L2; ~p10 of original state steps
MIN_INTERNAL_RUN = 15  # frames at 30 Hz = 0.5 s
START_RUN = 3  # consecutive moving steps to end prefix/suffix
MIN_EPISODE_FRAMES = 64
MISMATCH_THR = 0.10  # rad; ignore ordinary tracking noise
MISMATCH_MIN_RUN = 3
ARM_AGREE = 0.03  # arm maxabs below this => streams share the same clock/pose
JUMP_THR = 0.10  # discontinuous onset on one stream
SMOOTH_ONSET = 0.04
VIDEO_KEYS = (
    "observation.images.head_view",
    "observation.images.left_wrist_view",
    "observation.images.right_wrist_view",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(
            "/gaozt-test1/zhanxch/FITWAM-dewov9-20260828/data/grasp_anything_joint_absolute"
        ),
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(
            "/gaozt-test1/zhanxch/FITWAM-dewov9-20260828/data/grasp_anything_joint_absolute_clean"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stack_list_column(table: pa.Table, name: str) -> np.ndarray:
    return np.stack(table.column(name).to_pylist()).astype(np.float64)


def first_sustained_motion(delta: np.ndarray, tau: float, run: int) -> int:
    n = len(delta)
    if n < run:
        return 0
    for i in range(0, n - run + 1):
        if np.all(delta[i : i + run] >= tau):
            return i
    return n


def last_sustained_motion_end(delta: np.ndarray, tau: float, run: int) -> int:
    n = len(delta)
    if n < run:
        return n
    for i in range(n - run, -1, -1):
        if np.all(delta[i : i + run] >= tau):
            return i + run
    return 0


def runs_from_mask(mask: np.ndarray) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j, j - i))
            i = j
        else:
            i += 1
    return out


def keep_indices(action: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    t_len = len(action)
    if t_len < 2:
        return np.arange(t_len), {
            "prefix_dropped": 0,
            "suffix_dropped": 0,
            "internal_dropped": 0,
            "internal_runs": [],
        }

    d_action = np.linalg.norm(np.diff(action, axis=0), axis=1)
    d_state = np.linalg.norm(np.diff(state, axis=0), axis=1)

    prefix_steps = first_sustained_motion(d_action, TAU_ACTION, START_RUN)
    suffix_start_frame = last_sustained_motion_end(d_action, TAU_ACTION, START_RUN) + 1
    suffix_start_frame = max(suffix_start_frame, prefix_steps)
    suffix_start_frame = min(suffix_start_frame, t_len)

    drop = np.zeros(t_len, dtype=bool)
    drop[:prefix_steps] = True
    drop[suffix_start_frame:] = True

    still = np.zeros(t_len, dtype=bool)
    still[1:] = (d_action < TAU_ACTION) & (d_state < TAU_STATE_INTERNAL)
    still &= ~drop

    internal_runs = []
    for start, end, length in runs_from_mask(still):
        if length >= MIN_INTERNAL_RUN:
            drop[start:end] = True
            internal_runs.append({"start": int(start), "end": int(end), "length": int(length)})

    keep = np.flatnonzero(~drop)
    return keep, {
        "prefix_dropped": int(prefix_steps),
        "suffix_dropped": int(t_len - suffix_start_frame),
        "internal_dropped": int(sum(run["length"] for run in internal_runs)),
        "internal_runs": internal_runs,
    }


def _diagnose_mismatch(
    action: np.ndarray,
    state: np.ndarray,
    joint: int,
    start: int,
    end: int,
    arm_err: np.ndarray,
) -> tuple[str, str]:
    """Return (kind, trust) where trust is 'state' or 'action'."""
    t_len = len(action)
    arm_ok = float(np.median(arm_err[start:end])) < ARM_AGREE
    action_in = abs(action[start, joint] - action[start - 1, joint]) if start > 0 else 0.0
    state_in = abs(state[start, joint] - state[start - 1, joint]) if start > 0 else 0.0

    pre0 = max(0, start - 8)
    post1 = min(t_len, end + 8)
    action_mid = float(np.median(action[start:end, joint]))
    state_mid = float(np.median(state[start:end, joint]))
    action_post = (
        float(np.median(action[end:post1, joint])) if post1 > end else float(action[end - 1, joint])
    )
    state_post = (
        float(np.median(state[end:post1, joint])) if post1 > end else float(state[end - 1, joint])
    )

    rejoin_action = abs(state_post - action_mid) < abs(state_post - state_mid) and abs(
        action_post - action_mid
    ) < 0.08
    rejoin_state = abs(action_post - state_mid) < abs(action_post - action_mid) and abs(
        state_post - state_mid
    ) < 0.08
    state_jump = state_in > JUMP_THR and action_in < SMOOTH_ONSET
    action_jump = action_in > JUMP_THR and state_in < SMOOTH_ONSET

    if state_jump and rejoin_action and arm_ok:
        return "state_glitch", "action"
    if action_jump and rejoin_state and arm_ok:
        return "action_glitch", "state"
    if arm_ok:
        if rejoin_action:
            return "tracking_lag_state_catches_action", "state"
        if rejoin_state:
            return "action_drift_snaps_back", "state"
        return "arm_agrees_trust_state", "state"

    pre = 0.5 * (action[start - 1, joint] + state[start - 1, joint]) if start else action_mid
    post_i = min(end, t_len - 1)
    post = 0.5 * (action[post_i, joint] + state[post_i, joint])
    bridge = np.linspace(pre, post, end - start)
    action_dev = float(np.mean(np.abs(action[start:end, joint] - bridge)))
    state_dev = float(np.mean(np.abs(state[start:end, joint] - bridge)))
    if action_dev < state_dev:
        return "arm_mismatch_trust_action_bridge", "action"
    return "arm_mismatch_trust_state_bridge", "state"


def patch_mismatch(
    action: np.ndarray, state: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Rewrite only mismatched hand joints. Arms are left untouched."""
    patched_action = action.copy()
    patched_state = state.copy()
    arm_err = np.max(np.abs(action[:, :14] - state[:, :14]), axis=1)
    patches: list[dict[str, Any]] = []
    for joint in range(14, ACTION_DIM):
        mask = np.abs(action[:, joint] - state[:, joint]) > MISMATCH_THR
        for start, end, length in runs_from_mask(mask):
            if length < MISMATCH_MIN_RUN:
                continue
            kind, trust = _diagnose_mismatch(action, state, joint, start, end, arm_err)
            if trust == "action":
                patched_state[start:end, joint] = action[start:end, joint]
            else:
                patched_action[start:end, joint] = state[start:end, joint]
            patches.append(
                {
                    "joint": int(joint),
                    "start": int(start),
                    "end": int(end),
                    "length": int(length),
                    "kind": kind,
                    "trust": trust,
                    "maxabs": float(np.max(np.abs(action[start:end, joint] - state[start:end, joint]))),
                }
            )
    return patched_action, patched_state, patches


def list_float_column(values: np.ndarray) -> pa.Array:
    values = np.asarray(values, dtype=np.float32)
    return pa.array(values.tolist(), type=pa.list_(pa.float32()))


def write_episode_parquet(
    path: Path,
    *,
    action: np.ndarray,
    state: np.ndarray,
    episode_index: int,
    global_start: int,
    task_index: np.ndarray,
    annotation: np.ndarray,
) -> None:
    n = len(action)
    frame_index = np.arange(n, dtype=np.int64)
    timestamp = frame_index.astype(np.float32) / np.float32(FPS)
    table = pa.table(
        {
            "observation.state": list_float_column(state),
            "action": list_float_column(action),
            "timestamp": pa.array(timestamp, type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(np.full(n, episode_index, dtype=np.int64), type=pa.int64()),
            "index": pa.array(np.arange(global_start, global_start + n, dtype=np.int64), type=pa.int64()),
            "task_index": pa.array(np.asarray(task_index, dtype=np.int64), type=pa.int64()),
            "annotation.human.action.task_description": pa.array(
                np.asarray(annotation, dtype=np.int64), type=pa.int64()
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(), path)


def rewrite_video(src: Path, dst: Path, keep: np.ndarray, fps: float) -> int:
    keep_set = set(int(i) for i in keep)
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    fps_rate = int(round(float(fps)))
    if fps_rate <= 0:
        fps_rate = int(FPS)
    in_container = av.open(str(src), mode="r")
    try:
        in_stream = in_container.streams.video[0]
        out_container = av.open(str(dst), mode="w")
        try:
            out_stream = out_container.add_stream("libx264", rate=fps_rate)
            out_stream.width = int(in_stream.width)
            out_stream.height = int(in_stream.height)
            out_stream.pix_fmt = "yuv420p"
            out_stream.options = {"crf": "21", "preset": "veryfast"}
            for index, frame in enumerate(in_container.decode(in_stream)):
                if index not in keep_set:
                    continue
                rgb = frame.to_ndarray(format="rgb24")
                video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in out_stream.encode(video_frame):
                    out_container.mux(packet)
                written += 1
            for packet in out_stream.encode():
                out_container.mux(packet)
        finally:
            out_container.close()
    finally:
        in_container.close()
    if written != len(keep):
        raise RuntimeError(
            f"{src}: wrote {written} frames, expected {len(keep)} (video/parquet mismatch)"
        )
    return written


def feature_stats(array: np.ndarray) -> dict[str, list[float]]:
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return {
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "q01": np.quantile(array, 0.01, axis=0).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).tolist(),
    }


def clean(src: Path, dst: Path, overwrite: bool) -> None:
    src = src.resolve()
    dst = dst.resolve()
    if src == dst:
        raise ValueError("Refusing to write the cleaned dataset onto the source path")
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} exists (pass --overwrite)")
        shutil.rmtree(dst)

    info = load_json(src / "meta" / "info.json")
    episodes_meta = load_jsonl(src / "meta" / "episodes.jsonl")
    fps = float(info.get("fps", FPS))

    (dst / "meta").mkdir(parents=True)
    (dst / "data" / "chunk-000").mkdir(parents=True)
    for key in VIDEO_KEYS:
        (dst / "videos" / "chunk-000" / key).mkdir(parents=True)
    shutil.copy2(src / "meta" / "tasks.jsonl", dst / "meta" / "tasks.jsonl")
    shutil.copy2(src / "meta" / "modality.json", dst / "meta" / "modality.json")

    all_actions: list[np.ndarray] = []
    all_states: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    out_episodes: list[dict[str, Any]] = []
    episode_reports: list[dict[str, Any]] = []
    global_start = 0
    dropped_episodes = 0

    parquet_dir = src / "data" / "chunk-000"
    n_src = len(episodes_meta)
    for src_ep, ep_meta in enumerate(episodes_meta):
        parquet = parquet_dir / f"episode_{src_ep:06d}.parquet"
        table = pq.read_table(parquet)
        action = stack_list_column(table, "action")
        state = stack_list_column(table, "observation.state")
        if action.shape[1] != ACTION_DIM or state.shape != action.shape:
            raise ValueError(f"{parquet}: expected (*, {ACTION_DIM}), got {action.shape} {state.shape}")
        task_index = np.asarray(table.column("task_index").to_pylist(), dtype=np.int64)
        if "annotation.human.action.task_description" in table.column_names:
            annotation = np.asarray(
                table.column("annotation.human.action.task_description").to_pylist(),
                dtype=np.int64,
            )
        else:
            annotation = task_index.copy()

        patched_action, patched_state, patches = patch_mismatch(action, state)
        t_len = len(action)
        d_orig = np.linalg.norm(np.diff(action, axis=0), axis=1) if t_len > 1 else np.array([])
        prefix_steps = first_sustained_motion(d_orig, TAU_ACTION, START_RUN) if t_len > 1 else 0
        suffix_start = last_sustained_motion_end(d_orig, TAU_ACTION, START_RUN) + 1 if t_len > 1 else 0
        suffix_start = min(max(suffix_start, prefix_steps), t_len)
        drop = np.zeros(t_len, dtype=bool)
        drop[:prefix_steps] = True
        drop[suffix_start:] = True
        internal_runs: list[dict[str, Any]] = []
        if t_len > 1:
            d_action = np.linalg.norm(np.diff(patched_action, axis=0), axis=1)
            d_state = np.linalg.norm(np.diff(patched_state, axis=0), axis=1)
            still = np.zeros(t_len, dtype=bool)
            still[1:] = (d_action < TAU_ACTION) & (d_state < TAU_STATE_INTERNAL)
            still &= ~drop
            for start, end, length in runs_from_mask(still):
                if length >= MIN_INTERNAL_RUN:
                    drop[start:end] = True
                    internal_runs.append({"start": int(start), "end": int(end), "length": int(length)})
        drop_info = {
            "prefix_dropped": int(prefix_steps),
            "suffix_dropped": int(t_len - suffix_start),
            "internal_dropped": int(sum(run["length"] for run in internal_runs)),
            "internal_runs": internal_runs,
            "mismatch_patches": len(patches),
            "mismatch_trust_state": int(sum(p["trust"] == "state" for p in patches)),
            "mismatch_trust_action": int(sum(p["trust"] == "action" for p in patches)),
        }
        keep = np.flatnonzero(~drop)
        if len(keep) < MIN_EPISODE_FRAMES:
            dropped_episodes += 1
            episode_reports.append(
                {
                    "source_episode_index": src_ep,
                    "status": "dropped_too_short",
                    "source_frames": int(t_len),
                    "kept_frames": int(len(keep)),
                    **drop_info,
                }
            )
            print(f"[skip] episode {src_ep:03d}: kept {len(keep)} < {MIN_EPISODE_FRAMES}")
            continue

        state_kept = patched_state[keep]
        action_kept = patched_action[keep]
        task_kept = task_index[keep]
        ann_kept = annotation[keep]
        dst_ep = len(out_episodes)

        write_episode_parquet(
            dst / "data" / "chunk-000" / f"episode_{dst_ep:06d}.parquet",
            action=action_kept,
            state=state_kept,
            episode_index=dst_ep,
            global_start=global_start,
            task_index=task_kept,
            annotation=ann_kept,
        )
        for key in VIDEO_KEYS:
            rewrite_video(
                src / "videos" / "chunk-000" / key / f"episode_{src_ep:06d}.mp4",
                dst / "videos" / "chunk-000" / key / f"episode_{dst_ep:06d}.mp4",
                keep,
                fps,
            )

        n = len(action_kept)
        timestamp = np.arange(n, dtype=np.float32) / np.float32(fps)
        all_actions.append(action_kept)
        all_states.append(state_kept)
        all_timestamps.append(timestamp)
        global_start += n

        out_meta = dict(ep_meta)
        out_meta["episode_index"] = dst_ep
        out_meta["length"] = n
        out_meta["source_episode_index"] = src_ep
        out_episodes.append(out_meta)
        episode_reports.append(
            {
                "source_episode_index": src_ep,
                "output_episode_index": dst_ep,
                "status": "kept",
                "source_frames": int(len(action)),
                "kept_frames": n,
                "dropped_frames": int(len(action) - n),
                "kept_ratio": float(n / len(action)),
                **drop_info,
            }
        )
        print(
            f"[keep] {src_ep:03d}->{dst_ep:03d}  {len(action)}->{n}  "
            f"prefix={drop_info['prefix_dropped']} suffix={drop_info['suffix_dropped']} "
            f"internal={drop_info['internal_dropped']}  "
            f"patch_state={drop_info['mismatch_trust_state']} patch_action={drop_info['mismatch_trust_action']}"
        )

    if not out_episodes:
        raise RuntimeError("No episodes survived cleaning")

    actions = np.concatenate(all_actions, axis=0)
    states = np.concatenate(all_states, axis=0)
    timestamps = np.concatenate(all_timestamps, axis=0)
    same_maxabs = np.max(np.abs(actions - states), axis=1)
    arm_maxabs = np.max(np.abs(actions[:, :14] - states[:, :14]), axis=1)
    hand_maxabs = np.max(np.abs(actions[:, 14:] - states[:, 14:]), axis=1)

    stats = {
        "action": feature_stats(actions),
        "observation.state": feature_stats(states),
        "timestamp": feature_stats(timestamps),
    }
    write_json(dst / "meta" / "stats.json", stats)
    write_jsonl(dst / "meta" / "episodes.jsonl", out_episodes)

    out_info = dict(info)
    out_info["total_episodes"] = len(out_episodes)
    out_info["total_frames"] = int(global_start)
    out_info["total_videos"] = len(out_episodes) * len(VIDEO_KEYS)
    out_info["total_chunks"] = 1
    out_info["splits"] = {"train": f"0:{len(out_episodes)}"}
    write_json(dst / "meta" / "info.json", out_info)

    manifest = {
        "source": str(src),
        "output": str(dst),
        "rules": {
            "mismatch": (
                "Per hand-joint run with |action-state|>0.10 rad for >=3 frames. "
                "Arm agreement is the pose/clock prior. Trust state (rewrite action) "
                "for tracking lag and action glitches; trust action (rewrite state) "
                "only for state glitches. Arms are never rewritten."
            ),
            "tau_action_l2": TAU_ACTION,
            "tau_state_internal_l2": TAU_STATE_INTERNAL,
            "min_internal_run_frames": MIN_INTERNAL_RUN,
            "start_run_frames": START_RUN,
            "min_episode_frames": MIN_EPISODE_FRAMES,
            "prefix_suffix": "drop while original action step L2 < tau_action until 3 consecutive moving steps",
            "internal": "drop runs >= 0.5s where both patched action and state steps are below tau",
            "wrist_reused_frames": "kept (dropping them would change apparent speed)",
        },
        "source_episodes": n_src,
        "output_episodes": len(out_episodes),
        "dropped_episodes": dropped_episodes,
        "source_frames": int(info["total_frames"]),
        "output_frames": int(global_start),
        "dropped_frames": int(info["total_frames"]) - int(global_start),
        "post_clean_check": {
            "action_vs_state_same_t_mae": float(np.mean(np.abs(actions - states))),
            "hand_maxabs_p50": float(np.median(hand_maxabs)),
            "hand_maxabs_p99": float(np.percentile(hand_maxabs, 99)),
            "hand_maxabs_max": float(np.max(hand_maxabs)),
            "arm_maxabs_max": float(np.max(arm_maxabs)),
            "frac_hand_maxabs_gt_0p20": float(np.mean(hand_maxabs > 0.20)),
        },
        "episodes": episode_reports,
    }
    write_json(dst / "meta" / "clean_manifest.json", manifest)
    print(
        f"Wrote {dst}\n"
        f"  episodes {n_src} -> {len(out_episodes)}  "
        f"frames {info['total_frames']} -> {global_start}  "
        f"dropped {int(info['total_frames']) - global_start}"
    )


def main() -> None:
    args = parse_args()
    clean(args.src, args.dst, args.overwrite)


if __name__ == "__main__":
    main()
