"""Receding-horizon DexJoCo rollout with independent action / value replan."""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fastwam.inference.obs import fastwam_action_to_dexjoco, safe_rgb_uint8


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def combine_preview_cameras(obs: dict[str, np.ndarray], size: int = 224) -> np.ndarray:
    from fastwam.inference.obs import resize_rgb_area

    missing = {"front", "wrist"} - set(obs)
    if missing:
        frame = safe_rgb_uint8(obs.get("front", next(iter(obs.values()))))
        return resize_rgb_area(frame, (size, size))
    return np.concatenate(
        [
            resize_rgb_area(safe_rgb_uint8(obs["front"]), (size, size)),
            resize_rgb_area(safe_rgb_uint8(obs["wrist"]), (size, size)),
        ],
        axis=1,
    )


def rollout_episode(
    env: Any,
    initial_obs: dict[str, np.ndarray],
    policy: Any,
    seed: int,
    repeat: int,
    max_steps: int = 1200,
    capture_frames: bool = True,
    dual_arm: bool = False,
    record_trajectory: bool = False,
    options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Execute receding-horizon action chunks; optionally query value more often.

    ``record_trajectory=True`` stores 22D rotvec actions, 23D proprio, and
    full-resolution front/wrist frames for LeRobot collect.
    """
    obs = initial_obs
    pending_actions: deque[np.ndarray] = deque()
    frames = [combine_preview_cameras(obs)] if capture_frames else []
    recorded_actions: list[np.ndarray] = []
    recorded_states: list[np.ndarray] = []
    recorded_fronts: list[np.ndarray] = []
    recorded_wrists: list[np.ndarray] = []
    success = False
    terminated = False
    truncated = False
    replan_index = 0
    inference_calls = 0
    value_calls = 0
    inference_seconds = 0.0
    simulation_seconds = 0.0
    rollout_start = time.perf_counter()
    cfg_values: list[float] = []
    cfg_value_rels: list[float] = []
    steps_since_value = 10**9
    value_replan = int(getattr(policy, "value_replan_steps", policy.replan_steps))
    episode_steps = 0
    value_prev: float | None = None
    gate_fired = False

    while episode_steps < max_steps:
        if not pending_actions:
            noise_seed = seed * 100_000 + repeat * 1_000 + replan_index
            infer_options = dict(options or {})
            infer_options["cfg_replan_index"] = replan_index
            if value_prev is not None:
                infer_options["cfg_value_prev"] = float(value_prev)
            infer_options["cfg_gate_fired"] = bool(gate_fired)
            infer_start = time.perf_counter()
            extras = policy.infer_with_extras(obs, noise_seed=noise_seed, options=infer_options)
            inference_seconds += time.perf_counter() - infer_start
            inference_calls += 1
            replan_index += 1
            chunk = np.asarray(extras["action"], dtype=np.float32)
            pending_actions.extend(chunk[: policy.replan_steps])
            if "cfg_value" in extras:
                value = float(np.asarray(extras["cfg_value"]).reshape(-1)[0])
                cfg_values.append(value)
                rel = extras.get("cfg_value_rel")
                if rel is None:
                    cfg_value_rels.append(None)
                else:
                    rel_f = float(np.asarray(rel).reshape(-1)[0])
                    cfg_value_rels.append(rel_f if np.isfinite(rel_f) else None)
                if extras.get("cfg_gate_g") is not None:
                    gate_fired = gate_fired or float(np.asarray(extras["cfg_gate_g"])) >= 1.0
                value_prev = value
                steps_since_value = 0
                value_calls += 1
            else:
                steps_since_value = 0

        elif steps_since_value >= value_replan:
            infer_start = time.perf_counter()
            extras = policy.infer_value(obs)
            inference_seconds += time.perf_counter() - infer_start
            if "cfg_value" in extras:
                value = float(np.asarray(extras["cfg_value"]).reshape(-1)[0])
                cfg_values.append(value)
                cfg_value_rels.append(None)
                value_prev = value
                value_calls += 1
            steps_since_value = 0

        rotvec_action = np.asarray(pending_actions.popleft(), dtype=np.float32)
        if record_trajectory:
            state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
            if state.shape[0] > 23:
                state = state[:23]
            recorded_actions.append(rotvec_action.reshape(-1)[:22].astype(np.float32))
            recorded_states.append(state.astype(np.float32))
            recorded_fronts.append(safe_rgb_uint8(obs["front"]))
            recorded_wrists.append(safe_rgb_uint8(obs["wrist"]))

        sim_start = time.perf_counter()
        env_action = (
            fastwam_action_to_dexjoco(rotvec_action)
            if not dual_arm
            else policy.adapter.rotvec_to_env_action(rotvec_action, dual_arm=True)
        )
        obs, _, terminated, truncated, info = env.step(env_action)
        simulation_seconds += time.perf_counter() - sim_start
        episode_steps += 1
        steps_since_value += 1
        if capture_frames:
            frames.append(combine_preview_cameras(obs))
        success = bool(info.get("succeed", False))
        if terminated or truncated:
            break

    row = {
        "seed": int(seed),
        "repeat": int(repeat),
        "success": success,
        "result": "success" if success else "failure",
        "episode_steps": episode_steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "hit_eval_max_steps": bool(not (terminated or truncated)),
        "inference_calls": inference_calls,
        "value_calls": value_calls,
        "replan_steps": int(policy.replan_steps),
        "value_replan_steps": value_replan,
        "cfg_values": cfg_values,
        "cfg_value_rels": cfg_value_rels,
        "inference_seconds": inference_seconds,
        "simulation_seconds": simulation_seconds,
        "wall_seconds": time.perf_counter() - rollout_start,
        "completed_at": _utc_now(),
    }
    if record_trajectory:
        if not recorded_actions:
            raise ValueError("record_trajectory=True produced an empty episode")
        row["actions"] = np.stack(recorded_actions, axis=0)
        row["states"] = np.stack(recorded_states, axis=0)
        row["frames"] = {
            "observation.images.front": recorded_fronts,
            "observation.images.wrist": recorded_wrists,
        }
        row["elapsed_s"] = float(row["wall_seconds"])
        row["steps"] = int(episode_steps)
    return row, frames


def save_rollout_mp4(frames: list[np.ndarray], path: str | Path, fps: int = 30) -> None:
    import imageio.v2 as imageio

    if not frames:
        raise ValueError("Cannot encode an empty frame list")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        path,
        fps=int(fps),
        codec="libx264",
        macro_block_size=16,
        output_params=["-preset", "veryfast", "-crf", "23"],
        ffmpeg_log_level="error",
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()
