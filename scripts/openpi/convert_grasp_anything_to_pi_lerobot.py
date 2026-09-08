#!/usr/bin/env python3
"""Convert the GR00T-style grasp_anything LeRobot dataset into OpenPI's expected layout.

OpenPI's fine-tune path (see examples/libero/convert_libero_data_to_lerobot.py and
examples/ur5/README.md) wants a LeRobot v2 dataset with:
  - proprio in `state`
  - actions in `actions`
  - cameras as top-level video keys
  - language via `task` / `task_index` (prompt_from_task=True)

The source dataset already is LeRobot v2, but uses GR00T keys
(`observation.state`, `action`, `observation.images.*`, annotation.*). This
script rewrites parquet/meta and symlinks videos so OpenPI can consume it
without re-encoding.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SRC_VIDEO_KEYS = {
    "image": "observation.images.head_view",
    "wrist_image_left": "observation.images.left_wrist_view",
    "wrist_image_right": "observation.images.right_wrist_view",
}

DROP_COLUMNS = {"annotation.human.action.task_description"}
RENAME_COLUMNS = {
    "observation.state": "state",
    "action": "actions",
}


def _rewrite_info(src_info: dict) -> dict:
    src_features = src_info["features"]
    old_video_keys = set(SRC_VIDEO_KEYS.values())
    features = {}
    for old_name, spec in src_features.items():
        if old_name in DROP_COLUMNS or old_name in old_video_keys:
            continue
        new_name = RENAME_COLUMNS.get(old_name, old_name)
        features[new_name] = spec

    for new_key, old_key in SRC_VIDEO_KEYS.items():
        features[new_key] = src_features[old_key]

    info = dict(src_info)
    info["features"] = features
    info["robot_type"] = "wuji_astribot"
    info["total_videos"] = src_info["total_episodes"] * len(SRC_VIDEO_KEYS)
    return info


def _rewrite_stats(src_stats: dict) -> dict:
    out = {}
    if "observation.state" in src_stats:
        out["state"] = src_stats["observation.state"]
    if "action" in src_stats:
        out["actions"] = src_stats["action"]
    if "timestamp" in src_stats:
        out["timestamp"] = src_stats["timestamp"]
    return out


def _rewrite_episodes(src_path: Path, dst_path: Path) -> None:
    with src_path.open() as fin, dst_path.open("w") as fout:
        for line in fin:
            ep = json.loads(line)
            fout.write(
                json.dumps(
                    {
                        "episode_index": ep["episode_index"],
                        "tasks": ep["tasks"],
                        "length": ep["length"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _rewrite_parquet(src: Path, dst: Path) -> None:
    table = pq.read_table(src)
    names = []
    arrays = []
    for name, column in zip(table.column_names, table.columns):
        if name in DROP_COLUMNS:
            continue
        names.append(RENAME_COLUMNS.get(name, name))
        arrays.append(column)
    # Cast state/actions from list<float> to list<float32> when possible.
    out = pa.table(dict(zip(names, arrays)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, dst)


def _symlink_videos(src_root: Path, dst_root: Path, n_episodes: int) -> None:
    for new_key, old_key in SRC_VIDEO_KEYS.items():
        src_dir = src_root / "videos" / "chunk-000" / old_key
        dst_dir = dst_root / "videos" / "chunk-000" / new_key
        dst_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_episodes):
            name = f"episode_{i:06d}.mp4"
            src = (src_dir / name).resolve()
            if not src.exists():
                raise FileNotFoundError(src)
            dst = dst_dir / name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)


def convert(src_root: Path, dst_root: Path) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)

    src_meta = src_root / "meta"
    dst_meta = dst_root / "meta"
    dst_meta.mkdir()

    info = json.loads((src_meta / "info.json").read_text())
    n_episodes = int(info["total_episodes"])
    (dst_meta / "info.json").write_text(json.dumps(_rewrite_info(info), indent=2) + "\n")
    (dst_meta / "stats.json").write_text(
        json.dumps(_rewrite_stats(json.loads((src_meta / "stats.json").read_text())), indent=4) + "\n"
    )
    shutil.copy2(src_meta / "tasks.jsonl", dst_meta / "tasks.jsonl")
    _rewrite_episodes(src_meta / "episodes.jsonl", dst_meta / "episodes.jsonl")

    src_data = src_root / "data" / "chunk-000"
    dst_data = dst_root / "data" / "chunk-000"
    dst_data.mkdir(parents=True)
    for parquet in sorted(src_data.glob("episode_*.parquet")):
        _rewrite_parquet(parquet, dst_data / parquet.name)

    _symlink_videos(src_root, dst_root, n_episodes)
    print(f"Wrote OpenPI LeRobot dataset to {dst_root}")
    print(f"  episodes={n_episodes} frames={info['total_frames']} fps={info['fps']}")
    print("  cameras: image, wrist_image_left, wrist_image_right")
    print("  state/actions dim=54 (7+7 arm joints, 20+20 hand joints)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/gaozt-test1/zhanxch/FITWAM-dewov9-20260828/data/grasp_anything_joint_absolute"),
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/gaozt-test1/zhanxch/FITWAM-dewov9-20260828/data/pi/grasp_anything"),
    )
    args = parser.parse_args()
    convert(args.src.resolve(), args.dst.resolve())


if __name__ == "__main__":
    main()
