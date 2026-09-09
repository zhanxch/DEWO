"""Build a sparse Pass@M value table for D0 collect success episodes.

Keys are episode indices from the collect LeRobot dataset. Frame keys are
exact scan nodes (stringified ints). There is no interpolation: a window
start that is not a scanned node has no V label.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

FORMAT = "D0CollectValueIndex"
VERSION = 1


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _episode_index(row: Mapping[str, Any]) -> int:
    if row.get("source_episode_index") is not None:
        return int(row["source_episode_index"])
    return int(row["source_failure_episode_index"])


def build_d0_value_index(
    prefix_results: Sequence[Mapping[str, Any]],
    *,
    pass_m: int = 10,
    scan_root: Path | str | None = None,
) -> dict[str, Any]:
    episodes: dict[str, dict[str, float]] = {}
    skipped_incomplete = 0
    for row in prefix_results:
        m = max(int(row.get("pass_m") or pass_m), 1)
        evaluated = int(row.get("replicates_evaluated") or 0)
        if evaluated < m:
            skipped_incomplete += 1
            continue
        ep = str(_episode_index(row))
        frame = str(int(row["prefix_frame"]))
        k = int(row["success_count"])
        episodes.setdefault(ep, {})[frame] = float(k) / float(m)
    return {
        "format": FORMAT,
        "version": VERSION,
        "pass_m": int(pass_m),
        "scan_root": None if scan_root is None else str(scan_root),
        "num_episodes": len(episodes),
        "num_labeled_frames": sum(len(table) for table in episodes.values()),
        "num_incomplete_prefixes": skipped_incomplete,
        "episodes": episodes,
    }


def write_d0_value_index(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_d0_value_index(path: Path) -> dict[int, dict[int, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("episodes", payload)
    out: dict[int, dict[int, float]] = {}
    for ep, table in dict(raw).items():
        out[int(ep)] = {int(frame): float(value) for frame, value in dict(table).items()}
    return out


def lookup_scan_value(table: Mapping[Any, Any] | None, window_start: int) -> float | None:
    """Return Pass@M V at an exact scan node, or None if unlabeled."""

    if not table:
        return None
    key_str = str(int(window_start))
    if key_str in table:
        return float(table[key_str])
    key_int = int(window_start)
    if key_int in table:
        return float(table[key_int])
    return None
