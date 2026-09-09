"""DEWO v9.1 pool geometry: scan windows, CFG events, no stitch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

MIN_EVENT_FRAMES = 33
FAIL_CLIFF_POST = 24
CFG_MIN_DROP = 3
REPLAN = 24

POOL_ROLES = frozenset({"d0", "d_scan", "d_fail", "dplus"})


def is_cfg_event(k_prev: int, k: int, *, min_drop: int = CFG_MIN_DROP) -> bool:
    """Pass@10 drop of at least ``min_drop`` while still recoverable."""

    return int(k_prev) - int(k) >= int(min_drop) and int(k) >= 1


def crop_span(
    start: int,
    length: int,
    *,
    min_len: int = MIN_EVENT_FRAMES,
) -> tuple[int, int] | None:
    """Half-open ``[start, start+min_len)`` on a source episode, or None."""

    lo = int(start)
    hi = min(int(length), lo + int(min_len))
    if hi - lo < int(min_len):
        return None
    return lo, hi


def fail_cliff_span(
    t_star: int,
    m_first_zero: int,
    fail_len: int,
    *,
    min_len: int = MIN_EVENT_FRAMES,
    post: int = FAIL_CLIFF_POST,
) -> tuple[int, int]:
    """Half-open ``[lo, hi)`` on the factual fail episode.

    v9.1 D_fail starts at the cliff ``M`` (``m_first_zero``), not ``t_star``.
    ``t_star`` is accepted for the v9 caller; v9.1 passes ``M`` as both.
    """

    lo = int(m_first_zero)
    hi = min(int(fail_len), int(m_first_zero) + int(post))
    if hi < lo:
        raise ValueError(f"fail cliff inverted: M={lo} M+post={hi} len={fail_len}")
    if hi - lo < min_len:
        hi = min(int(fail_len), lo + min_len)
    if hi - lo < min_len:
        lo = max(0, hi - min_len)
    if hi - lo < min_len:
        raise ValueError(
            f"fail episode length {fail_len} cannot host {min_len} cliff frames"
        )
    return lo, hi


def prefix_dir(scan_root: Path, episode_index: int, prefix_frame: int) -> Path:
    return (
        Path(scan_root)
        / "prefixes"
        / f"ep{int(episode_index):06d}_f{int(prefix_frame):04d}"
    )


def first_success_tau(
    scan_root: Path,
    episode_index: int,
    prefix_frame: int,
    successful_replicate_indices: Sequence[int] | None,
) -> dict[str, Any] | None:
    """First successful replicate with RGB + npz, lowest index first."""

    indices = [int(i) for i in (successful_replicate_indices or ())]
    root = prefix_dir(scan_root, episode_index, prefix_frame)
    for rep in indices:
        replicate_dir = root / f"replicate_{rep:02d}"
        npz = replicate_dir / "trajectory.npz"
        front = replicate_dir / "continuation_front.mp4"
        wrist = replicate_dir / "continuation_wrist.mp4"
        if npz.is_file() and front.is_file() and wrist.is_file():
            return {
                "replicate": rep,
                "npz": npz,
                "front": front,
                "wrist": wrist,
            }
    return None


def adjacent_cfg_events(
    prefixes: Sequence[Mapping[str, Any]],
    *,
    replan: int = REPLAN,
    min_drop: int = CFG_MIN_DROP,
) -> list[dict[str, Any]]:
    """CFG events on a sorted-per-episode prefix list."""

    by_frame = sorted(
        prefixes,
        key=lambda row: int(row["prefix_frame"]),
    )
    events: list[dict[str, Any]] = []
    prev: Mapping[str, Any] | None = None
    for row in by_frame:
        t = int(row["prefix_frame"])
        k = int(row["success_count"])
        if prev is not None:
            t_prev = int(prev["prefix_frame"])
            k_prev = int(prev["success_count"])
            if t - t_prev == int(replan) and is_cfg_event(
                k_prev, k, min_drop=min_drop
            ):
                events.append(
                    {
                        "prefix_frame": t,
                        "success_count": k,
                        "prev_frame": t_prev,
                        "prev_success_count": k_prev,
                    }
                )
        prev = row
    return events


def pool_role_of(unit: Mapping[str, Any]) -> str | None:
    role = unit.get("pool_role")
    if role in POOL_ROLES:
        return str(role)
    return None


def build_v91_pool_specs(
    *,
    scan_root: Path,
    episodes: Mapping[int, Mapping[str, Any]],
    prefix_labels: Sequence[Mapping[str, Any]],
    pass_m: int = 10,
) -> dict[str, Any]:
    """Scan windows + CFG D+ specs. Does not stitch."""

    scan_root = Path(scan_root)
    by_ep: dict[int, list[dict[str, Any]]] = {}
    for row in prefix_labels:
        ep = int(row["source_failure_episode_index"])
        by_ep.setdefault(ep, []).append(dict(row))

    scan_windows: list[dict[str, Any]] = []
    dplus: list[dict[str, Any]] = []
    skipped_short = 0
    skipped_no_tau = 0
    for ep, rows in sorted(by_ep.items()):
        fail_len = int(episodes[ep]["length"])
        frames = {int(r["prefix_frame"]): r for r in rows}
        cliff = None
        zeros = [int(r["prefix_frame"]) for r in rows if int(r["success_count"]) == 0]
        if zeros:
            cliff = min(zeros)
        for row in sorted(rows, key=lambda r: int(r["prefix_frame"])):
            t = int(row["prefix_frame"])
            k = int(row["success_count"])
            m_pass = int(row.get("pass_m") or pass_m)
            span = crop_span(t, fail_len)
            if span is None:
                skipped_short += 1
                continue
            is_cliff = cliff is not None and t == cliff
            scan_windows.append(
                {
                    "source_failure_episode_index": ep,
                    "prefix_frame": t,
                    "success_count": k,
                    "pass_m": m_pass,
                    "value_target": float(k) / float(m_pass),
                    "fail_span": [span[0], span[1]],
                    "is_cliff": is_cliff,
                    "seed": row.get("seed"),
                }
            )
        for event in adjacent_cfg_events(rows):
            t = int(event["prefix_frame"])
            k = int(event["success_count"])
            src = frames[t]
            tau = first_success_tau(
                scan_root,
                ep,
                t,
                src.get("successful_replicate_indices"),
            )
            if tau is None:
                skipped_no_tau += 1
                continue
            dplus.append(
                {
                    "source_failure_episode_index": ep,
                    "prefix_frame": t,
                    "success_count": k,
                    "prev_frame": event["prev_frame"],
                    "prev_success_count": event["prev_success_count"],
                    "pass_m": int(src.get("pass_m") or pass_m),
                    "replicate": tau["replicate"],
                    "npz": str(tau["npz"]),
                    "front": str(tau["front"]),
                    "wrist": str(tau["wrist"]),
                    "seed": src.get("seed"),
                }
            )
    return {
        "scan_windows": scan_windows,
        "dplus": dplus,
        "counts": {
            "d_scan": len(scan_windows),
            "d_fail": sum(1 for row in scan_windows if row["is_cliff"]),
            "dplus": len(dplus),
            "skipped_short_scan": skipped_short,
            "skipped_dplus_no_rgb": skipped_no_tau,
        },
    }
