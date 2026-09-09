#!/usr/bin/env python3
"""Build the DEWO v9.1 train manifest: D0 (expert ∪ collect) + D_scan + D_fail + D+.

``--expert-manifest`` / ``--collect-manifest`` are success episode manifests.
``--pool-index`` is ``pair_lerobot/pool_index.json`` from materialize_v91.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from fastwam.everobot_schema import validate_manifest, with_manifest_hash

NUM_FRAMES = 33


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_expert_d0(row: Mapping[str, Any]) -> bool:
    return "expert_success" in str(row.get("dataset_id") or "")


def _tag_d0(
    row: dict[str, Any],
    *,
    d0_values: dict[int, dict[int, float]] | None = None,
) -> dict[str, Any]:
    out = dict(row)
    out["pool_role"] = "d0"
    out["batch_role"] = "primary"
    out["action_loss"] = "enabled"
    out["value_loss_weight"] = 0.0
    out["value_target"] = None
    if d0_values and not _is_expert_d0(row):
        table = d0_values.get(int(row["episode_index"]))
        if table:
            out["scan_value_by_frame"] = {
                str(int(frame)): float(value) for frame, value in table.items()
            }
    return out


def _event_unit(
    *,
    dataset_id: str,
    dataset_root: str,
    prompt: str,
    recipe: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    ep_idx = int(row["episode_index"])
    kind = str(row["kind"])
    is_dplus = kind == "dplus"
    is_fail = kind == "d_fail"
    length = int(row.get("length") or NUM_FRAMES)
    sample_id = f"{dataset_id}_ep{ep_idx:06d}_{kind}"
    return {
        "sample_type": "event",
        "sample_id": sample_id,
        "event_id": sample_id,
        "dataset_id": dataset_id,
        "dataset_root": dataset_root,
        "episode_id": f"{dataset_id}_ep{ep_idx:06d}",
        "episode_index": ep_idx,
        "round_id": f"{dataset_id}::r1",
        "collection_round": 1,
        "task": prompt,
        "start_frame": 0,
        "end_frame": length,
        "sample_stride": 1,
        "split": "train",
        "window_selection": "core_start_anchor",
        "core_start_frame": 0,
        "core_end_frame": length,
        "pool_role": "dplus" if is_dplus else ("d_fail" if is_fail else "d_scan"),
        "batch_role": "primary" if is_dplus else "auxiliary",
        "action_loss": "enabled" if is_dplus else "disabled",
        "episode_outcome": "success" if is_dplus else "failure",
        "event_outcome": "success" if is_dplus else "failure",
        "event_type": (
            "success_event" if is_dplus else ("failure_event" if is_fail else "scan_value")
        ),
        "sample_role": (
            "success_event_primary"
            if is_dplus
            else ("failure_context" if is_fail else "scan_value")
        ),
        "source_failure_episode_index": row.get("source_failure_episode_index"),
        "prefix_frame": row.get("prefix_frame"),
        "success_count": row.get("success_count"),
        "pass_m": row.get("pass_m"),
        "value_target": None if is_dplus else row.get("value_target"),
        "value_loss_weight": 0.0 if is_dplus else 1.0,
        "source_window_rule": recipe,
        "effector": "global",
        "pair_id": sample_id,
        "seed": row.get("seed"),
        "replicate": row.get("replicate"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-manifest", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, default=None)
    parser.add_argument("--pool-index", type=Path, required=True)
    parser.add_argument("--pool-dataset", type=Path, required=True)
    parser.add_argument("--pool-dataset-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--d0-value-index",
        type=Path,
        default=None,
        help="Optional Pass@M table for collect D0 windows (expert stays V-off).",
    )
    args = parser.parse_args(argv)

    d0_values = None
    if args.d0_value_index is not None:
        from dewo_v2.d0_collect_value import load_d0_value_index

        d0_values = load_d0_value_index(args.d0_value_index)

    collect = load_json(args.collect_manifest)
    samples = [
        _tag_d0(row, d0_values=d0_values)
        for row in collect.get("samples", [])
        if row.get("episode_outcome") == "success"
        and row.get("batch_role", "primary") == "primary"
        and row.get("split", "train") == "train"
    ]
    dataset_roots = dict(collect.get("dataset_roots") or {})
    expert: dict[str, Any] | None = None
    if args.expert_manifest is not None:
        expert = load_json(args.expert_manifest)
        samples.extend(
            _tag_d0(row, d0_values=None)
            for row in expert.get("samples", [])
            if row.get("episode_outcome") == "success"
            and row.get("split", "train") == "train"
        )
        dataset_roots.update(expert.get("dataset_roots") or {})

    pool_root = args.pool_dataset.expanduser().resolve()
    pool = load_json(args.pool_index)
    for row in pool.get("units") or []:
        samples.append(
            _event_unit(
                dataset_id=str(args.pool_dataset_id),
                dataset_root=str(pool_root),
                prompt=str(args.prompt),
                recipe=str(args.recipe),
                row=row,
            )
        )
    dataset_roots[str(args.pool_dataset_id)] = str(pool_root)

    n_d0 = sum(1 for s in samples if s.get("pool_role") == "d0")
    n_scan = sum(1 for s in samples if s.get("pool_role") == "d_scan")
    n_fail = sum(1 for s in samples if s.get("pool_role") == "d_fail")
    n_plus = sum(1 for s in samples if s.get("pool_role") == "dplus")
    source_hashes = dict(collect.get("source_hashes") or {})
    if expert is not None and not source_hashes:
        source_hashes = dict(expert.get("source_hashes") or {})
    payload = {
        "schema_version": collect.get("schema_version", "0.2"),
        "format": collect.get("format", "EveRobotTrainManifest"),
        "manifest_name": "offline_v91_scratch_pool",
        "eve_root": collect.get("eve_root"),
        "frame_interval": collect.get("frame_interval", "half_open"),
        "dataset_roots": dataset_roots,
        "source_round_ids": sorted({str(row["round_id"]) for row in samples}),
        "source_hashes": source_hashes,
        "samples": samples,
        "selection": {
            "recipe": args.recipe,
            "primary": "expert_plus_collect_d0",
            "d0": n_d0,
            "d_scan": n_scan,
            "d_fail": n_fail,
            "dplus": n_plus,
        },
    }
    payload = with_manifest_hash(payload)
    validate_manifest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output} d0={n_d0} d_scan={n_scan} d_fail={n_fail} dplus={n_plus}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
