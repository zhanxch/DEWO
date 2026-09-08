#!/usr/bin/env python3
"""Concatenate per-task DEWO pair Eve manifests + VAE/text caches into one mixed pool.

Usage:
  python scripts/dewo_v2/concat_mixed_scratch_pool.py \\
    --queue-file logs/mixed_scratch_pipeline.env \\
    --output-root data/mixed_5task_dewo_scratch_<stamp>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from fastwam.everobot_schema import validate_manifest, with_manifest_hash

ROOT = Path(__file__).resolve().parents[2]
_EXPORT_RE = re.compile(r"^(?:export\s+)?([A-Z][A-Z0-9_]*)=(.*)$")


def _unquote(raw: str) -> str:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    if text.startswith("$'") and text.endswith("'") and len(text) >= 3:
        return text[2:-1]
    return text


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _EXPORT_RE.match(stripped)
        if not match:
            continue
        values[match.group(1)] = _unquote(match.group(2))
    return values


def parse_queue_file(path: Path) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = _unquote(value.strip())
        if key == "TASK" and current:
            blocks.append(current)
            current = {}
        current[key] = value
    if current:
        blocks.append(current)
    return blocks


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def concat_manifests(
    manifests: list[dict[str, Any]],
    *,
    manifest_name: str,
    recipe: str,
    eve_root: Path,
) -> dict[str, Any]:
    if not manifests:
        raise ValueError("No manifests to concatenate")
    samples: list[dict[str, Any]] = []
    dataset_roots: dict[str, str] = {}
    source_hashes_parts: list[str] = []
    for manifest in manifests:
        for sample in manifest.get("samples", []):
            samples.append(dict(sample))
        for dataset_id, root in dict(manifest.get("dataset_roots") or {}).items():
            if dataset_id in dataset_roots and dataset_roots[dataset_id] != str(root):
                raise ValueError(
                    f"dataset_id {dataset_id!r} maps to both {dataset_roots[dataset_id]} and {root}"
                )
            dataset_roots[str(dataset_id)] = str(root)
        source_hashes_parts.append(json.dumps(manifest.get("source_hashes") or {}, sort_keys=True))
    sample_ids = [str(row.get("sample_id")) for row in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("concatenated manifest has duplicate sample_id values")
    source_round_ids = sorted({str(row["round_id"]) for row in samples})
    first = manifests[0]
    combined_hash = _sha256_text("\n".join(source_hashes_parts))
    payload = {
        "schema_version": first.get("schema_version", "0.2"),
        "format": first.get("format", "EveRobotTrainManifest"),
        "manifest_name": manifest_name,
        "eve_root": str(eve_root),
        "frame_interval": first.get("frame_interval", "half_open"),
        "dataset_roots": dataset_roots,
        "source_round_ids": source_round_ids,
        "source_hashes": {
            "round_meta_sha256": combined_hash,
            "episode_meta_sha256": combined_hash,
            "event_meta_sha256": combined_hash,
        },
        "samples": samples,
        "selection": {
            "recipe": recipe,
            "primary": "mixed_5task_concat",
            "auxiliary": "pair_success_video_plus_pair_failure_video",
            "include_s0_success_rollouts": True,
            "primary_source": "all_success_seeds",
            "source_manifests": len(manifests),
        },
    }
    hashed = with_manifest_hash(payload)
    validate_manifest(hashed, verify_hash=True)
    return hashed


def symlink_tree_files(src: Path, dst: Path, *, pattern: str) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    if not src.is_dir():
        return 0
    for path in sorted(src.glob(pattern)):
        if not path.is_file():
            continue
        target = dst / path.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(path.resolve())
        count += 1
    return count


def write_mixed_env(
    path: Path,
    *,
    output_root: Path,
    train_manifest: Path,
    val_manifest: Path,
    text_cache: Path,
    vae_cache: Path,
    base_dataset: str,
    rollout_raw: str,
    stats: str,
    source_config: str,
    action_dit: str,
    text_sha: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# Generated by concat_mixed_scratch_pool.py
export TASK=mixed_5task
export DEWO_TASK_NAME=mixed_5task
export BASE_DATASET={base_dataset}
export ROLLOUT_RAW={rollout_raw}
export PAIR_DATASET={rollout_raw}
export PRIMARY_KIND=all_success_seeds
export INIT_WEIGHTS={action_dit}
export SOURCE_CHECKPOINT={action_dit}
export STATS={stats}
export PRETRAINED_NORM_STATS={stats}
export FASTWAM_SOURCE_CONFIG={source_config}
export EVE_MANIFEST_PATH={train_manifest}
export B1_MANIFEST_PATH={train_manifest}
export EVE_VAL_MANIFEST_PATH={val_manifest}
export TEXT_EMBEDDING_CACHE_DIR={text_cache}
export TEXT_EMBEDDING_CACHE_SHA256={text_sha}
export B1_VIDEO_EXPERIMENT_ROOT={output_root}
export USE_VAE_LATENT_CACHE=1
export VAE_LATENT_CACHE_DIR={vae_cache}
export REQUIRE_VAE_LATENT_CACHE=1
export FILL_VAE_LATENT_CACHE=0
export SKIP_VAE_PREENCODE=1
export DEWO_TASK=dexjoco/dexjoco_dewo_scratch_joint
export DEWO_VARIANT=DEWO-scratch-joint
export DEWO_PROTOCOL=mixed_5task_dewo_scratch_joint
export DEWO_OUTPUT_DIR=./runs/dexjoco_mixed_5task_dewo_scratch_joint
export FITWAM_WANDB_GROUP=mixed_5task_dewo_scratch_joint
export NORM_STATS_SOURCE=compute
export SUCCESS_PROMPT='mixed 5-task DEWO scratch pool'
""",
        encoding="utf-8",
    )


def hash_text_cache(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file() or path.is_symlink())
    if not files:
        raise SystemExit(f"Text cache is empty: {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--action-dit",
        type=Path,
        default=ROOT / "checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=ROOT / "artifacts/mixed_5task/dataset_stats.json",
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=ROOT / "configs/eval/dexjoco/mixed_5task_fastwam_joint/config.yaml",
    )
    args = parser.parse_args(argv)

    blocks = parse_queue_file(args.queue_file.expanduser().resolve())
    if len(blocks) < 2:
        raise SystemExit(f"Need at least 2 prepared tasks in {args.queue_file}, got {len(blocks)}")

    output_root = args.output_root.expanduser().resolve()
    eve_root = output_root / "eve_v02"
    manifest_dir = eve_root / "manifests"
    protocol = eve_root / "protocol"
    text_cache = output_root / "text_embeds_cache"
    vae_cache = output_root / "vae_latent_cache"
    for directory in (manifest_dir, protocol, text_cache, vae_cache, output_root / "logs"):
        directory.mkdir(parents=True, exist_ok=True)

    train_manifests: list[dict[str, Any]] = []
    val_manifests: list[dict[str, Any]] = []
    env_payloads: list[dict[str, str]] = []
    for block in blocks:
        env_path = Path(block["ENV_FILE"]).expanduser().resolve()
        env = parse_env_file(env_path)
        env_payloads.append(env)
        train_manifests.append(load_json(Path(env["EVE_MANIFEST_PATH"])))
        val_manifests.append(load_json(Path(env["EVE_VAL_MANIFEST_PATH"])))
        n_text = symlink_tree_files(Path(env["TEXT_EMBEDDING_CACHE_DIR"]), text_cache, pattern="*")
        n_vae = symlink_tree_files(Path(env["VAE_LATENT_CACHE_DIR"]), vae_cache, pattern="*.pt")
        print(
            f"[concat] {block.get('TASK', env_path)} text+={n_text} vae+={n_vae}",
            flush=True,
        )

    train = concat_manifests(
        train_manifests,
        manifest_name="offline_b1_jump_fast_pair_mixed_5task",
        recipe="mixed_5task_dewo_scratch_recoverability_pairs",
        eve_root=eve_root,
    )
    val = concat_manifests(
        val_manifests,
        manifest_name="offline_selection_primary_success_mixed_5task",
        recipe="mixed_5task_dewo_scratch_val",
        eve_root=eve_root,
    )
    train_path = manifest_dir / "offline_b1_jump_fast_pair.json"
    val_path = manifest_dir / "offline_selection_primary_success.json"
    train_path.write_text(json.dumps(train, indent=2, sort_keys=True) + "\n")
    val_path.write_text(json.dumps(val, indent=2, sort_keys=True) + "\n")

    text_sha = hash_text_cache(text_cache)
    first = env_payloads[0]
    env_out = protocol / "offline_v1_b1_jump_fast.env"
    write_mixed_env(
        env_out,
        output_root=output_root,
        train_manifest=train_path,
        val_manifest=val_path,
        text_cache=text_cache,
        vae_cache=vae_cache,
        base_dataset=first["BASE_DATASET"],
        rollout_raw=first.get("ROLLOUT_RAW", first["BASE_DATASET"]),
        stats=str(args.stats.expanduser().resolve()),
        source_config=str(args.source_config.expanduser().resolve()),
        action_dit=str(args.action_dit.expanduser().resolve()),
        text_sha=text_sha,
    )
    summary = {
        "n_tasks": len(blocks),
        "n_train_samples": len(train["samples"]),
        "n_val_samples": len(val["samples"]),
        "env_file": str(env_out),
        "train_manifest": str(train_path),
        "val_manifest": str(val_path),
        "text_cache": str(text_cache),
        "vae_cache": str(vae_cache),
        "n_vae_files": len(list(vae_cache.glob("*.pt"))),
    }
    (output_root / "concat_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
