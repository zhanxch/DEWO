"""Resolve run-dir paths, stats, and inference horizons. Torch-free."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

DEFAULT_INFER_NUM_FRAMES = 33
EVEROOBOT_FULL_EPISODE_DATASET = "EveRobotFullEpisodeDataset"
DEFAULT_SLIDING_WINDOW_ACTION_HORIZON = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_run_dir(run_dir: str | Path) -> Path:
    """Resolve to the training/eval directory that owns config.yaml."""
    run_dir = Path(run_dir).expanduser().resolve()
    if run_dir.is_file():
        if run_dir.name == "config.yaml":
            return run_dir.parent
        raise ValueError(
            f"model_config must be a RUN_DIR or .../config.yaml, got {run_dir}"
        )
    if (run_dir / "config.yaml").exists():
        return run_dir
    for parent in run_dir.parents:
        if (parent / "config.yaml").exists():
            return parent
    raise FileNotFoundError(
        f"Training config not found under {run_dir} or its parents. "
        "Pass a run directory containing config.yaml."
    )


def resolve_checkpoint_path(run_dir: Path, checkpoint: str | Path) -> Path:
    raw = Path(checkpoint).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                Path.cwd() / raw,
                run_dir / raw,
                run_dir / "checkpoints" / "weights" / raw.name,
            ]
        )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    tried = "\n  - ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(
        f"Checkpoint not found for {checkpoint!r}. Tried:\n  - {tried}"
    )


def resolve_stats_path(run_dir: Path, dataset_stats_path: str | None) -> Path:
    if dataset_stats_path:
        stats = Path(dataset_stats_path).expanduser().resolve()
        if not stats.is_file() or stats.stat().st_size <= 0:
            raise FileNotFoundError(
                f"dataset_stats_path must be a non-empty file: {stats}"
            )
        return stats
    default = run_dir / "dataset_stats.json"
    if not default.is_file() or default.stat().st_size <= 0:
        raise FileNotFoundError(
            f"Non-empty dataset_stats.json not found under run dir {run_dir}. "
            "Pass dataset_stats_path explicitly."
        )
    return default


def resolve_meta_stats_dir(
    run_dir: Path,
    configured_meta_dir: Any,
    norm_stats_meta_dir: str | None,
) -> Path:
    if norm_stats_meta_dir:
        meta_dir = Path(norm_stats_meta_dir).expanduser().resolve()
    else:
        if configured_meta_dir is None or not str(configured_meta_dir).strip():
            raise ValueError(
                "norm_stats_source=meta requires processor.norm_stats_meta_dir "
                "or --norm-stats-meta-dir."
            )
        raw = Path(str(configured_meta_dir)).expanduser()
        meta_dir = raw.resolve() if raw.is_absolute() else (run_dir / raw).resolve()
    for name in ("stats.json", "modality.json"):
        path = meta_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(
                f"norm_stats_source=meta requires a non-empty {path}. "
                "Pass --norm-stats-meta-dir when the frozen artifacts were relocated."
            )
    return meta_dir


def resolve_normalization_binding(
    processor_cfg: Any,
    *,
    run_dir: Path,
    dataset_stats_path: str | None,
    norm_stats_meta_dir: str | None,
) -> tuple[str, Path]:
    get = processor_cfg.get if hasattr(processor_cfg, "get") else processor_cfg.__getitem__
    norm_stats_source = str(get("norm_stats_source", "compute")).strip().lower()
    if dataset_stats_path is not None and norm_stats_meta_dir is not None:
        raise ValueError(
            "--dataset-stats-path and --norm-stats-meta-dir are mutually exclusive"
        )
    if norm_stats_source == "meta":
        if dataset_stats_path is not None:
            raise ValueError(
                "The resolved config selects norm_stats_source=meta; "
                "--dataset-stats-path is not allowed."
            )
        meta_dir = resolve_meta_stats_dir(
            run_dir,
            get("norm_stats_meta_dir") if "norm_stats_meta_dir" in processor_cfg else None,
            norm_stats_meta_dir,
        )
        processor_cfg["norm_stats_meta_dir"] = str(meta_dir)
        return "meta", meta_dir
    if norm_stats_meta_dir is not None:
        raise ValueError(
            f"The resolved config selects norm_stats_source={norm_stats_source!r}; "
            "--norm-stats-meta-dir is not allowed."
        )
    return "dataset_stats", resolve_stats_path(run_dir, dataset_stats_path)


def is_everobot_full_episode_train(train_data: dict[str, Any] | Any) -> bool:
    target = ""
    if hasattr(train_data, "get"):
        target = str(train_data.get("_target_", ""))
    elif hasattr(train_data, "_target_"):
        target = str(train_data._target_)
    return EVEROOBOT_FULL_EPISODE_DATASET in target


def resolve_inference_horizons(
    train_data: Any,
    *,
    action_horizon: int | None,
) -> tuple[int, int]:
    """Return ``(action_horizon, num_video_frames)``."""
    num_frames = None
    if hasattr(train_data, "get"):
        num_frames = train_data.get("num_frames")
    elif hasattr(train_data, "num_frames"):
        num_frames = train_data.num_frames
    if num_frames is not None:
        num_video_frames = int(num_frames)
        resolved_action_horizon = (
            int(action_horizon) if action_horizon is not None else num_video_frames - 1
        )
        return resolved_action_horizon, num_video_frames
    if action_horizon is not None:
        resolved_action_horizon = int(action_horizon)
        return resolved_action_horizon, resolved_action_horizon + 1
    if is_everobot_full_episode_train(train_data):
        raise ValueError(
            "EveRobot full-episode run config has no fixed `num_frames`. "
            "Pass action_horizon when starting inference."
        )
    num_video_frames = DEFAULT_INFER_NUM_FRAMES
    return num_video_frames - 1, num_video_frames


def resolve_num_video_frames_from_cfg(cfg: Any, *, fallback: int) -> int:
    """Prefer EVALUATION / top-level ``num_video_frames`` over train ``num_frames``.

    Joint DexJoCo eval uses 9 video frames (latent_t=3). Training configs still
    describe the dataset with ``data.train.num_frames: 33``.
    """
    evaluation_cfg = None
    if hasattr(cfg, "get"):
        evaluation_cfg = cfg.get("EVALUATION")
        top_level = cfg.get("num_video_frames")
    else:
        evaluation_cfg = getattr(cfg, "EVALUATION", None)
        top_level = getattr(cfg, "num_video_frames", None)
    eval_value = None
    if evaluation_cfg is not None and hasattr(evaluation_cfg, "get"):
        eval_value = evaluation_cfg.get("num_video_frames")
    elif evaluation_cfg is not None:
        eval_value = getattr(evaluation_cfg, "num_video_frames", None)
    if eval_value is not None:
        return int(eval_value)
    if top_level is not None:
        return int(top_level)
    return int(fallback)


def resolve_eval_action_horizon(
    train_data: dict[str, Any],
    *,
    action_horizon_override: int | None = None,
) -> int:
    if action_horizon_override is not None:
        return int(action_horizon_override)
    num_frames = train_data.get("num_frames") if hasattr(train_data, "get") else None
    if num_frames is not None:
        return int(num_frames) - 1
    if is_everobot_full_episode_train(train_data):
        raise ValueError(
            "EveRobot full-episode training config has no fixed `num_frames`. "
            "Pass --action-horizon to specify inference chunk size (e.g. 32 or 180)."
        )
    return DEFAULT_SLIDING_WINDOW_ACTION_HORIZON
