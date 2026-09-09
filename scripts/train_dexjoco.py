#!/usr/bin/env python3
"""DEWO v9.1 scratch (FastWAMJoint) / v9 s0 adapter trainer after ``prepare_dexjoco``.

In-process CLI, same house style as ``eval_dexjoco.py`` / ``collect_dexjoco.py`` /
``prepare_dexjoco.py``. CFG mixing lives here, not in the prepare protocol env.

  scratch — full FastWAMJoint from Wan + ActionDiT (``full_dit``). Default.
  s0      — frozen mixed-S0 MoT + text-side K/V adapter (``dewo_v9_uncond_adapter``)

Example (v9.1 scratch on GPUs 0-3)::

    python scripts/train_dexjoco.py \\
      --task-name fold_glasses \\
      --init scratch \\
      --prepare-dir prepare_results/dexjoco/fold_glasses/<stamp> \\
      --gpus 0,1,2,3

Example (v9 adapter from frozen S0)::

    python scripts/train_dexjoco.py \\
      --task-name fold_glasses \\
      --init s0 \\
      --prepare-dir prepare_results/dexjoco/fold_glasses/<stamp> \\
      --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (
    REPO_ROOT / "src",
    REPO_ROOT / "scripts",
):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import eval_dexjoco as eval_entry
import prepare_dexjoco as prepare_entry
from dewo_v2.tasks import CfgRecipe

ENV_NAME = "offline_v1_b1_jump_fast.env"
ENV_REL = Path("eve_v02") / "protocol" / ENV_NAME
STEP_DIR_RE = re.compile(r"^step_(\d+)$")
HYDRA_S0 = "dexjoco/dexjoco_dewo_v9_offline_b1_jump_fast_uncond"
HYDRA_SCRATCH = "dexjoco/dexjoco_dewo_scratch_joint"
ACTION_DIT = REPO_ROOT / "checkpoints" / "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
JOINT_STATS = REPO_ROOT / "artifacts" / "mixed_5task" / "dataset_stats.json"
JOINT_CFG = REPO_ROOT / "configs" / "eval" / "dexjoco" / "mixed_5task_fastwam_joint" / "config.yaml"
CFG_COMPACT_KEYS = ("CFG_PRIMARY", "CFG_AUX_SUCCESS", "CFG_AUX_FAIL")
CFG_MIX_KEYS = (
    "CFG_PRIMARY_OUTCOME",
    "CFG_PRIMARY_FAST",
    "CFG_PRIMARY_BASE",
    "CFG_AUX_SUCCESS_OUTCOME",
    "CFG_AUX_SUCCESS_FAST",
    "CFG_AUX_SUCCESS_BASE",
    "CFG_AUX_FAIL_OUTCOME",
    "CFG_AUX_FAIL_FAST",
    "CFG_AUX_FAIL_BASE",
    "CFG_SUCCESS_SUFFIX",
    "CFG_FAILURE_SUFFIX",
    "CFG_DROPOUT",
    "CFG_RECIPE_NAME",
    "CFG_FAST_MODEL_ID",
    "CFG_FAST_MAX_TOKENS",
    "CFG_FAST_FAIL_CLOSED",
)
TRAIN_RUN = REPO_ROOT / "scripts" / "dewo_v2" / "train_run.sh"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gpu_csv(gpus: list[int]) -> str:
    return ",".join(str(item) for item in gpus)


def _assert_not_lora(label: str, value: str) -> None:
    text = str(value)
    if "lora" in text.lower():
        raise ValueError(
            f"{label}={value} is a LoRA path. Use --init s0 (adapter) or --init scratch (joint)."
        )


def cfg_env(recipe: CfgRecipe | None = None) -> dict[str, str]:
    cfg = recipe or CfgRecipe()
    return {
        "CFG_SUCCESS_SUFFIX": str(cfg.success_suffix),
        "CFG_FAILURE_SUFFIX": str(cfg.failure_suffix or ""),
        "CFG_DROPOUT": str(cfg.dropout),
        "CFG_PRIMARY_OUTCOME": str(cfg.primary[0]),
        "CFG_PRIMARY_FAST": str(cfg.primary[1]),
        "CFG_PRIMARY_BASE": str(cfg.primary[2]),
        "CFG_AUX_SUCCESS_OUTCOME": str(cfg.aux_success[0]),
        "CFG_AUX_SUCCESS_FAST": str(cfg.aux_success[1]),
        "CFG_AUX_SUCCESS_BASE": str(cfg.aux_success[2]),
        "CFG_AUX_FAIL_OUTCOME": str(cfg.aux_fail[0]),
        "CFG_AUX_FAIL_FAST": str(cfg.aux_fail[1]),
        "CFG_AUX_FAIL_BASE": str(cfg.aux_fail[2]),
    }


def strip_cfg_mix(values: dict[str, str]) -> dict[str, str]:
    cleaned = dict(values)
    for key in (*CFG_COMPACT_KEYS, *CFG_MIX_KEYS):
        cleaned.pop(key, None)
    return cleaned


def resolve_prepare_layout(
    prepare_dir: Path,
    checkpoint_steps: list[int] | None = None,
) -> dict[str, Any]:
    """Map a prepare stamp / Eve root / protocol env onto ``step -> env file``."""

    prepare_dir = Path(prepare_dir).expanduser().resolve()
    if prepare_dir.is_file():
        if prepare_dir.name != ENV_NAME:
            raise FileNotFoundError(f"Expected {ENV_NAME}, got {prepare_dir}")
        return {
            "layout": "env_file",
            "prepare_dir": prepare_dir.parent,
            "steps": [],
            "env_by_step": {},
            "env_file": prepare_dir,
        }
    if not prepare_dir.exists():
        raise FileNotFoundError(f"Missing prepare dir: {prepare_dir}")

    env_by_step: dict[int, Path] = {}
    for step_dir in sorted(prepare_dir.glob("step_*")):
        match = STEP_DIR_RE.match(step_dir.name)
        if match is None:
            continue
        env_file = step_dir / ENV_REL
        if env_file.is_file():
            env_by_step[int(match.group(1))] = env_file.resolve()

    if env_by_step:
        if checkpoint_steps:
            steps = [int(step) for step in checkpoint_steps]
        else:
            steps = sorted(env_by_step)
        missing = [step for step in steps if step not in env_by_step]
        if missing:
            raise FileNotFoundError(
                f"{prepare_dir} is missing {ENV_NAME} for steps {missing}"
            )
        return {
            "layout": "prepare_dexjoco",
            "prepare_dir": prepare_dir,
            "steps": steps,
            "env_by_step": {step: env_by_step[step] for step in steps},
            "env_file": env_by_step[steps[-1]],
        }

    for candidate, layout in (
        (prepare_dir / ENV_REL, "prepare_step"),
        (prepare_dir / "protocol" / ENV_NAME, "eve_root"),
        (prepare_dir / ENV_NAME, "protocol_dir"),
        (prepare_dir / "eve_v02" / "protocol" / ENV_NAME, "legacy_prepare"),
    ):
        if candidate.is_file():
            step = None
            match = STEP_DIR_RE.match(prepare_dir.name)
            if match is not None:
                step = int(match.group(1))
            return {
                "layout": layout,
                "prepare_dir": prepare_dir,
                "steps": [step] if step is not None else [],
                "env_by_step": {step: candidate.resolve()} if step is not None else {},
                "env_file": candidate.resolve(),
            }

    raise FileNotFoundError(
        f"No {ENV_NAME} under {prepare_dir}. "
        "Pass a prepare_dexjoco stamp, a step_XXXXX dir, or --env-file."
    )


def pick_env_file(
    layout: dict[str, Any],
    checkpoint_steps: list[int] | None,
) -> Path:
    env_file = layout.get("env_file")
    env_by_step: dict[Any, Path] = layout.get("env_by_step") or {}
    steps = [int(step) for step in (checkpoint_steps or layout.get("steps") or [])]
    if checkpoint_steps:
        if len(steps) != 1:
            raise ValueError("train one checkpoint step at a time; pass a single --checkpoint-steps")
        if steps[0] not in env_by_step:
            raise FileNotFoundError(f"No {ENV_NAME} for step {steps[0]}")
        return Path(env_by_step[steps[0]])
    if len(steps) > 1:
        chosen = max(steps)
        print(
            f"[train] prepare stamp has steps {steps}; using step_{chosen:06d}. "
            "Pass --checkpoint-steps to pick another.",
            flush=True,
        )
        return Path(env_by_step[chosen])
    if env_file is None:
        raise FileNotFoundError("Could not resolve protocol env file")
    return Path(env_file)


@dataclass(frozen=True)
class TrainRecipe:
    init: str
    hydra_task: str
    variant: str
    protocol: str
    output_dir: Path
    wandb_group: str
    tmux_session: str
    wandb_run_prefix: str
    hydra_base: str
    source_sha: str


def build_recipe(
    *,
    task_name: str,
    init: str,
    variant: str | None = None,
    protocol: str | None = None,
    output_dir: Path | None = None,
    wandb_group: str | None = None,
    tmux_session: str | None = None,
) -> TrainRecipe:
    init = str(init).strip().lower()
    if init == "lora":
        raise ValueError("LoRA recipes are removed. Use --init s0 or --init scratch.")
    if init not in {"s0", "scratch"}:
        raise ValueError(f"--init must be s0 or scratch, got {init}")
    _assert_not_lora("init", init)
    if init == "scratch":
        recipe = TrainRecipe(
            init="scratch",
            hydra_task=HYDRA_SCRATCH,
            variant=variant or "DEWO-scratch-joint",
            protocol=protocol or f"{task_name}_dewo_scratch_joint",
            output_dir=Path(output_dir or f"./runs/dexjoco_{task_name}_dewo_scratch_joint"),
            wandb_group=wandb_group or f"{task_name}_dewo_scratch_joint",
            tmux_session=tmux_session or f"{task_name}_dewo_scratch_joint",
            wandb_run_prefix=f"{task_name}_dewo_scratch",
            hydra_base="eval_every=0 resume=null",
            source_sha="scratch",
        )
    else:
        recipe = TrainRecipe(
            init="s0",
            hydra_task=HYDRA_S0,
            variant=variant or "B1-jump-fast-v9-uncond-adapter",
            protocol=protocol or f"{task_name}_dewo_v9_uncond_adapter_isolated",
            output_dir=Path(output_dir or f"./runs/dexjoco_{task_name}_dewo_v9"),
            wandb_group=wandb_group or f"{task_name}_dewo_v9_opensource",
            tmux_session=tmux_session or f"{task_name}_dewo_v9_uncond",
            wandb_run_prefix=f"{task_name}_dewo_v9",
            hydra_base="eval_every=0",
            source_sha="s0",
        )
    _assert_not_lora("DEWO_TASK", recipe.hydra_task)
    _assert_not_lora("DEWO_VARIANT", recipe.variant)
    _assert_not_lora("DEWO_PROTOCOL", recipe.protocol)
    return recipe


def build_hydra_overrides(
    recipe: TrainRecipe,
    args: argparse.Namespace,
) -> str:
    parts = [recipe.hydra_base]
    if args.learning_rate is not None:
        parts.append(f"learning_rate={args.learning_rate}")
    if args.max_steps is not None:
        parts.append(f"max_steps={args.max_steps}")
    if args.batch_size is not None:
        parts.append(f"batch_size={args.batch_size}")
    if args.primary_per_batch is not None:
        parts.append(f"role_balanced_sampling.primary_per_batch={args.primary_per_batch}")
    if recipe.init != "scratch":
        if args.adapter_rank is not None:
            parts.append(f"model.uncond_adapter.rank={args.adapter_rank}")
        if args.adapter_alpha is not None:
            parts.append(f"model.uncond_adapter.alpha={args.adapter_alpha}")
    if args.use_vae:
        parts.append("model.load_vae=false model.fill_vae_latent_cache=false")
    else:
        parts.append("model.load_vae=true model.fill_vae_latent_cache=false")
    extra = [item for item in (args.hydra_overrides or []) if item not in {"--", ""}]
    parts.extend(extra)
    return " ".join(part for part in parts if part).strip()


def apply_vae_policy(
    *,
    use_vae: bool,
    skip_preencode: bool | None,
    vae_cache: Path | None,
) -> dict[str, str]:
    if not use_vae:
        return {
            "USE_VAE_LATENT_CACHE": "0",
            "SKIP_VAE_PREENCODE": "1",
            "FILL_VAE_LATENT_CACHE": "0",
            "REQUIRE_VAE_LATENT_CACHE": "0",
        }
    cache_ready = vae_cache is not None and any(Path(vae_cache).glob("*.pt"))
    skip = skip_preencode if skip_preencode is not None else cache_ready
    values = {
        "USE_VAE_LATENT_CACHE": "1",
        "SKIP_VAE_PREENCODE": "1" if skip else "0",
        "FILL_VAE_LATENT_CACHE": "0",
        "REQUIRE_VAE_LATENT_CACHE": "1",
    }
    if vae_cache is not None:
        values["VAE_LATENT_CACHE_DIR"] = str(Path(vae_cache).resolve())
    return values


def resolve_init_weights(args: argparse.Namespace, exports: dict[str, str], recipe: TrainRecipe) -> Path:
    if args.init_weights is not None:
        return Path(args.init_weights).expanduser().resolve()
    if recipe.init == "scratch":
        return ACTION_DIT.resolve()
    for key in ("INIT_WEIGHTS", "CKPT", "SOURCE_CHECKPOINT"):
        raw = exports.get(key, "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
    raise FileNotFoundError("s0 train needs INIT_WEIGHTS in the protocol env or --init-weights")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task-name", required=True)
    parser.add_argument(
        "--init",
        choices=("s0", "scratch"),
        default="scratch",
        help="scratch = full FastWAMJoint (default). s0 = frozen S0 + v9 uncond adapter.",
    )
    parser.add_argument(
        "--prepare-dir",
        type=Path,
        default=None,
        help="prepare_dexjoco stamp, step dir, Eve root, or protocol dir.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="offline_v1_b1_jump_fast.env from prepare. Alternative to --prepare-dir.",
    )
    parser.add_argument("--checkpoint-steps", type=eval_entry._parse_int_list, default=None)
    parser.add_argument("--gpus", type=eval_entry._parse_int_list, default=[0, 1, 2, 3])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--init-weights", type=Path, default=None)
    parser.add_argument("--eve-manifest", type=Path, default=None)
    parser.add_argument("--eve-val-manifest", type=Path, default=None)
    parser.add_argument("--pretrained-norm-stats", type=Path, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optimizer steps (Hydra max_steps). Not the env episode horizon.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--primary-per-batch", type=int, default=None)
    parser.add_argument("--adapter-rank", type=int, default=None)
    parser.add_argument("--adapter-alpha", type=int, default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--tmux-session", default=None)
    parser.add_argument(
        "--tmux",
        action="store_true",
        help="Launch inside tmux (bash scripts/dewo_v2/train.sh default).",
    )
    parser.add_argument(
        "--inline",
        action="store_true",
        help="Run in this process (default for this Python entry).",
    )
    parser.add_argument(
        "--vae",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="use_vae",
        help="Read VAE latent cache (default). --no-vae encodes online.",
    )
    parser.add_argument(
        "--skip-vae-preencode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default: skip when the prepare VAE cache already has .pt files.",
    )
    parser.add_argument(
        "--dewo-version",
        default="v9.1",
        choices=("v9", "v9.1"),
        help="v9.1 = recoverability k/10 pool. v9 = legacy progress-return stitch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print recipe / hydra / env and exit without launching.",
    )
    parser.add_argument(
        "hydra_overrides",
        nargs=argparse.REMAINDER,
        help="Extra Hydra overrides after --.",
    )
    return parser


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.tmux and args.inline:
        raise ValueError("pass only one of --tmux / --inline")
    if args.env_file is None and args.prepare_dir is None:
        raise ValueError("pass --prepare-dir or --env-file")
    version = str(args.dewo_version).strip()
    if version not in {"v9", "v9.1"}:
        raise ValueError(f"DEWO version must be v9 or v9.1, got {args.dewo_version}")
    args.dewo_version = version
    for attr in ("variant", "protocol"):
        value = getattr(args, attr)
        if value and "lora" in str(value).lower():
            print(f"[train] ignoring LoRA --{attr.replace('_', '-')}={value}", flush=True)
            setattr(args, attr, None)
    if args.env_file is not None:
        args.env_file = Path(args.env_file).expanduser().resolve()
        args._layout = {
            "layout": "env_file",
            "prepare_dir": args.env_file.parent,
            "steps": [],
            "env_by_step": {},
            "env_file": args.env_file,
        }
    else:
        args.prepare_dir = Path(args.prepare_dir).expanduser()
        args._layout = resolve_prepare_layout(args.prepare_dir, args.checkpoint_steps)
        args.env_file = pick_env_file(args._layout, args.checkpoint_steps)
    if not args.env_file.is_file():
        raise FileNotFoundError(f"Missing protocol env: {args.env_file}")
    args._exports = strip_cfg_mix(prepare_entry._load_exports(args.env_file))
    if args.eve_manifest is not None:
        args._exports["EVE_MANIFEST_PATH"] = str(Path(args.eve_manifest).expanduser().resolve())
    if args.eve_val_manifest is not None:
        args._exports["EVE_VAL_MANIFEST_PATH"] = str(
            Path(args.eve_val_manifest).expanduser().resolve()
        )
    if args.pretrained_norm_stats is not None:
        args._exports["PRETRAINED_NORM_STATS"] = str(
            Path(args.pretrained_norm_stats).expanduser().resolve()
        )
    args._recipe = build_recipe(
        task_name=args.task_name,
        init=args.init,
        variant=args.variant,
        protocol=args.protocol,
        output_dir=args.output_dir,
        wandb_group=args.wandb_group,
        tmux_session=args.tmux_session,
    )
    args._init_weights = resolve_init_weights(args, args._exports, args._recipe)
    vae_cache = args._exports.get("VAE_LATENT_CACHE_DIR") or ""
    args._vae_cache = Path(vae_cache).expanduser() if vae_cache else None
    if args._vae_cache is None and args.use_vae:
        step_dir = args.env_file.parents[2] if len(args.env_file.parents) >= 3 else None
        if step_dir is not None:
            candidate = step_dir / "vae_latent_cache"
            if candidate.is_dir():
                args._vae_cache = candidate
    args._vae_policy = apply_vae_policy(
        use_vae=bool(args.use_vae),
        skip_preencode=args.skip_vae_preencode,
        vae_cache=args._vae_cache,
    )
    args._hydra = build_hydra_overrides(args._recipe, args)
    run_id = args.run_id or datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_{args._recipe.variant}")
    args._run_id = run_id
    args._wandb_run_name = f"{args._recipe.wandb_run_prefix}_{run_id}"
    args._run_inline = not args.tmux
    return args


def _require_file(path: Path, label: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _validate_args(args: argparse.Namespace) -> None:
    if not args.gpus:
        raise ValueError("--gpus must be a non-empty comma-separated list")
    exports = args._exports
    _require_file(args._init_weights, "init-weights")
    manifest = exports.get("EVE_MANIFEST_PATH")
    val_manifest = exports.get("EVE_VAL_MANIFEST_PATH")
    text_cache = exports.get("TEXT_EMBEDDING_CACHE_DIR")
    stats = exports.get("PRETRAINED_NORM_STATS") or str(JOINT_STATS)
    if not manifest:
        raise FileNotFoundError(f"{args.env_file} has no EVE_MANIFEST_PATH")
    if not val_manifest:
        raise FileNotFoundError(f"{args.env_file} has no EVE_VAL_MANIFEST_PATH")
    if not text_cache:
        raise FileNotFoundError(f"{args.env_file} has no TEXT_EMBEDDING_CACHE_DIR")
    _require_file(Path(manifest), "EVE_MANIFEST_PATH")
    _require_file(Path(val_manifest), "EVE_VAL_MANIFEST_PATH")
    if not Path(text_cache).is_dir():
        raise FileNotFoundError(f"Missing text cache directory: {text_cache}")
    if not Path(stats).is_file():
        raise FileNotFoundError(f"Missing PRETRAINED_NORM_STATS: {stats}")
    if args.use_vae and args._vae_policy.get("SKIP_VAE_PREENCODE") == "1":
        cache = Path(args._vae_policy.get("VAE_LATENT_CACHE_DIR") or "")
        if not cache.is_dir() or not any(cache.glob("*.pt")):
            raise FileNotFoundError(
                f"VAE cache is empty ({cache or '<unset>'}). "
                "Re-run prepare, pass --no-skip-vae-preencode, or --no-vae."
            )


def build_launch_env(args: argparse.Namespace) -> dict[str, str]:
    recipe: TrainRecipe = args._recipe
    exports = dict(args._exports)
    env = prepare_entry._base_env()
    env.update(exports)
    env.update(cfg_env())
    env.update(args._vae_policy)
    if not args.use_vae:
        env.pop("VAE_LATENT_CACHE_DIR", None)
    stats = exports.get("PRETRAINED_NORM_STATS") or str(JOINT_STATS.resolve())
    source_cfg = exports.get("FASTWAM_SOURCE_CONFIG") or str(JOINT_CFG.resolve())
    env.update(
        {
            "TASK": args.task_name,
            "DEWO_TASK_NAME": args.task_name,
            "GPUS": _gpu_csv(args.gpus),
            "CUDA_VISIBLE_DEVICES": _gpu_csv(args.gpus),
            "DEWO_VERSION": str(args.dewo_version),
            "DEWO_INIT": recipe.init,
            "DEWO_TASK": recipe.hydra_task,
            "DEWO_VARIANT": recipe.variant,
            "DEWO_PROTOCOL": recipe.protocol,
            "DEWO_OUTPUT_DIR": str(recipe.output_dir),
            "FITWAM_WANDB_GROUP": recipe.wandb_group,
            "TMUX_SESSION": recipe.tmux_session,
            "RUN_ID": args._run_id,
            "WANDB_RUN_NAME": args._wandb_run_name,
            "WANDB_MODE": args.wandb_mode or env.get("WANDB_MODE") or "offline",
            "DEWO_HYDRA_OVERRIDES": args._hydra,
            "INIT_WEIGHTS": str(args._init_weights),
            "SOURCE_CHECKPOINT": exports.get("SOURCE_CHECKPOINT") or str(args._init_weights),
            "PRETRAINED_NORM_STATS": stats,
            "STATS": stats,
            "FASTWAM_SOURCE_CONFIG": source_cfg,
            "SOURCE_CONFIG": source_cfg,
            "DEWO_SOURCE_SHA": recipe.source_sha,
            "RUN_INLINE": "1" if args._run_inline else "0",
            "VAE_ENCODE_VAL": env.get("VAE_ENCODE_VAL") or "false",
            "ENV_FILE": str(args.env_file),
        }
    )
    for compact in CFG_COMPACT_KEYS:
        env.pop(compact, None)
    return env


def _train_config_payload(args: argparse.Namespace) -> dict[str, Any]:
    recipe: TrainRecipe = args._recipe
    return {
        "task_name": args.task_name,
        "init": recipe.init,
        "dewo_version": str(args.dewo_version),
        "hydra_task": recipe.hydra_task,
        "variant": recipe.variant,
        "protocol": recipe.protocol,
        "prepare_dir": str(args.prepare_dir) if args.prepare_dir else None,
        "env_file": str(args.env_file),
        "layout": args._layout.get("layout"),
        "gpus": list(args.gpus),
        "init_weights": str(args._init_weights),
        "output_dir": str(recipe.output_dir),
        "run_id": args._run_id,
        "hydra_overrides": args._hydra,
        "vae_policy": args._vae_policy,
        "cfg": CfgRecipe().as_json(),
        "inline": args._run_inline,
        "started_at": _utc_now(),
    }


def main() -> None:
    args = _resolve_args(build_parser().parse_args())
    _validate_args(args)
    recipe: TrainRecipe = args._recipe
    payload = _train_config_payload(args)
    run_dir = Path(recipe.output_dir) / args._run_id
    print(f"[train] TASK={args.task_name} INIT={recipe.init} DEWO_VERSION={args.dewo_version} GPUS={_gpu_csv(args.gpus)}")
    print(f"[train] DEWO_TASK={recipe.hydra_task}")
    print(f"[train] env_file={args.env_file}")
    print(f"[train] INIT_WEIGHTS={args._init_weights}")
    print(f"[train] hydra={args._hydra}")
    print(
        "[train] cfg primary="
        f"{CfgRecipe().primary[0]}/{CfgRecipe().primary[1]}/{CfgRecipe().primary[2]} "
        f"aux_s={CfgRecipe().aux_success[0]}/{CfgRecipe().aux_success[1]}/{CfgRecipe().aux_success[2]} "
        f"aux_f={CfgRecipe().aux_fail[0]}/{CfgRecipe().aux_fail[1]}/{CfgRecipe().aux_fail[2]}"
    )
    print(
        f"[train] vae_cache={args._vae_policy.get('USE_VAE_LATENT_CACHE', '1')} "
        f"skip_preencode={args._vae_policy.get('SKIP_VAE_PREENCODE')} "
        f"cache={args._vae_policy.get('VAE_LATENT_CACHE_DIR', '<none>')}"
    )
    print(f"[train] run_dir={run_dir} inline={args._run_inline} tmux={recipe.tmux_session}")
    if args.dry_run:
        print("[train] dry-run; not launching", flush=True)
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    eval_entry._write_json(run_dir / "train_config.json", payload)
    if not TRAIN_RUN.is_file():
        raise FileNotFoundError(f"Missing trainer worker: {TRAIN_RUN}")
    launch_env = build_launch_env(args)
    printable = " ".join(shlex.quote(part) for part in ["bash", str(TRAIN_RUN)])
    print(f"[train] exec {printable}", flush=True)
    os.execvpe("bash", ["bash", str(TRAIN_RUN)], launch_env)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
