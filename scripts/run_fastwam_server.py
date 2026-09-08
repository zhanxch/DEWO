#!/usr/bin/env python3
"""Launch a FastWAM inference policy server (ZMQ API).

Policy loading and ``infer_action`` live in ``fastwam.inference``.
This script is only the ZMQ process wrapper.
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastwam_policy_server import DEFAULT_SERVER_PORT, PolicyServer
from fastwam.inference.loader import (
    resolve_checkpoint_path as _resolve_checkpoint_path,
    resolve_inference_horizons as _resolve_inference_horizons,
    resolve_normalization_binding as _resolve_normalization_binding,
    resolve_run_dir as _resolve_run_dir,
    sha256_file as _sha256_file,
)
from fastwam.inference.policy import FastWAMPolicy, load_policy_from_run as _build_policy_from_run
from hydra.utils import instantiate  # noqa: F401  # tests patch this name on the module


class MockFastWAMPolicy:
    def __init__(self) -> None:
        self._reset_count = 0

    def get_action(self, observation: dict, options: dict | None = None) -> dict:
        del observation, options
        import numpy as np

        return {"action": np.zeros(7, dtype=np.float32)}, {"mock": True}

    def reset(self, options: dict | None = None) -> dict:
        del options
        self._reset_count += 1
        return {"reset_count": self._reset_count}

    def get_modality_config(self) -> dict:
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FastWAM ZMQ policy server.")
    parser.add_argument("--mock", action="store_true", help="Start mock policy (no model load).")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    normalization = parser.add_mutually_exclusive_group()
    normalization.add_argument("--dataset-stats-path", type=str, default=None)
    normalization.add_argument("--norm-stats-meta-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument(
        "--inference-seed",
        type=int,
        default=None,
        help="Override EVALUATION.seed for deterministic diffusion sampling.",
    )
    parser.add_argument(
        "--text-cfg-scale",
        type=float,
        default=None,
        help=(
            "Action CFG mix weight w in ε_base + w(ε_posi-ε_base). "
            "0=本体 (base prompt, adapter off), 1=纯优势 (success + adapter), "
            ">1=CFG guide (e.g. 2)."
        ),
    )
    parser.add_argument(
        "--adaptive-cfg-tau",
        type=float,
        default=None,
        help=(
            "If set, freeze mix from NFE0 exec RMS: E>tau uses --text-cfg-scale, "
            "else mix w=0 (本体). Requires text_cfg_scale != 0."
        ),
    )
    parser.add_argument(
        "--cfg-epsilon-l",
        "--epsilon-l",
        "--cfg-residual-epsilon",
        dest="cfg_epsilon_l",
        type=float,
        default=None,
        help=(
            "Bound the per-token action CFG residual before text-cfg scaling. "
            "None keeps legacy unbounded guidance; 0 is the base branch."
        ),
    )
    parser.add_argument(
        "--cfg-residual-clip-mode",
        choices=("rms", "elementwise"),
        default=None,
        help="How --cfg-epsilon-l bounds the residual (default: rms).",
    )
    parser.add_argument(
        "--cfg-exec-horizon",
        type=int,
        default=None,
        help="Action steps used for CFG residual energy (independent of replan_steps).",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Base prompt for prompt-mode CFG. Cached-context clients send their base context per request.",
    )
    parser.add_argument(
        "--failure-prompt",
        type=str,
        default=None,
        help="Failure-conditioned prompt for DEWO v7 CFG. Cached-context clients send failure context per request.",
    )
    parser.add_argument(
        "--load-text-encoder",
        dest="load_text_encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load text encoder/tokenizer so get_action can accept prompt strings.",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=str,
        default=None,
        help="Frozen base MoT for DEWO v5 CFG (adapter-off branch).",
    )
    parser.add_argument(
        "--uncond-adapter",
        type=str,
        default=None,
        help="DEWO v5 uncond-adapter weights. If omitted, --checkpoint may be the adapter file.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--api-token", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Starting FastWAM inference server...", flush=True)
    print(f"  Host: {args.host}", flush=True)
    print(f"  Port: {args.port}", flush=True)

    if args.mock:
        policy = MockFastWAMPolicy()
        print("  Policy: mock (no checkpoint)", flush=True)
    else:
        if not args.run_dir or not args.checkpoint:
            raise ValueError("--run-dir and --checkpoint are required unless --mock is set.")
        run_dir = Path(args.run_dir).expanduser().resolve()
        policy = _build_policy_from_run(
            run_dir=run_dir,
            checkpoint=args.checkpoint,
            dataset_stats_path=args.dataset_stats_path,
            norm_stats_meta_dir=args.norm_stats_meta_dir,
            device=args.device,
            action_horizon=args.action_horizon,
            num_inference_steps=args.num_inference_steps,
            load_text_encoder=args.load_text_encoder,
            inference_seed=args.inference_seed,
            text_cfg_scale=args.text_cfg_scale,
            negative_prompt=args.negative_prompt,
            failure_prompt=getattr(args, "failure_prompt", None),
            backbone_checkpoint=args.backbone_checkpoint,
            uncond_adapter=args.uncond_adapter,
            adaptive_cfg_tau=args.adaptive_cfg_tau,
            cfg_epsilon_l=args.cfg_epsilon_l,
            cfg_residual_clip_mode=args.cfg_residual_clip_mode,
            cfg_exec_horizon=args.cfg_exec_horizon,
        )
        print(f"  Run dir: {_resolve_run_dir(run_dir)}", flush=True)
        print(f"  Device: {args.device}", flush=True)

    server = PolicyServer(policy=policy, host=args.host, port=args.port, api_token=args.api_token)
    print(f"\n✓ Server ready — listening on {args.host}:{args.port}\n", flush=True)

    def _terminate(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _terminate)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...", flush=True)


if __name__ == "__main__":
    main()
