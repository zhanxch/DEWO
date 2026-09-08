#!/usr/bin/env python3
"""Run one FastWAM closed-loop DexJoCo episode and save MP4 + JSON.

Works with configs/model/{fastwam,fastwam_joint,fastwam_idm} run dirs.
Pass DEWO CFG / value knobs when the checkpoint implements them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fastwam.inference.config import InferenceConfig
from fastwam.inference.dexjoco import FastWAMDexJocoPolicy
from fastwam.inference.rollout import rollout_episode, save_rollout_mp4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--text-embedding", type=Path, default=None)
    parser.add_argument("--text-embedding-base", type=Path, default=None)
    parser.add_argument("--text-embedding-failure", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument(
        "--value-replan-steps",
        type=int,
        default=None,
        help="Independent value-head query stride. Default: same as --replan-steps.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--text-cfg-scale", type=float, default=0.0)
    parser.add_argument("--cfg-exec-horizon", type=int, default=24)
    parser.add_argument("--adaptive-cfg-tau", type=float, default=None)
    parser.add_argument("--cfg-epsilon-l", type=float, default=None)
    parser.add_argument("--cfg-residual-clip-mode", default="rms")
    parser.add_argument("--cfg-gate-mode", default="off")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--output-video", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--load-text-encoder", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = build_parser().parse_args()
    output_video = args.output_video or (
        REPO_ROOT
        / "evaluate_results"
        / "dexjoco"
        / args.task_name
        / f"seed_{args.seed:03d}_repeat_{args.repeat}.mp4"
    )
    output_json = args.output_json or output_video.with_suffix(".json")

    infer_cfg = InferenceConfig(
        action_horizon=args.action_horizon,
        replan_steps=args.replan_steps,
        value_replan_steps=args.value_replan_steps,
        num_inference_steps=args.num_inference_steps,
        text_cfg_scale=args.text_cfg_scale,
        cfg_exec_horizon=args.cfg_exec_horizon,
        adaptive_cfg_tau=args.adaptive_cfg_tau,
        cfg_epsilon_l=args.cfg_epsilon_l,
        cfg_residual_clip_mode=args.cfg_residual_clip_mode,
        cfg_gate_mode=args.cfg_gate_mode,
    )
    policy = FastWAMDexJocoPolicy(
        model_config=args.run_dir,
        checkpoint=args.checkpoint,
        dataset_stats=args.dataset_stats,
        text_embedding=args.text_embedding,
        text_embedding_base=args.text_embedding_base,
        text_embedding_failure=args.text_embedding_failure,
        device=args.device,
        action_horizon=args.action_horizon,
        replan_steps=args.replan_steps,
        value_replan_steps=args.value_replan_steps,
        num_inference_steps=args.num_inference_steps,
        task_name=args.task_name,
        load_text_encoder=args.load_text_encoder,
        inference_config=infer_cfg,
    )

    from dexjoco.tasks import CONFIG_MAPPING

    if args.task_name not in CONFIG_MAPPING:
        raise KeyError(
            f"Unknown DexJoco task {args.task_name!r}; available: {sorted(CONFIG_MAPPING)}"
        )
    env = CONFIG_MAPPING[args.task_name]().get_environment(
        policy_mode=True,
        render_mode="rgb_array",
        randomize=False,
        seed=args.seed,
        randomize_dynamics=False,
    )
    try:
        obs, _ = env.reset()
        row, frames = rollout_episode(
            env=env,
            initial_obs=obs,
            policy=policy,
            seed=args.seed,
            repeat=args.repeat,
            max_steps=args.max_steps,
        )
    finally:
        env.close()

    save_rollout_mp4(frames, output_video, fps=args.video_fps)
    row.update(
        {
            "task_name": args.task_name,
            "task_instruction": policy.prompt,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "run_dir": str(Path(args.run_dir).resolve()),
            "text_cfg_scale": float(args.text_cfg_scale),
            "video": str(output_video.resolve()),
        }
    )
    row.pop("cfg_values", None)
    row.pop("cfg_value_rels", None)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
