"""In-process DexJoCo policy: env obs → FastWAMPolicy → denormalized rotvec chunk."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fastwam.inference.config import InferenceConfig
from fastwam.inference.loader import resolve_run_dir
from fastwam.inference.obs import (
    CAMERA_MAPPING,
    DexJoCoFastWAMAdapter,
    fastwam_action_to_dexjoco,
    load_dexjoco_eval_settings,
    load_text_embedding_file,
)
from fastwam.inference.policy import FastWAMPolicy, load_policy_from_run


def _resolve_instruction(*, prompt: str | None, task_name: str | None) -> str:
    if prompt:
        return str(prompt)
    if task_name:
        try:
            from dewo_v2.tasks import get_task
        except ImportError as exc:
            raise ImportError(
                "task_name= requires scripts/dewo_v2/tasks.py on PYTHONPATH, "
                "or pass prompt= explicitly."
            ) from exc
        return get_task(str(task_name)).success_prompt
    raise ValueError("FastWAMDexJocoPolicy needs prompt= or task_name=")


def _maybe_cfg_prompts(
    *,
    instruction: str,
    cfg_base_prompt: str | None,
    cfg_failure_prompt: str | None,
    wants_cfg: bool,
) -> tuple[str, str | None, str | None]:
    """Return (success_prompt, base_prompt, failure_prompt)."""
    if cfg_base_prompt is not None or cfg_failure_prompt is not None:
        return instruction, cfg_base_prompt, cfg_failure_prompt
    if not wants_cfg:
        return instruction, None, None
    return (
        f"{instruction} Successful execution.",
        instruction,
        f"{instruction} Failed execution.",
    )


class FastWAMDexJocoPolicy:
    """Closed-loop DexJoCo helper used by collect/scan/eval.

    ``infer`` returns a denormalized ``(H, 22)`` rotvec chunk. Convert to
    DexJoCo quat actions with :func:`fastwam_action_to_dexjoco`.
    """

    def __init__(
        self,
        model_config: str | Path,
        checkpoint: str | Path,
        dataset_stats: str | Path,
        text_embedding: str | Path | None = None,
        text_embedding_base: str | Path | None = None,
        text_embedding_failure: str | Path | None = None,
        device: str = "cuda:0",
        action_horizon: int = 32,
        replan_steps: int = 24,
        value_replan_steps: int | None = None,
        num_inference_steps: int = 10,
        prompt: str | None = None,
        task_name: str | None = None,
        cfg_base_prompt: str | None = None,
        cfg_failure_prompt: str | None = None,
        load_text_encoder: bool | None = None,
        inference_config: InferenceConfig | None = None,
        **load_kwargs: Any,
    ) -> None:
        instruction = _resolve_instruction(prompt=prompt, task_name=task_name)
        cfg = inference_config or InferenceConfig(
            action_horizon=int(action_horizon),
            replan_steps=int(replan_steps),
            value_replan_steps=value_replan_steps,
            num_inference_steps=int(num_inference_steps),
        )
        self.prompt, self.cfg_base_prompt, self.cfg_failure_prompt = _maybe_cfg_prompts(
            instruction=instruction,
            cfg_base_prompt=cfg_base_prompt,
            cfg_failure_prompt=cfg_failure_prompt,
            wants_cfg=cfg.wants_cfg_mix(),
        )
        run_dir = resolve_run_dir(model_config)
        use_cached_text = text_embedding is not None
        if load_text_encoder is None:
            load_text_encoder = not use_cached_text
        self.inner: FastWAMPolicy = load_policy_from_run(
            run_dir=run_dir,
            checkpoint=str(Path(checkpoint).expanduser().resolve()),
            dataset_stats_path=str(Path(dataset_stats).expanduser().resolve()),
            norm_stats_meta_dir=None,
            device=str(device),
            action_horizon=cfg.action_horizon,
            num_inference_steps=cfg.num_inference_steps,
            load_text_encoder=bool(load_text_encoder),
            inference_seed=cfg.seed,
            text_cfg_scale=cfg.text_cfg_scale,
            negative_prompt=cfg.negative_prompt,
            failure_prompt=cfg.failure_prompt,
            inference_config=cfg,
            **load_kwargs,
        )
        settings = load_dexjoco_eval_settings(
            run_dir, action_horizon_override=cfg.action_horizon
        )
        settings["load_text_encoder"] = bool(load_text_encoder) and not use_cached_text
        self.adapter = DexJoCoFastWAMAdapter(settings)
        if text_embedding is not None:
            context, mask = load_text_embedding_file(text_embedding)
            self.adapter.bind_text_embedding(context, mask, kind="success")
        if text_embedding_base is not None:
            context, mask = load_text_embedding_file(text_embedding_base)
            self.adapter.bind_text_embedding(context, mask, kind="base")
        if text_embedding_failure is not None:
            context, mask = load_text_embedding_file(text_embedding_failure)
            self.adapter.bind_text_embedding(context, mask, kind="failure")
        self.action_horizon = int(self.inner.action_horizon)
        self.replan_steps = int(cfg.replan_steps)
        self.value_replan_steps = int(cfg.resolved_value_replan_steps)
        self.inference_config = cfg

    def _policy_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self.adapter.env_obs_to_policy_obs(
            obs,
            camera_key="front",
            camera_mapping=CAMERA_MAPPING,
            task_prompt=self.prompt,
            cfg_base_prompt=self.cfg_base_prompt,
            cfg_failure_prompt=self.cfg_failure_prompt,
        )

    def infer(
        self,
        obs: dict[str, Any],
        noise_seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> np.ndarray:
        payload = self.infer_with_extras(obs, noise_seed=noise_seed, options=options)
        return np.asarray(payload["action"], dtype=np.float32)

    def infer_with_extras(
        self,
        obs: dict[str, Any],
        noise_seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        policy_obs = self._policy_obs(obs)
        merged: dict[str, Any] = {} if options is None else dict(options)
        if noise_seed is not None:
            merged["seed"] = int(noise_seed)
        payload = self.inner.get_action(policy_obs, options=merged or None)
        chunk = self.adapter.parse_policy_response(payload)
        payload["action"] = np.asarray(chunk, dtype=np.float32)
        return payload

    def infer_value(
        self,
        obs: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.inner.infer_value(self._policy_obs(obs), options=options)
