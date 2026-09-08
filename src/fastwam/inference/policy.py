"""Architecture-agnostic FastWAM policy: load, infer_action, optional value/CFG."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.inference.config import InferenceConfig
from fastwam.inference.contract import (
    CFG_OPTION_KEYS,
    CFG_VALUE_KEYS,
    KEY_ACTION,
    KEY_CONTEXT,
    KEY_CONTEXT_MASK,
    KEY_FAILURE_CONTEXT,
    KEY_FAILURE_CONTEXT_MASK,
    KEY_FAILURE_PROMPT,
    KEY_INPUT_IMAGE,
    KEY_NEGATIVE_CONTEXT,
    KEY_NEGATIVE_CONTEXT_MASK,
    KEY_NEGATIVE_PROMPT,
    KEY_PROMPT,
    KEY_PROPRIO,
    to_inference_tensors,
    validate_policy_observation,
)
from fastwam.inference.kwargs import filter_infer_action_kwargs, supports_cfg_mix
from fastwam.inference.loader import (
    resolve_checkpoint_path,
    resolve_inference_horizons,
    resolve_normalization_binding,
    resolve_num_video_frames_from_cfg,
    resolve_run_dir,
    sha256_file,
)


class FastWAMPolicy:
    """Training-aligned policy. Observation keys are defined in ``contract.py``.

    ``get_action`` calls ``model.infer_action`` and drops kwargs the current
    architecture does not accept (Joint / IDM vs FastWAM+DEWO).
    ``infer_value`` queries ``value_head`` without running action diffusion.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        device: str,
        action_horizon: int,
        num_inference_steps: int,
        num_video_frames: int | None = None,
        text_cfg_scale: float = 0.0,
        negative_prompt: str | None = None,
        failure_prompt: str | None = None,
        sigma_shift: float | None = None,
        seed: int | None = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        adaptive_cfg_tau: float | None = None,
        cfg_exec_horizon: int = 24,
        cfg_epsilon_l: float | None = None,
        cfg_residual_clip_mode: str = "rms",
        model_provenance: dict[str, Any] | None = None,
        inference_config: InferenceConfig | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        cfg = inference_config
        self.action_horizon = int(action_horizon if cfg is None else cfg.action_horizon)
        self.num_inference_steps = int(
            num_inference_steps if cfg is None else cfg.num_inference_steps
        )
        self.num_video_frames = None if num_video_frames is None else int(num_video_frames)
        if cfg is not None and cfg.num_video_frames is not None:
            self.num_video_frames = int(cfg.num_video_frames)
        self.text_cfg_scale = float(text_cfg_scale if cfg is None else cfg.text_cfg_scale)
        self.negative_prompt = (
            None
            if (cfg.negative_prompt if cfg is not None else negative_prompt) in (None, "")
            else str(cfg.negative_prompt if cfg is not None else negative_prompt)
        )
        self.failure_prompt = (
            None
            if (cfg.failure_prompt if cfg is not None else failure_prompt) in (None, "")
            else str(cfg.failure_prompt if cfg is not None else failure_prompt)
        )
        self.sigma_shift = sigma_shift if cfg is None else cfg.sigma_shift
        self.seed = seed if cfg is None else cfg.seed
        self.rand_device = str(rand_device if cfg is None else cfg.rand_device)
        self.tiled = bool(tiled if cfg is None else cfg.tiled)
        if cfg is not None:
            self.adaptive_cfg_tau = cfg.adaptive_cfg_tau
        else:
            self.adaptive_cfg_tau = (
                None if adaptive_cfg_tau is None else float(adaptive_cfg_tau)
            )
        self.cfg_exec_horizon = int(cfg_exec_horizon if cfg is None else cfg.cfg_exec_horizon)
        self.cfg_epsilon_l = (
            None if cfg_epsilon_l is None else float(cfg_epsilon_l)
        ) if cfg is None else cfg.cfg_epsilon_l
        self.cfg_residual_clip_mode = str(
            cfg_residual_clip_mode if cfg is None else cfg.cfg_residual_clip_mode
        )
        self.inference_config = cfg
        self.model_provenance = dict(model_provenance or {})
        self._episode = 0
        infer_fn = getattr(self.model, "infer_action", None)
        self._infer_parameters = (
            inspect.signature(infer_fn).parameters if callable(infer_fn) else {}
        )

    def get_modality_config(self) -> dict:
        return {
            "shape_meta": OmegaConf.to_container(self.processor.shape_meta, resolve=True),
            "model_provenance": self.model_provenance,
        }

    def reset(self, options: dict | None = None) -> dict:
        del options
        self._episode += 1
        return {"episode": self._episode}

    def _select_modality_value(self, value: Any, meta_name: str, field_name: str) -> Any:
        if not isinstance(value, dict):
            return value
        meta = self.processor.shape_meta.get(meta_name, [])
        preferred_keys = [str(item["key"]) for item in meta if "key" in item]
        for key in preferred_keys:
            if key in value:
                return value[key]
        if len(value) == 1:
            return next(iter(value.values()))
        raise ValueError(
            f"observation['{field_name}'] is a nested dict with keys {sorted(value.keys())}; "
            f"expected one of training keys {preferred_keys}."
        )

    def _slice_proprio_to_state_keys(self, proprio: torch.Tensor) -> dict:
        state_meta = self.processor.shape_meta["state"]
        state_batch = {}
        for meta in state_meta:
            key = meta["key"]
            sl = meta.get("modality_slice")
            if sl is not None:
                state_batch[key] = proprio[..., sl[0] : sl[1]]
            else:
                state_batch[key] = proprio
        return state_batch

    def _merge_state_keys_to_proprio(self, state_batch: dict) -> torch.Tensor:
        merger = self.processor.action_state_merger
        out = merger.forward({"state": state_batch})
        return out["state"]

    def _normalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        state_batch = {"state": self._slice_proprio_to_state_keys(proprio)}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return self._merge_state_keys_to_proprio(state_batch["state"])

    def _normalize_observation(self, observation: dict) -> dict:
        normalized = dict(observation)
        if KEY_INPUT_IMAGE in normalized:
            normalized[KEY_INPUT_IMAGE] = self._select_modality_value(
                normalized[KEY_INPUT_IMAGE],
                "images",
                KEY_INPUT_IMAGE,
            )
        if KEY_PROPRIO in normalized:
            normalized[KEY_PROPRIO] = self._select_modality_value(
                normalized[KEY_PROPRIO],
                "state",
                KEY_PROPRIO,
            )
        return normalized

    def _prepare_tensors(self, observation: dict) -> dict[str, Any]:
        observation = self._normalize_observation(observation)
        validate_policy_observation(observation)
        tensors = to_inference_tensors(
            observation,
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        proprio = tensors.get(KEY_PROPRIO)
        if proprio is not None:
            # Match FastWAM-infer-in-DexJoco: z-score in float32, then cast.
            proprio = self._normalize_proprio(proprio.to(dtype=torch.float32))
            self._last_normalized_proprio = proprio.detach().to(
                dtype=torch.float32, device="cpu"
            )
            tensors[KEY_PROPRIO] = proprio.to(
                device=self.model.device, dtype=self.model.torch_dtype
            )
        return tensors

    def _collect_options(self, options: dict[str, Any] | None) -> dict[str, Any]:
        infer_seed = self.seed
        return_cfg_residual = False
        cfg_exec_horizon = int(self.cfg_exec_horizon)
        adaptive_cfg_tau = self.adaptive_cfg_tau
        cfg_epsilon_l = self.cfg_epsilon_l
        cfg_residual_clip_mode = self.cfg_residual_clip_mode
        forwarded: dict[str, Any] = {}
        if options:
            raw_seed = options.get("seed", options.get("inference_seed"))
            if raw_seed is not None:
                infer_seed = int(raw_seed)
            return_cfg_residual = bool(options.get("return_cfg_residual", False))
            if options.get("cfg_exec_horizon") is not None:
                cfg_exec_horizon = int(options["cfg_exec_horizon"])
            if options.get("adaptive_cfg_tau") is not None:
                adaptive_cfg_tau = float(options["adaptive_cfg_tau"])
            for key in ("cfg_epsilon_l", "epsilon_l", "cfg_residual_epsilon"):
                if options.get(key) is not None:
                    cfg_epsilon_l = float(options[key])
                    break
            if options.get("cfg_residual_clip_mode") is not None:
                cfg_residual_clip_mode = str(options["cfg_residual_clip_mode"])
            for key in CFG_OPTION_KEYS:
                if key in options and options[key] is not None:
                    forwarded[key] = options[key]
        cfg = self.inference_config
        if cfg is not None:
            if forwarded.get("cfg_gate_mode") is None and cfg.cfg_gate_mode not in {"", "off"}:
                forwarded["cfg_gate_mode"] = cfg.cfg_gate_mode
            forwarded.setdefault("cfg_v_high", cfg.cfg_v_high)
            forwarded.setdefault("cfg_drop_delta", cfg.cfg_drop_delta)
            forwarded.setdefault("cfg_growth_tau", cfg.cfg_growth_tau)
            forwarded.setdefault("cfg_growth_start_replan", cfg.cfg_growth_start_replan)
            forwarded.setdefault("cfg_growth_stop_replan", cfg.cfg_growth_stop_replan)
            forwarded.setdefault("cfg_low_value_threshold", cfg.cfg_low_value_threshold)
            forwarded.setdefault("cfg_growth_delta", cfg.cfg_growth_delta)
        return {
            "seed": infer_seed,
            "return_cfg_residual": return_cfg_residual,
            "cfg_exec_horizon": cfg_exec_horizon,
            "adaptive_cfg_tau": adaptive_cfg_tau,
            "cfg_epsilon_l": cfg_epsilon_l,
            "cfg_residual_clip_mode": cfg_residual_clip_mode,
            "forwarded": forwarded,
        }

    def _build_infer_kwargs(
        self,
        tensors: dict[str, Any],
        *,
        options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        collected = self._collect_options(options)
        infer_kwargs: dict[str, Any] = {
            KEY_INPUT_IMAGE: tensors[KEY_INPUT_IMAGE],
            "action_horizon": self.action_horizon,
            KEY_PROPRIO: tensors.get(KEY_PROPRIO),
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": collected["seed"],
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        if self.num_video_frames is not None:
            infer_kwargs["num_video_frames"] = int(self.num_video_frames)
        if collected["return_cfg_residual"]:
            infer_kwargs["return_cfg_residual"] = True
        if collected["adaptive_cfg_tau"] is not None:
            infer_kwargs["adaptive_cfg_tau"] = float(collected["adaptive_cfg_tau"])
        if collected["cfg_epsilon_l"] is not None:
            infer_kwargs["cfg_epsilon_l"] = float(collected["cfg_epsilon_l"])
            infer_kwargs["cfg_residual_clip_mode"] = str(collected["cfg_residual_clip_mode"])
        infer_kwargs["cfg_exec_horizon"] = int(collected["cfg_exec_horizon"])
        infer_kwargs.update(collected["forwarded"])

        if KEY_CONTEXT in tensors:
            infer_kwargs[KEY_CONTEXT] = tensors[KEY_CONTEXT]
            infer_kwargs[KEY_CONTEXT_MASK] = tensors[KEY_CONTEXT_MASK]
            infer_kwargs[KEY_PROMPT] = None
            if KEY_NEGATIVE_CONTEXT in tensors:
                infer_kwargs[KEY_NEGATIVE_CONTEXT] = tensors[KEY_NEGATIVE_CONTEXT]
                infer_kwargs[KEY_NEGATIVE_CONTEXT_MASK] = tensors[KEY_NEGATIVE_CONTEXT_MASK]
            if KEY_FAILURE_CONTEXT in tensors:
                infer_kwargs[KEY_FAILURE_CONTEXT] = tensors[KEY_FAILURE_CONTEXT]
                infer_kwargs[KEY_FAILURE_CONTEXT_MASK] = tensors[KEY_FAILURE_CONTEXT_MASK]
        else:
            infer_kwargs[KEY_PROMPT] = tensors[KEY_PROMPT]
            infer_kwargs[KEY_NEGATIVE_PROMPT] = tensors.get(
                KEY_NEGATIVE_PROMPT, self.negative_prompt
            )
            infer_kwargs["failure_prompt"] = tensors.get(
                KEY_FAILURE_PROMPT, self.failure_prompt
            )

        require_cfg = float(self.text_cfg_scale) != 0.0 or bool(
            collected["forwarded"].get("cfg_gate_mode")
            not in (None, "", "off")
        )
        return filter_infer_action_kwargs(
            self.model.infer_action,
            infer_kwargs,
            require_cfg_mix=require_cfg,
        )

    def _pack_prediction(self, pred: dict[str, Any], *, action_np: np.ndarray) -> dict[str, Any]:
        payload: dict[str, Any] = {KEY_ACTION: action_np, "action_horizon": self.action_horizon}
        for key in CFG_VALUE_KEYS:
            if key not in pred:
                continue
            value = pred[key]
            if isinstance(value, torch.Tensor):
                payload[key] = value.detach().to(device="cpu", dtype=torch.float32).numpy()
            else:
                payload[key] = np.asarray(value, dtype=np.float32)
        return payload

    def get_action(self, observation: dict, options: dict | None = None) -> dict[str, Any]:
        tensors = self._prepare_tensors(observation)
        infer_kwargs = self._build_infer_kwargs(tensors, options=options)
        try:
            with torch.no_grad():
                pred = self.model.infer_action(**infer_kwargs)
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        action_tensor = pred[KEY_ACTION]
        if action_tensor.ndim == 2:
            action_tensor = action_tensor.unsqueeze(0)
        action_np = self._denormalize_action(action_tensor)
        return self._pack_prediction(pred, action_np=action_np)

    def infer_value(self, observation: dict, options: dict | None = None) -> dict[str, Any]:
        """Query ``value_head`` without action diffusion.

        Used when ``value_replan_steps`` is independent of action replan.
        Joint/IDM models with an attached value head are supported.
        """
        del options
        value_head = getattr(self.model, "value_head", None)
        if value_head is None:
            return {}
        with torch.no_grad():
            tensors = self._prepare_tensors(observation)
            input_image = tensors[KEY_INPUT_IMAGE]
            first_frame_latents = self.model._encode_input_image_latents_tensor(
                input_image=input_image, tiled=self.tiled
            )
            encoder = str(getattr(self.model, "value_head_encoder", "vae_latents") or "vae_latents")
            if encoder == "vae_latents":
                value = float(value_head(first_frame_latents).reshape(-1)[0].item())
                return {"cfg_value": np.float32(value)}

            context, context_mask = self._value_context(tensors)
            fuse_flag = bool(
                getattr(self.model.video_expert, "fuse_vae_embedding_in_latents", False)
            )
            value_tokens = self.model._encode_value_video_tokens(
                first_frame_latents,
                context,
                context_mask,
                fuse_flag,
            )
            value = float(value_head(value_tokens).reshape(-1)[0].item())
            return {"cfg_value": np.float32(value)}

    def _value_context(self, tensors: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if KEY_NEGATIVE_CONTEXT in tensors:
            context, mask = tensors[KEY_NEGATIVE_CONTEXT], tensors[KEY_NEGATIVE_CONTEXT_MASK]
        elif KEY_CONTEXT in tensors:
            context, mask = tensors[KEY_CONTEXT], tensors[KEY_CONTEXT_MASK]
        else:
            prompt = tensors.get(KEY_NEGATIVE_PROMPT) or tensors[KEY_PROMPT]
            context, mask = self.model.encode_prompt(str(prompt))
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        proprio = tensors.get(KEY_PROPRIO)
        if proprio is not None and getattr(self.model, "proprio_encoder", None) is not None:
            context, mask = self.model._append_proprio_to_context(
                context=context,
                context_mask=mask,
                proprio=proprio,
            )
        return context, mask

    def _denormalize_action(self, action_tensor: torch.Tensor) -> np.ndarray:
        action_meta = self.processor.shape_meta["action"]
        has_transforms = self.processor.action_state_transforms is not None
        action_tensor = action_tensor.to(dtype=torch.float32, device="cpu")
        if len(action_meta) == 1 and not has_transforms:
            action_key = action_meta[0]["key"]
            normalizer = self.processor.normalizer.normalizers["action"][action_key]
            return normalizer.backward(action_tensor).numpy()[0]

        proprio = getattr(self, "_last_normalized_proprio", None)
        if proprio is None:
            proprio = torch.zeros(
                1,
                self.processor.num_obs_steps,
                action_tensor.shape[-1],
                dtype=action_tensor.dtype,
                device=action_tensor.device,
            )
        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(0)
        data = {"action": action_tensor, "proprio": proprio}
        data["state"] = data.pop("proprio")
        data = self.processor.action_state_merger.backward(data)
        data = self.processor.normalizer.backward(data)
        if self.processor.action_state_transforms is not None:
            for trans in reversed(self.processor.action_state_transforms):
                data = trans.backward(data)
        action_flat = torch.cat([data["action"][m["key"]] for m in action_meta], dim=-1)
        return action_flat[0].numpy()


def load_policy_from_run(
    run_dir: Path,
    checkpoint: str,
    dataset_stats_path: str | None,
    norm_stats_meta_dir: str | None,
    device: str,
    action_horizon: int | None,
    num_inference_steps: int | None,
    load_text_encoder: bool,
    inference_seed: int | None = None,
    text_cfg_scale: float | None = None,
    negative_prompt: str | None = None,
    failure_prompt: str | None = None,
    backbone_checkpoint: str | None = None,
    uncond_adapter: str | None = None,
    adaptive_cfg_tau: float | None = None,
    cfg_epsilon_l: float | None = None,
    cfg_residual_clip_mode: str | None = None,
    cfg_exec_horizon: int | None = None,
    inference_config: InferenceConfig | None = None,
) -> FastWAMPolicy:
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision

    run_dir = resolve_run_dir(run_dir)
    checkpoint_path = resolve_checkpoint_path(run_dir, checkpoint)
    config_path = run_dir / "config.yaml"

    cfg = OmegaConf.load(config_path)
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg.load_text_encoder = bool(load_text_encoder)
    model_cfg.load_vae = True
    model_cfg.skip_dit_load_from_pretrain = True
    model_cfg.action_dit_pretrained_path = None
    mixed_precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    processor_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.data.train.processor, resolve=True)
    )
    normalization_kind, normalization_path = resolve_normalization_binding(
        processor_cfg,
        run_dir=run_dir,
        dataset_stats_path=dataset_stats_path,
        norm_stats_meta_dir=norm_stats_meta_dir,
    )

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU.", flush=True)
        device = "cpu"

    print(f"  Config: {config_path}", flush=True)
    print(f"  Checkpoint: {checkpoint_path}", flush=True)
    print(f"  Load text encoder: {model_cfg.load_text_encoder}", flush=True)
    print(f"  Load VAE: {model_cfg.load_vae} (forced true for live image encode)", flush=True)
    print(f"Loading FastWAM model on {device} ...", flush=True)
    model = instantiate(model_cfg, model_dtype=model_dtype, device=device)
    if getattr(model, "uncond_adapter_injected", False):
        from fastwam.models.wan22.uncond_adapter import (
            load_uncond_adapter_state_dict,
            resolve_backbone_and_adapter_paths,
            set_uncond_adapter_enabled,
        )

        resume_raw = cfg.get("resume", None)
        config_resume = None if resume_raw in (None, "null", "") else str(resume_raw)
        backbone_arg = None
        if backbone_checkpoint:
            backbone_arg = str(resolve_checkpoint_path(run_dir, backbone_checkpoint))
        adapter_arg = None
        if uncond_adapter:
            adapter_arg = str(resolve_checkpoint_path(run_dir, uncond_adapter))
        config_resume_resolved = None
        if config_resume:
            try:
                config_resume_resolved = str(resolve_checkpoint_path(run_dir, config_resume))
            except FileNotFoundError:
                config_resume_resolved = config_resume
        backbone_path, adapter_path = resolve_backbone_and_adapter_paths(
            str(checkpoint_path),
            backbone_checkpoint=backbone_arg,
            adapter_checkpoint=adapter_arg,
            config_resume=config_resume_resolved,
        )
        backbone_path = str(resolve_checkpoint_path(run_dir, backbone_path))
        adapter_path = str(resolve_checkpoint_path(run_dir, adapter_path))
        print(f"  DEWO v5 backbone: {backbone_path}", flush=True)
        print(f"  DEWO v5 adapter: {adapter_path}", flush=True)
        model.load_checkpoint(backbone_path)
        load_uncond_adapter_state_dict(model, adapter_path)
        set_uncond_adapter_enabled(model, False)
        checkpoint_sha256 = sha256_file(Path(adapter_path))
    else:
        model.load_checkpoint(str(checkpoint_path))
        checkpoint_sha256 = sha256_file(checkpoint_path)
    model.eval()
    config_sha256 = sha256_file(config_path)

    processor: FastWAMProcessor = instantiate(processor_cfg)
    processor.eval()
    if normalization_kind == "meta":
        processor.set_normalizer_from_modality_stats()
        print(f"  Modality stats (GR00T-style) from: {normalization_path}", flush=True)
    else:
        processor.set_normalizer_from_stats(
            load_dataset_stats_from_json(str(normalization_path))
        )
        print(f"  Dataset stats: {normalization_path}", flush=True)

    resolved_action_horizon, num_video_frames = resolve_inference_horizons(
        cfg.data.train, action_horizon=action_horizon
    )
    num_video_frames = resolve_num_video_frames_from_cfg(
        cfg, fallback=num_video_frames
    )
    if num_inference_steps is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 10))
    evaluation_cfg = cfg.get("EVALUATION", {}) or {}
    resolved_inference_seed = (
        int(inference_seed) if inference_seed is not None else evaluation_cfg.get("seed")
    )
    print(f"  Action horizon: {resolved_action_horizon}", flush=True)
    print(f"  Num video frames: {num_video_frames}", flush=True)
    print(f"  Num inference steps: {num_inference_steps}", flush=True)
    print(f"  Inference seed: {resolved_inference_seed}", flush=True)
    resolved_text_cfg_scale = (
        float(text_cfg_scale)
        if text_cfg_scale is not None
        else float(evaluation_cfg.get("text_cfg_scale", 0.0))
    )
    resolved_negative_prompt = (
        negative_prompt if negative_prompt is not None else evaluation_cfg.get("negative_prompt")
    )
    resolved_failure_prompt = (
        failure_prompt if failure_prompt is not None else evaluation_cfg.get("failure_prompt")
    )
    print(
        f"  Text CFG scale: {resolved_text_cfg_scale} (w in ε_base + w(ε_posi-ε_base); 0=本体, 1=优势, >1=guide)",
        flush=True,
    )
    if supports_cfg_mix(model.infer_action):
        print("  infer_action: FastWAM CFG/value signature", flush=True)
    else:
        print("  infer_action: action-only signature (Joint/IDM)", flush=True)
    resolved_adaptive_tau = None if adaptive_cfg_tau is None else float(adaptive_cfg_tau)
    resolved_epsilon_l = (
        None
        if cfg_epsilon_l is None and evaluation_cfg.get("cfg_epsilon_l") is None
        else float(
            cfg_epsilon_l
            if cfg_epsilon_l is not None
            else evaluation_cfg.get("cfg_epsilon_l")
        )
    )
    raw_clip_mode = (
        cfg_residual_clip_mode
        if cfg_residual_clip_mode is not None
        else evaluation_cfg.get("cfg_residual_clip_mode", "rms")
    )
    resolved_clip_mode = "rms" if raw_clip_mode in (None, "") else str(raw_clip_mode)
    resolved_cfg_exec = (
        int(cfg_exec_horizon)
        if cfg_exec_horizon is not None
        else int(evaluation_cfg.get("cfg_exec_horizon", 24))
    )
    if inference_config is not None:
        inference_config.action_horizon = resolved_action_horizon
        inference_config.num_inference_steps = int(num_inference_steps)
        inference_config.num_video_frames = int(num_video_frames)
        inference_config.text_cfg_scale = resolved_text_cfg_scale
        inference_config.seed = resolved_inference_seed
        if inference_config.negative_prompt is None:
            inference_config.negative_prompt = resolved_negative_prompt
        if inference_config.failure_prompt is None:
            inference_config.failure_prompt = resolved_failure_prompt

    return FastWAMPolicy(
        model=model,
        processor=processor,
        device=device,
        action_horizon=resolved_action_horizon,
        num_inference_steps=int(num_inference_steps),
        num_video_frames=num_video_frames,
        text_cfg_scale=resolved_text_cfg_scale,
        negative_prompt=resolved_negative_prompt,
        failure_prompt=resolved_failure_prompt,
        sigma_shift=evaluation_cfg.get("sigma_shift"),
        seed=resolved_inference_seed,
        rand_device=str(evaluation_cfg.get("rand_device", "cpu")),
        tiled=bool(evaluation_cfg.get("tiled", False)),
        adaptive_cfg_tau=resolved_adaptive_tau,
        cfg_epsilon_l=resolved_epsilon_l,
        cfg_residual_clip_mode=resolved_clip_mode,
        cfg_exec_horizon=resolved_cfg_exec,
        inference_config=inference_config,
        model_provenance={
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "config_path": str(config_path.resolve()),
            "config_sha256": config_sha256,
        },
    )


# Compatibility alias used by scripts/run_fastwam_server.py and tests.
build_policy_from_run = load_policy_from_run
_build_policy_from_run = load_policy_from_run
