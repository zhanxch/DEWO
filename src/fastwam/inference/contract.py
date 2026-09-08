"""Canonical policy observation contract.

Keys match ``model.infer_action``. DexJoCo / sim adapters convert into this
shape before calling :class:`~fastwam.inference.policy.FastWAMPolicy`.
This module is torch-free so eval clients can import the key names.
"""

from __future__ import annotations

from typing import Any

KEY_INPUT_IMAGE = "input_image"
KEY_PROPRIO = "proprio"
KEY_PROMPT = "prompt"
KEY_CONTEXT = "context"
KEY_CONTEXT_MASK = "context_mask"
KEY_NEGATIVE_PROMPT = "negative_prompt"
KEY_NEGATIVE_CONTEXT = "negative_context"
KEY_NEGATIVE_CONTEXT_MASK = "negative_context_mask"
KEY_FAILURE_PROMPT = "failure_prompt"
KEY_FAILURE_CONTEXT = "failure_context"
KEY_FAILURE_CONTEXT_MASK = "failure_context_mask"
KEY_ACTION = "action"

CFG_VALUE_KEYS = (
    "cfg_value",
    "cfg_value_rel",
    "cfg_gate_g",
    "cfg_mix_weight",
    "cfg_gate_exec_rms",
    "cfg_token_rms_nfe",
    "cfg_chunk_rms_nfe",
    "cfg_token_rms",
    "cfg_chunk_rms",
    "cfg_exec_rms",
    "cfg_epsilon_l",
    "cfg_residual_clip_mode",
)

# Gate / residual options forwarded into FastWAM.infer_action when present.
CFG_OPTION_KEYS = (
    "cfg_gate_mode",
    "cfg_value_prev",
    "cfg_gate_fired",
    "cfg_v_high",
    "cfg_drop_delta",
    "cfg_replan_index",
    "cfg_growth_tau",
    "cfg_growth_start_replan",
    "cfg_growth_stop_replan",
    "cfg_low_value_threshold",
    "cfg_growth_delta",
    "return_cfg_residual",
    "cfg_exec_horizon",
    "adaptive_cfg_tau",
    "cfg_epsilon_l",
    "epsilon_l",
    "cfg_residual_epsilon",
    "cfg_residual_clip_mode",
)

_FORBIDDEN_SIM_KEYS = {"rgb", "video", "state", "language", "instruction"}


def validate_policy_observation(observation: dict[str, Any]) -> None:
    """Reject sim-style payloads; callers must send training-aligned tensors."""
    if not isinstance(observation, dict):
        raise TypeError(f"observation must be a dict, got {type(observation)}")

    overlap = _FORBIDDEN_SIM_KEYS & observation.keys()
    if overlap:
        raise ValueError(
            f"observation contains sim-only keys {sorted(overlap)}. "
            "Convert via fastwam.inference.obs before calling the policy."
        )

    if KEY_INPUT_IMAGE not in observation:
        raise ValueError(f"observation must contain '{KEY_INPUT_IMAGE}'")

    img = observation[KEY_INPUT_IMAGE]
    if isinstance(img, dict):
        raise ValueError(
            "observation['input_image'] must be a tensor/ndarray [1,3,H,W] in [-1,1], "
            "not a nested dict."
        )

    has_context = KEY_CONTEXT in observation
    has_context_mask = KEY_CONTEXT_MASK in observation
    has_text = has_context and has_context_mask
    has_prompt = KEY_PROMPT in observation
    if has_context != has_context_mask:
        raise ValueError(
            f"Provide both '{KEY_CONTEXT}' and '{KEY_CONTEXT_MASK}' together."
        )
    if not has_text and not has_prompt:
        raise ValueError(
            f"observation must contain '{KEY_PROMPT}' or "
            f"('{KEY_CONTEXT}', '{KEY_CONTEXT_MASK}')."
        )
    if has_text and has_prompt:
        raise ValueError(
            f"Provide either '{KEY_PROMPT}' or cached text tensors, not both."
        )

    _validate_optional_pair(
        observation,
        prompt_key=KEY_NEGATIVE_PROMPT,
        context_key=KEY_NEGATIVE_CONTEXT,
        mask_key=KEY_NEGATIVE_CONTEXT_MASK,
        label="negative",
        has_prompt=has_prompt,
        has_text=has_text,
    )
    _validate_optional_pair(
        observation,
        prompt_key=KEY_FAILURE_PROMPT,
        context_key=KEY_FAILURE_CONTEXT,
        mask_key=KEY_FAILURE_CONTEXT_MASK,
        label="failure",
        has_prompt=has_prompt,
        has_text=has_text,
    )


def _validate_optional_pair(
    observation: dict[str, Any],
    *,
    prompt_key: str,
    context_key: str,
    mask_key: str,
    label: str,
    has_prompt: bool,
    has_text: bool,
) -> None:
    has_context = context_key in observation or mask_key in observation
    has_prompt_value = prompt_key in observation
    if has_context and not (context_key in observation and mask_key in observation):
        raise ValueError(f"Provide both '{context_key}' and '{mask_key}' together.")
    if has_context and has_prompt_value:
        raise ValueError(
            f"Provide either '{prompt_key}' or cached {label} text tensors, not both."
        )
    if has_prompt and has_context:
        raise ValueError(
            f"Prompt input requires a {label} prompt, not cached {label} context."
        )
    if has_text and has_prompt_value:
        raise ValueError(
            f"Cached context input requires cached {label} context, not a {label} prompt."
        )


def to_inference_tensors(
    observation: dict[str, Any],
    *,
    device: Any,
    dtype: Any,
) -> dict[str, Any]:
    """Convert numpy payloads to torch tensors for ``model.infer_action``."""
    import torch

    validate_policy_observation(observation)

    input_image = _image_to_inference_tensor(
        observation[KEY_INPUT_IMAGE], device=device, dtype=dtype
    )

    out: dict[str, Any] = {KEY_INPUT_IMAGE: input_image}

    proprio = observation.get(KEY_PROPRIO)
    if proprio is not None:
        if not isinstance(proprio, torch.Tensor):
            proprio = torch.as_tensor(proprio, dtype=torch.float32)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        out[KEY_PROPRIO] = proprio.to(device=device, dtype=dtype)

    if KEY_CONTEXT in observation:
        out[KEY_CONTEXT] = _as_tensor(observation[KEY_CONTEXT], device, dtype)
        out[KEY_CONTEXT_MASK] = _as_bool_mask(observation[KEY_CONTEXT_MASK], device)
        if KEY_NEGATIVE_CONTEXT in observation:
            out[KEY_NEGATIVE_CONTEXT] = _as_tensor(
                observation[KEY_NEGATIVE_CONTEXT], device, dtype
            )
            out[KEY_NEGATIVE_CONTEXT_MASK] = _as_bool_mask(
                observation[KEY_NEGATIVE_CONTEXT_MASK], device
            )
        if KEY_FAILURE_CONTEXT in observation:
            out[KEY_FAILURE_CONTEXT] = _as_tensor(
                observation[KEY_FAILURE_CONTEXT], device, dtype
            )
            out[KEY_FAILURE_CONTEXT_MASK] = _as_bool_mask(
                observation[KEY_FAILURE_CONTEXT_MASK], device
            )
    else:
        out[KEY_PROMPT] = str(observation[KEY_PROMPT])
        if KEY_NEGATIVE_PROMPT in observation:
            out[KEY_NEGATIVE_PROMPT] = str(observation[KEY_NEGATIVE_PROMPT])
        if KEY_FAILURE_PROMPT in observation:
            out[KEY_FAILURE_PROMPT] = str(observation[KEY_FAILURE_PROMPT])

    return out


def _image_to_inference_tensor(input_image: Any, *, device: Any, dtype: Any) -> Any:
    """Convert policy ``input_image`` to ``[1,3,H,W]`` in ``[-1, 1]``.

    uint8 images follow FastWAM-infer-in-DexJoco Joint eval: cast to the model
    dtype first, then ``x * (2/255) - 1``. Float images are already in ``[-1, 1]``.
    """
    import numpy as np
    import torch

    is_uint8 = False
    if isinstance(input_image, torch.Tensor):
        tensor = input_image
        is_uint8 = tensor.dtype == torch.uint8
    else:
        arr = np.asarray(input_image)
        is_uint8 = arr.dtype == np.uint8
        if is_uint8:
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = np.transpose(arr, (2, 0, 1))
            elif arr.ndim == 4 and arr.shape[-1] == 3:
                arr = np.transpose(arr, (0, 3, 1, 2))
            tensor = torch.from_numpy(np.ascontiguousarray(arr.copy()))
        else:
            tensor = torch.as_tensor(arr, dtype=torch.float32)

    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(
            f"input_image must be [1,3,H,W] or [3,H,W], got {tuple(tensor.shape)}"
        )
    if is_uint8:
        tensor = tensor.to(device=device, dtype=dtype)
        return tensor * (2.0 / 255.0) - 1.0
    return tensor.to(device=device, dtype=dtype)


def _as_tensor(value: Any, device: Any, dtype: Any) -> Any:
    import torch

    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=torch.float32)
    return value.to(device=device, dtype=dtype)


def _as_bool_mask(value: Any, device: Any) -> Any:
    import torch

    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=torch.bool)
    return value.to(device=device, dtype=torch.bool)
