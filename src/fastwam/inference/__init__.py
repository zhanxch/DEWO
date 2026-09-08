"""Unified FastWAM closed-loop inference.

Import submodules directly when you need a torch-free client::

    from fastwam.inference.obs import DexJoCoFastWAMAdapter
    from fastwam.inference.contract import KEY_ACTION

``from fastwam.inference import FastWAMPolicy`` still works (lazy).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FastWAMDexJocoPolicy",
    "FastWAMPolicy",
    "InferenceConfig",
    "DexJoCoFastWAMAdapter",
    "KEY_ACTION",
    "KEY_CONTEXT",
    "KEY_CONTEXT_MASK",
    "KEY_FAILURE_CONTEXT",
    "KEY_FAILURE_CONTEXT_MASK",
    "KEY_FAILURE_PROMPT",
    "KEY_INPUT_IMAGE",
    "KEY_NEGATIVE_CONTEXT",
    "KEY_NEGATIVE_CONTEXT_MASK",
    "KEY_NEGATIVE_PROMPT",
    "KEY_PROMPT",
    "KEY_PROPRIO",
    "fastwam_action_to_dexjoco",
    "filter_infer_action_kwargs",
    "load_dexjoco_eval_settings",
    "load_policy_from_run",
    "rollout_episode",
    "rotvec_action_to_env_quat",
    "save_rollout_mp4",
    "supports_cfg_mix",
    "to_inference_tensors",
    "validate_policy_observation",
]

_LAZY = {
    "FastWAMDexJocoPolicy": ("fastwam.inference.dexjoco", "FastWAMDexJocoPolicy"),
    "FastWAMPolicy": ("fastwam.inference.policy", "FastWAMPolicy"),
    "InferenceConfig": ("fastwam.inference.config", "InferenceConfig"),
    "DexJoCoFastWAMAdapter": ("fastwam.inference.obs", "DexJoCoFastWAMAdapter"),
    "KEY_ACTION": ("fastwam.inference.contract", "KEY_ACTION"),
    "KEY_CONTEXT": ("fastwam.inference.contract", "KEY_CONTEXT"),
    "KEY_CONTEXT_MASK": ("fastwam.inference.contract", "KEY_CONTEXT_MASK"),
    "KEY_FAILURE_CONTEXT": ("fastwam.inference.contract", "KEY_FAILURE_CONTEXT"),
    "KEY_FAILURE_CONTEXT_MASK": ("fastwam.inference.contract", "KEY_FAILURE_CONTEXT_MASK"),
    "KEY_FAILURE_PROMPT": ("fastwam.inference.contract", "KEY_FAILURE_PROMPT"),
    "KEY_INPUT_IMAGE": ("fastwam.inference.contract", "KEY_INPUT_IMAGE"),
    "KEY_NEGATIVE_CONTEXT": ("fastwam.inference.contract", "KEY_NEGATIVE_CONTEXT"),
    "KEY_NEGATIVE_CONTEXT_MASK": ("fastwam.inference.contract", "KEY_NEGATIVE_CONTEXT_MASK"),
    "KEY_NEGATIVE_PROMPT": ("fastwam.inference.contract", "KEY_NEGATIVE_PROMPT"),
    "KEY_PROMPT": ("fastwam.inference.contract", "KEY_PROMPT"),
    "KEY_PROPRIO": ("fastwam.inference.contract", "KEY_PROPRIO"),
    "fastwam_action_to_dexjoco": ("fastwam.inference.obs", "fastwam_action_to_dexjoco"),
    "filter_infer_action_kwargs": ("fastwam.inference.kwargs", "filter_infer_action_kwargs"),
    "load_dexjoco_eval_settings": ("fastwam.inference.obs", "load_dexjoco_eval_settings"),
    "load_policy_from_run": ("fastwam.inference.policy", "load_policy_from_run"),
    "rollout_episode": ("fastwam.inference.rollout", "rollout_episode"),
    "rotvec_action_to_env_quat": ("fastwam.inference.obs", "rotvec_action_to_env_quat"),
    "save_rollout_mp4": ("fastwam.inference.rollout", "save_rollout_mp4"),
    "supports_cfg_mix": ("fastwam.inference.kwargs", "supports_cfg_mix"),
    "to_inference_tensors": ("fastwam.inference.contract", "to_inference_tensors"),
    "validate_policy_observation": ("fastwam.inference.contract", "validate_policy_observation"),
}


def __getattr__(name: str) -> Any:
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = spec
    from importlib import import_module

    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value
