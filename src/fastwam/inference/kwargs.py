"""Filter ``infer_action`` kwargs so one caller works on all three architectures."""

from __future__ import annotations

import inspect
from typing import Any, Callable

# FastWAM.infer_action implements DEWO CFG mix + value gating.
# FastWAMJoint / FastWAMIDM expose a shorter action-only signature.
CFG_CAPABILITY_KEYS = ("negative_context", "cfg_gate_mode", "failure_context")


def infer_action_parameters(infer_fn: Callable[..., Any]) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(infer_fn).parameters)


def supports_cfg_mix(infer_fn: Callable[..., Any]) -> bool:
    params = infer_action_parameters(infer_fn)
    return any(key in params for key in CFG_CAPABILITY_KEYS)


def supports_value_head(model: Any) -> bool:
    return getattr(model, "value_head", None) is not None


def filter_infer_action_kwargs(
    infer_fn: Callable[..., Any],
    kwargs: dict[str, Any],
    *,
    require_cfg_mix: bool = False,
) -> dict[str, Any]:
    """Drop kwargs the current ``infer_action`` does not accept.

    Joint/IDM signatures omit DEWO CFG fields; passing them would raise
    ``TypeError``. Baseline action inference still works. Requesting a CFG
    mix on a model that cannot mix is an error.
    """
    params = infer_action_parameters(infer_fn)
    if require_cfg_mix and not supports_cfg_mix(infer_fn):
        raise ValueError(
            "This model's infer_action does not implement DEWO CFG mix "
            "(needs negative_context / cfg_gate_mode). Use create_fastwam "
            "with an uncond adapter, or set text_cfg_scale=0 and cfg_gate_mode=off."
        )
    if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in params.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in params}
