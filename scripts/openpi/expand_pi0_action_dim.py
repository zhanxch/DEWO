#!/usr/bin/env python3
"""Expand a 32-D π0 JAX checkpoint's action/state heads to a larger action_dim.

π0_base ships with action_dim=32. The Wuji joint-absolute robot is 54-D, so the
action_in_proj / action_out_proj / state_proj matrices must grow. The original
32 pretrained columns/rows are copied; new dimensions use the same NNX Linear
initialization as Pi0.
"""

from __future__ import annotations

from pathlib import Path

import flax.nnx as nnx
import jax
import numpy as np
import orbax.checkpoint as ocp
import tyro

import openpi.models.model as _model


def _to_mutable(tree):
    """Orbax may restore FrozenDict / read-only arrays; we need a writable tree."""
    if isinstance(tree, dict) or type(tree).__name__ in {"FrozenDict", "FlatMap"}:
        return {k: _to_mutable(v) for k, v in dict(tree).items()}
    if isinstance(tree, np.ndarray):
        return np.array(tree, copy=True)
    return tree


def _expand_linear_in(params: dict, name: str, target_in: int, rng: jax.Array) -> None:
    kernel = params[name]["kernel"]
    if kernel.ndim != 2:
        raise ValueError(f"{name}.kernel ndim={kernel.ndim}")
    src_in, hidden = kernel.shape
    if src_in == target_in:
        return
    if src_in != 32:
        raise ValueError(f"Expected {name} in_dim=32, got {kernel.shape}")
    init = nnx.Linear(target_in, hidden, rngs=nnx.Rngs(rng))
    new_kernel = np.array(init.kernel.value, dtype=kernel.dtype, copy=True)
    new_kernel[:src_in, :] = np.asarray(kernel)
    params[name]["kernel"] = new_kernel
    # bias is (hidden,) and does not depend on in_dim.


def _expand_linear_out(params: dict, name: str, target_out: int, rng: jax.Array) -> None:
    kernel = params[name]["kernel"]
    bias = params[name]["bias"]
    if kernel.ndim != 2 or bias.ndim != 1:
        raise ValueError(f"Unexpected {name} ranks")
    hidden, src_out = kernel.shape
    if src_out == target_out:
        return
    if src_out != 32 or bias.shape[0] != 32:
        raise ValueError(f"Expected {name} out_dim=32, got kernel={kernel.shape} bias={bias.shape}")
    init = nnx.Linear(hidden, target_out, rngs=nnx.Rngs(rng))
    new_kernel = np.array(init.kernel.value, dtype=kernel.dtype, copy=True)
    new_bias = np.array(init.bias.value, dtype=bias.dtype, copy=True)
    new_kernel[:, :src_out] = np.asarray(kernel)
    new_bias[:src_out] = np.asarray(bias)
    params[name]["kernel"] = new_kernel
    params[name]["bias"] = new_bias


def main(
    input_path: Path,
    output_path: Path,
    target_action_dim: int = 54,
) -> None:
    input_params_path = Path(input_path) / "params"
    output_params_path = Path(output_path) / "params"
    params = _to_mutable(_model.restore_params(str(input_params_path), restore_type=np.ndarray))

    _expand_linear_in(params, "action_in_proj", target_action_dim, jax.random.key(0))
    _expand_linear_out(params, "action_out_proj", target_action_dim, jax.random.key(1))
    if "state_proj" in params:
        _expand_linear_in(params, "state_proj", target_action_dim, jax.random.key(2))

    if output_params_path.exists():
        raise FileExistsError(output_params_path)
    output_params_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy assets (norm stats for other robots) next to the expanded params.
    assets_src = Path(input_path) / "assets"
    assets_dst = Path(output_path) / "assets"
    if assets_src.exists() and not assets_dst.exists():
        import shutil

        shutil.copytree(assets_src, assets_dst)

    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(str(output_params_path), args=ocp.args.PyTreeSave(item={"params": params}))  # type: ignore

    print(f"Saved expanded params to {output_params_path}")
    print(
        "action_in_proj.kernel",
        params["action_in_proj"]["kernel"].shape,
        "action_out_proj.kernel",
        params["action_out_proj"]["kernel"].shape,
    )
    if "state_proj" in params:
        print("state_proj.kernel", params["state_proj"]["kernel"].shape)


if __name__ == "__main__":
    tyro.cli(main)
