"""Compatibility shim. Canonical policy: ``fastwam.inference.dexjoco``."""

from __future__ import annotations

from typing import Any

from fastwam.inference.loader import resolve_run_dir as _resolve_run_dir
from fastwam.inference.obs import fastwam_action_to_dexjoco

__all__ = ["FastWAMDexJocoPolicy", "fastwam_action_to_dexjoco", "_resolve_run_dir"]


def __getattr__(name: str) -> Any:
    if name == "FastWAMDexJocoPolicy":
        from fastwam.inference.dexjoco import FastWAMDexJocoPolicy

        return FastWAMDexJocoPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
