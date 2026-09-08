"""Closed-loop inference knobs shared by all three FastWAM architectures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InferenceConfig:
    """Action chunking, independent value queries, and DEWO CFG coefficients.

    ``replan_steps`` is how many predicted actions are executed before the
    next ``infer_action``. ``value_replan_steps`` is how often the value head
    is queried; it may be denser or sparser than action replan. ``None``
    means "same cadence as ``replan_steps``".
    """

    action_horizon: int = 32
    replan_steps: int = 24
    value_replan_steps: int | None = None
    num_inference_steps: int = 10
    num_video_frames: int | None = None
    text_cfg_scale: float = 0.0
    adaptive_cfg_tau: float | None = None
    cfg_epsilon_l: float | None = None
    cfg_residual_clip_mode: str = "rms"
    cfg_exec_horizon: int = 24
    cfg_gate_mode: str = "off"
    cfg_v_high: float | None = None
    cfg_drop_delta: float = 0.15
    cfg_growth_tau: float = 0.05
    cfg_growth_start_replan: int = 2
    cfg_growth_stop_replan: int | None = None
    cfg_low_value_threshold: float = 0.10
    cfg_growth_delta: float = 0.01
    cfg_growth_once: bool = False
    negative_prompt: str | None = None
    failure_prompt: str | None = None
    sigma_shift: float | None = None
    seed: int | None = None
    rand_device: str = "cpu"
    tiled: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.action_horizon = int(self.action_horizon)
        self.replan_steps = int(self.replan_steps)
        self.num_inference_steps = int(self.num_inference_steps)
        self.cfg_exec_horizon = int(self.cfg_exec_horizon)
        self.text_cfg_scale = float(self.text_cfg_scale)
        if self.replan_steps < 1 or self.replan_steps > self.action_horizon:
            raise ValueError(
                "replan_steps must be in [1, action_horizon], "
                f"got {self.replan_steps} vs horizon {self.action_horizon}"
            )
        if self.value_replan_steps is not None:
            self.value_replan_steps = int(self.value_replan_steps)
            if self.value_replan_steps < 1:
                raise ValueError("value_replan_steps must be >= 1")
        if self.num_video_frames is not None:
            self.num_video_frames = int(self.num_video_frames)
        if self.negative_prompt == "":
            self.negative_prompt = None
        if self.failure_prompt == "":
            self.failure_prompt = None

    @property
    def resolved_value_replan_steps(self) -> int:
        return int(self.replan_steps if self.value_replan_steps is None else self.value_replan_steps)

    def wants_cfg_mix(self) -> bool:
        return float(self.text_cfg_scale) != 0.0 or str(self.cfg_gate_mode) not in {"", "off"}
