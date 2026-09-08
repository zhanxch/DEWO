"""DexJoCo sim extras for closed-loop eval (constraints, env wrapper).

Observation / action conversion lives in ``fastwam.inference.obs``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fastwam.inference.contract import (
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
)
from fastwam.inference.loader import (
    DEFAULT_SLIDING_WINDOW_ACTION_HORIZON,
    EVEROOBOT_FULL_EPISODE_DATASET,
    is_everobot_full_episode_train,
    resolve_eval_action_horizon,
)
from fastwam.inference.obs import (
    DEFAULT_PROMPT,
    DexJoCoFastWAMAdapter,
    DexJoCoTaskConfig,
    concat_robotwin_rgb,
    fastwam_action_to_dexjoco,
    hwc_rgb_to_input_image_np,
    load_dexjoco_eval_settings,
    load_text_context_arrays,
    quat_wxyz_to_rotvec as _quat_wxyz_to_rotvec,
    resize_rgb,
    resolve_env_camera_keys,
    rgb_to_input_image_np,
    rotvec_action_to_env_quat,
    rotvec_to_quat_wxyz as _rotvec_to_quat_wxyz,
    safe_rgb_uint8 as _safe_rgb_uint8,
)

DEFAULT_TASK_CONFIG_DIR = Path("third_party/dexjoco/configs/rand_obj")

CLICK_MOUSE_ALIGN_ROTVEC = np.array(
    [
        -4.4294e-01,
        1.3729e-06,
        1.5170e00,
        -3.14156462e00,
        -6.91584035e-05,
        -1.40317984e-03,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.263,
        0,
        0,
        0,
    ],
    dtype=np.float64,
)
CLICK_MOUSE_ALIGN_STEPS = 30


def load_task_configs(config_dir: Path) -> list[dict[str, Any]]:
    config_dir = config_dir.resolve()
    if not config_dir.exists():
        raise FileNotFoundError(f"Task config directory not found: {config_dir}")
    configs: list[dict[str, Any]] = []
    for path in sorted(config_dir.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        cfg["_config_path"] = str(path)
        configs.append(cfg)
    if not configs:
        raise FileNotFoundError(f"No task YAML files under {config_dir}")
    return configs


def clamp_rotvec_action_to_state(
    action_rotvec: np.ndarray,
    state: np.ndarray,
    *,
    dual_arm: bool,
    max_displacement: float,
    max_dz_down: float | None = None,
) -> np.ndarray:
    action = np.asarray(action_rotvec, dtype=np.float64).copy()
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    arm_specs = [(slice(0, 3), 0), (slice(22, 25), 7)] if dual_arm else [(slice(0, 3), 0)]
    for xyz_sl, state_start in arm_specs:
        state_xyz = state[state_start : state_start + 3]
        delta = action[xyz_sl] - state_xyz
        if max_dz_down is not None and delta[2] < -max_dz_down:
            delta[2] = -max_dz_down
        norm = float(np.linalg.norm(delta))
        if norm > max_displacement:
            delta = delta * (max_displacement / norm)
        action[xyz_sl] = state_xyz + delta
    return action.astype(np.float32)


def state_to_rotvec_reference(state: np.ndarray, *, dual_arm: bool) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if dual_arm:
        r_arm = state[:7]
        l_arm = state[7:14]
        r_hand = state[14:30]
        l_hand = state[30:46]
        return np.concatenate(
            [
                r_arm[:3],
                _quat_wxyz_to_rotvec(r_arm[3:7]),
                r_hand,
                l_arm[:3],
                _quat_wxyz_to_rotvec(l_arm[3:7]),
                l_hand,
            ]
        ).astype(np.float32)
    arm = state[:7]
    hand = state[7:23]
    return np.concatenate([arm[:3], _quat_wxyz_to_rotvec(arm[3:7]), hand]).astype(np.float32)


@dataclass(frozen=True)
class ActionConstraintConfig:
    max_xyz_step: float = 0.05
    max_rot_step: float = 0.0
    max_hand_step: float = 0.0
    max_dz_down: float | None = 0.03
    clip_to_dataset_bounds: bool = False


def _load_action_bounds(stats_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    import json

    payload = json.loads(Path(stats_path).expanduser().read_text(encoding="utf-8"))
    action_stats = payload["action"]["default"]
    action_min = np.asarray(action_stats["global_min"], dtype=np.float64)
    action_max = np.asarray(action_stats["global_max"], dtype=np.float64)
    return action_min, action_max


def constrain_rotvec_action(
    action_rotvec: np.ndarray,
    state: np.ndarray,
    *,
    dual_arm: bool,
    config: ActionConstraintConfig,
    action_min: np.ndarray | None = None,
    action_max: np.ndarray | None = None,
) -> np.ndarray:
    action = np.asarray(action_rotvec, dtype=np.float64).copy()
    reference = np.asarray(state_to_rotvec_reference(state, dual_arm=dual_arm), dtype=np.float64)
    if dual_arm:
        arm_specs = [
            (slice(0, 3), slice(3, 6), slice(6, 22)),
            (slice(22, 25), slice(25, 28), slice(28, 44)),
        ]
    else:
        arm_specs = [(slice(0, 3), slice(3, 6), slice(6, 22))]

    for xyz_sl, rot_sl, hand_sl in arm_specs:
        ref_xyz = reference[xyz_sl]
        delta_xyz = action[xyz_sl] - ref_xyz
        if config.max_dz_down is not None and delta_xyz[2] < -config.max_dz_down:
            delta_xyz[2] = -config.max_dz_down
        xyz_norm = float(np.linalg.norm(delta_xyz))
        if xyz_norm > config.max_xyz_step:
            delta_xyz = delta_xyz * (config.max_xyz_step / xyz_norm)
        action[xyz_sl] = ref_xyz + delta_xyz
        if config.max_rot_step > 0.0:
            delta_rot = action[rot_sl] - reference[rot_sl]
            rot_norm = float(np.linalg.norm(delta_rot))
            if rot_norm > config.max_rot_step:
                delta_rot = delta_rot * (config.max_rot_step / rot_norm)
            action[rot_sl] = reference[rot_sl] + delta_rot
        if config.max_hand_step > 0.0:
            delta_hand = action[hand_sl] - reference[hand_sl]
            hand_norm = float(np.linalg.norm(delta_hand))
            if hand_norm > config.max_hand_step:
                delta_hand = delta_hand * (config.max_hand_step / hand_norm)
            action[hand_sl] = reference[hand_sl] + delta_hand

    if config.clip_to_dataset_bounds and action_min is not None and action_max is not None:
        action = np.clip(action, action_min, action_max)
    return action.astype(np.float32)


class DexJoCoFastWAMEvalEnv:
    """Thin DexJoCo wrapper for synchronous FastWAM closed-loop evaluation."""

    def __init__(
        self,
        task: DexJoCoTaskConfig,
        *,
        seed: int,
        randomize: bool = False,
        randomize_dynamics: bool = False,
        render_mode: str = "rgb_array",
    ) -> None:
        from dexjoco.tasks.mappings import CONFIG_MAPPING

        self.task = task
        self.seed = seed
        self._done = False
        self._success = False
        self._last_stay_state: np.ndarray | None = None
        self._latest_obs: dict[str, Any] = {}

        config = CONFIG_MAPPING[task.env_name]()
        env_kwargs: dict[str, Any] = {}
        if task.env_name == "bimanual_unlock_ipad" and task.password is not None:
            env_kwargs["password"] = task.password

        self.env = config.get_environment(
            policy_mode=True,
            render_mode=render_mode,
            randomize=randomize,
            seed=seed,
            randomize_dynamics=randomize_dynamics,
            **env_kwargs,
        )

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def reset(self) -> dict[str, Any]:
        obs, _ = self.env.reset()
        self._done = False
        self._success = False
        self._last_stay_state = None
        self._latest_obs = copy.deepcopy(obs)
        return self._latest_obs

    def get_camera_frame(self) -> np.ndarray:
        return _safe_rgb_uint8(self._latest_obs[self.task.camera_key])

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def is_success(self) -> bool:
        return self._success

    def click_mouse_warmup(self) -> None:
        if self.task.env_name != "click_mouse":
            return
        align = rotvec_action_to_env_quat(
            CLICK_MOUSE_ALIGN_ROTVEC,
            dual_arm=self.task.dual_arm,
        )
        for _ in range(CLICK_MOUSE_ALIGN_STEPS):
            self._step_env(align)

    def step_rotvec(self, rotvec_action: np.ndarray) -> None:
        env_action = rotvec_action_to_env_quat(rotvec_action, dual_arm=self.task.dual_arm)
        self._step_env(env_action)

    def stay(self, *, continue_stay: bool = False) -> np.ndarray:
        if continue_stay and self._last_stay_state is not None:
            stay_state = self._last_stay_state
        else:
            state = np.asarray(self._latest_obs["state"], dtype=np.float64).reshape(-1)
            stay_state = state[:46] if self.task.dual_arm else state[:23]
            self._last_stay_state = stay_state

        if self.task.dual_arm:
            r_arm = stay_state[:7]
            l_arm = stay_state[7:14]
            r_hand = stay_state[14:30]
            l_hand = stay_state[30:46]
            rotvec_action = np.concatenate(
                [
                    r_arm[:3],
                    _quat_wxyz_to_rotvec(r_arm[3:7]),
                    r_hand,
                    l_arm[:3],
                    _quat_wxyz_to_rotvec(l_arm[3:7]),
                    l_hand,
                ]
            )
        else:
            arm = stay_state[:7]
            hand = stay_state[7:23]
            rotvec_action = np.concatenate([arm[:3], _quat_wxyz_to_rotvec(arm[3:7]), hand])

        self.step_rotvec(rotvec_action)
        return rotvec_action.astype(np.float32)

    def build_policy_obs(self, adapter: DexJoCoFastWAMAdapter) -> dict[str, Any]:
        return adapter.env_obs_to_policy_obs(
            self._latest_obs,
            camera_key=self.task.camera_key,
            camera_mapping=self.task.camera_mapping,
            task_prompt=self.task.prompt,
            cfg_base_prompt=self.task.cfg_base_prompt,
            cfg_failure_prompt=self.task.cfg_failure_prompt,
        )

    def _step_env(self, env_action: np.ndarray) -> None:
        obs, _reward, terminated, truncated, info = self.env.step(env_action)
        self._done = bool(terminated or truncated)
        self._success = bool(info.get("succeed", False))
        self._latest_obs = copy.deepcopy(obs)
