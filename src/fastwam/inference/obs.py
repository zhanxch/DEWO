"""DexJoCo observation / action conversion. Torch-free (numpy + PIL)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from fastwam.inference.contract import (
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
from fastwam.inference.loader import resolve_eval_action_horizon


def _as_numpy(value: Any, *, dtype) -> np.ndarray:
    """Convert torch/numpy T5 caches to numpy.

    Official caches are often ``bfloat16`` tensors; ``np.asarray(..., float32)``
    cannot cast that dtype and raises ``TypeError: unsupported ScalarType``.
    """
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        tensor = value.detach().cpu()
        if dtype is np.float32 and hasattr(tensor, "float"):
            tensor = tensor.float()
        elif dtype is bool and hasattr(tensor, "bool"):
            tensor = tensor.bool()
        value = tensor.numpy()
    return np.asarray(value, dtype=dtype)

DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)

ROBOTWIN_TOP_SIZE_WH = (320, 256)
ROBOTWIN_WRIST_SIZE_WH = (160, 128)

CAMERA_MAPPING = {"front": "front", "wrist": "wrist", "base": "front"}


def resize_rgb(rgb: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    return np.asarray(image.resize(size_wh, resample=Image.BILINEAR), dtype=np.uint8)


def resize_rgb_area(rgb: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Downsample with cv2 INTER_AREA (FastWAM-infer-in-DexJoco joint eval)."""
    import cv2

    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")
    width, height = int(size_wh[0]), int(size_wh[1])
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def hwc_rgb_to_input_image_np(rgb: np.ndarray) -> np.ndarray:
    """HWC uint8 → [1,3,H,W] float32 in [-1, 1] (no resize)."""
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    tensor = rgb.transpose(2, 0, 1).astype(np.float32)
    tensor = tensor * (2.0 / 255.0) - 1.0
    return tensor[np.newaxis, ...]


def rgb_to_input_image_np(rgb: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    return hwc_rgb_to_input_image_np(resize_rgb(rgb, size_wh))


def concat_robotwin_rgb(top: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    head = resize_rgb(top, ROBOTWIN_TOP_SIZE_WH)
    wrist_left = resize_rgb(left, ROBOTWIN_WRIST_SIZE_WH)
    wrist_right = resize_rgb(right, ROBOTWIN_WRIST_SIZE_WH)
    bottom = np.concatenate([wrist_left, wrist_right], axis=1)
    return np.ascontiguousarray(np.concatenate([head, bottom], axis=0), dtype=np.uint8)


def safe_rgb_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr, arr, arr], axis=2)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            if np.nanmax(arr) <= 1.0:
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            else:
                arr = np.clip(arr, 0.0, 255.0)
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def resolve_env_camera_keys(
    image_keys: list[str],
    camera_mapping: dict[str, str],
) -> list[str]:
    env_keys: list[str] = []
    for key in image_keys:
        if key in camera_mapping:
            env_keys.append(camera_mapping[key])
        elif key == "ego" and "base" in camera_mapping:
            env_keys.append(camera_mapping["base"])
        elif key in camera_mapping.values():
            env_keys.append(key)
        else:
            raise ValueError(
                f"Cannot map training image key {key!r} via camera_mapping {camera_mapping}"
            )
    return env_keys


def rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """Match FastWAM-infer-in-DexJoco: SciPy rotvec → xyzw → wxyz."""
    from scipy.spatial.transform import Rotation

    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    quaternion_xyzw = Rotation.from_rotvec(rotvec).as_quat()
    return np.asarray(quaternion_xyzw[[3, 0, 1, 2]], dtype=np.float64)


def quat_wxyz_to_rotvec(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    quat = quat / norm
    w = float(np.clip(quat[0], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(max(0.0, 1.0 - w * w))
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float64)
    axis = quat[1:4] / sin_half
    return axis * angle


def rotvec_action_to_env_quat(action_rotvec: np.ndarray, *, dual_arm: bool = False) -> np.ndarray:
    """Convert policy rotvec action (22/44-dim) to DexJoCo quat action (23/46-dim)."""
    action_rotvec = np.asarray(action_rotvec, dtype=np.float64)
    if dual_arm:
        r_xyz = action_rotvec[:3]
        r_rotvec = action_rotvec[3:6]
        r_hand = action_rotvec[6:22]
        l_xyz = action_rotvec[22:25]
        l_rotvec = action_rotvec[25:28]
        l_hand = action_rotvec[28:44]
        return np.concatenate(
            [
                r_xyz,
                rotvec_to_quat_wxyz(r_rotvec),
                l_xyz,
                rotvec_to_quat_wxyz(l_rotvec),
                r_hand,
                l_hand,
            ]
        )
    xyz = action_rotvec[:3]
    rotvec = action_rotvec[3:6]
    hand = action_rotvec[6:22]
    return np.concatenate([xyz, rotvec_to_quat_wxyz(rotvec), hand])


def fastwam_action_to_dexjoco(action_rotvec: np.ndarray) -> np.ndarray:
    return rotvec_action_to_env_quat(action_rotvec, dual_arm=False)


def load_text_context_arrays(
    instruction: str,
    *,
    text_embedding_cache_dir: str | Path,
    context_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached T5 context as numpy arrays (.npz preferred, .pt optional)."""
    cache_dir = Path(text_embedding_cache_dir)
    hashed = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    stem = f"{hashed}.t5_len{context_len}"
    npz_path = cache_dir / f"{stem}.npz"
    pt_matches = sorted(cache_dir.glob(f"{stem}*.pt"))
    if npz_path.exists():
        payload = np.load(npz_path)
        context = payload["context"].astype(np.float32)
        context_mask = payload["mask"].astype(bool)
    elif pt_matches:
        import torch

        payload = torch.load(pt_matches[0], map_location="cpu", weights_only=False)
        context = _as_numpy(payload["context"], dtype=np.float32)
        context_mask = _as_numpy(payload["mask"], dtype=bool)
    else:
        raise FileNotFoundError(
            f"Missing text embedding cache for instruction hash {hashed} "
            f"under {cache_dir} (looked for {npz_path.name} and {stem}*.pt)."
        )
    if context.ndim > 2:
        context = context.reshape(context.shape[-2], context.shape[-1])
    if context_mask.ndim > 1:
        context_mask = context_mask.reshape(-1)
    context = context.copy()
    context[~context_mask] = 0.0
    context_mask = np.ones_like(context_mask, dtype=bool)
    return context, context_mask


def load_text_embedding_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if path.suffix == ".npz":
        payload = np.load(path)
        context = payload["context"].astype(np.float32)
        context_mask = payload["mask"].astype(bool)
    else:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        context = _as_numpy(payload["context"], dtype=np.float32)
        context_mask = _as_numpy(payload["mask"], dtype=bool)
    if context.ndim > 2:
        context = np.squeeze(context)
    if context_mask.ndim > 1:
        context_mask = np.squeeze(context_mask)
    context = context.copy()
    context[~context_mask] = 0.0
    context_mask = np.ones_like(context_mask, dtype=bool)
    return context, context_mask


def _path_from_config(value: Any, *, run_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in (Path.cwd(), run_dir):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def _as_path_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _resolve_train_dataset_roots(train_data: dict[str, Any], *, run_dir: Path) -> list[Path]:
    roots = [
        _path_from_config(path, run_dir=run_dir)
        for path in _as_path_list(train_data.get("dataset_dirs"))
    ]
    if roots:
        return roots
    manifest_path = train_data.get("manifest_path")
    if manifest_path:
        path = _path_from_config(manifest_path, run_dir=run_dir)
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            return [
                _path_from_config(root, run_dir=run_dir)
                for root in manifest.get("dataset_roots", {}).values()
            ]
    return []


def _load_eve_action_schema(train_data: dict[str, Any], *, run_dir: Path) -> dict[str, Any]:
    for root in _resolve_train_dataset_roots(train_data, run_dir=run_dir):
        schema_path = root / "meta" / "eve" / "action_schema.json"
        if schema_path.exists():
            with schema_path.open("r", encoding="utf-8") as handle:
                schema = json.load(handle)
            schema["_schema_path"] = str(schema_path)
            return schema
    return {}


def load_dexjoco_eval_settings(
    run_dir: Path,
    *,
    action_horizon_override: int | None = None,
    text_embedding_cache_dir_override: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(run_dir) / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing training config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    train_data = cfg["data"]["train"]
    image_size = tuple(int(x) for x in train_data["video_size"])
    processor = train_data["processor"]
    shape_meta = train_data["shape_meta"]
    image_keys = [str(item["key"]) for item in shape_meta["images"]]
    image_sizes_wh = [
        (int(item["shape"][2]), int(item["shape"][1])) for item in shape_meta["images"]
    ]
    action_schema = _load_eve_action_schema(train_data, run_dir=Path(run_dir))
    evaluation_cfg = cfg.get("EVALUATION") or {}
    model_target = str((cfg.get("model") or {}).get("_target_", ""))
    is_joint = "create_fastwam_joint" in model_target
    image_resize = str(
        evaluation_cfg.get("image_resize") or ("area" if is_joint else "bilinear")
    )
    keep_uint8_image = bool(evaluation_cfg.get("keep_uint8_image", is_joint))
    policy_action_output_dim = int(
        action_schema.get("policy_action_dim", processor["action_output_dim"])
    )
    control_action_slice = action_schema.get("control_action_slice")
    if control_action_slice is None:
        prefix_dim = int(action_schema.get("policy_action_prefix_dim", 0))
        control_action_slice = [prefix_dim, policy_action_output_dim]
    return {
        "image_size_wh": (image_size[1], image_size[0]),
        "action_horizon": resolve_eval_action_horizon(
            train_data,
            action_horizon_override=action_horizon_override,
        ),
        "action_output_dim": policy_action_output_dim,
        "policy_action_prefix_dim": int(action_schema.get("policy_action_prefix_dim", 0)),
        "policy_action_control_slice": [
            int(control_action_slice[0]),
            int(control_action_slice[1]),
        ],
        "proprio_output_dim": int(processor["proprio_output_dim"]),
        "text_embedding_cache_dir": (
            str(Path(text_embedding_cache_dir_override).expanduser().resolve())
            if text_embedding_cache_dir_override is not None
            else train_data.get("text_embedding_cache_dir")
        ),
        "context_len": int(train_data.get("context_len", 128)),
        "load_text_encoder": bool(cfg.get("model", {}).get("load_text_encoder", False)),
        "concat_multi_camera": train_data.get("concat_multi_camera"),
        "image_keys": image_keys,
        "image_sizes_wh": image_sizes_wh,
        "image_resize": image_resize,
        "keep_uint8_image": keep_uint8_image,
    }


@dataclass
class DexJoCoTaskConfig:
    env_name: str
    prompt: str
    cfg_base_prompt: str | None
    cfg_failure_prompt: str | None
    dual_arm: bool
    camera_key: str
    camera_mapping: dict[str, str]
    password: list[int] | None = None

    @classmethod
    def from_yaml(cls, cfg: dict[str, Any]) -> "DexJoCoTaskConfig":
        camera_mapping = {str(k): str(v) for k, v in cfg["camera_mapping"].items()}
        base_key = camera_mapping.get("base")
        if base_key is None:
            raise ValueError(f"camera_mapping must contain 'base' key: {cfg}")
        return cls(
            env_name=str(cfg["env_name"]),
            prompt=str(cfg["prompt"]),
            cfg_base_prompt=(
                None if cfg.get("cfg_base_prompt") is None else str(cfg["cfg_base_prompt"])
            ),
            cfg_failure_prompt=(
                None
                if cfg.get("cfg_failure_prompt") is None
                else str(cfg["cfg_failure_prompt"])
            ),
            dual_arm=str(cfg.get("robot_type", "single_arm")) == "dual_arm",
            camera_key=str(base_key),
            camera_mapping=camera_mapping,
            password=cfg.get("password"),
        )


class DexJoCoFastWAMAdapter:
    """Maps DexJoCo env observations/actions ↔ FastWAM policy I/O."""

    def __init__(self, eval_settings: dict[str, Any]) -> None:
        self.image_size_wh: tuple[int, int] = eval_settings["image_size_wh"]
        self.action_horizon = int(eval_settings["action_horizon"])
        self.policy_action_output_dim = int(eval_settings["action_output_dim"])
        self.policy_action_prefix_dim = int(eval_settings.get("policy_action_prefix_dim", 0))
        control_slice = eval_settings.get("policy_action_control_slice")
        if control_slice is None:
            control_slice = [self.policy_action_prefix_dim, self.policy_action_output_dim]
        if not isinstance(control_slice, (list, tuple)) or len(control_slice) != 2:
            raise ValueError(f"Invalid policy_action_control_slice: {control_slice}")
        self.policy_action_control_start = int(control_slice[0])
        self.policy_action_control_end = int(control_slice[1])
        self.action_output_dim = (
            self.policy_action_control_end - self.policy_action_control_start
        )
        self.proprio_output_dim = int(eval_settings["proprio_output_dim"])
        self.text_embedding_cache_dir = eval_settings.get("text_embedding_cache_dir")
        self.context_len = int(eval_settings["context_len"])
        self.use_prompt = bool(eval_settings["load_text_encoder"])
        self.concat_multi_camera = eval_settings.get("concat_multi_camera")
        self.image_keys: list[str] = list(eval_settings.get("image_keys") or [])
        self.image_sizes_wh: list[tuple[int, int]] = [
            tuple(map(int, item)) for item in eval_settings.get("image_sizes_wh", [])
        ]
        self.image_resize = str(eval_settings.get("image_resize") or "bilinear")
        self.keep_uint8_image = bool(eval_settings.get("keep_uint8_image", False))
        self._cached_embeddings: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def task_prompt(self, task_prompt: str) -> str:
        return DEFAULT_PROMPT.format(task=task_prompt)

    def bind_text_embedding(
        self,
        context: np.ndarray,
        context_mask: np.ndarray,
        *,
        kind: str = "success",
    ) -> None:
        self._cached_embeddings[kind] = (
            np.asarray(context, dtype=np.float32),
            np.asarray(context_mask, dtype=bool),
        )
        self.use_prompt = False

    def _load_instruction_context(self, instruction: str) -> tuple[np.ndarray, np.ndarray]:
        if self.text_embedding_cache_dir is None:
            raise FileNotFoundError("No text embedding cache dir configured")
        return load_text_context_arrays(
            instruction,
            text_embedding_cache_dir=self.text_embedding_cache_dir,
            context_len=self.context_len,
        )

    def _build_input_image(
        self,
        env_obs: dict[str, Any],
        *,
        camera_key: str,
        camera_mapping: dict[str, str],
    ) -> np.ndarray:
        if self.concat_multi_camera == "robotwin":
            if len(self.image_keys) != 3:
                raise ValueError(
                    f"concat_multi_camera='robotwin' requires 3 image keys, got {self.image_keys}"
                )
            env_cam_keys = resolve_env_camera_keys(self.image_keys, camera_mapping)
            top = safe_rgb_uint8(env_obs[env_cam_keys[0]])
            left = safe_rgb_uint8(env_obs[env_cam_keys[1]])
            right = safe_rgb_uint8(env_obs[env_cam_keys[2]])
            return self._to_input_image(concat_robotwin_rgb(top, left, right))

        if self.concat_multi_camera in {"horizontal", "vertical"} and len(self.image_keys) > 1:
            if len(self.image_sizes_wh) != len(self.image_keys):
                raise ValueError(
                    "Multi-camera eval requires one shape_meta image size per image key, "
                    f"got keys={self.image_keys}, sizes={self.image_sizes_wh}"
                )
            env_cam_keys = resolve_env_camera_keys(self.image_keys, camera_mapping)
            tiles = [
                self._resize_rgb(safe_rgb_uint8(env_obs[env_key]), size_wh)
                for env_key, size_wh in zip(env_cam_keys, self.image_sizes_wh)
            ]
            axis = 1 if self.concat_multi_camera == "horizontal" else 0
            rgb = np.ascontiguousarray(np.concatenate(tiles, axis=axis), dtype=np.uint8)
            if (rgb.shape[1], rgb.shape[0]) != self.image_size_wh:
                rgb = self._resize_rgb(rgb, self.image_size_wh)
            return self._to_input_image(rgb)

        rgb = safe_rgb_uint8(env_obs[camera_key])
        return self._to_input_image(self._resize_rgb(rgb, self.image_size_wh))

    def _resize_rgb(self, rgb: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
        if self.image_resize == "area":
            return resize_rgb_area(rgb, size_wh)
        return resize_rgb(rgb, size_wh)

    def _to_input_image(self, rgb_hwc: np.ndarray) -> np.ndarray:
        rgb = np.ascontiguousarray(np.asarray(rgb_hwc, dtype=np.uint8))
        if self.keep_uint8_image:
            return rgb.transpose(2, 0, 1)[np.newaxis, ...]
        return hwc_rgb_to_input_image_np(rgb)

    def _extract_proprio(self, env_obs: dict[str, Any]) -> np.ndarray:
        state = np.asarray(env_obs["state"], dtype=np.float32).reshape(-1)
        if state.shape[0] >= self.proprio_output_dim:
            return state[: self.proprio_output_dim]
        padded = np.zeros(self.proprio_output_dim, dtype=np.float32)
        padded[: state.shape[0]] = state
        return padded

    def env_obs_to_policy_obs(
        self,
        env_obs: dict[str, Any],
        *,
        camera_key: str = "front",
        camera_mapping: dict[str, str] | None = None,
        task_prompt: str,
        cfg_base_prompt: str | None = None,
        cfg_failure_prompt: str | None = None,
    ) -> dict[str, Any]:
        camera_mapping = camera_mapping or CAMERA_MAPPING
        policy_obs: dict[str, Any] = {
            KEY_INPUT_IMAGE: self._build_input_image(
                env_obs,
                camera_key=camera_key,
                camera_mapping=camera_mapping,
            ),
            KEY_PROPRIO: self._extract_proprio(env_obs).astype(np.float32),
        }
        instruction = self.task_prompt(task_prompt)
        base_instruction = (
            None if cfg_base_prompt is None else self.task_prompt(cfg_base_prompt)
        )
        fail_instruction = (
            None if cfg_failure_prompt is None else self.task_prompt(cfg_failure_prompt)
        )
        if self.use_prompt or (
            self.text_embedding_cache_dir is None and "success" not in self._cached_embeddings
        ):
            policy_obs[KEY_PROMPT] = instruction
            if base_instruction is not None:
                policy_obs[KEY_NEGATIVE_PROMPT] = base_instruction
            if fail_instruction is not None:
                policy_obs[KEY_FAILURE_PROMPT] = fail_instruction
            return policy_obs

        if "success" in self._cached_embeddings:
            context, context_mask = self._cached_embeddings["success"]
        else:
            context, context_mask = self._load_instruction_context(instruction)
        policy_obs[KEY_CONTEXT] = context
        policy_obs[KEY_CONTEXT_MASK] = context_mask
        if base_instruction is not None or "base" in self._cached_embeddings:
            if "base" in self._cached_embeddings:
                negative_context, negative_mask = self._cached_embeddings["base"]
            else:
                negative_context, negative_mask = self._load_instruction_context(
                    str(base_instruction)
                )
            policy_obs[KEY_NEGATIVE_CONTEXT] = negative_context
            policy_obs[KEY_NEGATIVE_CONTEXT_MASK] = negative_mask
        if fail_instruction is not None or "failure" in self._cached_embeddings:
            if "failure" in self._cached_embeddings:
                failure_context, failure_mask = self._cached_embeddings["failure"]
            else:
                failure_context, failure_mask = self._load_instruction_context(
                    str(fail_instruction)
                )
            policy_obs[KEY_FAILURE_CONTEXT] = failure_context
            policy_obs[KEY_FAILURE_CONTEXT_MASK] = failure_mask
        return policy_obs

    def parse_policy_response(self, response: Any) -> np.ndarray:
        if isinstance(response, (list, tuple)) and len(response) >= 1:
            action_dict = response[0]
        elif isinstance(response, dict):
            action_dict = response
        else:
            raise RuntimeError(f"Unexpected policy response type: {type(response)}")
        from fastwam.inference.contract import KEY_ACTION

        chunk = np.asarray(action_dict[KEY_ACTION], dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        if chunk.shape[-1] != self.policy_action_output_dim:
            raise ValueError(
                f"Policy action dim {chunk.shape[-1]} != expected {self.policy_action_output_dim}"
            )
        if (
            self.policy_action_control_start != 0
            or self.policy_action_control_end != self.policy_action_output_dim
        ):
            chunk = chunk[:, self.policy_action_control_start : self.policy_action_control_end]
        return chunk

    def rotvec_to_env_action(self, rotvec_action: np.ndarray, *, dual_arm: bool) -> np.ndarray:
        return rotvec_action_to_env_quat(rotvec_action, dual_arm=dual_arm)
