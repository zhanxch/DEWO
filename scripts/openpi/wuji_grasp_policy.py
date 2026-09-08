"""OpenPI input/output transforms for the Wuji bimanual + dexterous-hand robot.

Camera slots follow the π0 convention: one third-person view and two wrist views.
Action layout (54-D, matching the converted LeRobot dataset):
  0:7    left arm joints
  7:14   right arm joints
  14:34  left hand joints
  34:54  right hand joints
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

WUJI_ACTION_DIM = 54
WUJI_ARM_JOINTS = 14  # 7 left + 7 right
WUJI_HAND_JOINTS = 40  # 20 left + 20 right


def make_wuji_delta_action_mask() -> tuple[bool, ...]:
    """Delta for arm joints, absolute for finger joints (π0 gripper convention)."""
    return transforms.make_bool_mask(WUJI_ARM_JOINTS, -WUJI_HAND_JOINTS)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class WujiGraspInputs(transforms.DataTransformFn):
    """Map Wuji LeRobot / runtime dicts onto π0 model inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["image"])
        left_wrist = _parse_image(data["wrist_image_left"])
        right_wrist = _parse_image(data["wrist_image_right"])

        inputs = {
            "state": np.asarray(data["state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class WujiGraspOutputs(transforms.DataTransformFn):
    """Keep the 54 robot action dimensions; drop any model-side padding."""

    action_dim: int = WUJI_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
