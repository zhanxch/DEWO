"""Standard FastWAM policy-server observation contract.

Canonical implementation: ``fastwam.inference.contract``.
"""

from fastwam.inference.contract import *  # noqa: F403
from fastwam.inference.contract import (  # noqa: F401
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
