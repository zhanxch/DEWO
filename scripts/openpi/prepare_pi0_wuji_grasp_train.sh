#!/usr/bin/env bash
# Prepare official JAX π0_base full fine-tuning on the Wuji grasp_anything dataset.
# Does not launch training.
set -euo pipefail

ROOT=/gaozt-test1/zhanxch/FITWAM-dewov9-20260828
OPENPI=/gaozt-test1/zhanxch/openpi
CKPT_DIR="${ROOT}/checkpoints/openpi"
PI0_BASE="${CKPT_DIR}/pi0_base"
PI0_BASE_54="${CKPT_DIR}/pi0_base_action_dim_54"
DATASET="${ROOT}/data/pi/grasp_anything"
# Cluster JAX env matching official openpi pins (jax 0.5.3 / flax 0.10.2). Used when
# `uv sync` cannot fetch GitHub-pinned lerobot.
FALLBACK_PY=/gaozt-test1/chenzirui/777pi05jax/.venv/bin/python

export HF_LEROBOT_HOME="${ROOT}/data/pi"
export OPENPI_DATA_HOME="${CKPT_DIR}"
export GIT_LFS_SKIP_SMUDGE=1
export PYTHONPATH="${OPENPI}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${OPENPI}"

if [[ -x "${OPENPI}/.venv/bin/python" ]] && "${OPENPI}/.venv/bin/python" -c 'import jax, flax, orbax.checkpoint, lerobot' 2>/dev/null; then
  PY="${OPENPI}/.venv/bin/python"
elif [[ -x "${FALLBACK_PY}" ]]; then
  echo "[prepare] using cluster JAX env ${FALLBACK_PY}"
  PY="${FALLBACK_PY}"
else
  echo "[prepare] uv sync (official OpenPI JAX env)"
  uv sync
  uv pip install -e .
  PY="${OPENPI}/.venv/bin/python"
fi

if [[ ! -f "${PI0_BASE}/params/_METADATA" ]]; then
  echo "Missing ${PI0_BASE}. Run scripts/openpi/download_pi0_base.sh"
  exit 1
fi

if [[ ! -d "${DATASET}/data/chunk-000" ]]; then
  echo "[prepare] converting LeRobot dataset to OpenPI keys"
  python3 "${ROOT}/scripts/openpi/convert_grasp_anything_to_pi_lerobot.py"
fi

if [[ ! -f "${PI0_BASE_54}/params/_METADATA" ]]; then
  echo "[prepare] expanding π0_base action head 32 -> 54"
  "${PY}" "${OPENPI}/scripts/expand_pi0_action_dim.py" \
    --input-path "${PI0_BASE}" \
    --output-path "${PI0_BASE_54}" \
    --target-action-dim 54
fi

if [[ ! -f "${CKPT_DIR}/assets/pi0_wuji_grasp_anything/grasp_anything/norm_stats.json" ]]; then
  echo "[prepare] computing norm stats"
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "${PY}" "${OPENPI}/scripts/compute_norm_stats.py" \
    --config-name pi0_wuji_grasp_anything --max-frames 20000
fi

cat <<EOF

Ready to train (not started):

  cd ${OPENPI}
  conda deactivate 2>/dev/null || true
  export HF_LEROBOT_HOME=${HF_LEROBOT_HOME}
  export OPENPI_DATA_HOME=${OPENPI_DATA_HOME}
  export PYTHONPATH=${OPENPI}/src
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
  # Full-parameter JAX fine-tune. One H20 96GB is enough; add --fsdp-devices 2 if OOM.
  ${PY} scripts/train.py pi0_wuji_grasp_anything --exp-name grasp_anything_sft --overwrite

EOF
