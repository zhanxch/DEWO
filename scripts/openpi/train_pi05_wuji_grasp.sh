#!/usr/bin/env bash
# Full-parameter JAX π0.5 fine-tune on GPUs 4-7 (clean grasp_anything).
set -euo pipefail

ROOT=/gaozt-test1/zhanxch/FITWAM-dewov9-20260828
OPENPI=/gaozt-test1/zhanxch/openpi
PY=/gaozt-test1/chenzirui/777pi05jax/.venv/bin/python
LOG_DIR="${ROOT}/checkpoints/openpi/pi05_ckpts"
LOG="${LOG_DIR}/train_pi05_gpu.log"

mkdir -p "${LOG_DIR}"
cd "${OPENPI}"
conda deactivate 2>/dev/null || true
unset JAX_PLATFORMS JAX_PLATFORM_NAME
export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_LEROBOT_HOME="${ROOT}/data/pi"
export OPENPI_DATA_HOME="${ROOT}/checkpoints/openpi"
export PYTHONPATH="${OPENPI}/src"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

echo "[$(date -Is)] starting π0.5 on GPUs 4,5,6,7 (JAX_PLATFORMS unset)" | tee -a "${LOG}"
set +e
"${PY}" scripts/train.py pi05_wuji_grasp_anything --exp-name grasp_anything_clean_sft --overwrite 2>&1 | tee -a "${LOG}"
rc=${PIPESTATUS[0]}
echo "[$(date -Is)] π0.5 train.py exited with code ${rc}" | tee -a "${LOG}"
exit "${rc}"
