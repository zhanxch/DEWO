#!/usr/bin/env bash
# Thin wrapper: DEWO v9 prepare → scripts/prepare_dexjoco.py
#
#   TASK=fold_glasses COLLECT_ROOT=... GPUS=4,5,6,7 \
#     RUN_SCAN=0 QUEUE_FILE=logs/mixed_v9_pipeline.env \
#     bash scripts/dewo_v2/prepare_v9_mixed_task.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

dewo_v2_require_task
dewo_v2_load_task "${TASK}"
dewo_v2_resolve_open_repo
dewo_v2_pin_joint_s0
dewo_v2_activate_fastwam

COLLECT_ROOT="${COLLECT_ROOT:?Set COLLECT_ROOT to a collect_dexjoco stamp or data/${TASK}_mixed_s0_collect_*}"
COLLECT_ROOT="$(realpath -e "${COLLECT_ROOT}")"
GPUS="${GPUS:-4,5,6,7}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/data/${TASK}_dewo_v9_pair_${STAMP}}"
PAIR_OUT="${PAIR_OUT:-${ROOT_DIR}/data/${TASK}_dewo_v9_pair_full_lerobot}"
QUEUE_FILE="${QUEUE_FILE:-${ROOT_DIR}/logs/mixed_v9_pipeline.env}"
RUN_SCAN="${RUN_SCAN:-0}"
dewo_v2_assert_path_for_task COLLECT_ROOT "${COLLECT_ROOT}"
dewo_v2_assert_path_for_task EXP_ROOT "${EXP_ROOT}"
dewo_v2_assert_path_for_task PAIR_OUT "${PAIR_OUT}"
mkdir -p "${EXP_ROOT}/logs" "$(dirname "${QUEUE_FILE}")"

flags=(
  --task-name "${TASK}"
  --collect-dir "${COLLECT_ROOT}"
  --gpus "${GPUS}"
  --output-dir "${EXP_ROOT}"
  --pair-dataset "${PAIR_OUT}"
  --queue-file "${QUEUE_FILE}"
  --run-dir "${SOURCE_CONFIG}"
  --checkpoint-dir "$(dirname "${CKPT}")"
  --dataset-stats "${STATS}"
)
if [[ -n "${SCAN_ROOT:-}" ]]; then
  flags+=(--scan-dir "${SCAN_ROOT}")
fi
ckpt_step="${CKPT##*/}"
ckpt_step="${ckpt_step#step_}"
ckpt_step="${ckpt_step%.pt}"
if [[ "${ckpt_step}" =~ ^[0-9]+$ ]]; then
  flags+=(--checkpoint-steps "${ckpt_step}")
fi
if [[ "${RUN_SCAN}" != "1" ]]; then
  flags+=(--require-existing-scan)
fi
if [[ "${SKIP_VAE_PREENCODE:-0}" == "1" || "${USE_VAE_LATENT_CACHE:-1}" == "0" ]]; then
  flags+=(--no-vae)
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  flags+=(--overwrite)
fi
if [[ -n "${TEXT_EMB:-}" && -f "${TEXT_EMB}" ]]; then
  flags+=(--text-embedding "${TEXT_EMB}")
fi

echo "[v9-prep ${TASK} $(date -Is)] python scripts/prepare_dexjoco.py ${flags[*]}"
exec "${FITWAM_ENV_PREFIX}/bin/python" "${ROOT_DIR}/scripts/prepare_dexjoco.py" "${flags[@]}"
