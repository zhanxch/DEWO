#!/usr/bin/env bash
# Opensource-aligned 4×50 collect (seeds 10086..10135 × 4 = 200).
# Defaults to in-repo mixed FastWAMJoint 55K + live T5 (no OPEN_REPO / T5 .pt).
#   TASK=water_plant GPUS=4,5,6,7 bash scripts/dewo_v2/collect_opensource_4x50.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

dewo_v2_require_task
dewo_v2_require_gpus
dewo_v2_load_task "${TASK}"
dewo_v2_resolve_open_repo
if [[ "${USE_JOINT_S0:-1}" == "1" ]]; then
  dewo_v2_pin_joint_s0
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/data/${TASK}_mixed_s0_collect_${STAMP}}"
MODEL_CONFIG="${MODEL_CONFIG:-${JOINT_S0_CFG:-${ROOT_DIR}/configs/eval/dexjoco/mixed_5task_fastwam_joint/config.yaml}}"

if [[ ! -f "${STATS}" ]]; then
  echo "[dewo-v2-collect] ERROR missing dataset_stats ${STATS}" >&2
  exit 2
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "[dewo-v2-collect] ERROR missing ckpt ${CKPT}" >&2
  exit 2
fi
if [[ ! -f "${MODEL_CONFIG}" ]]; then
  echo "[dewo-v2-collect] ERROR missing model config ${MODEL_CONFIG}" >&2
  exit 2
fi

# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/fitwam_env.sh"
fitwam_activate "${FITWAM_ENV:-fastwam}"
ENV_PREFIX="${FITWAM_ENV_PREFIX}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${ROOT_DIR}/checkpoints"

mkdir -p "${OUTPUT_DIR}/logs"
echo "[dewo-v2-collect] task=${TASK} ckpt=${CKPT}"
echo "[dewo-v2-collect] model_config=${MODEL_CONFIG}"
echo "[dewo-v2-collect] collect_max_steps=${COLLECT_MAX_STEPS:-${MAX_STEPS:-1000}} eval_max_steps=${MAX_STEPS:-1000}"
echo "[dewo-v2-collect] output=${OUTPUT_DIR} gpus=${GPUS}"

overwrite_flag=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  overwrite_flag=(--overwrite)
fi
text_emb_flag=()
if [[ -n "${TEXT_EMB:-}" && -f "${TEXT_EMB}" ]]; then
  text_emb_flag=(--text-embedding "${TEXT_EMB}")
fi

exec "${ENV_PREFIX}/bin/python" scripts/dexjoco/collect_opensource_4x50.py \
  --gpus "${GPUS}" \
  --seed-start "${SEED_START}" \
  --seed-end "${SEED_END}" \
  --repeats "${REPEATS}" \
  --max-steps "${COLLECT_MAX_STEPS:-${MAX_STEPS:-1000}}" \
  --action-horizon "${ACTION_HORIZON}" \
  --replan-steps "${REPLAN_STEPS}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --checkpoint "${CKPT}" \
  --model-config "${MODEL_CONFIG}" \
  --dataset-stats "${STATS}" \
  --source-dataset "${SOURCE_DATASET}" \
  --output-dir "${OUTPUT_DIR}" \
  --task-name "${TASK}" \
  --success-prompt "${SUCCESS_PROMPT}" \
  --skip-pin-check \
  "${text_emb_flag[@]}" \
  "${overwrite_flag[@]}"
