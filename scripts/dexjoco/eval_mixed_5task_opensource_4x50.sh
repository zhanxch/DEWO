#!/usr/bin/env bash
# Official 4×50 eval of mixed_5task joint 55K on the in-repo DEWOv9 async stack.
# Does not use FastWAM-infer-in-DexJoco.
#   GPUS=0,1,2,3 bash scripts/dexjoco/eval_mixed_5task_opensource_4x50.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT_DIR="${ROOT}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/dewo_v2/lib.sh"
cd "${ROOT}"

dewo_v2_require_gpus
dewo_v2_activate_fastwam

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/evaluate_results/dexjoco/mixed_5task_joint_4x50_${STAMP}}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/checkpoints/dexjoco/mixed_5task_fastwam_joint/weights}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-55000}"
CKPT="${CKPT:-${CKPT_DIR}/step_$(printf '%06d' "${CHECKPOINT_STEPS}").pt}"
STATS="${STATS:-${ROOT}/artifacts/mixed_5task/dataset_stats.json}"
RUN_DIR="${RUN_DIR:-${ROOT}/configs/eval/dexjoco/mixed_5task_fastwam_joint}"
CFG_SCALE="${CFG_SCALE:-0}"
LOAD_TEXT_ENCODER="${LOAD_TEXT_ENCODER:-1}"
METHOD="${METHOD:-mixed_5task_fastwam_joint}"
WAIT_IDLE="${WAIT_IDLE:-0}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/orchestrator.log"
log() { echo "[mixed-5task-joint $(date -Is)] $*" | tee -a "${MASTER_LOG}"; }

log "OUT_ROOT=${OUT_ROOT} GPUS=${GPUS} STATS=${STATS}"
log "CKPT=${CKPT} RUN_DIR=${RUN_DIR} CFG_SCALE=${CFG_SCALE} LOAD_TEXT_ENCODER=${LOAD_TEXT_ENCODER}"
log "CFG_TASK_DIR=${CFG_TASK_DIR:-<dewo_v9 default>}"
[[ -f "${STATS}" ]] || { log "ERROR missing mixed stats ${STATS}"; exit 1; }
[[ -e "${CKPT}" ]] || { log "ERROR missing mixed ckpt ${CKPT}"; exit 1; }
[[ -f "${RUN_DIR}/config.yaml" ]] || { log "ERROR missing ${RUN_DIR}/config.yaml"; exit 1; }

JOBS=(
  "mixed_fold_glasses|fold_glasses"
  "mixed_hammer_nail|hammer_nail"
  "mixed_pick_bucket|pick_bucket"
  "mixed_pinch_tongs|pinch_tongs"
  "mixed_water_plant|water_plant"
)

fail=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r jid task <<<"${job}"
  log "START ${jid}"
  if ! TASK="${task}" \
    GPUS="${GPUS}" \
    CKPT="${CKPT}" \
    RUN_DIR="${RUN_DIR}" \
    PRETRAINED_NORM_STATS="${STATS}" \
    STATS="${STATS}" \
    CFG_SCALE="${CFG_SCALE}" \
    LOAD_TEXT_ENCODER="${LOAD_TEXT_ENCODER}" \
    METHOD="${METHOD}" \
    CFG_TASK_DIR="${CFG_TASK_DIR:-}" \
    WAIT_IDLE=0 \
    OUT_ROOT="${OUT_ROOT}/${jid}" \
    bash "${ROOT}/scripts/dewo_v2/eval_cfg_official_4x50.sh"
  then
    log "FAIL ${jid}"
    fail=1
  fi
done

"${ENV_PREFIX}/bin/python" "${ROOT}/scripts/dexjoco/aggregate_opensource_baseline_results.py" \
  --out-root "${OUT_ROOT}" \
  --master-log "${MASTER_LOG}"

log "ALL complete fail=${fail} out=${OUT_ROOT}"
exit "${fail}"
