#!/usr/bin/env bash
# DEWO(scratch) FastWAMJoint pipeline: ActionDiT → 5-task collect → prepare → concat → train.
# Parameterized. Do not bake dated GPUs into this file.
#
#   GPUS=0,1,2,3,4,5,6,7 STAMP=20260906_210000 \
#     bash scripts/dewo_v2/run_scratch_joint_pipeline.sh
#
# Phases: actiondit,collect,prepare,concat,train,all
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PHASES="${PHASES:-all}"
TASKS_CSV="${TASKS:-fold_glasses,hammer_nail,pick_bucket,pinch_tongs,water_plant}"
MIN_COLLECT_SR="${MIN_COLLECT_SR:-0.20}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-300000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WAIT_IDLE="${WAIT_IDLE:-0}"
export GPUS STAMP

dewo_v2_require_gpus
dewo_v2_joint_s0_defaults
dewo_v2_resolve_open_repo
dewo_v2_activate_fastwam

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/mixed_scratch_pipeline_${STAMP}}"
QUEUE_FILE="${QUEUE_FILE:-${LOG_DIR}/prepare_queue.env}"
MIXED_ROOT="${MIXED_ROOT:-${ROOT_DIR}/data/mixed_5task_dewo_scratch_${STAMP}}"
MASTER="${LOG_DIR}/pipeline.log"
mkdir -p "${LOG_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${ROOT_DIR}/checkpoints"
export RUN_INLINE=1
export USE_VAE_LATENT_CACHE=1
export SKIP_VAE_PREENCODE=1
export REQUIRE_VAE_LATENT_CACHE=1

log() { echo "[scratch-pipe $(date -Is)] $*" | tee -a "${MASTER}"; }

want_phase() {
  local name="$1"
  [[ "${PHASES}" == "all" || ",${PHASES}," == *",${name},"* ]]
}

wan_dit_ready() {
  ls "${ROOT_DIR}/checkpoints/Wan-AI/Wan2.2-TI2V-5B"/diffusion_pytorch_model*.safetensors >/dev/null 2>&1
}

ensure_wan22_dit_dir() {
  local dit_dir="${ROOT_DIR}/checkpoints/Wan-AI/Wan2.2-TI2V-5B"
  if [[ -L "${dit_dir}" && ! -e "${dit_dir}" ]]; then
    log "replace dangling Wan2.2 symlink ${dit_dir}"
    rm -f "${dit_dir}"
  fi
  mkdir -p "${dit_dir}"
}

download_wan22_dit() {
  ensure_wan22_dit_dir
  if wan_dit_ready; then
    log "Wan DiT shards already present"
    return 0
  fi
  log "download Wan2.2 DiT safetensors (CPU/network, skip_download=false)"
  DIFFSYNTH_SKIP_DOWNLOAD=false DIFFSYNTH_MODEL_BASE_PATH="${ROOT_DIR}/checkpoints" \
    python - <<'PY'
from fastwam.models.wan22.helpers.io import ModelConfig

cfg = ModelConfig(
    model_id="Wan-AI/Wan2.2-TI2V-5B",
    origin_file_pattern="diffusion_pytorch_model*.safetensors",
)
cfg.download_if_necessary()
print(f"[scratch-pipe] Wan DiT path={cfg.path}", flush=True)
if not cfg.path:
    raise SystemExit("Wan DiT download produced an empty path")
PY
}

build_action_dit() {
  if [[ -f "${ACTION_DIT}" ]]; then
    log "reuse ActionDiT ${ACTION_DIT}"
    return 0
  fi
  download_wan22_dit
  log "build ActionDiT -> ${ACTION_DIT}"
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" python "${ROOT_DIR}/scripts/preprocess_action_dit_backbone.py" \
    --model-config "${ROOT_DIR}/configs/model/fastwam.yaml" \
    --output "${ACTION_DIT}" \
    --device cuda \
    --dtype bfloat16 \
    2>&1 | tee -a "${LOG_DIR}/action_dit.log"
  test -f "${ACTION_DIT}" || { log "ERROR ActionDiT missing after preprocess"; return 2; }
}

IFS=',' read -r -a TASK_ARR <<< "${TASKS_CSV}"

collect_sr() {
  python3 - "$1" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(2)
row = json.loads(p.read_text())
sr = row.get("pooled_success_rate")
if sr is None:
    raise SystemExit(3)
print(f"{float(sr):.6f}")
PY
}

if [[ "${WAIT_IDLE}" == "1" ]]; then
  dewo_v2_wait_gpus_idle "${MASTER}" "scratch-pipe"
fi

log "stamp=${STAMP} gpus=${GPUS} phases=${PHASES} tasks=${TASKS_CSV}"
log "ckpt=${JOINT_S0_CKPT}"
log "stats=${JOINT_S0_STATS}"
log "cfg=${JOINT_S0_CFG}"
test -f "${JOINT_S0_CKPT}" || { log "ERROR missing joint ckpt ${JOINT_S0_CKPT}"; exit 2; }
test -f "${JOINT_S0_STATS}" || { log "ERROR missing stats ${JOINT_S0_STATS}"; exit 2; }
test -f "${JOINT_S0_CFG}" || { log "ERROR missing cfg ${JOINT_S0_CFG}"; exit 2; }

# Collect does not need Wan DiT (eval yaml skip_dit_load_from_pretrain=true).
# ActionDiT interpolation does. Download DiT shards in the background while collect runs,
# unless another downloader (separate tmux) is already writing shards.
DIT_DOWNLOAD_PID=""
if want_phase actiondit && [[ ! -f "${ACTION_DIT}" ]]; then
  ensure_wan22_dit_dir
  if wan_dit_ready; then
    log "Wan DiT shards present; ActionDiT will build after collect"
  elif ls "${ROOT_DIR}/checkpoints/Wan-AI/Wan2.2-TI2V-5B"/diffusion_pytorch_model*.safetensors >/dev/null 2>&1 \
    || pgrep -f 'Wan2.2-TI2V-5B' >/dev/null 2>&1; then
    log "Wan DiT download already in progress; will wait after collect"
  else
    log "start background Wan DiT download"
    (
      download_wan22_dit
    ) >>"${LOG_DIR}/wan_dit_download.log" 2>&1 &
    DIT_DOWNLOAD_PID="$!"
  fi
fi

if want_phase collect; then
  for TASK in "${TASK_ARR[@]}"; do
    export TASK
    COLLECT_ROOT="${ROOT_DIR}/data/${TASK}_mixed_s0_collect_${STAMP}"
    AGG="${COLLECT_ROOT}/aggregate.json"
    if [[ -f "${AGG}" ]]; then
      sr="$(collect_sr "${AGG}")"
      log "reuse collect ${TASK} sr=${sr} ${COLLECT_ROOT}"
    else
      log "COLLECT ${TASK} -> ${COLLECT_ROOT}"
      TASK="${TASK}" GPUS="${GPUS}" STAMP="${STAMP}" OUTPUT_DIR="${COLLECT_ROOT}" \
        bash "${ROOT_DIR}/scripts/dewo_v2/collect_opensource_4x50.sh" \
        2>&1 | tee -a "${LOG_DIR}/collect_${TASK}.log"
    fi
    test -f "${AGG}" || { log "ERROR missing ${AGG}"; exit 2; }
    sr="$(collect_sr "${AGG}")"
    python3 - "${sr}" "${MIN_COLLECT_SR}" "${TASK}" <<'PY'
import sys
sr, lo, task = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
print(f"[scratch-pipe] {task} collect_sr={sr:.4f} min={lo:.4f}")
if sr < lo:
    raise SystemExit(
        f"Collect SR {sr:.4f} < {lo:.4f} for {task}. "
        "Wrong model/stats/ckpt (~12% means stop)."
    )
PY
  done
fi

if want_phase actiondit; then
  if [[ -n "${DIT_DOWNLOAD_PID}" ]]; then
    log "wait for Wan DiT download pid=${DIT_DOWNLOAD_PID}"
    if ! wait "${DIT_DOWNLOAD_PID}"; then
      log "ERROR Wan DiT download failed; see ${LOG_DIR}/wan_dit_download.log"
      exit 2
    fi
  fi
  if ! wan_dit_ready && [[ ! -f "${ACTION_DIT}" ]]; then
    log "wait for Wan DiT shards (external download)"
    for _i in $(seq 1 180); do
      if wan_dit_ready; then
        break
      fi
      sleep 20
    done
  fi
  build_action_dit || exit 2
fi

if want_phase prepare; then
  : > "${QUEUE_FILE}"
  for TASK in "${TASK_ARR[@]}"; do
    export TASK
    COLLECT_ROOT="${ROOT_DIR}/data/${TASK}_mixed_s0_collect_${STAMP}"
    EXP_ROOT="${ROOT_DIR}/data/${TASK}_dewo_v9_pair_${STAMP}"
    PAIR_OUT="${ROOT_DIR}/data/${TASK}_dewo_v9_pair_full_lerobot_${STAMP}"
    log "PREPARE ${TASK} collect=${COLLECT_ROOT}"
    TASK="${TASK}" COLLECT_ROOT="${COLLECT_ROOT}" EXP_ROOT="${EXP_ROOT}" \
      PAIR_OUT="${PAIR_OUT}" RUN_SCAN=1 GPUS="${GPUS}" STAMP="${STAMP}" \
      QUEUE_FILE="${QUEUE_FILE}" \
      bash "${ROOT_DIR}/scripts/dewo_v2/prepare_v9_mixed_task.sh" \
      2>&1 | tee -a "${LOG_DIR}/prepare_${TASK}.log"
  done
fi

if want_phase concat; then
  log "CONCAT mixed pool -> ${MIXED_ROOT}"
  python "${ROOT_DIR}/scripts/dewo_v2/concat_mixed_scratch_pool.py" \
    --queue-file "${QUEUE_FILE}" \
    --output-root "${MIXED_ROOT}" \
    --action-dit "${ACTION_DIT}" \
    --stats "${JOINT_S0_STATS}" \
    --source-config "${JOINT_S0_CFG}" \
    2>&1 | tee -a "${LOG_DIR}/concat.log"
fi

MIXED_ENV="${MIXED_ROOT}/eve_v02/protocol/offline_v1_b1_jump_fast.env"
if want_phase train; then
  test -f "${MIXED_ENV}" || { log "ERROR missing mixed env ${MIXED_ENV}"; exit 2; }
  test -f "${ACTION_DIT}" || { log "ERROR missing ActionDiT ${ACTION_DIT}"; exit 2; }
  n_vae="$(find "${MIXED_ROOT}/vae_latent_cache" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${n_vae}" -lt 1 ]]; then
    log "ERROR empty VAE cache ${MIXED_ROOT}/vae_latent_cache"
    exit 2
  fi
  log "TRAIN scratch joint steps=${TRAIN_MAX_STEPS} bs=${BATCH_SIZE} vae_files=${n_vae}"
  TASK=mixed_5task INIT=scratch DEWO_VERSION=v9 \
    GPUS="${GPUS}" ENV_FILE="${MIXED_ENV}" \
    TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS}" BATCH_SIZE="${BATCH_SIZE}" \
    USE_VAE_LATENT_CACHE=1 SKIP_VAE_PREENCODE=1 RUN_INLINE=1 \
    INIT_WEIGHTS="${ACTION_DIT}" \
    TMUX_SESSION="mixed_5task_dewo_scratch_joint_${STAMP}" \
    bash "${ROOT_DIR}/scripts/dewo_v2/train.sh" \
    2>&1 | tee -a "${LOG_DIR}/train.log"
fi

log "DONE phases=${PHASES} stamp=${STAMP}"
