#!/usr/bin/env bash
# Compat wrapper around the DexJoCo train CLI.
# Official entry (same house style as eval/collect/prepare):
#
#   python scripts/train_dexjoco.py \
#     --task-name fold_glasses --init s0 \
#     --prepare-dir prepare_results/dexjoco/fold_glasses/<stamp> \
#     --gpus 1,2,3,4
#
# Legacy env-var form still works:
#   TASK=fold_glasses INIT=s0 GPUS=1,2,3,4 \
#     ENV_FILE=.../offline_v1_b1_jump_fast.env \
#     bash scripts/dewo_v2/train.sh
#
# Scratch:
#   TASK=mixed_5task INIT=scratch GPUS=0,1,2,3,4,5,6,7 \
#     ENV_FILE=data/mixed_5task_dewo_scratch_*/eve_v02/protocol/offline_v1_b1_jump_fast.env \
#     bash scripts/dewo_v2/train.sh
set -euo pipefail

SCRIPT_DIR="$(realpath -e -- "$(dirname -- "${BASH_SOURCE[0]}")")"
ROOT_DIR="$(realpath -e -- "${SCRIPT_DIR}/../..")"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/fitwam_env.sh"
fitwam_activate "${FITWAM_ENV:-fastwam}"

: "${TASK:?Set TASK (fold_glasses|hammer_nail|water_plant|pick_bucket|pinch_tongs|mixed_5task)}"
: "${GPUS:?Set GPUS to a comma-separated list of physical GPU ids.}"
ENV_FILE="${ENV_FILE:-}"
PREPARE_DIR="${PREPARE_DIR:-}"
if [[ -z "${ENV_FILE}" && -z "${PREPARE_DIR}" ]]; then
  echo "[dewo-train] ERROR: set ENV_FILE or PREPARE_DIR" >&2
  exit 2
fi

ARGS=(
  --task-name "${TASK}"
  --init "${INIT:-s0}"
  --gpus "${GPUS}"
  --dewo-version "${DEWO_VERSION:-v9}"
)
if [[ -n "${ENV_FILE}" ]]; then
  ARGS+=(--env-file "${ENV_FILE}")
fi
if [[ -n "${PREPARE_DIR}" ]]; then
  ARGS+=(--prepare-dir "${PREPARE_DIR}")
fi
[[ -z "${DEWO_OUTPUT_DIR:-}" ]] || ARGS+=(--output-dir "${DEWO_OUTPUT_DIR}")
[[ -z "${INIT_WEIGHTS:-}" ]] || ARGS+=(--init-weights "${INIT_WEIGHTS}")
[[ -z "${EVE_MANIFEST_PATH:-}" ]] || ARGS+=(--eve-manifest "${EVE_MANIFEST_PATH}")
[[ -z "${EVE_VAL_MANIFEST_PATH:-}" ]] || ARGS+=(--eve-val-manifest "${EVE_VAL_MANIFEST_PATH}")
[[ -z "${PRETRAINED_NORM_STATS:-}" ]] || ARGS+=(--pretrained-norm-stats "${PRETRAINED_NORM_STATS}")
LR_VALUE="${LR:-${LEARNING_RATE:-}}"
[[ -z "${LR_VALUE}" ]] || ARGS+=(--learning-rate "${LR_VALUE}")
[[ -z "${TRAIN_MAX_STEPS:-}" ]] || ARGS+=(--max-steps "${TRAIN_MAX_STEPS}")
[[ -z "${BATCH_SIZE:-}" ]] || ARGS+=(--batch-size "${BATCH_SIZE}")
[[ -z "${PRIMARY_PER_BATCH:-}" ]] || ARGS+=(--primary-per-batch "${PRIMARY_PER_BATCH}")
[[ -z "${ADAPTER_RANK:-}" ]] || ARGS+=(--adapter-rank "${ADAPTER_RANK}")
[[ -z "${ADAPTER_ALPHA:-}" ]] || ARGS+=(--adapter-alpha "${ADAPTER_ALPHA}")
[[ -z "${DEWO_VARIANT:-}" ]] || ARGS+=(--variant "${DEWO_VARIANT}")
[[ -z "${DEWO_PROTOCOL:-}" ]] || ARGS+=(--protocol "${DEWO_PROTOCOL}")
[[ -z "${FITWAM_WANDB_GROUP:-}" ]] || ARGS+=(--wandb-group "${FITWAM_WANDB_GROUP}")
[[ -z "${WANDB_MODE:-}" ]] || ARGS+=(--wandb-mode "${WANDB_MODE}")
[[ -z "${RUN_ID:-}" ]] || ARGS+=(--run-id "${RUN_ID}")
[[ -z "${TMUX_SESSION:-}" ]] || ARGS+=(--tmux-session "${TMUX_SESSION}")

if [[ "${RUN_INLINE:-0}" == "1" ]]; then
  ARGS+=(--inline)
else
  ARGS+=(--tmux)
fi
if [[ "${USE_VAE_LATENT_CACHE:-1}" == "0" ]]; then
  ARGS+=(--no-vae)
fi
if [[ "${SKIP_VAE_PREENCODE:-}" == "1" ]]; then
  ARGS+=(--skip-vae-preencode)
elif [[ "${SKIP_VAE_PREENCODE:-}" == "0" ]]; then
  ARGS+=(--no-skip-vae-preencode)
fi
if [[ -n "${DEWO_HYDRA_OVERRIDES:-}" ]]; then
  # shellcheck disable=SC2206
  extra=(${DEWO_HYDRA_OVERRIDES})
  ARGS+=(-- "${extra[@]}")
fi

exec python "${ROOT_DIR}/scripts/train_dexjoco.py" "${ARGS[@]}"
