#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT_DIR}/../miniconda3/envs/fastwam/bin/python"
SOURCE_CONFIG="${ROOT_DIR}/configs/eval/dexjoco/fastwam_dexjoco.yaml"
STATS="${ROOT_DIR}/artifacts/mixed_5task/dataset_stats.json"
QUEUE_LOG="${ROOT_DIR}/logs/expert_vae_queue_20260907.log"

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${ROOT_DIR}/checkpoints"

mkdir -p "${ROOT_DIR}/logs"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "[$(date -Is)] queue started; waiting for hammer_nail VAE workers"
while pgrep -f '[p]recompute_vae_latents.py.*hammer_nail_expert_vae_20260907/vae_latent_cache' >/dev/null 2>&1; do
  echo "[$(date -Is)] hammer_nail still running"
  sleep 30
done
echo "[$(date -Is)] hammer_nail VAE workers finished; starting remaining expert tasks"

for task in fold_glasses pick_bucket pinch_tongs water_plant; do
  DATASET="${ROOT_DIR}/data/dexjoco/dexjoco_lerobot_datasets/${task}"
  OUT="${ROOT_DIR}/data/${task}_expert_vae_20260907"
  EVE="${OUT}/eve_v02"
  LOG="${OUT}/logs"
  DATASET_ID="${task}_expert_success"
  mkdir -p "${EVE}/splits" "${EVE}/manifests" "${EVE}/protocol" "${LOG}" \
    "${OUT}/text_embeds_cache" "${OUT}/vae_latent_cache"

  if [[ ! -f "${EVE}/splits/episode_splits.jsonl" ]]; then
    echo "[$(date -Is)] ${task}: building episode splits"
    "${PYTHON}" "${ROOT_DIR}/scripts/everobot/build_episode_split.py" \
      --dataset "${DATASET_ID}=${DATASET}" \
      --force-success-dataset-id "${DATASET_ID}" \
      --val-fraction 0.2 --seed 20260812 \
      --output "${EVE}/splits/episode_splits.jsonl" \
      --report "${EVE}/splits/episode_splits.report.json"
  fi
  if [[ ! -f "${EVE}/episode_meta.jsonl" ]]; then
    echo "[$(date -Is)] ${task}: initializing expert sidecar"
    "${PYTHON}" "${ROOT_DIR}/scripts/everobot/build_eve_sidecar.py" init-base \
      --dataset-root "${DATASET}" --dataset-id "${DATASET_ID}" --eve-root "${EVE}" \
      --task-name "${task}" --source-type expert_success --source-policy expert \
      --collection-round -1 --force-success --split-map "${EVE}/splits/episode_splits.jsonl" \
      --config-path "${SOURCE_CONFIG}" --code-commit unknown
  fi
  if [[ ! -f "${EVE}/manifests/offline_expert_success.json" ]]; then
    echo "[$(date -Is)] ${task}: building train manifest"
    "${PYTHON}" "${ROOT_DIR}/scripts/everobot/build_eve_sidecar.py" build-manifest \
      --eve-root "${EVE}" --manifest-name offline_expert_success \
      --include-outcomes success --success-dataset-ids "${DATASET_ID}" \
      --success-sample-mode episode_only --splits train
  fi
  if [[ ! -f "${EVE}/manifests/offline_selection_primary_success.json" ]]; then
    echo "[$(date -Is)] ${task}: building val manifest"
    "${PYTHON}" "${ROOT_DIR}/scripts/everobot/build_eve_sidecar.py" build-manifest \
      --eve-root "${EVE}" --manifest-name offline_selection_primary_success \
      --include-outcomes success --success-dataset-ids "${DATASET_ID}" \
      --success-sample-mode episode_only --splits val
  fi

  export BASE_DATASET="${DATASET}"
  export ROLLOUT_RAW="${DATASET}"
  export EVE_MANIFEST_PATH="${EVE}/manifests/offline_expert_success.json"
  export EVE_VAL_MANIFEST_PATH="${EVE}/manifests/offline_selection_primary_success.json"
  export PRETRAINED_NORM_STATS="${STATS}"
  export TEXT_EMBEDDING_CACHE_DIR="${OUT}/text_embeds_cache"
  export VAE_LATENT_CACHE_DIR="${OUT}/vae_latent_cache"
  export REQUIRE_VAE_LATENT_CACHE=0

  echo "[$(date -Is)] ${task}: starting VAE shards on GPUs 0,1,2,3"
  pids=()
  for rank in 0 1 2 3; do
    (
      export CUDA_VISIBLE_DEVICES="${rank}"
      unset WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT GROUP_RANK LOCAL_WORLD_SIZE
      "${PYTHON}" "${ROOT_DIR}/scripts/precompute_vae_latents.py" \
        task=dexjoco/dexjoco_dewo_v9_offline_b1_jump_fast_uncond \
        "+vae_latent_cache_dir=${VAE_LATENT_CACHE_DIR}" \
        "+vae_shard_rank=${rank}" "+vae_shard_world=4" "+encode_val=false" \
        >"${LOG}/precompute_vae_latents_expert.shard${rank}.log" 2>&1
    ) &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then failed=1; fi
  done
  count="$(find "${VAE_LATENT_CACHE_DIR}" -type f -name '*.pt' | wc -l | tr -d ' ')"
  echo "[$(date -Is)] ${task}: VAE workers done failed=${failed} files=${count} cache=${VAE_LATENT_CACHE_DIR}"
  if [[ "${failed}" -ne 0 ]]; then
    echo "[$(date -Is)] ${task}: ERROR; stopping queue"
    exit 2
  fi
done

echo "[$(date -Is)] all remaining expert VAE tasks completed"
