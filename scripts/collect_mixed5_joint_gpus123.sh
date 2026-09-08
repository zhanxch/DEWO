#!/usr/bin/env bash
# Temporary: sequential Joint collect on GPUs 1,2,3 for the five DexJoCo tasks.
set -euo pipefail

ROOT=/gaozt-test1/zhanxch/FITWAM-dewov9-20260828
ENV=/gaozt-test1/zhanxch/miniconda3/envs/fastwam
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"

export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${ROOT}/checkpoints"
export PYTHONPATH="${ROOT}/src:${ROOT}/scripts:${ROOT}/third_party/dexjoco/dexjoco:${PYTHONPATH:-}"

echo "[collect] stamp=${STAMP} gpus=1,2,3"
for task in fold_glasses hammer_nail pick_bucket pinch_tongs water_plant; do
  shopt -s nullglob
  t5_files=("${ROOT}/third_party/FastWAM-infer-in-DexJoco/artifacts/${task}"/*.t5_len128.wan22ti2v5b.pt)
  shopt -u nullglob
  if [[ ${#t5_files[@]} -ne 1 ]]; then
    echo "[collect] ERROR: expected exactly one T5 cache for ${task}, got ${#t5_files[@]}" >&2
    printf '  %s\n' "${t5_files[@]}" >&2
    exit 1
  fi
  t5="${t5_files[0]}"
  echo "[collect] ${task}  start=$(date -Is)"
  "${ENV}/bin/python" "${ROOT}/scripts/collect_dexjoco.py" \
    --task-name "${task}" \
    --run-dir "${ROOT}/configs/eval/dexjoco/mixed_5task_fastwam_joint" \
    --checkpoint-dir "${ROOT}/checkpoints/dexjoco/mixed_5task_fastwam_joint/weights" \
    --checkpoint-steps 55000 \
    --dataset-stats "${ROOT}/artifacts/mixed_5task/dataset_stats.json" \
    --text-embedding "${t5}" \
    --no-load-text-encoder \
    --gpus 1,2,3 \
    --text-cfg-scale 0 \
    --seed-start 10086 \
    --seed-end 10135 \
    --repeats 4 \
    --action-horizon 32 \
    --replan-steps 24 \
    --num-inference-steps 10 \
    --max-steps 1200 \
    --output-dir "${ROOT}/collect_results/dexjoco/${task}/${STAMP}"
  echo "[collect] ${task}  done=$(date -Is)"
done

echo "[collect] all five tasks finished stamp=${STAMP}"
