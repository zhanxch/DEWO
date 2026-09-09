#!/usr/bin/env bash
# Sequential Joint v9.1 prepare on GPUs 1,2,3 for the five DexJoCo tasks.
# Reads collect_results/dexjoco/<task>/${STAMP} from collect_mixed5_joint_gpus123.sh.
#
#   STAMP=20260907_141408 bash scripts/prepare_mixed5_joint_gpus123.sh
#
# Resume fold_glasses on an existing stamp (does not create a new directory)::
#
#   STAMP=20260907_141408 \
#   FOLD_OUTPUT=prepare_results/dexjoco/fold_glasses/20260908_090014 \
#   bash scripts/prepare_mixed5_joint_gpus123.sh
set -euo pipefail

ROOT=/gaozt-test1/zhanxch/FITWAM-dewov9-20260828
ENV=/gaozt-test1/zhanxch/miniconda3/envs/fastwam
STAMP="${STAMP:?Set STAMP to the collect_results stamp, e.g. 20260907_141408}"
GPUS="${GPUS:-1,2,3}"
FOLD_OUTPUT="${FOLD_OUTPUT:-}"

export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${ROOT}/checkpoints"
export PYTHONPATH="${ROOT}/src:${ROOT}/scripts:${ROOT}/third_party/dexjoco/dexjoco:${PYTHONPATH:-}"

echo "[prepare] stamp=${STAMP} gpus=${GPUS} fold_output=${FOLD_OUTPUT:-<new stamp>}"
for task in fold_glasses hammer_nail pick_bucket pinch_tongs water_plant; do
  collect_dir="${ROOT}/collect_results/dexjoco/${task}/${STAMP}"
  if [[ ! -f "${collect_dir}/collect_config.json" ]]; then
    echo "[prepare] ERROR missing collect stamp ${collect_dir}" >&2
    exit 1
  fi
  extra=()
  if [[ "${task}" == fold_glasses && -n "${FOLD_OUTPUT}" ]]; then
    extra+=(--output-dir "${FOLD_OUTPUT}")
  fi
  echo "[prepare] ${task}  start=$(date -Is)"
  "${ENV}/bin/python" "${ROOT}/scripts/prepare_dexjoco.py" \
    --task-name "${task}" \
    --collect-dir "${collect_dir}" \
    --gpus "${GPUS}" \
    "${extra[@]}"
  echo "[prepare] ${task}  done=$(date -Is)"
done

echo "[prepare] all five tasks finished stamp=${STAMP}"
