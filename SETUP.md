ROOT=/gaozt-test1/zhanxch/FITWAM-dewov9-20260828
ENV=/gaozt-test1/zhanxch/miniconda3/envs/fastwam

export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${ROOT}/checkpoints"
export PYTHONPATH="${ROOT}/src:${ROOT}/scripts:${ROOT}/third_party/dexjoco/dexjoco:${PYTHONPATH:-}"

"${ENV}/bin/python" "${ROOT}/scripts/eval_dexjoco.py" \
  --task-name fold_glasses \
  --run-dir "${ROOT}/configs/eval/dexjoco/mixed_5task_fastwam" \
  --checkpoint-dir "${ROOT}/checkpoints/dexjoco/mixed_5task_fastwam/weights" \
  --checkpoint-steps 55000 \
  --dataset-stats "${ROOT}/artifacts/mixed_5task/dataset_stats.json" \
  --text-embedding "${ROOT}/third_party/FastWAM-infer-in-DexJoco/artifacts/fold_glasses/0c3367ce1d74848cc46b93c6d2eee5e2097dca410a2c95f3da48bd8c8673fa20.t5_len128.wan22ti2v5b.pt" \
  --no-load-text-encoder \
  --gpus 4,5,6,7 \
  --seed-start 0 \
  --seed-end 49 \
  --repeats 4 \
  --action-horizon 32 \
  --replan-steps 24 \
  --num-inference-steps 10 \
  --max-steps 1200



# Joint eval
ROOT=/gaozt-test1/zhanxch/FITWAM-dewov9-20260828
ENV=/gaozt-test1/zhanxch/miniconda3/envs/fastwam

export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${ROOT}/checkpoints"
export PYTHONPATH="${ROOT}/src:${ROOT}/scripts:${ROOT}/third_party/dexjoco/dexjoco:${PYTHONPATH:-}"

"${ENV}/bin/python" "${ROOT}/scripts/eval_dexjoco.py" \
  --task-name fold_glasses \
  --run-dir "${ROOT}/configs/eval/dexjoco/mixed_5task_fastwam_joint" \
  --checkpoint-dir "${ROOT}/checkpoints/dexjoco/mixed_5task_fastwam_joint/weights" \
  --checkpoint-steps 55000 \
  --dataset-stats "${ROOT}/artifacts/mixed_5task/dataset_stats.json" \
  --text-embedding "${ROOT}/third_party/FastWAM-infer-in-DexJoco/artifacts/fold_glasses/0c3367ce1d74848cc46b93c6d2eee5e2097dca410a2c95f3da48bd8c8673fa20.t5_len128.wan22ti2v5b.pt" \
  --no-load-text-encoder \
  --gpus 1,2,3,4,5,6,7 \
  --text-cfg-scale 0 \
  --seed-start 0 \
  --seed-end 49 \
  --repeats 4 \
  --action-horizon 32 \
  --replan-steps 24 \
  --num-inference-steps 10 \
  --max-steps 1200

# Joint collect (same inference stack as eval_dexjoco.py; writes LeRobot rollout_raw)
# Seeds 10086-10135 × 4 so they do not overlap eval 0-49.

"${ENV}/bin/python" "${ROOT}/scripts/collect_dexjoco.py" \
  --task-name fold_glasses \
  --run-dir "${ROOT}/configs/eval/dexjoco/mixed_5task_fastwam_joint" \
  --checkpoint-dir "${ROOT}/checkpoints/dexjoco/mixed_5task_fastwam_joint/weights" \
  --checkpoint-steps 55000 \
  --dataset-stats "${ROOT}/artifacts/mixed_5task/dataset_stats.json" \
  --text-embedding "${ROOT}/third_party/FastWAM-infer-in-DexJoco/artifacts/fold_glasses/0c3367ce1d74848cc46b93c6d2eee5e2097dca410a2c95f3da48bd8c8673fa20.t5_len128.wan22ti2v5b.pt" \
  --no-load-text-encoder \
  --gpus 1,2,3,4,5,6,7 \
  --text-cfg-scale 0 \
  --seed-start 10086 \
  --seed-end 10135 \
  --repeats 4 \
  --action-horizon 32 \
  --replan-steps 24 \
  --num-inference-steps 10 \
  --max-steps 1200

# Joint collect: five tasks in order on GPUs 1,2,3. Wait until those cards are free.
# Seeds 10086-10135 × 4. Outputs: collect_results/dexjoco/<task>/<STAMP>/

ROOT=/gaozt-test1/zhanxch/FITWAM-dewov9-20260828
ENV=/gaozt-test1/zhanxch/miniconda3/envs/fastwam
STAMP=$(date -u +%Y%m%d_%H%M%S)

export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${ROOT}/checkpoints"
export PYTHONPATH="${ROOT}/src:${ROOT}/scripts:${ROOT}/third_party/dexjoco/dexjoco:${PYTHONPATH:-}"

set -euo pipefail
for task in fold_glasses hammer_nail pick_bucket pinch_tongs water_plant; do
  t5=$(ls "${ROOT}/third_party/FastWAM-infer-in-DexJoco/artifacts/${task}"/*.t5_len128.wan22ti2v5b.pt)
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



# Joint prepare: five tasks in order on GPUs 1,2,3 (same cards as collect).
# Reads collect_results/dexjoco/<task>/${STAMP}/. Outputs: prepare_results/dexjoco/<task>/<new-stamp>/

STAMP=20260907_141408 bash /gaozt-test1/zhanxch/FITWAM-dewov9-20260828/scripts/prepare_mixed5_joint_gpus123.sh

