#!/usr/bin/env bash
# Wait for the running JAX π0 job to finish, then start π0.5 on GPUs 4-7
# using the CLEAN grasp_anything dataset. Does not touch the π0 process.
set -euo pipefail

ROOT=/gaozt-test1/zhanxch/FITWAM-dewov9-20260828
OPENPI=/gaozt-test1/zhanxch/openpi
PY=/gaozt-test1/chenzirui/777pi05jax/.venv/bin/python
PI0_CKPT_DIR="${ROOT}/checkpoints/openpi/pi0_ckpts/pi0_wuji_grasp_anything/grasp_anything_sft"
PI05_READY="${ROOT}/checkpoints/openpi/assets/pi05_wuji_grasp_anything/grasp_anything_clean/norm_stats.json"
PI05_WEIGHTS="${ROOT}/checkpoints/openpi/pi05_base_action_dim_54/params/_METADATA"
LOG_DIR="${ROOT}/checkpoints/openpi/pi05_ckpts"
LOG="${LOG_DIR}/watcher_pi0_then_pi05.log"
PI0_PATTERN='scripts/train.py pi0_wuji_grasp_anything'
PI05_PATTERN='scripts/train.py pi05_wuji_grasp_anything'
FINAL_STEP=29999  # openpi saves the last ckpt at num_train_steps-1

mkdir -p "${LOG_DIR}"
exec >>"${LOG}" 2>&1

log() { echo "[$(date -Is)] $*"; }

pi0_running() {
  pgrep -f "${PI0_PATTERN}" >/dev/null 2>&1
}

latest_pi0_step() {
  python3 - "${PI0_CKPT_DIR}" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
if not root.exists():
    print(-1)
    raise SystemExit
steps = []
for p in root.iterdir():
    name = p.name
    if name.isdigit():
        steps.append(int(name))
print(max(steps) if steps else -1)
PY
}

log "watcher start; waiting for π0 (${PI0_PATTERN}) to finish"
while pi0_running; do
  step="$(latest_pi0_step)"
  log "π0 still running; latest ckpt step=${step}"
  sleep 120
done

step="$(latest_pi0_step)"
log "π0 process exited; latest ckpt step=${step}"
if [[ "${step}" -lt "${FINAL_STEP}" ]]; then
  log "ERROR: π0 did not reach final checkpoint ${FINAL_STEP}; not starting π0.5"
  exit 1
fi

if [[ ! -f "${PI05_WEIGHTS}" ]]; then
  log "ERROR: missing expanded π0.5 weights at ${PI05_WEIGHTS}"
  exit 1
fi
if [[ ! -f "${PI05_READY}" ]]; then
  log "ERROR: missing π0.5 clean-data norm stats at ${PI05_READY}"
  exit 1
fi

if pgrep -f "${PI05_PATTERN}" >/dev/null 2>&1; then
  log "π0.5 train.py already running; watcher exiting"
  exit 0
fi

log "waiting 30s for GPU memory to release"
sleep 30

cd "${OPENPI}"
conda deactivate 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_LEROBOT_HOME="${ROOT}/data/pi"
export OPENPI_DATA_HOME="${ROOT}/checkpoints/openpi"
export PYTHONPATH="${OPENPI}/src"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

log "starting π0.5 full FT on clean grasp_anything, GPUs 4,5,6,7"
"${PY}" scripts/train.py pi05_wuji_grasp_anything --exp-name grasp_anything_clean_sft --overwrite
log "π0.5 train.py exited with code $?"
