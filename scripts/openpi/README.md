# Official JAX π0 / π0.5 full fine-tune on Wuji grasp_anything

Dataset is adapted to OpenPI's LeRobot layout (do not train on the GR00T-key original).

## What is already prepared

| Item | Path |
| --- | --- |
| Official JAX `pi0_base` (11.19 GiB, 33/33 files) | `FITWAM-dewov9-20260828/checkpoints/openpi/pi0_base` |
| Action/state heads expanded 32 → 54 | `.../pi0_base_action_dim_54` |
| PI-layout dataset (keys `state`/`actions`/`image`/`wrist_image_*`) | `FITWAM-dewov9-20260828/data/pi/grasp_anything` |
| Official JAX `pi05_base` expanded 32 → 54 | `.../pi05_base_action_dim_54` |
| PI-layout **clean** dataset | `FITWAM-dewov9-20260828/data/pi/grasp_anything_clean` |
| π0.5 train config | `pi05_wuji_grasp_anything` |

π0.5 uses official `pi05_libero` conventions: `pi05=True`, `discrete_state_input=False`, **no** arm-delta transform (absolute 54-D). Watcher: `scripts/openpi/watch_pi0_then_train_pi05.sh` waits for π0 `grasp_anything_sft` to finish, then trains π0.5 on GPUs 4–7.

Convention used (π0, not π0.5): delta on 14 arm joints, absolute on 40 finger joints; 3 cameras mapped to `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb`; language from LeRobot `task`. Full-parameter FT (no LoRA).

## Train (not started)

```bash
cd /gaozt-test1/zhanxch/openpi
conda deactivate 2>/dev/null || true
export HF_LEROBOT_HOME=/gaozt-test1/zhanxch/FITWAM-dewov9-20260828/data/pi
export OPENPI_DATA_HOME=/gaozt-test1/zhanxch/FITWAM-dewov9-20260828/checkpoints/openpi
export PYTHONPATH=/gaozt-test1/zhanxch/openpi/src
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# Prefer this cluster JAX env until `uv sync` in ./openpi finishes (GitHub lerobot fetch is flaky).
/gaozt-test1/chenzirui/777pi05jax/.venv/bin/python scripts/train.py \
  pi0_wuji_grasp_anything --exp-name grasp_anything_sft --overwrite
```

Re-run prepare (norm stats, etc.) with:

`bash /gaozt-test1/zhanxch/FITWAM-dewov9-20260828/scripts/openpi/prepare_pi0_wuji_grasp_train.sh`
