TASK=hammer_nail GPUS=0,1,2,3 \
  ENV_FILE=data/hammer_nail_dewo_v2_pair_20260817_193358/eve_v02/protocol/offline_v1_b1_jump_fast.env \
  bash scripts/dewo_v2/train.sh
# 可选：LR=1e-4 MAX_STEPS=15000 BATCH_SIZE=16

## Runtime environments

The launchers resolve the user-owned Miniconda installation next to this
repository (`../miniconda3`) without requiring a prior `conda activate`:

- `fastwam` is the default for training, preprocessing, and policy servers.
- `dexjoco` is selected by DexJoCo client/evaluation launchers.

The shared resolver is `scripts/fitwam_env.sh`. Root's login shells also load
it automatically, so `python` from a new root session points to
`miniconda3/envs/fastwam/bin/python`. To switch explicitly inside a shell:

```bash
source /gaozt-test1/zhanxch/FITWAM-dewov9-20260828/scripts/fitwam_env.sh
fitwam_activate fastwam   # or: fitwam_activate dexjoco
```
