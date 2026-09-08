#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# The collector declares normalization sources as mutually exclusive. Use the
# metadata directory when explicitly provided; otherwise use dataset stats.
normalization_args=(
  --dataset-stats-path
  "${DATASET_STATS_PATH:-${ROOT_DIR}/artifacts/mixed_5task/dataset_stats.json}"
)
if [[ -n "${NORM_STATS_META_DIR:-}" ]]; then
  normalization_args=(--norm-stats-meta-dir "${NORM_STATS_META_DIR}")
fi

# S0 collection leaves actions unclipped. Required orchestration/model options
# are supplied by the caller; additional arguments pass through unchanged.
exec python "${ROOT_DIR}/scripts/dexjoco_async/run_multi_gpu_dexjoco_collect.py" \
  --tasks water_plant \
  "${normalization_args[@]}" \
  --no-action-clip \
  "$@"
