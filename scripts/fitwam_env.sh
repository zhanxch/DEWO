#!/usr/bin/env bash
# Shared environment selection for this workspace.
# Source this file from launchers instead of relying on the caller's conda
# activation state.

if [[ -z "${FITWAM_ROOT:-}" ]]; then
  FITWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
FITWAM_ROOT="$(cd "${FITWAM_ROOT}" && pwd)"

# The repository and its user-owned Miniconda installation are siblings.
FITWAM_CONDA_ROOT="${FITWAM_CONDA_ROOT:-$(cd "${FITWAM_ROOT}/.." && pwd)/miniconda3}"
FITWAM_CONDA_ROOT="$(cd "${FITWAM_CONDA_ROOT}" && pwd)"
export FITWAM_ROOT FITWAM_CONDA_ROOT

fitwam_env_prefix() {
  local env_name="${1:?environment name required (fastwam or dexjoco)}"
  case "${env_name}" in
    fastwam|dexjoco)
      printf '%s/envs/%s\n' "${FITWAM_CONDA_ROOT}" "${env_name}"
      ;;
    *)
      echo "[fitwam-env] unsupported environment: ${env_name}" >&2
      return 2
      ;;
  esac
}

fitwam_activate() {
  local env_name="${1:-fastwam}"
  local env_prefix
  env_prefix="$(fitwam_env_prefix "${env_name}")" || return
  if [[ ! -x "${env_prefix}/bin/python" ]]; then
    echo "[fitwam-env] missing Python: ${env_prefix}/bin/python" >&2
    return 2
  fi

  if [[ -f "${FITWAM_CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    # Use this installation explicitly; the root account may also have a
    # separate /root/miniconda3 that does not contain these environments.
    # shellcheck disable=SC1091
    source "${FITWAM_CONDA_ROOT}/etc/profile.d/conda.sh"
    if ! conda activate "${env_name}"; then
      echo "[fitwam-env] failed to activate ${env_name} from ${FITWAM_CONDA_ROOT}" >&2
      return 2
    fi
  else
    export PATH="${env_prefix}/bin:${FITWAM_CONDA_ROOT}/bin:${PATH:-}"
    export CONDA_PREFIX="${env_prefix}"
    export CONDA_DEFAULT_ENV="${env_name}"
    export CONDA_PREFIX_1="${FITWAM_CONDA_ROOT}"
  fi

  # Shell launchers use this path for child processes and generated workers.
  FITWAM_ENV_PREFIX="${env_prefix}"
  export FITWAM_ENV_PREFIX FITWAM_ACTIVE_ENV="${env_name}"
  export PATH="${env_prefix}/bin:${PATH:-}"
}

fitwam_python() {
  printf '%s/bin/python\n' "$(fitwam_env_prefix "${1:-fastwam}")"
}
