#!/usr/bin/env bash

set -euo pipefail

repo_root=$(
  cd "$(dirname "${BASH_SOURCE[0]}")/../.."
  pwd
)

resolve_ascend_home() {
  local candidates=()

  if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    candidates+=("${ASCEND_HOME_PATH}")
  fi
  candidates+=(
    "/usr/local/Ascend/cann"
    "/usr/local/Ascend/ascend-toolkit/latest"
    "/usr/local/Ascend/ascend-toolkit"
    "${HOME}/Ascend/cann"
  )

  local candidate=""
  for candidate in "${candidates[@]}"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

source_ascend_env() {
  local ascend_home="$1"
  local env_scripts=(
    "${ascend_home}/bin/setenv.bash"
    "${ascend_home}/set_env.sh"
    "${ascend_home}/ascend-toolkit/set_env.sh"
  )

  local env_script=""
  for env_script in "${env_scripts[@]}"; do
    if [[ -f "${env_script}" ]]; then
      # shellcheck disable=SC1090
      source "${env_script}"
      return 0
    fi
  done

  return 1
}

require_command() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: required command not found: ${cmd}" >&2
    exit 1
  fi
}

main() {
  local ascend_home=""
  if ! ascend_home="$(resolve_ascend_home)"; then
    echo "ERROR: ASCEND_HOME_PATH is not set and no default Ascend installation was found." >&2
    exit 1
  fi

  export ASCEND_HOME_PATH="${ascend_home}"
  if ! source_ascend_env "${ASCEND_HOME_PATH}"; then
    echo "WARNING: no Ascend environment script found under ${ASCEND_HOME_PATH}; continuing with current shell environment." >&2
  fi

  require_command python3
  require_command cmake
  require_command make

  cd "${repo_root}"

  echo "Repository root: ${repo_root}"
  echo "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"
  python3 --version
  cmake --version | head -n 1
  if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info || true
  fi

  chmod +x ./build.sh ./tests/run_st.sh
  bash ./build.sh --run_simple --a5 --npu
}

main "$@"
