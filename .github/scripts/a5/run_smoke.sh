#!/usr/bin/env bash

set -euo pipefail

repo_root=$(
  cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
  pwd
)

source_if_exists() {
  local script_path="$1"
  shift || true
  if [[ -f "${script_path}" ]]; then
    set +eu
    # shellcheck disable=SC1090
    source "${script_path}" "$@"
    set -eu
    return 0
  fi

  return 1
}

source_runner_env() {
  local conda_activate="${A5_CONDA_ACTIVATE_PATH:-}"
  local cann_setenv="${A5_CANN_SETENV_PATH:-}"

  if [[ -n "${conda_activate}" ]]; then
    source_if_exists "${conda_activate}" || true
  fi

  if [[ -n "${cann_setenv}" ]]; then
    source_if_exists "${cann_setenv}" || true
  fi
}

resolve_ascend_home() {
  if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    if [[ -d "${ASCEND_HOME_PATH}" ]]; then
      printf '%s\n' "${ASCEND_HOME_PATH}"
      return 0
    fi
    echo "ERROR: Explicitly set ASCEND_HOME_PATH='${ASCEND_HOME_PATH}' is not a valid directory." >&2
    return 1
  fi

  local candidates=()
  local glob_candidate=""

  candidates+=(
    "/usr/local/Ascend/cann"
    "/usr/local/Ascend/ascend-toolkit/latest"
    "/usr/local/Ascend/ascend-toolkit"
    "${HOME}/Ascend/cann"
  )
  shopt -s nullglob
  for glob_candidate in /usr/local/Ascend/cann-*; do
    candidates+=("${glob_candidate}")
  done
  shopt -u nullglob

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
    if source_if_exists "${env_script}"; then
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

cleanup_a5_root_artifacts() {
  local a5_build_dir="${repo_root}/tests/npu/a5/src/st/build"
  set +e
  if [[ -d "${a5_build_dir}" ]]; then
    echo "Cleaning root-owned A5 build directory: ${a5_build_dir}"
    rm -rf "${a5_build_dir}"
  fi
}

main() {
  source_runner_env
  # shellcheck disable=SC1091
  source "${repo_root}/.github/scripts/a5/setup_runner_env.sh"
  a5_bootstrap_runner_env

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
  require_command git
  require_command make

  cd "${repo_root}"
  trap cleanup_a5_root_artifacts EXIT

  echo "Repository root: ${repo_root}"
  echo "ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"
  python3 --version
  cmake --version | head -n 1
  git --version
  if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info || true
  fi

  local a5_build_dir="${repo_root}/tests/npu/a5/src/st/build"
  if [[ -d "${a5_build_dir}" ]]; then
    echo "Removing stale A5 build directory: ${a5_build_dir}"
    rm -rf "${a5_build_dir}"
  fi

  chmod +x ./build.sh ./tests/run_st.sh
  bash ./build.sh --run_simple --a5 --npu
}

main "$@"
