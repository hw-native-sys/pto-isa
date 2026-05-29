#!/usr/bin/env bash

set -euo pipefail

a5_runner_log() {
  echo "[a5-runner-init] $*"
}

a5_runner_warn() {
  echo "[a5-runner-init] WARN: $*" >&2
}

a5_runner_fail() {
  echo "[a5-runner-init] ERROR: $*" >&2
  return 1
}

a5_runner_has_command() {
  command -v "$1" >/dev/null 2>&1
}

a5_runner_prepend_path() {
  local dir="$1"
  if [[ -n "${dir}" && -d "${dir}" ]]; then
    export PATH="${dir}:${PATH}"
  fi
}

a5_runner_prepend_path_list() {
  local path_list="$1"
  local dir=""
  local entries=()

  IFS=':' read -r -a entries <<< "${path_list}"
  for ((idx=${#entries[@]} - 1; idx >= 0; idx--)); do
    dir="${entries[idx]}"
    a5_runner_prepend_path "${dir}"
  done
}

a5_runner_export_gtest_env() {
  local prefix="${A5_GTEST_PREFIX:-}"
  local cmake_dir=""
  local lib_dir=""

  if [[ -z "${prefix}" || ! -d "${prefix}" ]]; then
    return 0
  fi

  export CMAKE_PREFIX_PATH="${prefix}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

  for cmake_dir in \
    "${prefix}/lib/cmake/GTest" \
    "${prefix}/lib64/cmake/GTest"; do
    if [[ -d "${cmake_dir}" ]]; then
      export GTest_DIR="${cmake_dir}"
      break
    fi
  done

  for lib_dir in "${prefix}/lib64" "${prefix}/lib"; do
    if [[ -d "${lib_dir}" ]]; then
      export LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      export LIBRARY_PATH="${lib_dir}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    fi
  done
}

a5_runner_python_has_module() {
  local module_name="$1"
  python3 -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('${module_name}') else 1)"
}

a5_runner_require_python_package() {
  local module_name="$1"

  if ! a5_runner_python_has_module "${module_name}"; then
    a5_runner_fail "Python module '${module_name}' is required. Pre-install it in the runner environment."
  fi
}

a5_bootstrap_runner_env() {
  if [[ "${A5_RUNNER_SKIP_BOOTSTRAP:-0}" == "1" ]]; then
    a5_runner_log "Skipping bootstrap because A5_RUNNER_SKIP_BOOTSTRAP=1"
    return 0
  fi

  a5_runner_prepend_path_list "${A5_TOOLCHAIN_PATH_PREFIX:-}"
  a5_runner_prepend_path "${A5_PYTHON_BIN_DIR:-}"
  a5_runner_prepend_path "/usr/local/bin"
  a5_runner_prepend_path "/usr/bin"
  a5_runner_prepend_path "/bin"
  a5_runner_export_gtest_env

  a5_runner_has_command python3 || a5_runner_fail "python3 is required on the runner."
  a5_runner_has_command cmake || a5_runner_fail "cmake is required on the runner."
  a5_runner_has_command git || a5_runner_fail "git is required on the runner."
  a5_runner_has_command make || a5_runner_fail "make is required on the runner."

  a5_runner_require_python_package numpy
  a5_runner_require_python_package ml_dtypes
  a5_runner_require_python_package en_dtypes

  a5_runner_log "Using python: $(command -v python3)"
  a5_runner_log "Using git: $(command -v git)"
  a5_runner_log "Using cmake: $(command -v cmake)"
  if [[ -n "${GTest_DIR:-}" ]]; then
    a5_runner_log "Using GTest config: ${GTest_DIR}"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  a5_bootstrap_runner_env "$@"
fi
