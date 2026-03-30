#!/usr/bin/env bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

# ============================================================================
# Usage
# ============================================================================
usage() {
  cat <<EOF
Usage: $(basename "$0") [-n NPU_COUNT] [-v VERSION] [-t TESTCASE] [-a] [-d]

Options:
  -n NPU_COUNT   Number of NPUs (devices) available: 2, 4, or 8 (default: 8)
                 Only test cases requiring <= NPU_COUNT ranks will run.
  -v VERSION     SoC version: a3 (Ascend910B, default) or a5 (Ascend910_9599).
  -t TESTCASE    Run only the specified testcase (e.g. tput, treduce).
                 Can be specified multiple times. Default: run all.
  -a             Include async testcases (e.g. tput_async, tget_async).
                 Async tests are excluded by default as they require a
                 newer CANN version with SDMA opapi support.
  -d             Enable debug mode (extra logging at each sync point).
  -h             Show this help message.

Examples:
  $(basename "$0")                   # Run all non-async tests with 8 NPUs on a3
  $(basename "$0") -n 2              # Run only 2-rank non-async tests
  $(basename "$0") -a                # Run all tests including async
  $(basename "$0") -a -t tput_async  # Run only tput_async
  $(basename "$0") -v a5 -n 2 -t tput  # Run tput on A5 with 2 NPUs
  $(basename "$0") -d -t tput        # Run tput with debug output
EOF
  exit 0
}

# ============================================================================
# Parse arguments
# ============================================================================
NPU_COUNT=8
SOC_VERSION="a3"
DEBUG_FLAG=""
INCLUDE_ASYNC=false
declare -a SELECTED_TESTS=()

while getopts "n:v:t:adh" opt; do
  case "$opt" in
    n) NPU_COUNT="$OPTARG" ;;
    v) SOC_VERSION="$OPTARG" ;;
    t) SELECTED_TESTS+=("$OPTARG") ;;
    a) INCLUDE_ASYNC=true ;;
    d) DEBUG_FLAG="-d" ;;
    h) usage ;;
    *) usage ;;
  esac
done

if [[ "$NPU_COUNT" != 2 && "$NPU_COUNT" != 4 && "$NPU_COUNT" != 8 ]]; then
  echo "[ERROR] -n must be 2, 4, or 8 (got: ${NPU_COUNT})" >&2
  exit 1
fi

if [[ "$SOC_VERSION" != "a3" && "$SOC_VERSION" != "a5" ]]; then
  echo "[ERROR] -v must be a3 or a5 (got: ${SOC_VERSION})" >&2
  exit 1
fi

# ============================================================================
# Build gtest filter for a specific rank count.
#
# Each test expects an exact MPI world size (ForkAndRunWithHcclRootInfo checks
# mpiSize == nRanks). The script runs the binary once per distinct rank count,
# using gtest filters to select only the matching tests each time.
#
# All multi-rank tests follow the *_NRanks / *_Nranks naming convention,
# so a simple wildcard match is sufficient.
# ============================================================================

get_gtest_filter_for_nranks() {
  local nranks="$1"

  case "$nranks" in
    2) echo "*-*4Ranks*:*4ranks*:*8Ranks*:*8ranks*" ;;
    4) echo "*4Ranks*:*4ranks*" ;;
    8) echo "*8Ranks*:*8ranks*" ;;
  esac
}

# ============================================================================
# Discover testcases
# ============================================================================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$SOC_VERSION" == "a5" ]]; then
  ST_DIR="${ROOT_DIR}/tests/npu/a5/comm/st/testcase"
else
  ST_DIR="${ROOT_DIR}/tests/npu/a2a3/comm/st/testcase"
fi

if [[ ! -d "${ST_DIR}" ]]; then
  echo "[ERROR] testcase dir not found: ${ST_DIR}" >&2
  exit 1
fi

is_async_test() { [[ "$1" == *_async ]]; }

declare -a tests=()
if [[ "${#SELECTED_TESTS[@]}" -gt 0 ]]; then
  tests=("${SELECTED_TESTS[@]}")
else
  while IFS= read -r -d '' dir; do
    name="$(basename "${dir}")"
    if is_async_test "$name" && ! $INCLUDE_ASYNC; then
      continue
    fi
    tests+=("$name")
  done < <(find "${ST_DIR}" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
fi

if [[ "${#tests[@]}" -eq 0 ]]; then
  echo "[ERROR] No testcase directories found under ${ST_DIR}" >&2
  exit 1
fi

async_info=""
$INCLUDE_ASYNC && async_info=", async=on" || async_info=", async=off"
echo "[INFO] NPU_COUNT=${NPU_COUNT}, SOC=${SOC_VERSION}, DEBUG=${DEBUG_FLAG:-(off)}${async_info}, running ${#tests[@]} testcase(s): ${tests[*]}"

# ============================================================================
# Run
# ============================================================================
fail_count=0
total_runs=0

for t in "${tests[@]}"; do
  built=false
  for nranks in 2 4 8; do
    if (( nranks > NPU_COUNT )); then continue; fi

    gtest_filter="$(get_gtest_filter_for_nranks "$nranks")"
    [[ -z "$gtest_filter" ]] && continue

    echo "============================================================"
    echo "[INFO] Running testcase: ${t}  (nranks=${nranks}, GTEST_FILTER=${gtest_filter})"
    echo "============================================================"

    build_flag=""
    if $built; then
      build_flag="-w"
    fi
    built=true

    total_runs=$((total_runs + 1))
    if ! GTEST_FILTER="${gtest_filter}" \
         python3 "${ROOT_DIR}/tests/script/run_st.py" -r npu -v "${SOC_VERSION}" ${DEBUG_FLAG} ${build_flag} -n "${nranks}" -t "comm/${t}"; then
      echo "[ERROR] Testcase failed: ${t} (nranks=${nranks})" >&2
      fail_count=$((fail_count + 1))
    fi
  done
done

echo "============================================================"
if [[ "${fail_count}" -eq 0 ]]; then
  echo "[INFO] All ${total_runs} comm ST run(s) passed (NPU_COUNT=${NPU_COUNT}, SOC=${SOC_VERSION})."
  exit 0
else
  echo "[ERROR] ${fail_count}/${total_runs} run(s) failed."
  exit 1
fi
