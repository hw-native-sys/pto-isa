#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] [CASE]

Build and run TopK gtests. CASE selects a subset via --gtest_filter=*CASE*.

Options:
  -r, --run-mode MODE      sim|npu (default: sim)
  -v, --soc-version VER    e.g. Ascend950PR_9599 (default: Ascend950PR_9599)
  -n, --npu ID             npu id for sim (default: 0)
  -c, --case NAME          gtest filter substring (same as positional CASE)
  -k, --keep-build         reuse existing build/ (skip rm + cmake when binary exists)
  -d, --debug              cmake -DDEBUG_MODE=ON
  -l, --list               list known test names and exit
  -h, --help               this help

After each run, camodel dump logs are kept under build/<case_name>/
(core*_summary_log, *.dump, run.log, …). <case_name> is the -c/CASE
string, or "all" when no filter is set.

Examples:
  $(basename "$0") -r sim -v Ascend950PR_9599 -c iter_1x128_k3
  $(basename "$0") -c radix_1x4096_k512     # dumps → build/radix_1x4096_k512/
  $(basename "$0") -k -c iter_1x256_k8
  $(basename "$0")                         # dumps → build/all/

Known CASE names (must match launchTopK instantiations in topk_test_kernel.cpp /
gtest TEST_F in main.cpp; substring match):
  iter_1x128_k3 / iter_1x256_k8 / iter_1x256_k12 / iter_1x288_k8 / iter_1x384_k6
  iter_1x512_k12 / iter_1x896_k16
  radix_1x4096_k512 / radix_1x16384_k256 / radix_1x16384_k2048
EOF
}

# Cases wired in topk_test_kernel.cpp (explicit launchTopK instantiations) + main.cpp.
list_cases() {
    if [[ -x "${SCRIPT_DIR}/build/topk" ]]; then
        "${SCRIPT_DIR}/build/topk" --gtest_list_tests 2>/dev/null | sed -n 's/^  //p' | grep -v '^$'
    else
        cat <<EOF
iter_1x128_k3
iter_1x256_k8
iter_1x256_k12
iter_1x288_k8
iter_1x384_k6
iter_1x512_k12
iter_1x896_k16
radix_1x4096_k512
radix_1x16384_k256
radix_1x16384_k2048
EOF
    fi
}

RUN_MODE="sim"
SOC_VERSION="Ascend950PR_9599"
NPU_ID=0
CASE_FILTER=""
KEEP_BUILD=0
DEBUG_BUILD=0

SHORT=r:,v:,n:,c:,k,d,l,h
LONG=run-mode:,soc-version:,npu:,case:,keep-build,debug,list,help
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@") || { usage; exit 1; }
eval set -- "$OPTS"
while :
do
    case "$1" in
        (-r | --run-mode )
            RUN_MODE="$2"
            shift 2;;
        (-v | --soc-version )
            SOC_VERSION="$2"
            shift 2;;
        (-n | --npu )
            NPU_ID="$2"
            shift 2;;
        (-c | --case )
            CASE_FILTER="$2"
            shift 2;;
        (-k | --keep-build )
            KEEP_BUILD=1
            shift 1;;
        (-d | --debug )
            DEBUG_BUILD=1
            shift 1;;
        (-l | --list )
            list_cases
            exit 0;;
        (-h | --help )
            usage
            exit 0;;
        (--)
            shift
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1"
            usage
            exit 1;;
    esac
done

# Positional CASE overrides / fills -c when provided.
if [[ $# -ge 1 ]]; then
    CASE_FILTER="$1"
fi

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[ERROR] ASCEND_HOME_PATH is unset. source ~/source_cann.sh first."
    exit 1
fi

cd "${SCRIPT_DIR}"
set -euo pipefail

echo "[RUN.SH] RUN_MODE=${RUN_MODE}"
echo "[RUN.SH] SOC_VERSION=${SOC_VERSION}"
echo "[RUN.SH] NPU_ID=${NPU_ID}"
echo "[RUN.SH] CASE_FILTER=${CASE_FILTER:-<all>}"
echo "[RUN.SH] KEEP_BUILD=${KEEP_BUILD}"
echo "[RUN.SH] DEBUG=${DEBUG_BUILD}"

NEED_BUILD=1
if [[ "${KEEP_BUILD}" -eq 1 && -x build/topk ]]; then
    NEED_BUILD=0
    echo "[RUN.SH] reusing existing build/topk"
fi

if [[ "${NEED_BUILD}" -eq 1 ]]; then
    rm -rf build
    mkdir build
    cd build
    export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${LD_LIBRARY_PATH:-}
    CMAKE_EXTRA=()
    if [[ "${DEBUG_BUILD}" -eq 1 ]]; then
        CMAKE_EXTRA+=(-DDEBUG_MODE=ON)
    fi
    cmake -DRUN_MODE="${RUN_MODE}" -DSOC_VERSION="${SOC_VERSION}" "${CMAKE_EXTRA[@]}" ..
    make -j16
    cd ..
else
    export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${LD_LIBRARY_PATH:-}
fi

EXTRA_BIN_ARGS=()
if [[ "${RUN_MODE}" == "sim" ]]; then
    EXTRA_BIN_ARGS+=(--npu="${NPU_ID}")
fi
if [[ -n "${CASE_FILTER}" ]]; then
    EXTRA_BIN_ARGS+=(--gtest_filter="*${CASE_FILTER}*")
fi

# Sanitize case name for a directory under build/ (camodel dumps into CWD).
CASE_NAME="${CASE_FILTER:-all}"
CASE_NAME="${CASE_NAME//\//_}"
CASE_NAME="${CASE_NAME//\*/_}"
CASE_NAME="${CASE_NAME// /_}"
DUMP_DIR="${SCRIPT_DIR}/build/${CASE_NAME}"
rm -rf "${DUMP_DIR}"
mkdir -p "${DUMP_DIR}"

BIN="${SCRIPT_DIR}/build/topk"
if [[ ! -x "${BIN}" ]]; then
    echo "[ERROR] missing binary: ${BIN}"
    exit 1
fi

echo "[RUN.SH] ${BIN} ${EXTRA_BIN_ARGS[*]:-}"
echo "[RUN.SH] dump dir: ${DUMP_DIR}"
set +e
(
    cd "${DUMP_DIR}"
    time "${BIN}" "${EXTRA_BIN_ARGS[@]}" 2>&1 | tee run.log
    exit "${PIPESTATUS[0]}"
)
RC=$?
set -e
echo "[RUN.SH] dumps saved under ${DUMP_DIR} (exit=${RC})"
ls -la "${DUMP_DIR}" | head -40
exit "${RC}"
