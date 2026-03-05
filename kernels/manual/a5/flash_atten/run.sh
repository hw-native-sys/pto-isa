#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

SHORT=r:,v:,n:,c:,a:,p:,m:,i,d,k
LONG=run-mode:,soc-version:,npu:,case:,cases:,qk-preload:,mode:,intermediate,debug,mask,mode_dn
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
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
        (-a | --cases )
            CASES_RAW="$2"
            shift 2;;
        (-p | --qk-preload )
            QK_PRELOAD="$2"
            shift 2;;
        (-m | --mode )
            FIFO_MODE="$2"
            shift 2;;
        (-i | --intermediate )
            INTERMEDIATE=1
            shift 1;;
        (-d | --debug )
            DEBUG_BUILD=1
            shift 1;;
        (--mode_dn )
            MODE_DN=1
            shift 1;;
        (-k | --mask )
            CAUSAL_MASK=1
            shift 1;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done

pattern="^Ascend910_9599"
if [[ ! "$SOC_VERSION" =~ $pattern ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}, this folder only support A5."
    exit 1
fi

rm -rf build
mkdir build
cd build

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
set -euo pipefail

# default device id
: "${NPU_ID:=0}"
: "${QK_PRELOAD:=2}"
: "${FIFO_MODE:=1}"  # 0=ALL_GM_PATH, 1=ALL_UB_PATH, 2=QK_PV_UB_ONLY

GEN_CASE_ARGS=()
# Handle missing value after -c/--case (e.g. user passed -c and then --cases)
if [[ -n "${CASE_FILTER:-}" && "${CASE_FILTER}" == --* ]]; then
    CASE_FILTER=""
fi

if [[ -n "${CASES_RAW:-}" ]]; then
    IFS=';' read -ra CASE_ENTRIES <<< "${CASES_RAW}"
    for entry in "${CASE_ENTRIES[@]}"; do
        GEN_CASE_ARGS+=(--cases "$entry")
    done
elif [[ -n "${CASE_FILTER:-}" ]]; then
    # If only a single case filter was provided, ensure generation for numeric tuple filters
    if [[ "${CASE_FILTER}" == *","* ]]; then
        GEN_CASE_ARGS+=(--cases "${CASE_FILTER}")
    fi
fi

echo "[RUN.SH] CASE_FILTER=${CASE_FILTER:-}<none>"
echo "[RUN.SH] CASES_RAW=${CASES_RAW:-}<none>"
echo "[RUN.SH] NPU_ID=${NPU_ID}"
echo "[RUN.SH] QK_PRELOAD=${QK_PRELOAD}"
echo "[RUN.SH] FIFO_MODE=${FIFO_MODE} (0=ALL_GM, 1=ALL_UB, 2=QK_PV_UB_ONLY)"
echo "[RUN.SH] GEN_CASE_ARGS=${GEN_CASE_ARGS[*]:-<none>}"
echo "[RUN.SH] INTERMEDIATE=${INTERMEDIATE:-0}"
echo "[RUN.SH] CAUSAL_MASK=${CAUSAL_MASK:-0}"
echo "[RUN.SH] DEBUG=${DEBUG_BUILD:-0}"
echo "[RUN.SH] DN_MODE=${MODE_DN:-0}"

python3 ../scripts/generate_cases.py --qk-preload "${QK_PRELOAD}" "${GEN_CASE_ARGS[@]}" --causal-mask "${CAUSAL_MASK:-0}"

CMAKE_EXTRA=()
if [[ -n "${DEBUG_BUILD:-}" ]]; then
    CMAKE_EXTRA+=(-DDEBUG_MODE=ON)
fi
if [[ -n "${MODE_DN:-}" ]]; then
    CMAKE_EXTRA+=(-DMODE_DN=ON)
fi
CMAKE_EXTRA+=(-DFIFO_MODE=${FIFO_MODE})

cmake -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} "${CMAKE_EXTRA[@]}" ..
make -j16

EXTRA_BIN_ARGS=()
if [[ -n "${INTERMEDIATE:-}" ]]; then
    EXTRA_BIN_ARGS+=(--intermediate)
fi
EXTRA_BIN_ARGS+=(--sys_cnt_multiple=1.0)

if [[ -n "${MODE_DN:-}" ]]; then
    if [[ -n "${CASE_FILTER:-}" ]]; then
        python3 ../scripts/gen_data.py --case="${CASE_FILTER}" "${GEN_CASE_ARGS[@]}" --causal-mask "${CAUSAL_MASK:-0}"
        time ./fa_performance_dn --npu="${NPU_ID}" --case="${CASE_FILTER}" "${EXTRA_BIN_ARGS[@]}"
    elif [[ -n "${CASES_RAW:-}" ]]; then
        python3 ../scripts/gen_data.py "${GEN_CASE_ARGS[@]}" --causal-mask "${CAUSAL_MASK:-0}"
        time ./fa_performance_dn --npu="${NPU_ID}" --cases="${CASES_RAW}" "${EXTRA_BIN_ARGS[@]}"
    else
        python3 ../scripts/gen_data.py "${GEN_CASE_ARGS[@]}" --causal-mask "${CAUSAL_MASK:-0}"
        time ./fa_performance_dn --npu="${NPU_ID}" "${EXTRA_BIN_ARGS[@]}"
    fi
else
    if [[ -n "${CASE_FILTER:-}" ]]; then
        python3 ../scripts/gen_data.py --case="${CASE_FILTER}" "${GEN_CASE_ARGS[@]}" --causal-mask "${CAUSAL_MASK:-0}"
        time ./fa_performance --npu="${NPU_ID}" --case="${CASE_FILTER}" "${EXTRA_BIN_ARGS[@]}"
    elif [[ -n "${CASES_RAW:-}" ]]; then
        python3 ../scripts/gen_data.py "${GEN_CASE_ARGS[@]}" --causal-mask "${CAUSAL_MASK:-0}"
        time ./fa_performance --npu="${NPU_ID}" --cases="${CASES_RAW}" "${EXTRA_BIN_ARGS[@]}"
    else
        python3 ../scripts/gen_data.py "${GEN_CASE_ARGS[@]}" --causal-mask "${CAUSAL_MASK:-0}"
        time ./fa_performance --npu="${NPU_ID}" "${EXTRA_BIN_ARGS[@]}"
    fi
fi
