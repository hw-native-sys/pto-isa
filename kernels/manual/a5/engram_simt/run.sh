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

  # engram_simt usage: Perf-Mode Usage
  #   -r <run-mode>    : sim or npu (default: npu)
  #   -v <soc-version> : e.g. Ascend910_9599
  #   -c <case>        : run a specific case, e.g. "ENGRAMSIMTTest.baseline_E128_B1_T64K"
  #   -p               : enable perf-analysis mode (builds all 4D x 4B = 16 kernel instantiations,
  #                      generates 4D x 4B x 3T x 4P = 192 configs with varied table sizes and
  #                      access patterns: RAND, SEQ, SAME, STRIDE; omit -c to run all 384 cases)

SHORT=r:,v:,n:,c:,a:,p
LONG=run-mode:,soc-version:,npu:,case:,cases:,perf-analysis
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
        (-p | --perf-analysis )
            PERF_ANALYSIS=1
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
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}, this folder only supports A5."
    exit 1
fi

rm -rf build
mkdir build
cd build

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
set -euo pipefail

: "${NPU_ID:=0}"

GEN_CASE_ARGS=()
if [[ -n "${PERF_ANALYSIS:-}" ]]; then
    GEN_CASE_ARGS+=(--perf-analysis)
fi
if [[ -n "${CASES_RAW:-}" ]]; then
    IFS=';' read -ra CASE_ENTRIES <<< "${CASES_RAW}"
    for entry in "${CASE_ENTRIES[@]}"; do
        GEN_CASE_ARGS+=(--cases "$entry")
    done
fi

python3 ../scripts/generate_cases.py "${GEN_CASE_ARGS[@]}"

CMAKE_EXTRA=()
if [[ -n "${PERF_ANALYSIS:-}" ]]; then
    CMAKE_EXTRA+=(-DPERF_ANALYSIS=ON)
fi

cmake -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} "${CMAKE_EXTRA[@]}" ..
make -j16

GEN_DATA_ENV=""
if [[ -n "${PERF_ANALYSIS:-}" ]]; then
    GEN_DATA_ENV="PERF_ANALYSIS=1"
fi

if [[ -n "${CASE_FILTER:-}" ]]; then
    env ${GEN_DATA_ENV} python3 ../scripts/gen_data.py ../data "${CASE_FILTER}"
elif [[ -n "${CASES_RAW:-}" ]]; then
    env ${GEN_DATA_ENV} python3 ../scripts/gen_data.py ../data
else
    env ${GEN_DATA_ENV} python3 ../scripts/gen_data.py ../data
fi

EXTRA_ARGS=()
EXTRA_ARGS+=(--npu="${NPU_ID}")
if [[ -n "${CASE_FILTER:-}" ]]; then
    EXTRA_ARGS+=(--case="${CASE_FILTER}")
elif [[ -n "${CASES_RAW:-}" ]]; then
    EXTRA_ARGS+=(--cases="${CASES_RAW}")
fi

time ./engram-simt "${EXTRA_ARGS[@]}"
