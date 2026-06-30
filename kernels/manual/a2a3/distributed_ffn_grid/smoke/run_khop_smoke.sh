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

# GridPipe routed K-hop unicast smoke test.  A 1 x cols row of cells; each cell
# pushes a stamped fp32 tile `dist` hops EAST and the +dist cell pops/stores it.
# Verifies out[c] == in[c-dist] in-process (no data files).

: "${ASCEND_CANN_PATH:=$(ls -1d /usr/local/Ascend/cann-*/set_env.sh 2>/dev/null | sort -V | tail -1)}"
if [ -z "${ASCEND_CANN_PATH}" ]; then
    echo "[ERROR] Cannot find CANN set_env.sh.  Set ASCEND_CANN_PATH explicitly."
    exit 1
fi
source "${ASCEND_CANN_PATH}"

SHORT=r:,v:,d:
LONG=run-mode:,soc-version:,device-id:,grid-rows:,grid-cols:,dist:,token-tile:,model-tile:,build-only
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

BUILD_ONLY=0
while :; do
    case "$1" in
        (-r | --run-mode)    RUN_MODE="$2"; shift 2;;
        (-v | --soc-version) SOC_VERSION="$2"; shift 2;;
        (-d | --device-id)   DEVICE_ID="$2"; shift 2;;
        (--grid-rows)        KHOP_ROWS="$2"; shift 2;;
        (--grid-cols)        KHOP_COLS="$2"; shift 2;;
        (--dist)             KHOP_DIST="$2"; shift 2;;
        (--token-tile)       KHOP_T="$2"; shift 2;;
        (--model-tile)       KHOP_W="$2"; shift 2;;
        (--build-only)       BUILD_ONLY=1; shift;;
        (--) shift; break;;
        (*) echo "[ERROR] Unexpected option: $1"; exit 1;;
    esac
done

: "${RUN_MODE:=npu}"
: "${SOC_VERSION:=Ascend910B1}"
: "${KHOP_ROWS:=1}"
: "${KHOP_COLS:=4}"
: "${KHOP_DIST:=2}"
: "${KHOP_T:=16}"
: "${KHOP_W:=64}"
: "${DEVICE_ID:=${ASCEND_DEVICE_ID:-${DEVICE_ID:-0}}}"

if [[ ! "${SOC_VERSION}" =~ ^Ascend ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}"
    exit 1
fi

rm -rf /dev/shm/sem.hccl* 2>/dev/null
ipcrm -a 2>/dev/null

echo "=== GridPipe routed K-hop unicast smoke ==="
echo "  RUN_MODE: ${RUN_MODE}  SOC_VERSION: ${SOC_VERSION}  DEVICE_ID: ${DEVICE_ID}"
echo "  Grid: ${KHOP_ROWS}x${KHOP_COLS}  DIST: ${KHOP_DIST}  Tile: ${KHOP_T}x${KHOP_W}"
echo "==========================================="

# CMakeLists.txt lives in the parent demo directory; build from there.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_DIR}"

rm -rf build
mkdir build
cd build

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${LD_LIBRARY_PATH:-}
set -euo pipefail

cmake -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} \
      -DKHOP_ROWS=${KHOP_ROWS} -DKHOP_COLS=${KHOP_COLS} -DKHOP_DIST=${KHOP_DIST} \
      -DKHOP_T=${KHOP_T} -DKHOP_W=${KHOP_W} \
      ..
make -j16 khop_smoke

if [ "${BUILD_ONLY}" -eq 1 ]; then
    echo "[INFO] --build-only requested; skipping run."
    exit 0
fi

echo ""
echo "=== Running GridPipe routed K-hop unicast smoke ==="
./khop_smoke --device-id "${DEVICE_ID}"
