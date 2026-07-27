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

# GridPipe single-source broadcast smoke test.  One source cell TPUSH<ROW|COL>-es
# a stamped fp32 tile to its whole span in a single multicast (batched writes +
# one publish fence + batched doorbells); every other cell TPOP<dir,dist>s and
# stores it.  Verifies out[cell] == in[span-source] in-process (no data files).

: "${ASCEND_CANN_PATH:=$(ls -1d /usr/local/Ascend/cann-*/set_env.sh 2>/dev/null | sort -V | tail -1)}"
if [ -z "${ASCEND_CANN_PATH}" ]; then
    echo "[ERROR] Cannot find CANN set_env.sh.  Set ASCEND_CANN_PATH explicitly."
    exit 1
fi
source "${ASCEND_CANN_PATH}"

SHORT=r:,v:,d:
LONG=run-mode:,soc-version:,device-id:,grid-rows:,grid-cols:,src:,span-col:,subrect:,rect-r0:,rect-r1:,rect-c0:,rect-c1:,rect-src:,token-tile:,model-tile:,build-only
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

BUILD_ONLY=0
while :; do
    case "$1" in
        (-r | --run-mode)    RUN_MODE="$2"; shift 2;;
        (-v | --soc-version) SOC_VERSION="$2"; shift 2;;
        (-d | --device-id)   DEVICE_ID="$2"; shift 2;;
        (--grid-rows)        BCAST_ROWS="$2"; shift 2;;
        (--grid-cols)        BCAST_COLS="$2"; shift 2;;
        (--src)              BCAST_SRC="$2"; shift 2;;
        (--span-col)         BCAST_SPAN_COL="$2"; shift 2;;
        (--subrect)          BCAST_SUBRECT="$2"; shift 2;;
        (--rect-r0)          BCAST_RECT_R0="$2"; shift 2;;
        (--rect-r1)          BCAST_RECT_R1="$2"; shift 2;;
        (--rect-c0)          BCAST_RECT_C0="$2"; shift 2;;
        (--rect-c1)          BCAST_RECT_C1="$2"; shift 2;;
        (--rect-src)         BCAST_RECT_SRC="$2"; shift 2;;
        (--token-tile)       BCAST_T="$2"; shift 2;;
        (--model-tile)       BCAST_W="$2"; shift 2;;
        (--build-only)       BUILD_ONLY=1; shift;;
        (--) shift; break;;
        (*) echo "[ERROR] Unexpected option: $1"; exit 1;;
    esac
done

: "${RUN_MODE:=npu}"
: "${SOC_VERSION:=Ascend910B1}"
: "${BCAST_ROWS:=1}"
: "${BCAST_COLS:=5}"
: "${BCAST_SRC:=2}"
: "${BCAST_SPAN_COL:=0}"
: "${BCAST_SUBRECT:=0}"
: "${BCAST_RECT_R0:=0}"
: "${BCAST_RECT_R1:=${BCAST_ROWS}}"
: "${BCAST_RECT_C0:=0}"
: "${BCAST_RECT_C1:=${BCAST_COLS}}"
: "${BCAST_RECT_SRC:=0}"
: "${BCAST_T:=16}"
: "${BCAST_W:=64}"
: "${DEVICE_ID:=${ASCEND_DEVICE_ID:-${DEVICE_ID:-0}}}"

if [[ ! "${SOC_VERSION}" =~ ^Ascend ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}"
    exit 1
fi

rm -rf /dev/shm/sem.hccl* 2>/dev/null
ipcrm -a 2>/dev/null

echo "=== GridPipe single-source broadcast smoke ==="
echo "  RUN_MODE: ${RUN_MODE}  SOC_VERSION: ${SOC_VERSION}  DEVICE_ID: ${DEVICE_ID}"
echo "  Grid: ${BCAST_ROWS}x${BCAST_COLS}  SRC: ${BCAST_SRC}  SPAN_COL: ${BCAST_SPAN_COL}  SUBRECT: ${BCAST_SUBRECT}  Rect: [r${BCAST_RECT_R0}:${BCAST_RECT_R1},c${BCAST_RECT_C0}:${BCAST_RECT_C1}] RectSRC: ${BCAST_RECT_SRC}  Tile: ${BCAST_T}x${BCAST_W}"
echo "=============================================="

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
      -DBCAST_ROWS=${BCAST_ROWS} -DBCAST_COLS=${BCAST_COLS} -DBCAST_SRC=${BCAST_SRC} \
      -DBCAST_SPAN_COL=${BCAST_SPAN_COL} -DBCAST_SUBRECT=${BCAST_SUBRECT} \
      -DBCAST_RECT_R0=${BCAST_RECT_R0} -DBCAST_RECT_R1=${BCAST_RECT_R1} \
      -DBCAST_RECT_C0=${BCAST_RECT_C0} -DBCAST_RECT_C1=${BCAST_RECT_C1} \
      -DBCAST_RECT_SRC=${BCAST_RECT_SRC} \
      -DBCAST_T=${BCAST_T} -DBCAST_W=${BCAST_W} \
      ..
make -j16 bcast_smoke

if [ "${BUILD_ONLY}" -eq 1 ]; then
    echo "[INFO] --build-only requested; skipping run."
    exit 0
fi

echo ""
echo "=== Running GridPipe single-source broadcast smoke ==="
./bcast_smoke --device-id "${DEVICE_ID}"
