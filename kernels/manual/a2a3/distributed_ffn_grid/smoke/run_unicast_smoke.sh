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

# GridPipe unicast time-division HANDOVER smoke test.  Two producers take turns on
# one consumer channel and the second takes it over while the first one's tiles are
# still undrained -- the case relay counting exists for.  See
# smoke/unicast_smoke_config.hpp for the schedule and what each knob changes.

: "${ASCEND_CANN_PATH:=$(ls -1d /usr/local/Ascend/cann-*/set_env.sh 2>/dev/null | sort -V | tail -1)}"
if [ -z "${ASCEND_CANN_PATH}" ]; then
    echo "[ERROR] Cannot find CANN set_env.sh.  Set ASCEND_CANN_PATH explicitly."
    exit 1
fi
source "${ASCEND_CANN_PATH}"

SHORT=r:,v:,d:
LONG=run-mode:,soc-version:,device-id:,tiles:,slot-count:,max-spins:,token-tile:,model-tile:,build-only
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@") || {
    echo "[ERROR] bad arguments"; exit 2;
}
eval set -- "$OPTS"

BUILD_ONLY=0
while :; do
    case "$1" in
        (-r | --run-mode)    RUN_MODE="$2"; shift 2;;
        (-v | --soc-version) SOC_VERSION="$2"; shift 2;;
        (-d | --device-id)   DEVICE_ID="$2"; shift 2;;
        (--tiles)            UNICAST_TILES="$2"; shift 2;;
        (--slot-count)       UNICAST_SLOT_COUNT="$2"; shift 2;;
        (--max-spins)        UNICAST_MAX_SPINS="$2"; shift 2;;
        (--token-tile)       UNICAST_T="$2"; shift 2;;
        (--model-tile)       UNICAST_W="$2"; shift 2;;
        (--build-only)       BUILD_ONLY=1; shift;;
        (--) shift; break;;
        (*) echo "[ERROR] Unexpected option: $1"; exit 1;;
    esac
done

: "${RUN_MODE:=npu}"
: "${SOC_VERSION:=Ascend910B1}"
# Tiles each producer publishes on its turn; the last one carries CLOSE.
: "${UNICAST_TILES:=2}"
# Ring depth.  >= 2*TILES keeps the two turns in disjoint slots; == TILES makes the
# ring wrap, so the new owner must wait on the credit baseline it was handed --
# that variant is the payload-safety proof.
: "${UNICAST_SLOT_COUNT:=4}"
# 0 = block forever (the shipping path).  Non-zero bounds every wait, so a handover
# that never happens reports a fault code instead of hanging.
: "${UNICAST_MAX_SPINS:=0}"
: "${UNICAST_T:=16}"
: "${UNICAST_W:=64}"
: "${DEVICE_ID:=${TASK_DEVICE:-${ASCEND_DEVICE_ID:-${DEVICE_ID:-0}}}}"

if [[ ! "${SOC_VERSION}" =~ ^Ascend ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}"
    exit 1
fi

rm -rf /dev/shm/sem.hccl* 2>/dev/null
ipcrm -a 2>/dev/null

echo "=== GridPipe unicast handover smoke ==="
echo "  RUN_MODE: ${RUN_MODE}  SOC_VERSION: ${SOC_VERSION}  DEVICE_ID: ${DEVICE_ID}"
echo "  Tiles/turn: ${UNICAST_TILES}  Slots: ${UNICAST_SLOT_COUNT}  MaxSpins: ${UNICAST_MAX_SPINS}  Tile: ${UNICAST_T}x${UNICAST_W}"
echo "======================================="

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_DIR}"

rm -rf build
mkdir build
cd build

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${LD_LIBRARY_PATH:-}
set -euo pipefail

cmake -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} \
      -DUNICAST_TILES=${UNICAST_TILES} -DUNICAST_SLOT_COUNT=${UNICAST_SLOT_COUNT} \
      -DUNICAST_MAX_SPINS=${UNICAST_MAX_SPINS} \
      -DUNICAST_T=${UNICAST_T} -DUNICAST_W=${UNICAST_W} \
      ..
make -j16 unicast_smoke

if [ "${BUILD_ONLY}" -eq 1 ]; then
    echo "[INFO] --build-only requested; skipping run."
    exit 0
fi

echo ""
echo "=== Running GridPipe unicast handover smoke ==="
./unicast_smoke --device-id "${DEVICE_ID}"
