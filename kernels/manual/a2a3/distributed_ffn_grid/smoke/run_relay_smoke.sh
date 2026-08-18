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

# GridPipe reduce <-> unicast CHANNEL RELAY smoke test.  ONE channel in the shared
# pool carries a group reduce, then a unicast flow, then a group reduce again.
# The sequence runs first as one stage per launch, then entirely inside one launch,
# so both handover rules and the explicit isLastRound release run.  See
# smoke/relay_smoke_config.hpp for the schedule and what each boundary proves.

: "${ASCEND_CANN_PATH:=$(ls -1d /usr/local/Ascend/cann-*/set_env.sh 2>/dev/null | sort -V | tail -1)}"
if [ -z "${ASCEND_CANN_PATH}" ]; then
    echo "[ERROR] Cannot find CANN set_env.sh.  Set ASCEND_CANN_PATH explicitly."
    exit 1
fi
source "${ASCEND_CANN_PATH}"

SHORT=r:,v:,d:
LONG=run-mode:,soc-version:,device-id:,rounds:,tiles:,side-tiles:,slot-count:,max-spins:,token-tile:,model-tile:,build-only
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
        (--rounds)           RELAY_ROUNDS="$2"; shift 2;;
        (--tiles)            RELAY_TILES="$2"; shift 2;;
        (--side-tiles)       RELAY_SIDE_TILES="$2"; shift 2;;
        (--slot-count)       RELAY_SLOT_COUNT="$2"; shift 2;;
        (--max-spins)        RELAY_MAX_SPINS="$2"; shift 2;;
        (--token-tile)       RELAY_T="$2"; shift 2;;
        (--model-tile)       RELAY_W="$2"; shift 2;;
        (--build-only)       BUILD_ONLY=1; shift;;
        (--) shift; break;;
        (*) echo "[ERROR] Unexpected option: $1"; exit 1;;
    esac
done

: "${RUN_MODE:=npu}"
: "${SOC_VERSION:=Ascend910B1}"
# Rounds each reduce phase folds.  > 1 so phase 2's members exercise the credit
# wait against the NON-ZERO baseline the retiring unicast flow left behind.
: "${RELAY_ROUNDS:=2}"
# Tiles the middle unicast phase pushes (the last carries CLOSE).
: "${RELAY_TILES:=3}"
# Tiles of the concurrent side flow.  Deliberately != --tiles, so one member enters
# phase 2 with a credit leftover that does NOT match the fold baseline; 0 disables.
: "${RELAY_SIDE_TILES:=1}"
: "${RELAY_SLOT_COUNT:=2}"
# 0 = block forever (the shipping path).  Non-zero bounds every wait, so a handover
# that never happens reports a fault code instead of hanging -- worth setting when
# bisecting a change to the handover rules.
: "${RELAY_MAX_SPINS:=0}"
: "${RELAY_T:=16}"
: "${RELAY_W:=64}"
: "${DEVICE_ID:=${TASK_DEVICE:-${ASCEND_DEVICE_ID:-${DEVICE_ID:-0}}}}"

if [[ ! "${SOC_VERSION}" =~ ^Ascend ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}"
    exit 1
fi

rm -rf /dev/shm/sem.hccl* 2>/dev/null
ipcrm -a 2>/dev/null

echo "=== GridPipe reduce/unicast channel relay smoke ==="
echo "  RUN_MODE: ${RUN_MODE}  SOC_VERSION: ${SOC_VERSION}  DEVICE_ID: ${DEVICE_ID}"
echo "  Rounds/reduce: ${RELAY_ROUNDS}  Unicast tiles: ${RELAY_TILES}  Side tiles: ${RELAY_SIDE_TILES}"
echo "  Slots: ${RELAY_SLOT_COUNT}  MaxSpins: ${RELAY_MAX_SPINS}  Tile: ${RELAY_T}x${RELAY_W}"
echo "=================================================="

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_DIR}"

rm -rf build
mkdir build
cd build

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${LD_LIBRARY_PATH:-}
set -euo pipefail

cmake -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} \
      -DRELAY_ROUNDS=${RELAY_ROUNDS} -DRELAY_TILES=${RELAY_TILES} \
      -DRELAY_SIDE_TILES=${RELAY_SIDE_TILES} -DRELAY_SLOT_COUNT=${RELAY_SLOT_COUNT} \
      -DRELAY_MAX_SPINS=${RELAY_MAX_SPINS} \
      -DRELAY_T=${RELAY_T} -DRELAY_W=${RELAY_W} \
      ..
make -j16 relay_smoke

if [ "${BUILD_ONLY}" -eq 1 ]; then
    echo "[INFO] --build-only requested; skipping run."
    exit 0
fi

echo ""
echo "=== Running GridPipe reduce/unicast channel relay smoke ==="
./relay_smoke --device-id "${DEVICE_ID}"
