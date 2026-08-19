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

# GridPipe single-source broadcast smoke test.  One source cell TBROADCASTs a
# stamped fp32 tile to its whole group (batched writes + one publish fence +
# batched doorbells); every other cell drains its source lane and stores it.
# Verifies out[cell] == in[group-source] in-process (no data files).

: "${ASCEND_CANN_PATH:=$(ls -1d /usr/local/Ascend/cann-*/set_env.sh 2>/dev/null | sort -V | tail -1)}"
if [ -z "${ASCEND_CANN_PATH}" ]; then
    echo "[ERROR] Cannot find CANN set_env.sh.  Set ASCEND_CANN_PATH explicitly."
    exit 1
fi
source "${ASCEND_CANN_PATH}"

SHORT=r:,v:,d:
LONG=run-mode:,soc-version:,device-id:,grid-rows:,grid-cols:,src:,span-col:,subrect:,rect-r0:,rect-r1:,rect-c0:,rect-c1:,rect-src:,rounds:,all-sources:,max-spins:,ticket-batch:,slot-count:,token-tile:,model-tile:,build-only
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@") || {
    # getopt already named the offending flag.  Bail out instead of falling back to
    # the defaults: a silently ignored option looks exactly like a passing run of a
    # configuration that was never built.
    echo "[ERROR] bad arguments"; exit 2;
}
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
        (--rounds)           BCAST_ROUNDS="$2"; shift 2;;
        (--all-sources)      BCAST_ALL_SOURCES="$2"; shift 2;;
        (--max-spins)        BCAST_MAX_SPINS="$2"; shift 2;;
        (--ticket-batch)     BCAST_TICKET_BATCH="$2"; shift 2;;
        (--slot-count)       BCAST_SLOT_COUNT="$2"; shift 2;;
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
: "${BCAST_ROUNDS:=1}"
: "${BCAST_ALL_SOURCES:=0}"
# 0 = public TBROADCAST/TPOP (block forever); n > 0 = TRY forms bounded at n, which
# report a fault code instead of hanging.  Use --max-spins 1000 to debug a hang.
: "${BCAST_MAX_SPINS:=0}"
# Publishers one receiver may have in flight.  It is the grant WINDOW width, so it
# is also the collective's concurrency degree; the pipe clamps it to SlotCount, so
# the default means "as concurrent as the ring allows".  Pass 1 to exercise the
# strictly-serialised grant path (every receiver hands out one basek at a time).
: "${BCAST_TICKET_BATCH:=32}"
# Broadcast ring depth in slots; 0 = one per group member.  Set it BELOW the group
# width to exercise the wave loop (publishers no longer all fit in the ring).
: "${BCAST_SLOT_COUNT:=0}"
: "${BCAST_T:=16}"
: "${BCAST_W:=64}"
# TASK_DEVICE first, like the four run_*.sh next door: on this server an NPU run is
# submitted through task-submit, which locks a card and exports its id there.  Without
# it this script silently falls back to device 0 and runs on whatever card that is --
# very likely one another user's task already holds.
: "${DEVICE_ID:=${TASK_DEVICE:-${ASCEND_DEVICE_ID:-${DEVICE_ID:-0}}}}"

if [[ ! "${SOC_VERSION}" =~ ^Ascend ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}"
    exit 1
fi

rm -rf /dev/shm/sem.hccl* 2>/dev/null
ipcrm -a 2>/dev/null

echo "=== GridPipe single-source broadcast smoke ==="
echo "  RUN_MODE: ${RUN_MODE}  SOC_VERSION: ${SOC_VERSION}  DEVICE_ID: ${DEVICE_ID}"
echo "  Grid: ${BCAST_ROWS}x${BCAST_COLS}  SRC: ${BCAST_SRC}  SPAN_COL: ${BCAST_SPAN_COL}  SUBRECT: ${BCAST_SUBRECT}  Rect: [r${BCAST_RECT_R0}:${BCAST_RECT_R1},c${BCAST_RECT_C0}:${BCAST_RECT_C1}] RectSRC: ${BCAST_RECT_SRC}  Rounds: ${BCAST_ROUNDS}  AllSrc: ${BCAST_ALL_SOURCES}  Tile: ${BCAST_T}x${BCAST_W}  TicketBatch: ${BCAST_TICKET_BATCH}  Slots: ${BCAST_SLOT_COUNT}"
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
      -DBCAST_ROUNDS=${BCAST_ROUNDS} -DBCAST_ALL_SOURCES=${BCAST_ALL_SOURCES} \
      -DBCAST_MAX_SPINS=${BCAST_MAX_SPINS} -DBCAST_TICKET_BATCH=${BCAST_TICKET_BATCH} -DBCAST_SLOT_COUNT=${BCAST_SLOT_COUNT} \
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
