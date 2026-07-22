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

# Single-device multi-block FFN GridPipe AllGather demo.  grid-rows x
# grid-cols is the logical grid launched on one NPU.  Columns are
# model-parallel FFN shards; hidden shards are AllGathered before down, and
# down writes output-H shards directly.

: "${ASCEND_CANN_PATH:=$(ls -1d /usr/local/Ascend/cann-*/set_env.sh 2>/dev/null | sort -V | tail -1)}"
if [ -z "${ASCEND_CANN_PATH}" ]; then
    echo "[ERROR] Cannot find CANN set_env.sh.  Set ASCEND_CANN_PATH explicitly."
    exit 1
fi
source "${ASCEND_CANN_PATH}"

SHORT=r:,v:,n:,d:
LONG=run-mode:,soc-version:,n-ranks:,device-id:,grid-rows:,grid-cols:,token-tile:,model-tile:,ffn-tile:,build-only
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

BUILD_ONLY=0
while :; do
    case "$1" in
        (-r | --run-mode)    RUN_MODE="$2"; shift 2;;
        (-v | --soc-version) SOC_VERSION="$2"; shift 2;;
        (-n | --n-ranks)     N_RANKS="$2"; shift 2;;
        (-d | --device-id)   DEVICE_ID="$2"; shift 2;;
        (--grid-rows)        GRID_ROWS="$2"; shift 2;;
        (--grid-cols)        GRID_COLS="$2"; shift 2;;
        (--token-tile)       TOKEN_TILE="$2"; shift 2;;
        (--model-tile)       MODEL_TILE="$2"; shift 2;;
        (--ffn-tile)         FFN_TILE="$2"; shift 2;;
        (--build-only)       BUILD_ONLY=1; shift;;
        (--) shift; break;;
        (*) echo "[ERROR] Unexpected option: $1"; exit 1;;
    esac
done

: "${RUN_MODE:=npu}"
: "${SOC_VERSION:=Ascend910B1}"
: "${GRID_ROWS:=2}"
: "${GRID_COLS:=2}"
: "${TOKEN_TILE:=16}"
: "${MODEL_TILE:=64}"
: "${FFN_TILE:=64}"
: "${N_RANKS:=1}"
: "${DEVICE_ID:=${FFN_GRID_DEVICE_ID:-${ASCEND_DEVICE_ID:-${DEVICE_ID:-0}}}}"

if [ "${N_RANKS}" -ne 1 ]; then
    echo "[ERROR] Single-device multi-block mode requires -n/--n-ranks 1."
    exit 1
fi
if [ $((MODEL_TILE % GRID_COLS)) -ne 0 ]; then
    echo "[ERROR] AllGather mode requires --model-tile divisible by --grid-cols."
    exit 1
fi

if [[ ! "${SOC_VERSION}" =~ ^Ascend ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}"
    exit 1
fi
if [[ "${SOC_VERSION}" =~ ^Ascend910B4-1 ]] && [ "${RUN_MODE}" == "sim" ]; then
    echo "[ERROR] SocVersion: ${SOC_VERSION} can not support sim mode, please use Ascend910B4."
    exit 1
fi

rm -rf /dev/shm/sem.hccl* 2>/dev/null
ipcrm -a 2>/dev/null

echo "=== Single-device Multi-block FFN GridPipe AllGather Demo ==="
echo "  RUN_MODE: ${RUN_MODE}  SOC_VERSION: ${SOC_VERSION}"
echo "  Grid: ${GRID_ROWS}x${GRID_COLS}  N_RANKS: ${N_RANKS}  DEVICE_ID: ${DEVICE_ID}"
echo "  Tile: ${TOKEN_TILE}x${MODEL_TILE}  FfnTile: ${FFN_TILE}"
echo "==================================================="

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}"

# Regenerate full-tensor Batcher data + SwiGLU golden before running.  The host
# Batcher (batcher.hpp) loads the full weights/x, splits them column-parallel,
# and collects the output; the host maps all grid_rows * grid_cols cells to
# blocks on the selected single device.
if [ "${BUILD_ONLY}" -eq 0 ]; then
    python3 "${SCRIPT_DIR}/scripts/gen_data.py" \
        --grid-rows "${GRID_ROWS}" --grid-cols "${GRID_COLS}" \
        --t "${TOKEN_TILE}" --h "${MODEL_TILE}" --fi "${FFN_TILE}" \
        --act silu --split-mode allgather \
        --output-dir "${SCRIPT_DIR}/out"
fi

rm -rf build
mkdir build
cd build

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${LD_LIBRARY_PATH:-}
set -euo pipefail

cmake -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} \
      -DGRID_ROWS=${GRID_ROWS} -DGRID_COLS=${GRID_COLS} \
      -DTOKEN_TILE=${TOKEN_TILE} -DMODEL_TILE=${MODEL_TILE} -DFFN_TILE=${FFN_TILE} \
      ..
make -j16 distributed_ffn_grid_allgather

if [ "${BUILD_ONLY}" -eq 1 ]; then
    echo "[INFO] --build-only requested; skipping run."
    exit 0
fi

echo ""
echo "=== Running Single-device Multi-block FFN GridPipe AllGather ==="
export N_RANKS=${N_RANKS}
export FFN_GRID_DATA_DIR="${SCRIPT_DIR}/out"
./distributed_ffn_grid_allgather --device-id "${DEVICE_ID}"
