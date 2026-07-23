#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the software repository for the full text of the License.
# --------------------------------------------------------------------------------

# Pure 1D N-cut 32-cell WSE FFN emulation -- TPUSH AllGather variant.
# Verifies the GridPipe TPUSH/TPOP unicast primitives on a distributed-FFN AllGather:
# the two gather phases are nearest-neighbor TPUSH/TPOP relays (a fan-in-1 DAG)
# instead of the MPSC TBROADCAST collective, and the output is compared bit-exact
# against the same SwiGLU golden.  grid-rows x grid-cols is the logical grid launched
# on one NPU; columns are model-parallel FFN shards, hidden shards are relay-gathered
# before down, and down writes output-H shards directly.

: "${ASCEND_CANN_PATH:=$(ls -1d /usr/local/Ascend/cann-*/set_env.sh 2>/dev/null | sort -V | tail -1)}"
if [ -z "${ASCEND_CANN_PATH}" ]; then
    echo "[ERROR] Cannot find CANN set_env.sh.  Set ASCEND_CANN_PATH explicitly."
    exit 1
fi
source "${ASCEND_CANN_PATH}"

SHORT=r:,v:,n:,d:
LONG=run-mode:,soc-version:,n-ranks:,device-id:,grid-rows:,grid-cols:,token-tile:,model-tile:,ffn-tile:,phys-cores:,build-only
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
        (--phys-cores)       PHYS_CORES="$2"; shift 2;;
        (--build-only)       BUILD_ONLY=1; shift;;
        (--) shift; break;;
        (*) echo "[ERROR] Unexpected option: $1"; exit 1;;
    esac
done

# Defaults = real DeepSeek-v4 Pro FFN on a 4x8 = 32-cell mesh.
# ffn-tile is the per-cell I_shard; full I = ffn-tile * grid_rows * grid_cols.
: "${RUN_MODE:=npu}"
: "${SOC_VERSION:=Ascend910B1}"
: "${GRID_ROWS:=4}"
: "${GRID_COLS:=8}"
: "${TOKEN_TILE:=8}"
: "${MODEL_TILE:=7168}"
: "${FFN_TILE:=96}"
: "${N_RANKS:=1}"
: "${PHYS_CORES:=24}"
: "${DEVICE_ID:=${TASK_DEVICE:-${FFN_GRID_DEVICE_ID:-${ASCEND_DEVICE_ID:-0}}}}"

NCUT_I=$(( FFN_TILE * GRID_ROWS * GRID_COLS ))

if [ "${N_RANKS}" -ne 1 ]; then
    echo "[ERROR] Single-device multi-block mode requires -n/--n-ranks 1."
    exit 1
fi
if [ $((MODEL_TILE % (GRID_ROWS * GRID_COLS))) -ne 0 ]; then
    echo "[ERROR] pure-N-cut requires --model-tile (H) divisible by grid_rows*grid_cols (cells)."
    exit 1
fi
if [ $((NCUT_I % (GRID_ROWS * GRID_COLS))) -ne 0 ]; then
    echo "[ERROR] pure-N-cut requires full I (=ffn-tile*cells) divisible by cells."
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

echo "=== Pure 1D N-cut 32-cell WSE FFN emulation -- TPUSH AllGather ==="
echo "  RUN_MODE: ${RUN_MODE}  SOC_VERSION: ${SOC_VERSION}"
echo "  Grid: ${GRID_ROWS}x${GRID_COLS}  Cells: $((GRID_ROWS*GRID_COLS))  PhysCores: ${PHYS_CORES}"
echo "  T=${TOKEN_TILE}  H=${MODEL_TILE}  I_shard=${FFN_TILE}  I_full=${NCUT_I}"
echo "==================================================="

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}"

# Regenerate flat full-tensor Batcher data + SwiGLU golden before running (same
# flat tensors as the other variants).
if [ "${BUILD_ONLY}" -eq 0 ]; then
    python3 "${SCRIPT_DIR}/scripts/gen_data.py" \
        --pure-ncut \
        --grid-rows "${GRID_ROWS}" --grid-cols "${GRID_COLS}" \
        --t "${TOKEN_TILE}" --h "${MODEL_TILE}" --fi "${FFN_TILE}" \
        --act silu \
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
      -DNCUT_GRID_ROWS=${GRID_ROWS} -DNCUT_GRID_COLS=${GRID_COLS} \
      -DNCUT_T=${TOKEN_TILE} -DNCUT_H=${MODEL_TILE} -DNCUT_I=${NCUT_I} \
      ..
make -j16 distributed_ffn_grid_tpush_allgather

if [ "${BUILD_ONLY}" -eq 1 ]; then
    echo "[INFO] --build-only requested; skipping run."
    exit 0
fi

echo ""
echo "=== Running 32-cell N-cut FFN GridPipe TPUSH AllGather (emulated on ${PHYS_CORES} phys AICores) ==="
export N_RANKS=${N_RANKS}
export FFN_GRID_DATA_DIR="${SCRIPT_DIR}/out"
./distributed_ffn_grid_tpush_allgather --device-id "${DEVICE_ID}" --phys-cores "${PHYS_CORES}"
