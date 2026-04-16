#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

# GEMM AllReduce A5 Demo — Build and Run (HCCL backend)

# Ascend CANN environment: honor ASCEND_CANN_PATH or auto-detect
: "${ASCEND_CANN_PATH:=$(ls -1d /usr/local/Ascend/cann-*/set_env.sh 2>/dev/null | sort -V | tail -1)}"
if [ -z "${ASCEND_CANN_PATH}" ]; then
    echo "[ERROR] Cannot find CANN set_env.sh. Set ASCEND_CANN_PATH to <cann-install>/set_env.sh"
    exit 1
fi
source "${ASCEND_CANN_PATH}"
export CMAKE_PREFIX_PATH="$HOME/.local/lib64/cmake:${CMAKE_PREFIX_PATH:-}"

# MPI setup: search common mpich install locations.
# Override with MPI_SEARCH_DIRS (space-separated list of bin/ directories).
if [ -z "${MPI_SEARCH_DIRS:-}" ]; then
    MPI_SEARCH_DIRS="/usr/local/mpich/bin /home/mpich/bin"
    for candidate in /home/*/mpich/bin /home/*/*/mpich/bin; do
        [ -d "$candidate" ] && MPI_SEARCH_DIRS="$MPI_SEARCH_DIRS $candidate"
    done
fi
for d in ${MPI_SEARCH_DIRS}; do
    if [ -x "$d/mpirun" ]; then
        export PATH="$d:$PATH"
        MPI_LIB_DIR="$(dirname "$d")/lib"
        export LD_LIBRARY_PATH="$MPI_LIB_DIR:${LD_LIBRARY_PATH:-}"
        export MPI_LIB_PATH="$MPI_LIB_DIR/libmpi.so"
        break
    fi
done

SHORT=r:,v:,n:,d:
LONG=run-mode:,soc-version:,nranks:,ndevices:,compute-blocks:,comm-blocks:,
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
        (-n | --nranks )
            NRANKS="$2"
            shift 2;;
        (-d | --ndevices )
            NDEVICES="$2"
            shift 2;;
        (--compute-blocks )
            COMPUTE_BLOCKS="$2"
            shift 2;;
        (--comm-blocks )
            COMM_BLOCKS="$2"
            shift 2;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done

: "${NRANKS:=2}"
: "${NDEVICES:=2}"
: "${RUN_MODE:=npu}"
: "${SOC_VERSION:=Ascend910B1}"

# Clean stale HCCL shared-memory state from any previous crashed run
rm -rf /dev/shm/sem.hccl* 2>/dev/null
ipcrm -a 2>/dev/null

if [[ ! "${SOC_VERSION}" =~ ^Ascend ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}"
    exit 1
fi

if [[ "${SOC_VERSION}" =~ ^Ascend910B4-1 ]] && [ "${RUN_MODE}" == "sim" ]; then
    echo "[ERROR] SocVersion: ${SOC_VERSION} can not support sim mode, please use Ascend910B4."
    exit 1
fi

: "${G_M:=5416}"
  : "${G_K:=6144}"
: "${G_N:=1408}"
: "${G_BASE_M:=}"
: "${G_BASE_N:=}"


# Pad M/N to G_BASE_M / G_BASE_N (defaults 128 / 256 match gemm_ar_config.h). G_BASE_K is fixed at 64 in kernel.
BM=${G_BASE_M:-128}
BN=${G_BASE_N:-256}
PAD_M=$(( ((G_M + BM - 1) / BM) * BM ))
PAD_N=$(( ((G_N + BN - 1) / BN) * BN ))

# HCCL window = reduced_output (M*N*2) + signal_matrix (64B) + margin
NEEDED_MB=$(( PAD_M * PAD_N * 2 / 1024 / 1024 + 64 ))
CURRENT_BUFFSIZE="${HCCL_BUFFSIZE:-200}"
if [ "${CURRENT_BUFFSIZE}" -lt "${NEEDED_MB}" ]; then
    echo "[INFO] Raising HCCL_BUFFSIZE from ${CURRENT_BUFFSIZE} to ${NEEDED_MB} MB for M=${G_M}(pad=${PAD_M}) N=${G_N}(pad=${PAD_N}) nranks=${NRANKS}"
    export HCCL_BUFFSIZE="${NEEDED_MB}"
fi

echo "=== GEMM AllReduce A5 Demo (HCCL) ==="
echo "  RUN_MODE: ${RUN_MODE}  SOC_VERSION: ${SOC_VERSION}"
echo "  NRANKS: ${NRANKS}  NDEVICES: ${NDEVICES}"
echo "  HCCL_BUFFSIZE: ${HCCL_BUFFSIZE:-200} MB"
echo "  COMPUTE_BLOCKS: ${COMPUTE_BLOCKS:-default}  COMM_BLOCKS: ${COMM_BLOCKS:-default}"
echo "==========================="

rm -rf build
mkdir build
cd build

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${LD_LIBRARY_PATH:-}

if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${CONDA_PREFIX}/aarch64-conda-linux-gnu/lib:${LD_LIBRARY_PATH}
fi

set -euo pipefail

BLOCK_OPTS=""
if [ -n "${COMPUTE_BLOCKS:-}" ]; then
    BLOCK_OPTS="$BLOCK_OPTS -DCOMPUTE_BLOCKS=${COMPUTE_BLOCKS}"
fi
if [ -n "${COMM_BLOCKS:-}" ]; then
    BLOCK_OPTS="$BLOCK_OPTS -DCOMM_BLOCKS=${COMM_BLOCKS}"
fi

TILE_OPTS=""
[ -n "${G_BASE_M}" ] && TILE_OPTS="$TILE_OPTS -DCONFIG_G_BASE_M=${G_BASE_M}"
[ -n "${G_BASE_N}" ] && TILE_OPTS="$TILE_OPTS -DCONFIG_G_BASE_N=${G_BASE_N}"

cmake -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} \
      -DCONFIG_G_M=${G_M} -DCONFIG_G_K=${G_K} -DCONFIG_G_N=${G_N} \
      ${BLOCK_OPTS} ${TILE_OPTS} ..
make -j16

echo ""
echo "=== Running GEMM AllReduce A5 (HCCL, mpirun) ==="

FIRST_DEVICE="${FIRST_DEVICE:-0}"
export GEMM_AR_DIR="$(cd .. && pwd)"
mpirun -n ${NRANKS} ./gemm_allreduce --first-device ${FIRST_DEVICE}
