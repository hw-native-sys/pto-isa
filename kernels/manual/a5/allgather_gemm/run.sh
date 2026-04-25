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

# AllGather GEMM Demo - Build and Run (HCCL backend)

# Ascend CANN environment: honor ASCEND_CANN_PATH or auto-detect
: "${ASCEND_CANN_PATH:=$(ls -1d /usr/local/Ascend/cann-*/set_env.sh 2>/dev/null | sort -V | tail -1)}"
if [ -z "${ASCEND_CANN_PATH}" ]; then
    echo "[ERROR] Cannot find CANN set_env.sh. Set ASCEND_CANN_PATH to <cann-install>/set_env.sh"
    exit 1
fi
source "${ASCEND_CANN_PATH}"

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

SHORT=r:,v:,n:,m:,k:,
LONG=run-mode:,soc-version:,n-ranks:,gm:,gk:,gn:,base-m:,base-n:,compute-blocks:,comm-blocks:,
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
        (-n | --n-ranks )
            N_RANKS="$2"
            shift 2;;
        (--gm )
            G_M="$2"
            shift 2;;
        (--gk )
            G_K="$2"
            shift 2;;
        (--gn )
            G_N="$2"
            shift 2;;
        (--base-m )
            G_BASE_M="$2"
            shift 2;;
        (--base-n )
            G_BASE_N="$2"
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

: "${N_RANKS:=2}"
: "${RUN_MODE:=npu}"
: "${SOC_VERSION:=Ascend950PR_958b}"
: "${G_M:=2048}"
: "${G_K:=2048}"
: "${G_N:=1024}"
: "${G_BASE_M:=128}"
: "${G_BASE_N:=256}"

if [[ ! "${SOC_VERSION}" =~ ^Ascend ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION}"
    exit 1
fi
if [[ "${SOC_VERSION}" =~ ^Ascend950PR ]] && [ "${RUN_MODE}" == "sim" ]; then
    echo "[WARN] Simulator support depends on the installed ${SOC_VERSION} package. Proceeding with sim mode."
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "${SCRIPT_DIR}"

if [ $((G_BASE_N % 4)) -ne 0 ]; then
    echo "[ERROR] --base-n (${G_BASE_N}) must be divisible by 4 (K pack step); BASE_K is derived as BASE_N/4."
    exit 1
fi
if [ $((G_M % G_BASE_M)) -ne 0 ] || [ $((G_K % G_BASE_N)) -ne 0 ] || [ $((G_N % G_BASE_N)) -ne 0 ]; then
    echo "[ERROR] Require G_M%G_BASE_M==0, G_K%G_BASE_N==0, G_N%G_BASE_N==0 (got M=${G_M} K=${G_K} N=${G_N} BASE_M=${G_BASE_M} BASE_N=${G_BASE_N})."
    exit 1
fi

# Clean stale HCCL shared-memory state from any previous crashed run
rm -rf /dev/shm/sem.hccl* 2>/dev/null
ipcrm -a 2>/dev/null

echo "=== AllGather GEMM Demo (HCCL) ==="
echo "  RUN_MODE: ${RUN_MODE}  SOC_VERSION: ${SOC_VERSION}"
echo "  N_RANKS: ${N_RANKS}  G_M: ${G_M}  G_K: ${G_K}  G_N: ${G_N}"
echo "  G_BASE_M: ${G_BASE_M}  G_BASE_N: ${G_BASE_N}  COMPUTE_BLOCKS: ${COMPUTE_BLOCKS:-default}  COMM_BLOCKS: ${COMM_BLOCKS:-default}"
echo "==========================="

python3 scripts/gen_data.py --n-ranks "${N_RANKS}" --m "${G_M}" --k "${G_K}" --n "${G_N}" --output-dir ./out

rm -rf build
mkdir build
cd build

set -euo pipefail
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:${LD_LIBRARY_PATH:-}

BLOCK_OPTS=""
if [ -n "${COMPUTE_BLOCKS:-}" ]; then
    BLOCK_OPTS="$BLOCK_OPTS -DCOMPUTE_BLOCKS=${COMPUTE_BLOCKS}"
fi
if [ -n "${COMM_BLOCKS:-}" ]; then
    BLOCK_OPTS="$BLOCK_OPTS -DCOMM_BLOCKS=${COMM_BLOCKS}"
fi

cmake -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} \
      -DG_M=${G_M} -DG_K=${G_K} -DG_N=${G_N} \
      -DG_BASE_M=${G_BASE_M} -DG_BASE_N=${G_BASE_N} \
      ${BLOCK_OPTS} ..
make -j16

echo ""
echo "=== Running AllGather GEMM (HCCL, mpirun) ==="

export N_RANKS=${N_RANKS}
export ALLGATHER_GEMM_DATA_DIR="${SCRIPT_DIR}/out"
mpirun -n "${N_RANKS}" ./allgather_gemm
