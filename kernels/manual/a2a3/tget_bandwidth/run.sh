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

# TGET bandwidth example — build and run (HCCL + MPICH)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Ascend CANN environment: keep an already-sourced env; otherwise auto-detect.
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    : "${ASCEND_CANN_PATH:=$(ls -1d /usr/local/Ascend/cann-*/set_env.sh 2>/dev/null | sort -V | tail -1)}"
    if [ -n "${ASCEND_CANN_PATH}" ] && [ -f "${ASCEND_CANN_PATH}" ]; then
        # shellcheck disable=SC1090
        source "${ASCEND_CANN_PATH}"
    fi
fi

# MPI setup: search common MPICH install locations.
# Override with MPI_SEARCH_DIRS (space-separated list of bin/ directories).
if [ -z "${MPI_SEARCH_DIRS:-}" ]; then
    MPI_SEARCH_DIRS="/usr/local/mpich/bin /home/mpich/bin /usr/lib64/mpich/bin"
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

if ! command -v mpirun >/dev/null 2>&1; then
    echo "[ERROR] mpirun not found. Install MPICH or set MPI_SEARCH_DIRS / MPI_HOME."
    exit 1
fi

SHORT=r:,v:,n:,
LONG=run-mode:,soc-version:,nranks:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

RUN_MODE="npu"
NRANKS=2

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
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done

if [[ "${SOC_VERSION}" == "a3" || "${SOC_VERSION}" == "A3" ]]; then
    SOC_VERSION="Ascend910B1"
fi

if [[ ! "${SOC_VERSION}" =~ ^Ascend ]]; then
    echo "[ERROR] Unsupported SocVersion: ${SOC_VERSION} (use a3 or an Ascend* SoC string)"
    exit 1
fi

rm -rf build
mkdir build
cd build

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH

cmake -DRUN_MODE=${RUN_MODE} -DSOC_VERSION=${SOC_VERSION} ..
make -j16

echo "=== Running TGET bandwidth (MPICH, mpirun -n ${NRANKS}) ==="
mpirun -n "${NRANKS}" ./tget_bandwidth
