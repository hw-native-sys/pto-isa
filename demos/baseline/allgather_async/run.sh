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

# Allgather Async Demo — Build and Run Script
#
# Prerequisites:
#   1. CANN toolkit installed and set_env.sh sourced (ASCEND_HOME_PATH set)
#   2. MPI (mpich) available in PATH
#   3. At least 2 NPU devices available (N_RANKS <= number of devices)
#
# Usage:
#   ./run.sh                             # 8 ranks, default SoC
#   ./run.sh 4                           # 4 ranks
#   ./run.sh 2 Ascend910_9599            # 2 ranks, A5 SoC

set -e

if [ -z "${ASCEND_HOME_PATH}" ]; then
    echo "ASCEND_HOME_PATH not set, sourcing /usr/local/Ascend/ascend-toolkit/set_env.sh ..."
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
N_RANKS="${1:-${N_RANKS:-8}}"
SOC_VERSION="${2:-${SOC_VERSION:-ascend910b1}}"

echo "=== Allgather Async Demo: Building (SOC_VERSION=${SOC_VERSION}) ==="

cd "${SCRIPT_DIR}"
rm -rf build
mkdir -p build && cd build
cmake .. -DSOC_VERSION="${SOC_VERSION}" 2>&1
make -j"$(nproc)" 2>&1
cd "${SCRIPT_DIR}"

echo ""
echo "=== Allgather Async Demo: Running with ${N_RANKS} ranks ==="
echo ""

mpirun -n "${N_RANKS}" ./build/bin/allgather_demo
