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

set -euo pipefail

WORLD_SIZE=8
SOC=Ascend910B
M=2048
K=7168
N=4096
TOPK=8
EXPERTS=16
MAX_OUTPUT_SIZE=81940
AIC_NUM=24
AIV_NUM=48
SEED=20260515
ATOL=1e-4
RTOL=1e-3
WARMUP_ITERS=${DISPATCH_MEGA_COMBINE_WARMUP_ITERS:-3}
MEASURE_ITERS=${DISPATCH_MEGA_COMBINE_MEASURE_ITERS:-5}
GOLDEN_BACKEND=${DISPATCH_MEGA_COMBINE_GOLDEN_BACKEND:-python-batch}
GOLDEN_CHUNK_ROWS=${DISPATCH_MEGA_COMBINE_GOLDEN_CHUNK_ROWS:-512}
GOLDEN_PROFILE=${DISPATCH_MEGA_COMBINE_GOLDEN_PROFILE:-0}
REUSE_DATA=${DISPATCH_MEGA_COMBINE_REUSE_DATA:-0}

: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH must be set before running run.sh}"
CMAKE_COMPILER=${CMAKE_COMPILER:-bisheng}
MPI_ENV_BIN=${MPI_ENV_BIN:-/home/ntlab/miniconda3/envs/ltr_pto/bin}
MPI_ENV_LIB=${MPI_ENV_LIB:-/home/ntlab/miniconda3/envs/ltr_pto/lib}
MPI_LIB_PATH=${MPI_LIB_PATH:-${MPI_ENV_LIB}/libmpi.so}
MPI_RUNNER=${MPI_RUNNER:-mpirun}

export ASCEND_HOME_PATH
export PATH="${MPI_ENV_BIN}:$PATH"
export LD_LIBRARY_PATH="${MPI_ENV_LIB}:${LD_LIBRARY_PATH:-}"
export MPI_LIB_PATH

while [[ $# -gt 0 ]]; do
  case "$1" in
    --soc) SOC="$2"; shift 2 ;;
    --world-size) WORLD_SIZE="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --topk) TOPK="$2"; shift 2 ;;
    --experts) EXPERTS="$2"; shift 2 ;;
    --max-output-size) MAX_OUTPUT_SIZE="$2"; shift 2 ;;
    --aic-num) AIC_NUM="$2"; shift 2 ;;
    --aiv-num) AIV_NUM="$2"; shift 2 ;;
    --atol) ATOL="$2"; shift 2 ;;
    --rtol) RTOL="$2"; shift 2 ;;
    --golden-backend) GOLDEN_BACKEND="$2"; shift 2 ;;
    --golden-chunk-rows) GOLDEN_CHUNK_ROWS="$2"; shift 2 ;;
    --golden-profile) GOLDEN_PROFILE=1; shift ;;
    --reuse-data) REUSE_DATA=1; shift ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_DIR="${SCRIPT_DIR}/out"
BUILD_DIR="${SCRIPT_DIR}/build"

GEN_DATA_EXTRA_ARGS=()
GEN_DATA_EXTRA_ARGS+=(--golden-backend "${GOLDEN_BACKEND}")
GEN_DATA_EXTRA_ARGS+=(--golden-chunk-rows "${GOLDEN_CHUNK_ROWS}")
if [[ "${GOLDEN_PROFILE}" != "0" ]]; then
  GEN_DATA_EXTRA_ARGS+=(--golden-profile)
fi
if [[ "${REUSE_DATA}" != "0" ]]; then
  GEN_DATA_EXTRA_ARGS+=(--reuse-data)
fi

MIB=$((1024 * 1024))
PACKED_OFFSET_A_BYTES=$((MAX_OUTPUT_SIZE * (K + 32)))
OFFSET_A_WINDOW_BYTES=$((PACKED_OFFSET_A_BYTES * 3))
OFFSET_D_BYTES=$((MAX_OUTPUT_SIZE * K * 2))
OFFSET_D_WINDOW_BYTES=$((((OFFSET_D_BYTES + 3 * MIB + 511) * 3 + 1) / 2))
NEEDED_WINDOW_BYTES="${OFFSET_A_WINDOW_BYTES}"
if [[ "${OFFSET_D_WINDOW_BYTES}" -gt "${NEEDED_WINDOW_BYTES}" ]]; then
  NEEDED_WINDOW_BYTES="${OFFSET_D_WINDOW_BYTES}"
fi
NEEDED_HCCL_BUFFSIZE_MB=$(((NEEDED_WINDOW_BYTES + MIB - 1) / MIB + 64))
CURRENT_HCCL_BUFFSIZE_MB="${HCCL_BUFFSIZE:-200}"
if [[ "${CURRENT_HCCL_BUFFSIZE_MB}" -lt "${NEEDED_HCCL_BUFFSIZE_MB}" ]]; then
  echo "[INFO] Raising HCCL_BUFFSIZE from ${CURRENT_HCCL_BUFFSIZE_MB} to ${NEEDED_HCCL_BUFFSIZE_MB} MB" \
    "for maxOutputSize=${MAX_OUTPUT_SIZE} K=${K}"
  export HCCL_BUFFSIZE="${NEEDED_HCCL_BUFFSIZE_MB}"
fi

python3 "${SCRIPT_DIR}/scripts/gen_data.py" \
  --output-dir "${OUT_DIR}" \
  --world-size "${WORLD_SIZE}" \
  --m "${M}" --k "${K}" --n "${N}" \
  --topk "${TOPK}" --experts "${EXPERTS}" \
  --max-output-size "${MAX_OUTPUT_SIZE}" \
  --aic-num "${AIC_NUM}" \
  --aiv-num "${AIV_NUM}" \
  --seed "${SEED}" \
  --atol "${ATOL}" \
  --rtol "${RTOL}" \
  "${GEN_DATA_EXTRA_ARGS[@]}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DSOC_VERSION="${SOC}" \
  -DCMAKE_COMPILER="${CMAKE_COMPILER}"
cmake --build "${BUILD_DIR}" --target dispatch_mega_combine -j16

export LD_LIBRARY_PATH="${BUILD_DIR}/lib:${LD_LIBRARY_PATH}"
export DISPATCH_MEGA_COMBINE_CASE_DIR="${OUT_DIR}"
export DISPATCH_MEGA_COMBINE_WARMUP_ITERS="${WARMUP_ITERS}"
export DISPATCH_MEGA_COMBINE_MEASURE_ITERS="${MEASURE_ITERS}"
"${MPI_RUNNER}" -n "${WORLD_SIZE}" "${BUILD_DIR}/dispatch_mega_combine"
