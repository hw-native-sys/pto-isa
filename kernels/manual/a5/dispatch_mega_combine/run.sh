#!/usr/bin/env bash
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

WORLD_SIZE=2
FIRST_DEVICE=${FIRST_DEVICE:-0}
SOC=Ascend910_9599
M=2048
K=7168
N=4096
TOPK=8
EXPERTS=16
MAX_OUTPUT_SIZE=81940
SEED=20260515
ATOL=1e-2
RTOL=1e-2
REUSE_DATA=${DISPATCH_MEGA_COMBINE_REUSE_DATA:-0}
OUTPUT_DIR=""
START_SYNC=${DISPATCH_MEGA_COMBINE_START_SYNC:-0}
WARMUP_ITERS=${DISPATCH_MEGA_COMBINE_WARMUP_ITERS:-3}
MEASURE_ITERS=${DISPATCH_MEGA_COMBINE_MEASURE_ITERS:-5}
BUILD_ONLY=${BUILD_ONLY:-0}
# 0 selects the runtime-reported core count. Nonzero values select a validated
# launch topology and must not exceed the physical device count.
AICORE_NUM=${DISPATCH_MEGA_COMBINE_AICORE_NUM:-0}
export HCCL_WHITELIST_DISABLE=1

: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH must be set before running run.sh}"
CMAKE_COMPILER=${CMAKE_COMPILER:-bisheng}
MPI_RUNNER=${MPI_RUNNER:-mpirun}

export ASCEND_HOME_PATH

while [[ $# -gt 0 ]]; do
  case "$1" in
    --soc) SOC="$2"; shift 2 ;;
    --world-size) WORLD_SIZE="$2"; shift 2 ;;
    --first-device) FIRST_DEVICE="$2"; shift 2 ;;
    --m) M="$2"; shift 2 ;;
    --k) K="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --topk) TOPK="$2"; shift 2 ;;
    --experts) EXPERTS="$2"; shift 2 ;;
    --max-output-size) MAX_OUTPUT_SIZE="$2"; shift 2 ;;
    --atol) ATOL="$2"; shift 2 ;;
    --rtol) RTOL="$2"; shift 2 ;;
    --aicore-num|--aic-num) AICORE_NUM="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --reuse-data) REUSE_DATA=1; shift ;;
    --build-only) BUILD_ONLY=1; shift ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

if [[ ! "${WORLD_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--world-size must be a positive integer, got: ${WORLD_SIZE}" >&2
  exit 1
fi
if [[ ! "${FIRST_DEVICE}" =~ ^[0-9]+$ ]]; then
  echo "--first-device must be a non-negative device ID, got: ${FIRST_DEVICE}" >&2
  exit 1
fi
if [[ ! "${AICORE_NUM}" =~ ^(0|28|32|36)$ ]]; then
  echo "--aicore-num must be one of 0, 28, 32, or 36, got: ${AICORE_NUM}" >&2
  exit 1
fi
FIRST_DEVICE=$((10#${FIRST_DEVICE}))
if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  echo "[INFO] Ignoring ASCEND_RT_VISIBLE_DEVICES; binding physical devices with --first-device"
  unset ASCEND_RT_VISIBLE_DEVICES
fi
echo "[INFO] Mapping MPI ranks 0-$((WORLD_SIZE - 1)) to physical NPU devices" \
  "${FIRST_DEVICE}-$((FIRST_DEVICE + WORLD_SIZE - 1))"

case "${SOC}" in
  Ascend910_9599|Ascend950PR_*) ;;
  *)
    echo "A5 requires an A5 SoC alias (Ascend910_9599 or Ascend950PR_*), got: ${SOC}" >&2
    exit 1
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/out"}
BUILD_DIR="${SCRIPT_DIR}/build"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
  -DSOC_VERSION="${SOC}" \
  -DCMAKE_COMPILER="${CMAKE_COMPILER}" \
  -DCMAKE_C_COMPILER="${CMAKE_COMPILER}" \
  -DCMAKE_CXX_COMPILER="${CMAKE_COMPILER}"
cmake --build "${BUILD_DIR}" --target dispatch_mega_combine -j16

if [[ "${BUILD_ONLY}" != "0" ]]; then
  echo "[INFO] BUILD_ONLY set; skipping data generation and mpirun execution."
  exit 0
fi

GEN_DATA_EXTRA_ARGS=()
if [[ "${REUSE_DATA}" != "0" ]]; then
  GEN_DATA_EXTRA_ARGS+=(--reuse-data)
fi

MIB=$((1024 * 1024))
HCCL_WINDOW_HEAD_GUARD_BYTES=4096
ROUTE_COUNT=$((M * TOPK))
QUANT_DATA_STORAGE_BYTES=$((((K + 255) / 256) * 256))
QUANT_SCALE_COLS=$((K / 32))
QUANT_SCALE_STORAGE_BYTES=$((((QUANT_SCALE_COLS + 31) / 32) * 32))
PACKED_ROW_STRIDE=$((QUANT_DATA_STORAGE_BYTES + QUANT_SCALE_STORAGE_BYTES))
SOURCE_TOKEN_RECORD_BYTES=$((M * PACKED_ROW_STRIDE))
ROUTE_MASK_BYTES=$(((((ROUTE_COUNT + 7) / 8) + 31) / 32 * 32))
FRONT_AIV_NUM=$((AICORE_NUM == 0 ? 72 : AICORE_NUM * 2))
MASK_BLOCK_COUNT=$((ROUTE_MASK_BYTES / 32))
GLOBAL_EXPERTS=$((WORLD_SIZE * EXPERTS))
MAX_ALLOCATED_MASK_LANES=$(((FRONT_AIV_NUM + GLOBAL_EXPERTS - 1) / GLOBAL_EXPERTS))
MASK_LANE_CAPACITY=$((MASK_BLOCK_COUNT < MAX_ALLOCATED_MASK_LANES ? MASK_BLOCK_COUNT : MAX_ALLOCATED_MASK_LANES))
ROUTE_MASK_SLOT_BYTES=$((ROUTE_MASK_BYTES + MASK_LANE_CAPACITY * 32))
EXPERTS_PER_RANK=${EXPERTS}
ROUTE_MASK_REGION_BYTES=$((EXPERTS_PER_RANK * WORLD_SIZE * ROUTE_MASK_SLOT_BYTES))
ROUTE_MASK_OFFSET=$((((SOURCE_TOKEN_RECORD_BYTES + 511) / 512) * 512))
COMBINE_OUTPUT_OFFSET=$((((ROUTE_MASK_OFFSET + ROUTE_MASK_REGION_BYTES + 511) / 512) * 512))
COMBINE_OUTPUT_BYTES=$((ROUTE_COUNT * K * 2))
PRESUM_OFFSET=$((((COMBINE_OUTPUT_OFFSET + COMBINE_OUTPUT_BYTES + 511) / 512) * 512))
PRESUM_BYTES=$(((((GLOBAL_EXPERTS * 4) + 511) / 512) * 512))
PEER_DATA_BYTES=$((((PRESUM_OFFSET + PRESUM_BYTES + 511) / 512) * 512))
NEEDED_WINDOW_BYTES=$((PEER_DATA_BYTES + MIB))
NEEDED_HCCL_BUFFSIZE_MB=$(((NEEDED_WINDOW_BYTES + HCCL_WINDOW_HEAD_GUARD_BYTES + MIB - 1) / MIB + 64))
CURRENT_HCCL_BUFFSIZE_MB="${HCCL_BUFFSIZE:-200}"
if [[ "${CURRENT_HCCL_BUFFSIZE_MB}" -lt "${NEEDED_HCCL_BUFFSIZE_MB}" ]]; then
  echo "[INFO] Raising HCCL_BUFFSIZE from ${CURRENT_HCCL_BUFFSIZE_MB} to ${NEEDED_HCCL_BUFFSIZE_MB} MB" \
    "for mask-pull M=${M} topK=${TOPK} K=${K} hcclHeadGuard=${HCCL_WINDOW_HEAD_GUARD_BYTES}"
  export HCCL_BUFFSIZE="${NEEDED_HCCL_BUFFSIZE_MB}"
fi

python3 "${SCRIPT_DIR}/scripts/gen_data.py" \
  --output-dir "${OUT_DIR}" \
  --world-size "${WORLD_SIZE}" \
  --m "${M}" --k "${K}" --n "${N}" \
  --topk "${TOPK}" --experts "${EXPERTS}" \
  --max-output-size "${MAX_OUTPUT_SIZE}" \
  --seed "${SEED}" \
  --atol "${ATOL}" \
  --rtol "${RTOL}" \
  "${GEN_DATA_EXTRA_ARGS[@]}"

export LD_LIBRARY_PATH="${BUILD_DIR}/lib:${LD_LIBRARY_PATH:-}"
export DISPATCH_MEGA_COMBINE_CASE_DIR="${OUT_DIR}"
export DISPATCH_MEGA_COMBINE_AICORE_NUM="${AICORE_NUM}"
export DISPATCH_MEGA_COMBINE_START_SYNC="${START_SYNC}"
export DISPATCH_MEGA_COMBINE_WARMUP_ITERS="${WARMUP_ITERS}"
export DISPATCH_MEGA_COMBINE_MEASURE_ITERS="${MEASURE_ITERS}"
if ! command -v "${MPI_RUNNER}" >/dev/null 2>&1; then
  echo "MPICH launcher not found: ${MPI_RUNNER}. Source the project environment before running." >&2
  exit 1
fi
MPI_VERSION=$("${MPI_RUNNER}" --version 2>&1 || true)
case "${MPI_VERSION}" in
  *HYDRA*|*MPICH*) ;;
  *)
    echo "dispatch_mega_combine requires MPICH; ${MPI_RUNNER} is not an MPICH launcher." >&2
    exit 1
    ;;
esac
"${MPI_RUNNER}" -n "${WORLD_SIZE}" "${BUILD_DIR}/dispatch_mega_combine" --first-device "${FIRST_DEVICE}"
