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
URMA=${DISPATCH_MEGA_COMBINE_URMA:-0}
RANKS_PER_SERVER=${DISPATCH_MEGA_COMBINE_RANKS_PER_SERVER:-0}
LOCAL_DEVICE_MAPPING=${DISPATCH_MEGA_COMBINE_LOCAL_DEVICE_MAPPING:-0}
HOSTFILE=""
# 0 selects the runtime-reported core count. Nonzero values select a validated
# launch topology and must not exceed the physical device count.
AICORE_NUM=${DISPATCH_MEGA_COMBINE_AICORE_NUM:-0}
: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH must be set before running run.sh}"
CMAKE_COMPILER=${CMAKE_COMPILER:-bisheng}
MPI_RUNNER=${MPI_RUNNER:-mpirun}

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
    --urma) URMA=1; shift ;;
    --rank-num-per-server|--ranks-per-server) RANKS_PER_SERVER="${2:?ranks-per-server needs a value}"; URMA=1; shift 2 ;;
    --device-map)
      case "${2:?device-map needs a value}" in global) LOCAL_DEVICE_MAPPING=0;; server-local) LOCAL_DEVICE_MAPPING=1;;
        *) echo "--device-map must be global or server-local" >&2; exit 2;; esac; shift 2 ;;
    --hostfile) HOSTFILE="${2:?hostfile needs a path}"; LOCAL_DEVICE_MAPPING=1; URMA=1; shift 2 ;;
    --start-sync) START_SYNC=1; shift ;;
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
if [[ ! "$URMA" =~ ^[01]$ || ! "$LOCAL_DEVICE_MAPPING" =~ ^[01]$ || ! "$RANKS_PER_SERVER" =~ ^[0-9]+$ ]]; then
  echo "invalid URMA/topology option" >&2; exit 2
fi
RANKS_PER_SERVER=$((10#$RANKS_PER_SERVER))
if (( RANKS_PER_SERVER > 0 )); then URMA=1; fi
if (( LOCAL_DEVICE_MAPPING != 0 && (URMA == 0 || RANKS_PER_SERVER == 0) )); then
  echo "server-local mapping requires explicit --rank-num-per-server and URMA" >&2; exit 2
fi
if (( URMA != 0 )); then
  if (( RANKS_PER_SERVER == 0 )); then RANKS_PER_SERVER=$WORLD_SIZE; fi
  if (( RANKS_PER_SERVER > WORLD_SIZE || WORLD_SIZE % RANKS_PER_SERVER != 0 )); then
    echo "ranks-per-server must divide world-size" >&2; exit 2
  fi
fi
if [[ -n "$HOSTFILE" && ! -f "$HOSTFILE" ]]; then echo "hostfile does not exist: $HOSTFILE" >&2; exit 2; fi
export DISPATCH_MEGA_COMBINE_URMA="$URMA"
export DISPATCH_MEGA_COMBINE_RANKS_PER_SERVER="$RANKS_PER_SERVER"
export DISPATCH_MEGA_COMBINE_LOCAL_DEVICE_MAPPING="$LOCAL_DEVICE_MAPPING"
export DISPATCH_MEGA_COMBINE_WORLD_SIZE="$WORLD_SIZE"
if (( LOCAL_DEVICE_MAPPING != 0 )); then
  echo "[INFO] Each server maps rank%$RANKS_PER_SERVER to devices $FIRST_DEVICE-$((FIRST_DEVICE+RANKS_PER_SERVER-1))"
else
  echo "[INFO] Global ranks map to physical devices $FIRST_DEVICE-$((FIRST_DEVICE+WORLD_SIZE-1))"
fi

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

# Validate runtime options before data generation or MPI; CMake already requires the SharedPool SDK.
"${BUILD_DIR}/dispatch_mega_combine" --check-transport

GEN_DATA_EXTRA_ARGS=()
if [[ "${REUSE_DATA}" != "0" ]]; then
  GEN_DATA_EXTRA_ARGS+=(--reuse-data)
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

export DISPATCH_MEGA_COMBINE_CASE_DIR="${OUT_DIR}"
export DISPATCH_MEGA_COMBINE_AICORE_NUM="${AICORE_NUM}"
export DISPATCH_MEGA_COMBINE_START_SYNC="${START_SYNC}"
export DISPATCH_MEGA_COMBINE_WARMUP_ITERS="${WARMUP_ITERS}"
export DISPATCH_MEGA_COMBINE_MEASURE_ITERS="${MEASURE_ITERS}"
if ! command -v "${MPI_RUNNER}" >/dev/null 2>&1; then
  echo "MPICH launcher not found: ${MPI_RUNNER}. Add MPICH to PATH or set MPI_RUNNER." >&2
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
MPI_HOST_ARGS=()
if [[ -n "$HOSTFILE" ]]; then MPI_HOST_ARGS=(-f "$HOSTFILE" -ppn "$RANKS_PER_SERVER"); fi
"${MPI_RUNNER}" "${MPI_HOST_ARGS[@]}" -n "${WORLD_SIZE}" "${BUILD_DIR}/dispatch_mega_combine" --first-device "${FIRST_DEVICE}"
