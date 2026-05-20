#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0
#
# Build the pto-dsl flash-attention runtime-S1 .so. The generated kernel loops
# over s1 / S1_TILE at runtime, so one fa.{mlir,cpp,so} covers all supported
# benchmark lengths.
#
# Usage:
#   bash compile.sh
#   bash compile.sh --remove-vec-barriers line1,line2,...      # remove selected PIPE_V barriers
#   bash compile.sh --remove-vec-barrier-patterns gu,softmax-sum-add
#   FA_REMOVE_VEC_BARRIERS=line1,line2,... bash compile.sh     # same via env
#   FA_REMOVE_VEC_BARRIER_PATTERNS=gu,softmax-sum-add bash compile.sh
#   PTO_LIB_PATH=/abs/pto-isa bash compile.sh                  # override include path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="${SCRIPT_DIR}/build_artifacts"
PTO_LIB_PATH="${PTO_LIB_PATH:-/sources/pto-isa}"
PTOAS="${PTOAS:-ptoas}"
BISHENG="${BISHENG:-bisheng}"
REMOVE_VEC_BARRIER_LINES="${FA_REMOVE_VEC_BARRIERS:-}"
REMOVE_VEC_BARRIER_PATTERNS="${FA_REMOVE_VEC_BARRIER_PATTERNS:-gu}"

usage() {
    cat >&2 <<EOF
Usage: $0 [--remove-vec-barriers line1,line2,...] [--remove-vec-barrier-patterns gu,softmax-exp-sum,softmax-sum-add]

Environment:
  FA_REMOVE_VEC_BARRIERS          Comma-separated generated-C++ line numbers to patch.
  FA_REMOVE_VEC_BARRIER_PATTERNS  Comma-separated pattern names. Defaults to stable pattern: gu.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --remove-vec-barriers)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "--remove-vec-barriers requires a comma-separated line list" >&2
                usage
                exit 2
            fi
            REMOVE_VEC_BARRIER_LINES="$2"
            shift 2
            ;;
        --remove-vec-barrier-patterns)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "--remove-vec-barrier-patterns requires a comma-separated pattern list" >&2
                usage
                exit 2
            fi
            REMOVE_VEC_BARRIER_PATTERNS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

mkdir -p "${ARTIFACT_DIR}"

MLIR_PATH="${ARTIFACT_DIR}/fa.mlir"
GENERATED_CPP="${ARTIFACT_DIR}/fa.cpp"
PATCHED_CPP="${ARTIFACT_DIR}/fa_patched.cpp"
LIB_PATH="${ARTIFACT_DIR}/fa.so"

echo "==> Building runtime-S1 fa -> ${LIB_PATH}"
rm -f "${MLIR_PATH}" "${GENERATED_CPP}" "${PATCHED_CPP}" "${LIB_PATH}"

python3 "${SCRIPT_DIR}/kernels/fa_builder.py" > "${MLIR_PATH}"
"${PTOAS}" --pto-arch=a3 --enable-insert-sync "${MLIR_PATH}" > "${GENERATED_CPP}"

COMPILE_CPP="${GENERATED_CPP}"
if [[ -n "${REMOVE_VEC_BARRIER_LINES}${REMOVE_VEC_BARRIER_PATTERNS}" ]]; then
    python3 "${SCRIPT_DIR}/scripts/patch_vec_barriers.py" \
        "${GENERATED_CPP}" \
        "${PATCHED_CPP}" \
        --remove-vec-barriers "${REMOVE_VEC_BARRIER_LINES}" \
        --remove-vec-barrier-patterns "${REMOVE_VEC_BARRIER_PATTERNS}"
    COMPILE_CPP="${PATCHED_CPP}"
fi

"${BISHENG}" \
    -I"${PTO_LIB_PATH}/include" \
    -fPIC -shared -D_FORTIFY_SOURCE=2 -O2 \
    -Wno-macro-redefined -Wno-ignored-attributes -fstack-protector-strong \
    -xcce -Xhost-start -Xhost-end \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=true \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -cce-enable-mix \
    --npu-arch=dav-2201 -DMEMORY_BASE \
    -std=gnu++17 \
    -DKERNEL_CPP="\"${COMPILE_CPP}\"" \
    "${SCRIPT_DIR}/caller.cpp" \
    -o "${LIB_PATH}"

echo "Generated ${GENERATED_CPP}"
if [[ "${COMPILE_CPP}" != "${GENERATED_CPP}" ]]; then
    echo "Patched ${COMPILE_CPP}"
fi
echo "Done. Built ${LIB_PATH}"
