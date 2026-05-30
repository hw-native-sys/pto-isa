#!/bin/bash
# ============================================================================
# MoE Dispatch PTO-ISA Operator — Build & Run Script
#
# Usage:
#   ./run.sh [build|run|all] [OPTIONS]
#
# Options:
#   --ep N              Number of ranks (default: 2)
#   --experts N         Experts per rank (default: 1)
#   --hidden N          Hidden size (default: 128)
#   --tokens N          Max tokens per rank (default: 64)
#   --max-output N      Max output size (default: 512)
#   --first-device N     First device ID (default: 0)
#   --debug             Enable debug logging
#   --clean             Clean build directory first
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
PTO_ROOT="${SCRIPT_DIR}/../../../.."

# Default parameters
EP=2
EXPERT_PER_RANK=1
HIDDEN_SIZE=128
MAX_TOKENS=64
MAX_OUTPUT=512
FIRST_DEVICE=0
DEBUG_MODE=""
CLEAN=""
ACTION="all"
DISPATCH_MODE="direct"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        build|run|all) ACTION=$1; shift ;;
        --ep) EP=$2; shift 2 ;;
        --experts) EXPERT_PER_RANK=$2; shift 2 ;;
        --hidden) HIDDEN_SIZE=$2; shift 2 ;;
        --tokens) MAX_TOKENS=$2; shift 2 ;;
        --max-output) MAX_OUTPUT=$2; shift 2 ;;
        --first-device) FIRST_DEVICE=$2; shift 2 ;;
        --mode) DISPATCH_MODE=$2; shift 2 ;;
        --debug) DEBUG_MODE="-DDEBUG_MODE=ON"; shift ;;
        --clean) CLEAN=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== MoE Dispatch PTO-ISA Operator ==="
echo "EP=${EP}, expertPerRank=${EXPERT_PER_RANK}, hiddenSize=${HIDDEN_SIZE}"
echo "maxTokens=${MAX_TOKENS}, maxOutput=${MAX_OUTPUT}, firstDevice=${FIRST_DEVICE}, mode=${DISPATCH_MODE}"
echo ""

# Check environment
if [[ -z "${ASCEND_HOME_PATH}" ]]; then
    echo "[ERROR] ASCEND_HOME_PATH not set. Please run: source set_env_new.sh"
    exit 1
fi

do_build() {
    echo "[BUILD] Building moe_dispatch..."

    if [[ -n "${CLEAN}" ]]; then
        rm -rf "${BUILD_DIR}"
    fi

    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    cmake "${SCRIPT_DIR}" \
        -DRUN_MODE=npu \
        -DCONFIG_EP=${EP} \
        -DCONFIG_EXPERT_PER_RANK=${EXPERT_PER_RANK} \
        -DCONFIG_HIDDEN_SIZE=${HIDDEN_SIZE} \
        -DCONFIG_MAX_TOKENS_PER_RANK=${MAX_TOKENS} \
        -DCONFIG_MAX_OUTPUT_SIZE=${MAX_OUTPUT} \
        -DCONFIG_FIRST_DEVICE_ID=${FIRST_DEVICE} \
        ${DEBUG_MODE}

    make -j$(nproc)

    echo "[BUILD] Done. Binary: ${BUILD_DIR}/moe_dispatch"
}

do_run() {
    echo "[RUN] Launching with mpirun -n ${EP} (mode=${DISPATCH_MODE})..."

    if [[ ! -f "${BUILD_DIR}/moe_dispatch" ]]; then
        echo "[ERROR] Binary not found. Run with 'build' first."
        exit 1
    fi

    cd "${BUILD_DIR}"
    DISPATCH_MODE=${DISPATCH_MODE} mpirun -n ${EP} ./moe_dispatch

    echo "[RUN] Done."
}

case ${ACTION} in
    build) do_build ;;
    run) do_run ;;
    all) do_build && do_run ;;
esac
