#!/usr/bin/env bash
# Build the default PTO-DSL GEMM shared library.
#
# Usage:
#   bash compile.sh
#   PTO_LIB_PATH=/abs/pto-isa bash compile.sh
#
# This script emits the build() default artifacts under build_artifacts/. Case
# builds used by run.py are emitted separately under case_builds/<case>/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACT_DIR="${SCRIPT_DIR}/build_artifacts"
PTO_LIB_PATH="${PTO_LIB_PATH:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
PTOAS="${PTOAS:-ptoas}"
BISHENG="${BISHENG:-bisheng}"
NPU_ARCH="${NPU_ARCH:-dav-2201}"

mkdir -p "${ARTIFACT_DIR}"

PTO_PATH="${ARTIFACT_DIR}/gemm.pto"
GENERATED_CPP="${ARTIFACT_DIR}/gemm.cpp"
LIB_PATH="${ARTIFACT_DIR}/gemm.so"

if ! command -v "${BISHENG}" >/dev/null 2>&1; then
    # Try common local Ascend installations before reporting a missing compiler.
    if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
        # shellcheck disable=SC1091
        source /usr/local/Ascend/ascend-toolkit/set_env.sh || true
    elif [[ -f /usr/local/Ascend/cann-8.5.0/set_env.sh ]]; then
        # shellcheck disable=SC1091
        source /usr/local/Ascend/cann-8.5.0/set_env.sh || true
    fi
fi

if [[ -n "${PTOAS_ROOT:-}" ]]; then
    # PTOAS_ROOT can point either to a bin-containing install root or directly
    # to a directory that contains ptoas.
    export PATH="${PTOAS_ROOT}/bin:${PTOAS_ROOT}:${PATH}"
    export LD_LIBRARY_PATH="${PTOAS_ROOT}/lib:${LD_LIBRARY_PATH:-}"
fi

if ! command -v "${BISHENG}" >/dev/null 2>&1; then
    echo "[ERROR] Missing executable: ${BISHENG}"
    echo "Source Ascend CANN environment or add bisheng to PATH before running compile.sh."
    exit 1
fi

if ! command -v "${PTOAS}" >/dev/null 2>&1; then
    echo "[ERROR] Missing executable: ${PTOAS}"
    echo "Ensure ptoas is on PATH before running compile.sh."
    exit 1
fi

if ! python3 -c "import mlir.ir; from mlir.dialects import pto" >/dev/null 2>&1; then
    echo "[ERROR] Missing Python MLIR/PTO modules."
    echo "Ensure the MLIR Python environment is available before running compile.sh."
    exit 1
fi

echo "==> Building GEMM -> ${LIB_PATH}"
rm -f "${PTO_PATH}" "${GENERATED_CPP}" "${LIB_PATH}"

python3 "${SCRIPT_DIR}/kernels/gemm_builder.py" > "${PTO_PATH}"
"${PTOAS}" --enable-insert-sync "${PTO_PATH}" -o "${GENERATED_CPP}"

"${BISHENG}" -fPIC -shared -xcce -O2 -std=c++17 \
    --npu-arch="${NPU_ARCH}" \
    -I"${PTO_LIB_PATH}/include" \
    -DKERNEL_CPP="\"${GENERATED_CPP}\"" \
    "${SCRIPT_DIR}/caller.cpp" \
    -o "${LIB_PATH}"

echo "Done. Run: python3 run.py"
