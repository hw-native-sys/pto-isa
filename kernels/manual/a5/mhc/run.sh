#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

BUILD_DIR=build
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. -DRUN_MODE=npu
make -j$(nproc)

echo ""
echo "=== Running expand_to_mhc_fwd verification ==="
./mhc_test
