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

set -eo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
run_mode=sim
soc_version=Ascend950PR_9599
build_dir="${script_dir}/build"
jobs=8
tile_n=128
while (($#)); do
    case "$1" in
        --run-mode|-r|--soc-version|-v|--build-dir|--jobs|-j|--tile-n)
            if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
            case "$1" in
                --run-mode|-r) run_mode=$2 ;;
                --soc-version|-v) soc_version=$2 ;;
                --build-dir) build_dir=$2 ;;
                --jobs|-j) jobs=$2 ;;
                --tile-n) tile_n=$2 ;;
            esac
            shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--build-dir DIR] [--run-mode sim|npu] [--soc-version SOC] [--jobs N]"
            echo "  [--tile-n 64|128]"
            echo "Default build directory: ${script_dir}/build"
            echo "Default: tile128. Fixed: BF16 Q/K/V/P, FP4 cache, paired gather32, QK/PV unit flags."
            echo "Tile layouts are checked against device memory capacity at compile time."
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$run_mode" == sim || "$run_mode" == npu ]] || { echo "Invalid runtime: $run_mode" >&2; exit 2; }
[[ "$soc_version" =~ ^Ascend950 ]] || { echo "Requires A5 / Ascend950" >&2; exit 2; }
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs must be positive" >&2; exit 2; }
[[ "$tile_n" == 64 || "$tile_n" == 128 ]] || { echo "--tile-n must be 64 or 128" >&2; exit 2; }
source "${script_dir}/tools/setup_cann.sh"
set -u
cmake -S "$script_dir" -B "$build_dir" \
    -DRUN_MODE="$run_mode" -DSOC_VERSION="$soc_version" \
    -DFLASH_MLA_TILE_N="$tile_n"
cmake --build "$build_dir" --parallel "$jobs"
