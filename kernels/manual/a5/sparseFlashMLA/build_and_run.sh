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
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir="${script_dir}/build"
tile_n=128
fixture_case=""
fixture_options=()
compile_options=()
run_options=()
while (($#)); do
    case "$1" in
        --run-mode|-r|--soc-version|-v|--build-dir|--tile-n|--jobs|-j|--case|--run-dir|--core-limit|--sim-cores)
            if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
            case "$1" in
                --run-mode|-r|--soc-version|-v)
                    compile_options+=("$1" "$2")
                    run_options+=("$1" "$2") ;;
                --build-dir) build_dir=$2 ;;
                --tile-n) tile_n=$2 ;;
                --case) fixture_case=$2 ;;
                --jobs|-j) compile_options+=("$1" "$2") ;;
                --run-dir|--core-limit|--sim-cores) run_options+=("$1" "$2") ;;
            esac
            shift 2 ;;
        --batch-size|--batch|--query-length|--sq|--num-heads|--heads|--top-k|--topk|--kv-length|--kv-pool|--page-size|--seed|--softmax-scale|--rescale-rate)
            if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
            fixture_options+=("$1" "$2")
            shift 2 ;;
        --validate-only|--keep-memory-logs) run_options+=("$1"); shift ;;
        --help|-h)
            echo "Usage: $0 [--run-mode sim|npu] [--soc-version SOC] [--tile-n 64|128]"
            echo "  [--build-dir DIR] [--jobs N] [--case NAME] [--core-limit N] [--sim-cores N]"
            echo "  [--run-dir DIR] [--keep-memory-logs] [--validate-only]"
            echo "  [--batch-size N] [--query-length N] [--num-heads 64|128]"
            echo "  [--top-k N] [--kv-length N] [--page-size N] [--seed N] [--softmax-scale F]"
            echo "  [--kv-pool per-batch|shared]"
            echo "  [--rescale-rate PERCENT]"
            echo "Compile, generate one golden fixture, then launch with matching build/runtime settings."
            echo "Defaults: sim, Ascend950PR_9599, tile128, topk1024_jobs2, one simulator AIC."
            echo "Default build directory: ${script_dir}/build"
            echo "Golden data: BUILD_DIR/golden/CASE. Run logs: fresh BUILD_DIR/runs/run.*."
            echo "--case accepts custom or the names listed by golden/gen_validation.py --help."
            echo "Shape/data options select custom automatically; they cannot modify named validation fixtures."
            echo "Custom defaults are listed by golden/gen_data.py --help."
            echo "Each batch owns a KV pool by default; query positions share their batch's pool."
            echo "Custom inputs are random unless --rescale-rate requests a controlled fixture. Both report the reference rate."
            echo "--validate-only checks the generated fixture without launching the kernel."
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

if ((${#fixture_options[@]})); then
    if [[ -n "$fixture_case" && "$fixture_case" != custom ]]; then
        echo "Shape/data options require --case custom or no --case; named validation fixtures have fixed inputs." >&2
        exit 2
    fi
    fixture_case=custom
fi
fixture_case=${fixture_case:-topk1024_jobs2}
golden_dir="${build_dir}/golden/${fixture_case}"

echo "[1/3] Compiling the kernel"
bash "${script_dir}/compile.sh" --build-dir "$build_dir" --tile-n "$tile_n" "${compile_options[@]}"

echo "[2/3] Generating golden data: $fixture_case"
if [[ "$fixture_case" == custom ]]; then
    python3 "${script_dir}/golden/gen_data.py" \
        --output "$golden_dir" --tile-n "$tile_n" "${fixture_options[@]}"
else
    python3 "${script_dir}/golden/gen_validation.py" \
        --output "${build_dir}/golden" --case "$fixture_case" --tile-n "$tile_n"
fi

echo "[3/3] Running the kernel runner"
exec bash "${script_dir}/run.sh" --build-dir "$build_dir" \
    --golden-dir "$golden_dir" "${run_options[@]}"
