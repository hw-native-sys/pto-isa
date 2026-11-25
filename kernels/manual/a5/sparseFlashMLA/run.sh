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
golden_dir="${script_dir}/build/golden/topk1024_jobs2"
run_dir=""
core_limit=0
sim_cores=1
validate_only=0
keep_memory_logs=0
while (($#)); do
    case "$1" in
        --run-mode|-r|--soc-version|-v|--build-dir|--golden-dir|--run-dir|--core-limit|--sim-cores)
            if (($# < 2)); then echo "Missing value for $1" >&2; exit 2; fi
            case "$1" in
                --run-mode|-r) run_mode=$2 ;;
                --soc-version|-v) soc_version=$2 ;;
                --build-dir) build_dir=$2 ;;
                --golden-dir) golden_dir=$2 ;;
                --run-dir) run_dir=$2 ;;
                --core-limit) core_limit=$2 ;;
                --sim-cores) sim_cores=$2 ;;
            esac
            shift 2 ;;
        --validate-only) validate_only=1; shift ;;
        --keep-memory-logs) keep_memory_logs=1; shift ;;
        --help|-h)
            echo "Usage: $0 [--golden-dir DIR] [--core-limit N] [--build-dir DIR] [--run-dir DIR]"
            echo "  [--run-mode sim|npu] [--soc-version SOC] [--sim-cores N] [--validate-only] [--keep-memory-logs]"
            echo "Default build directory: ${script_dir}/build"
            echo "Default golden directory: ${script_dir}/build/golden/topk1024_jobs2"
            echo "Simulator default: one physical AIC and two AIVs in a private cached toolkit copy."
            echo "FLASH_MLA_SIM_LIB overrides simulator libraries/configuration."
            echo "Successful simulator runs remove memory dumps; instruction/timing/error logs remain."
            echo "Failed runs preserve all logs. Default run directory: fresh BUILD_DIR/runs/run.*."
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$run_mode" == sim || "$run_mode" == npu ]] || { echo "Invalid runtime: $run_mode" >&2; exit 2; }
[[ -d "$golden_dir" ]] || {
    echo "Golden directory not found: $golden_dir" >&2
    echo "Generate it with golden/gen_validation.py --case topk1024_jobs2, or pass --golden-dir DIR." >&2
    exit 2
}
[[ "$core_limit" =~ ^[0-9]+$ ]] || { echo "--core-limit must be nonnegative" >&2; exit 2; }
[[ "$sim_cores" =~ ^[1-9][0-9]*$ ]] || { echo "--sim-cores must be positive" >&2; exit 2; }
golden_dir=$(cd -- "$golden_dir" && pwd)
build_dir=$(cd -- "$build_dir" && pwd)
[[ -x "${build_dir}/flash_mla_bf16_test" ]] || { echo "Build first with compile.sh" >&2; exit 2; }
[[ -f "${build_dir}/flash_mla_run_mode.txt" && -f "${build_dir}/flash_mla_soc_version.txt" ]] || {
    echo "Build runtime metadata missing; rebuild with compile.sh" >&2; exit 2;
}
[[ "$(< "${build_dir}/flash_mla_run_mode.txt")" == "$run_mode" && "$(< "${build_dir}/flash_mla_soc_version.txt")" == "$soc_version" ]] || {
    echo "Build runtime/platform differs; rebuild with matching compile.sh options" >&2; exit 2;
}
source "${script_dir}/tools/setup_cann.sh"
set -u
export LD_LIBRARY_PATH="${build_dir}/lib:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
if [[ "$run_mode" == sim ]]; then
    if [[ -z "${FLASH_MLA_SIM_LIB:-}" ]]; then
        FLASH_MLA_SIM_LIB=$(python3 "${script_dir}/tools/prepare_simulator.py" \
            --toolkit "$ASCEND_HOME_PATH" --soc-version "$soc_version" --cores "$sim_cores" \
            --cache-dir "${build_dir}/simulator")
        export FLASH_MLA_SIM_LIB
    else
        sim_cores=$(python3 - "$FLASH_MLA_SIM_LIB" <<'PY'
import json
from pathlib import Path
import sys
print(int(json.loads((Path(sys.argv[1]) / "config_stars.json").read_text())["model_top"]["num_aic"]))
PY
        )
    fi
    if ((core_limit == 0)); then core_limit=$sim_cores; fi
    ((core_limit <= sim_cores)) || { echo "--core-limit exceeds the simulator's $sim_cores physical AICs" >&2; exit 2; }
    export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/tools/simulator/${soc_version}/lib:${ASCEND_HOME_PATH}/simulator/${soc_version}/lib:${LD_LIBRARY_PATH}"
    if [[ -n "${FLASH_MLA_SIM_LIB:-}" ]]; then
        [[ -d "$FLASH_MLA_SIM_LIB" ]] || { echo "FLASH_MLA_SIM_LIB is not a directory" >&2; exit 2; }
        export LD_LIBRARY_PATH="${FLASH_MLA_SIM_LIB}:${LD_LIBRARY_PATH}"
    fi
fi
if [[ -z "$run_dir" ]]; then
    mkdir -p -- "${build_dir}/runs"
    run_dir=$(mktemp -d "${build_dir}/runs/run.XXXXXX")
else
    mkdir -p -- "$run_dir"
    if [[ -n "$(ls -A -- "$run_dir")" ]]; then
        echo "--run-dir must be empty to keep each run's logs separate" >&2
        exit 2
    fi
fi
run_dir=$(cd -- "$run_dir" && pwd)
arguments=(--golden-dir "$golden_dir" --core-limit "$core_limit")
if ((validate_only)); then arguments+=(--validate-only); fi
echo "Run directory: $run_dir"
cd -- "$run_dir"
sim_lib=${FLASH_MLA_SIM_LIB:-${ASCEND_HOME_PATH}/tools/simulator/${soc_version}/lib}
python3 "${script_dir}/tools/capture_run.py" \
    --source-dir "$script_dir" --build-dir "$build_dir" --golden-dir "$golden_dir" \
    --sim-lib "$sim_lib" --run-mode "$run_mode" --soc-version "$soc_version" --output manifest.json \
    -- "${build_dir}/flash_mla_bf16_test" "${arguments[@]}"
status=0
"${build_dir}/flash_mla_bf16_test" "${arguments[@]}" > >(tee stdout.log) 2> >(tee stderr.log >&2) || status=$?
echo "$status" > exit_status.txt
if [[ "$status" == 0 && "$run_mode" == sim && "$validate_only" == 0 && "$keep_memory_logs" == 0 ]]; then
    python3 "${script_dir}/tools/cleanup_sim_logs.py" --root "$run_dir" --apply --summary cleanup_summary.json
fi
exit "$status"
