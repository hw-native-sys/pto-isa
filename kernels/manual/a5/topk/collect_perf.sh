# --------------------------------------------------------------------------------
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

# Run each TopK testcase (ITER_SELECT vs RADIX_SELECT) in the simulator and collect
# camodel performance logs. Same methodology as topk_simdvf_radix/collect_perf.sh.
# Usage: bash collect_perf.sh [--quick]   (--quick skips iter_1x16384_k2048, ~2.75h)
set -u
cd "$(dirname "$0")"

QUICK=0
if [[ "${1:-}" == "--quick" ]]; then
    QUICK=1
fi

# name|gtest_filter|timeout_seconds
CASES=(
    "iter_1x128_k3|*iter_1x128_k3*|1200"
    "iter_1x256_k8|*iter_1x256_k8*|1200"
    "iter_1x256_k12|*iter_1x256_k12*|1200"
    "iter_1x288_k8|*iter_1x288_k8*|1200"
    "iter_1x384_k6|*iter_1x384_k6*|1200"
    "iter_1x512_k12|*iter_1x512_k12*|1200"
    "iter_1x896_k16|*iter_1x896_k16*|1200"
    "radix_1x4096_k512|*radix_1x4096_k512*|1200"
    "radix_1x16384_k256|*radix_1x16384_k256*|3600"
    "radix_1x16384_k2048|*radix_1x16384_k2048*|3600"
)

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[ERROR] ASCEND_HOME_PATH is unset. source ~/source_cann.sh first."
    exit 1
fi
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib:${LD_LIBRARY_PATH:-}

BIN="$(pwd)/build/topk"
if [[ ! -x "${BIN}" ]]; then
    echo "[INFO] ${BIN} not found; building via ./run.sh -c iter_1x128_k3 ..."
    if ! ./run.sh -c iter_1x128_k3; then
        echo "[ERROR] build/run failed; cannot collect perf."
        exit 1
    fi
fi
if [[ ! -x "${BIN}" ]]; then
    echo "[ERROR] missing binary after build: ${BIN}"
    echo "        run: ./run.sh -c iter_1x128_k3"
    exit 1
fi

mkdir -p perf
FAIL=0

for ENTRY in "${CASES[@]}"; do
    IFS='|' read -r CASE FILTER TMO <<< "$ENTRY"
    if [[ $QUICK == 1 && "$CASE" == "iter_1x16384_k2048" ]]; then
        echo "===== SKIP $CASE (quick mode) ====="
        continue
    fi
    echo "===== RUN $CASE ====="
    DIR="perf/${CASE}"
    rm -rf "$DIR"
    mkdir -p "$DIR"
    (
        cd "$DIR"
        set +e
        OUT=$(timeout "$TMO" "$BIN" --npu=0 --gtest_filter="$FILTER" 2>&1)
        RC=$?
        set -e
        echo "$OUT" | grep -E "Model RUN TIME|\[       OK \]|\[  FAILED" | sed 's/^/  /' || true
        if [[ $RC -ne 0 ]] || ! echo "$OUT" | grep -q '\[       OK \]'; then
            echo "  [ERROR] $CASE failed (rc=$RC)"
            echo "$OUT" | tail -40 | sed 's/^/  /'
            exit 1
        fi
        cp core*_summary_log . 2>/dev/null || true
        cp *su_perf_summary_log . 2>/dev/null || true
        echo "$OUT" | grep -E "Model RUN TIME" > model_runtime.txt || true
        {
            for f in core*.ccu.mte1_issque.dump core*.ccu.mte2_issque.dump core*.ccu.mte3_issque.dump; do
                [ -f "$f" ] || continue
                echo -n "$f mte_push=$(grep -c 'Push Instr' "$f") "
            done
            echo ""
            for f in core*.rvec.ISU.dump; do
                [ -f "$f" ] || continue
                echo "$f rvec_isu_recv=$(grep -c 'ISU_RECV' "$f")"
            done
        } > instr_counts.txt
        rm -f core*.dump core*_wave.vcd profile_*.toml mcu_log.dump stars_log0.dump ffts_verify_log*.log
    )
    if [[ $? -ne 0 ]]; then
        FAIL=1
    fi
done

if [[ $FAIL -ne 0 ]]; then
    echo "===== DONE (with failures) ====="
    exit 1
fi
echo "===== DONE ====="
