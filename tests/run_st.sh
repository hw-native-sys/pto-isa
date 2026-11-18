#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

set -e
TOTAL_CASES=0
FAILED_CASES=0
# Run ST cases with failure-tolerant counting: a single failing case must not
# abort the whole batch (set -e would do that); a summary is printed at the end.
python3() {
  case "$1" in
    *build_st.py*)
      command python3 "$@"
      ;;
    *)
      # Record the result so the parent can aggregate it later.
      #
      # Background calls (`&`) run in a subshell, so a counter increment here
      # would be lost; those results go to a file that the parent reads after
      # each `wait`. Foreground calls (the A3 / Kirin / comm paths) run in the
      # same shell, so they can increment directly — and must, because a file
      # keyed on BASHPID would be overwritten by the next serial call.
      if command python3 "$@"; then
        _status=PASS
      else
        _status=FAIL
        echo ">>> [FAIL] $*" >&2
      fi
      if [ "${BASHPID}" = "$$" ]; then
        TOTAL_CASES=$((TOTAL_CASES + 1))
        if [ "$_status" = "FAIL" ]; then
          FAILED_CASES=$((FAILED_CASES + 1))
        fi
      else
        _init_results_dir
        echo "$_status" > "${_RESULTS_DIR}/${BASHPID}.result"
      fi
      ;;
  esac
}


# Result files for background tasks: since `&` spawns a subshell, variable
# updates in python3() don't propagate to the parent. Instead each task writes
# its result to a unique file in _RESULTS_DIR; after every wait, the parent
# reads these files to update TOTAL_CASES/FAILED_CASES.
_RESULTS_DIR=""

_init_results_dir() {
  if [ -z "$_RESULTS_DIR" ]; then
    _RESULTS_DIR=$(mktemp -d -t pto_results_XXXXXX)
  fi
}
# Cleanup is called explicitly at script end (after SUMMARY), NOT via trap,
# because background subshells inherit the EXIT trap and would delete the
# results dir when the first one finishes.
_cleanup_results_dir() {
  # Must return 0 even when no result dir was ever created (the A3 / Kirin /
  # comm paths never create one). This is the last command of the script, so a
  # non-zero return here would fail the run under `set -e` after a green pass.
  if [ -n "$_RESULTS_DIR" ] && [ -d "$_RESULTS_DIR" ]; then
    rm -rf "$_RESULTS_DIR"
  fi
  return 0
}
_collect_results() {
  if [ -z "$_RESULTS_DIR" ] || [ ! -d "$_RESULTS_DIR" ]; then
    return
  fi
  for _rf in "$_RESULTS_DIR"/*.result; do
    [ -f "$_rf" ] || continue
    _status=$(cat "$_rf" 2>/dev/null)
    TOTAL_CASES=$((TOTAL_CASES + 1))
    if [ "$_status" = "FAIL" ]; then
      FAILED_CASES=$((FAILED_CASES + 1))
    fi
    rm -f "$_rf"
  done
}

# Case-count guard. Counting processes alone cannot detect a mistyped
# --gtest_filter: gtest exits 0 when a filter matches nothing, so the process
# would be counted as PASS while its cases silently never ran. run_st.py
# appends "<ran> <expected> <testcase>" per invocation to $PTO_CASE_COUNT_FILE;
# _check_case_counts sums them and fails when the totals disagree.
CASE_COUNT_FILE=""
_check_case_counts() {
  [ -n "$CASE_COUNT_FILE" ] && [ -f "$CASE_COUNT_FILE" ] || return 0
  _ran_total=0
  _exp_total=0
  _unknown=0
  _bad=""
  while read -r _ran _exp _tc; do
    [ -n "$_ran" ] || continue
    _ran_total=$((_ran_total + _ran))
    if [ "$_exp" -lt 0 ]; then
      _unknown=$((_unknown + 1))
    else
      _exp_total=$((_exp_total + _exp))
      if [ "$_ran" -ne "$_exp" ]; then
        _bad="${_bad} ${_tc}(${_ran}/${_exp})"
      fi
    fi
  done < "$CASE_COUNT_FILE"
  # unknown_filters = invocations whose expected count could not be derived
  # (an unfiltered run whose --gtest_list_tests failed, or a filter using
  # wildcards / negative patterns). Their cases are in _ran_total but not in
  # _exp_total, so the totals are not comparable. Fail closed instead of
  # skipping the comparison: a real A5 run resolves every invocation, so a
  # non-zero count means the guard could not verify part of the run.
  echo "[CASES] ran=${_ran_total} expected_from_filters=${_exp_total} unknown_filters=${_unknown}"
  if [ -n "$_bad" ]; then
    echo "[CASES] case count mismatch:${_bad}" >&2
    return 1
  fi
  if [ "$_unknown" -ne 0 ]; then
    echo "[CASES] ${_unknown} invocation(s) without a verifiable expected count" >&2
    return 1
  fi
  return 0
}
ENABLE_A3=false
ENABLE_A5=false
ENABLE_KIRIN9030=false
ENABLE_KIRINX90=false
ENABLE_KIRIN9050=false
ENABLE_SIMPLE=false
ENABLE_ALL=false
ENABLE_COMM=false
ARGS=" "
IS_AUTO_MODE=false

usage() {
  cat >&2 <<'EOF'
Usage: ./tests/run_st.sh <platform> <mode> [run-mode] [options]

Platform (choose at least one):
  --a3          A2/A3 (Ascend910B/910C)
  --a5          A5 (Ascend950)
  --a3_a5       both A2/A3 and A5
  --kirin9030   Kirin9030
  --kirinX90    KirinX90
  --kirin9050 Kirin9050

Mode (required for --a3/--a5/--a3_a5, choose one):
  --simple      run the curated single-case subset
  --all         run the full ST suite

Run mode (choose one; defaults to on-board npu if omitted):
  --sim         run against the simulator
  --npu         run on NPU hardware

Options:
  --comm        include communication ST
  --auto_mode   run ST in auto mode
  PTO_SLOTS=N   A5 --simple: number of NPU slots (cards); default: all visible NPUs.
                A value above the visible NPU count fails fast instead of downscaling.

Examples:
  ./tests/run_st.sh --a5 --npu --simple
  ulimit -n 65536 && ./tests/run_st.sh --a3 --sim --all
EOF
}

checkopts() {
  while true; do
    case "$1" in
      --a3)
        ENABLE_A3=true
        shift
        ;;
      --a5)
        ENABLE_A5=true
        shift
        ;;
      --a3_a5)
        ENABLE_A3=true
        ENABLE_A5=true
        shift
        ;;
      --kirin9030)
        ENABLE_KIRIN9030=true
        shift
        ;;
      --kirinX90)
        ENABLE_KIRINX90=true
        shift
        ;;
      --kirin9050)
        ENABLE_KIRIN9050=true
        shift
        ;;
      --sim)
        ARGS+=" -r sim "
        shift
        ;;
      --npu)
        ARGS+="-r npu "
        shift
        ;;
      --comm)
        ENABLE_COMM=true
        shift
        ;;
      --simple)
        ENABLE_SIMPLE=true
        shift
        ;;
      --all)
        ENABLE_ALL=true
        shift
        ;;
      --auto_mode)
        ARGS+="-a "
        IS_AUTO_MODE=true
        shift
        ;;
      --)
        shift
        break
        ;;
      "")
        break
        ;;
      *)
        echo "Error: unknown argument '$1'" >&2
        usage
        exit 1
        ;;
    esac
  done
}

checkopts "$@"

# Run mode defaults to on-board NPU when omitted (see usage above).
if [[ "$ARGS" != *" -r "* ]]; then
  ARGS+=" -r npu "
fi

# Validate the parsed options so that a mis-typed invocation fails loudly
# instead of silently exiting 0 without running any test.
if [ "$ENABLE_A3" = "false" ] && [ "$ENABLE_A5" = "false" ] && \
   [ "$ENABLE_KIRIN9030" = "false" ] && [ "$ENABLE_KIRINX90" = "false" ] && \
   [ "$ENABLE_KIRIN9050" = "false" ]; then
  echo "Error: no platform selected (expected one of --a3/--a5/--a3_a5/--kirin9030/--kirinX90/--kirin9050)." >&2
  usage
  exit 1
fi

if { [ "$ENABLE_A3" = "true" ] || [ "$ENABLE_A5" = "true" ]; } && \
   [ "$ENABLE_SIMPLE" = "false" ] && [ "$ENABLE_ALL" = "false" ] && [ "$ENABLE_COMM" = "false" ]; then
  echo "Error: --a3/--a5 requires a test mode: pass --simple or --all." >&2
  usage
  exit 1
fi

if [ "$ENABLE_A3" = "true" ]; then                 # A2A3
  if [ "$ENABLE_SIMPLE" = "true" ]; then           # 单个用例
    python3 tests/script/build_st.py $ARGS -v a3 -t all
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolgather -g TCOLGATHERTest.case_mask_half_16x64_16x64_P1111
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tconcatdstidx -g TCONCATTest.case_int16_16x32_16x16_16x16_8x16_8x16
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tconcatidx -g TCONCATTest.case_int16_16x32_16x16_16x16_8x16_8x16
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t taxpy -g TAXPYTest.case1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolexpand -g TCOLEXPANDTest.case1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolsum -g TCOLSUMTest.case1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolprod -g TCOLPRODTest.case1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolmax -g TCOLMAXTest.case1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolargmax -g TCOLCMAXTest.case01
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolmin -g TCOLMINTest.case1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolargmin -g TCOLCMINTest.case01
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trem -g TREMTest.case_float_16x64_16x128_16x128_16x64
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tfmod -g TFMODTest.case_float_16x64_16x128_16x128_16x64
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trems -g TREMSTest.case1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tfmods -g TFMODSTest.case1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsubs -g TSUBSTest.case1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmaxs -g TMAXSTest.case1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tlrelu -g TLRELUTest.case1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tgatherb -g TGATHERBTest.case_float_2x128_2x16_2x128
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmov -g TMOVTest.case14_scaling_dynamic_int32_int8_0_1_1_1_0_param
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmov_acc2mat -g TMOVTest.case_nz2nz_fb_quant_4
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmrgsort -g TMRGSORTTest.case_topk1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmul -g TMULTest.case_float_64x64_64x64_64x64
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tdiv -g TDIVTest.case_float_64x64_64x64_64x64
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tstore -g TStoreTest.ND_float_1_1_1_2_128_1_1_1_2_128
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tstore_acc2gm -g TStoreAcc2gmTest.case7
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tstore_mat2gm -g TStoreMat2GMTest.case_nd1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowsum -g TROWSUMTest.case1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpand -g TROWEXPANDTest.case0
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpandadd -g TROWEXPANDADDTest.case1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpanddiv -g TROWEXPANDDIVTest.case2
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpandmax -g TROWEXPANDMAXTest.case3
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpandmin -g TROWEXPANDMINTest.case4
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpandmul -g TROWEXPANDMULTest.case13
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpandsub -g TROWEXPANDSUBTest.case14
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpandexpdif -g TROWEXPANDEXPDIFTest.case7
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolexpandadd -g TColExpandAddTest.case_fp32_16_128_1_128
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolexpandmax -g TColExpandMaxTest.case_fp32_32_32_1_32
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolexpandmin -g TColExpandMinTest.case_fp16_4_256_1_256
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsels -g TSELSTest.case_uint16_uint8_2x16_2x32_2x16_2x16
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsels -g TSELSTest.case_float_uint16_2x8_2x16_2x8_2x8
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsort32 -g TSort32Test.case1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tadd -g TADDTest.case_float_64x64_64x64
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tand -g TANDTest.case1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tor -g TORTest.case2
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tpartargmax -g TPARTARGMAXTest.case_float_uint32_tile_diff_32k_small_0
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tpartargmin -g TPARTARGMINTest.case_float_uint32_tile_diff_32k_small_0
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tpows -g TPOWSTest.case11
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsel -g TSELTest.case1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tpartsel -g TPARTSELTest.float_tail
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmins
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trsqrt -g TRSQRTTest.case_float_64x64_64x64_64x64_inPlace_False
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsqrt -g TSQRTTest.case_float_64x64_64x64_64x64_inPlace_False
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t texp -g TEXPTest.case_float_64x64_64x64_64x64_inPlace_False
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tabs -g TABSTest.case_float_64x64_64x64_64x64_inPlace_False
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tlog -g TLOGTest.case_float_64x64_64x64_64x64_inPlace_False
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trecip -g TRECIPTest.case_float_64x64_64x64_64x64_inPlace_False
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tdivs -g TDIVSTest.case1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tdivs -g TDIVSTest.case4
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tdivs -g TDIVSTest.case5
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmuls -g TMULSTest.case1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tadds -g TADDSTest.case6
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t texpands -g TEXPANDSTest.case_float_64x64_64x64_64x64_PAD_VALUE_NULL
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tnot -g TNOTTest.case_int16_64x64_64x64_64x64
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tprelu -g TPRELUTest.case5
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trelu -g TRELUTest.case_int32_64x64_64x64_64x64
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tands -g TANDSTest.case_int16_64x64_64x64_64x64
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tors -g TORSTest.case_int16_64x64_64x64_64x64
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tshl -g TSHLTest.case1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tshr -g TSHRTest.case2
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tshls -g TSHLSTest.case_int16_64x64_64x64_64x64
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tshrs -g TSHRSTest.case_int16_64x64_64x64_64x64
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t txor -g TXORTest.case_int16_64x64_64x64_64x64_64x64
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t txors -g TXORSTest.case_int16_64x64_64x64_64x64
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tdequant -g TDEQUANTTest.case4
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tdequant -g TDEQUANTTest.case5
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tconcat -g TCONCATTest.case_half_16x128_16x64_16x64_16x63_16x64
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tconcat -g TCONCATTest.case_int16_32x256_32x128_32x128_32x127_32x128
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tconcat -g TCONCATTest.case_int32_64x128_64x64_64x64_64x64_64x64
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowargmax -g TROWARGMAXTest.case_uint32_float_16x1_13x16_1x8_13x13
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowargmax -g TROWARGMAXTest.case_uint32_float_8x1_3x4096_3x192_3x4095
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowargmax -g TROWARGMAXTest.case_uint32_float_8x1_8x1_1x16384_1x768_1x16381
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowargmin -g TROWARGMINTest.case_uint32_float_16x1_13x16_1x8_13x13
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowargmin -g TROWARGMINTest.case_uint32_float_8x1_3x4096_3x192_3x4095
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowargmin -g TROWARGMINTest.case_uint32_float_8x1_8x1_1x16384_1x768_1x16381
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_aligned_1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_aligned_4_bf16
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_partial_validrow
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_unaligned_validcol_full_row
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_aligned_int8_strided
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_nonpow2_half
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_nonpow2_partial_float_unaligned_idxrow
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_unalignedvalid_int16
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_unalignedvalid_float_smallthan32B
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_unalignedvalid_uint16_taillarge
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_scalar_4_int8
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_scalar_nonpow2_float
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nz_1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nz_multi_fractal_dst
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nz_half_large
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nz_nonpow2_float
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nz_partial_bf16
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nz_partial_float_unaligned_idxcol
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nz_unalignedvalid_validrow_half
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nz_scalar_5_int32
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nz_scalar_nonpow2_int8
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_aligned_1
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_aligned_4_bf16
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_partial_validrow
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_full_row_strided
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_aligned_uint16
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_partial_validboth_float
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_nonpow2_int8
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_nonpow2_int32
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_unalignedvalid_half
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_unalignedvalid_uint16_strided
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_unalignedvalid_float_smallthan32B
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_scalar_3_bf16
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_scalar_2_half
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nz_2
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nz_int32_partial
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nz_partial_float_unaligned_idxcol
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nz_nonpow2_partial_int8
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nz_unalignedvalid_int16
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nz_unalignedvalid_both_float
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nz_unalignedvalid_both_with_idx_bf16
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nz_scalar_5_int32
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nz_scalar_9_uint8_edge
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nz_scalar_nonpow2_int8
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmadd -g TMADDTest.case_float_32x128_32x192_32x256_32x127
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmula -g TMULATest.case_float_32x128_32x192_32x256_32x127
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_ndto2xnz -g TExtractNd2xNzTest.case_half
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_ndto2xnz -g TExtractNd2xNzTest.case_int32
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_ndto2xnz -g TExtractNd2xNzTest.case_int8_idx
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_ndto2xnz -g TExtractNd2xNzTest.case_int8_oddvalid
    fi
    if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_ndto2xnz -g TExtractNd2xNzTest.case_int8_1x1
    fi

    if [ "$IS_AUTO_MODE" = "false" ]; then
      # this testcase has to directly call CCE intrinsics now, which won't compile for auto mode;
      # besides, auto-sync doesn't work with CCE intrinsics
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpushpop_cv -g TPushPopCVTest.case1_half_single_tile
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpushpop_vc -g TPushPopVCTest.case1_int8_single_k_tile
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpushpop_cv_nosplit -g TPushPopCVNoSplitTest.case1_half_single_tile
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpushpop_vc_nosplit -g TPushPopVCNoSplitTest.case1_int8_single_k_tile
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpushpop_dir_both -g TPushPopDirBothTest.case1_float_dir_both
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpushpop_subtile -g TPushTpopSubtileTest.case1_half_128x512
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem2d_nz_float_16x16_2blk
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_row_nz_float_clamp_16x8_1blk
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_row_dyn_int32_3x16_8rows
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem2d_dyn_float_4x8_64size
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem_scalar_float_1x1_in_1x8_8size
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem2d_half_atomic_add_4x16_8size
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem2d_float_atomic_add_4x16_8size
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem2d_int32_unaligned_3x3_in_3x8_64size
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem2d_uint32_8x16_256size
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem2d_float_8x32_256size
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem2d_uint8_4x64_256size
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem_uint32_32_64size
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem_uint16_32_64size
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_row_uint8_unaligned_3x32_32rows
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_row_half_atomic_add_8x32_8rows
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_row_float_8x32_64rows
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_row_int32_wrap_8x16_8rows
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather -g MGATHERTest.case_row_float_8x32_64rows
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather -g MGATHERTest.case_row_half_16x64_64rows
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather -g MGATHERTest.case_row_float_partial_4x16_in_8x16
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather -g MGATHERTest.case_elem_half_64_128size
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather -g MGATHERTest.case_elem2d_half_4x32_256size
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather -g MGATHERTest.case_elem2d_int16_4x32_256size
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather -g MGATHERTest.case_row_dyn_int32_3x16_8rows
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather -g MGATHERTest.case_elem2d_nz_int32_16x8_1blk
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tfillpad -g TFILLPADTest.case_float_GT_128_127_VT_128_128_BLK1_PADMAX_PADMAX
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tfillpad -g TFILLPADTest.case_u16_GT_259_7_VT_260_32_BLK1_PADMIN_PADMAX_EXPAND -d
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tgather -g TGATHERTest.case1_float_P0101
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tload_gm2mat -g TLoadGM2L1Test.ND2NZ_bfloat16_t_1_1_1_1_1_1_1_1_1_1
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tmov_vect -g TMOVTest.vect_copy_case1
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpartadd -g TPARTADDTest.case_float_64x64_64x64_64x64
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpartmul -g TPARTMULTest.case_float_64x64_64x64_64x64
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpow -g TPOWTest.case11
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t trowprod -g TROWPRODTest.case1
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tscatter -g TSCATTERTest.case_mask_float_16x64_16x64_P1111
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t ttrans_3d -g TTRANS3DTest.case3_int32_17_3_3_2_2
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t ttrans_3d -g TTRANS3DTest.case1_float32_2_4_2_2_2
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t ttrans_3d -g TTRANS3DTest.case7_uint16_4_8_2_2_3
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t ttrans_3d -g TTRANS3DTest.case10_uint8_9_18_2_2_4
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t ttri -g TTRITest.case_float_128x128_128x31_1__444
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tcvt -g TCVTTest.case_fp16_fp32_2x64
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tcmp -g TCMPTest.case_eq_float_1x64_1x64_1x64
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tcmps -g TCMPSTest.case_gt_float_8x64_8x64_8x64
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolscatter -g TCOLSCATTERTest.case_mask_half_16x64_16x64_P1111
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t textract -g TEXTRACTTest.case1_half_0_1_16_16_32_param
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t ttrans -g TTRANSTest.case1_float_16_8_16_8
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "2" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t ttrans_conv -g TTRANSConvTest.int8_1_63_2_128
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tci -g TCITest.case1_int32
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tquant -g TQUANTTEST.case_int8_sym_fp32_128x128_nd
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tquant -g TQUANTTEST.case_int8_asym_fp32_128x128_nd
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t texpands_mat -g TEXPANDSTest.case1
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather_gm2l1 -g MGATHERGM2L1Test.case_row_float_16x16_64rows
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather_gm2l1 -g MGATHERGM2L1Test.case_row_int32_16x8_32rows
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather_gm2l1 -g MGATHERGM2L1Test.case_row_float_16x16_64rows
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather_gm2l1 -g MGATHERGM2L1Test.case_row_uint16_16x32_48rows
      fi
      if [ -z "$ST_PART" ] || [ "$ST_PART" = "1" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather_gm2l1 -g MGATHERGM2L1Test.case_elem_bfloat16_16x16_256size
      fi
    fi

  elif [ "$ENABLE_ALL" = "true" ]; then            # 所有用例
    python3 tests/script/build_st.py $ARGS -v a3 -t all
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tconcatdstidx
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tconcatidx
    python3 tests/script/run_st.py $ARGS -w -v a3 -t taxpy
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolexpand
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolsum
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolprod
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolmax
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolmin
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolargmin
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trem
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trems
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tfmod
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tfmods
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsubs
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmaxs
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tlrelu
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmatmul
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmov
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmov_acc2mat
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmrgsort
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tstore
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tstore_acc2gm
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tstore_mat2gm
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowsum
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpand
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpandadd
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpanddiv
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpandmax
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpandmin
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpandmul
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpandsub
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpandexpdif
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolexpandadd
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolexpandmax
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolexpandmin
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsort32
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tpows
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tpartargmax
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tpartargmin
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsel
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tload
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tadd
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tand
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tor
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsels
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tpartsel
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmins
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsub
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tgatherb
    python3 tests/script/run_st.py $ARGS -w -v a3 -t texp
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trsqrt
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsqrt
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tabs
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tlog
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trecip
    python3 tests/script/run_st.py $ARGS -w -v a3 -t texpands
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tnot
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tprelu
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trelu
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tands
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tors
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tshl
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tshr
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tshls
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tshrs
    python3 tests/script/run_st.py $ARGS -w -v a3 -t txor
    python3 tests/script/run_st.py $ARGS -w -v a3 -t txors
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tdequant
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tconcat
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowargmax
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowargmin
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_ndto2xnz
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmadd
    if [ "$IS_AUTO_MODE" = "false" ]; then
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolgather
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolscatter
      python3 tests/script/run_st.py $ARGS -w -v a3 -t trowprod
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolargmax
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tcvt
      python3 tests/script/run_st.py $ARGS -w -v a3 -t textract
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tgather
      python3 tests/script/run_st.py $ARGS -w -v a3 -t ttrans
      python3 tests/script/run_st.py $ARGS -w -v a3 -t ttrans_conv
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpartadd
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpow
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpartmul
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tload_gm2mat
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tci
      python3 tests/script/run_st.py $ARGS -w -v a3 -t texpands_mat
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tcmp
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tscatter
      python3 tests/script/run_st.py $ARGS -w -v a3 -t ttri
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tquant
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather
      python3 tests/script/run_st.py $ARGS -w -v a3 -t ttrans_3d
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter
      # this testcase has to directly call CCE intrinsics now, which won't compile for auto mode;
      # besides, auto-sync doesn't work with CCE intrinsics
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpushpop_cv
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpushpop_vc
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpushpop_cv_nosplit
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpushpop_vc_nosplit
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpushpop_dir_both
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpushpop_subtile
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather_gm2l1
    fi
  fi
fi

if [ "$ENABLE_A5" = "true" ]; then
  if [ "$ENABLE_SIMPLE" = "true" ]; then           # 单个用例
    # Initialize result dir in parent shell BEFORE any background task.
    # Background tasks (via python3() wrapper) inherit this dir and write
    # PASS/FAIL files here; _collect_results in the parent reads them after
    # each wait. Without this, _RESULTS_DIR stays empty in the parent and
    # _collect_results is a no-op — the gate silently fails open.
    _init_results_dir

    # Case-count guard wiring: run_st.py appends its gtest case count here.
    CASE_COUNT_FILE=$(mktemp -t pto_cases_XXXXXX)
    export PTO_CASE_COUNT_FILE="$CASE_COUNT_FILE"

    python3 tests/script/build_st.py $ARGS -v a5 -t all

    # Per-slot isolated build dir. Each concurrent process gets its OWN copy
    # of build/ so gen_data and golden files never race: two processes sharing
    # one build dir would concurrently write the same testcase/<tc>/golden dirs
    # and corrupt each other's goldens (the exact FileExistsError /
    # buffer-size-mismatch failure seen with the pre-gen experiment).
    #
    # Slot layout: one process per visible NPU. PTO_SLOTS pins the slot count;
    # otherwise use every visible device. build_st.py chdirs to
    # tests/npu/a5/src/st and builds there, so build/ lives at
    # tests/npu/a5/src/st/build/. run_st.py also chdirs there and reads
    # PTO_BUILD_DIR (relative to that cwd), so the copies must sit alongside
    # the original build/ in tests/npu/a5/src/st/.
    _A5_ST_DIR="tests/npu/a5/src/st"
    # One process per card. Running two processes per card was measured
    # (job-5d5bcc2f) and rejected: the per-process median invocation time went
    # from 13907ms to 26324ms (~1.9x) because aclInit serializes inside the
    # driver — 8 concurrent initializers queue behind the same lock instead of
    # 4 — while the test span only shrank by ~54s. The per-slot build dir
    # isolation below is kept because it is what makes the split safe.
    _available_ids=()
    if [ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
      IFS=',' read -r -a _candidate_ids <<< "$(echo "${ASCEND_RT_VISIBLE_DEVICES}" | tr -d ' ')"
      for _id in "${_candidate_ids[@]}"; do
        if [[ "$_id" =~ ^[0-9]+$ ]]; then
          _available_ids+=("$_id")
        fi
      done
    else
      for _dev in /dev/davinci[0-9]*; do
        [ -e "$_dev" ] || continue
        _available_ids+=("${_dev##*davinci}")
      done
    fi
    if [ ${#_available_ids[@]} -eq 0 ]; then
      _available_ids=(0)
      echo "[PARALLEL] no visible NPU detected; falling back to device 0"
    fi
    _SLOTS=${PTO_SLOTS:-${#_available_ids[@]}}
    if ! [[ "$_SLOTS" =~ ^[0-9]+$ ]] || [ "$_SLOTS" -lt 1 ]; then
      echo "[PARALLEL] invalid PTO_SLOTS='${PTO_SLOTS}'" >&2
      exit 1
    fi
    if [ "$_SLOTS" -gt ${#_available_ids[@]} ]; then
      echo "[PARALLEL] PTO_SLOTS=${_SLOTS} exceeds visible NPUs: ${_available_ids[*]}" >&2
      exit 1
    fi
    rm -rf ${_A5_ST_DIR}/build_card*
    for _s in $(seq 0 $(( _SLOTS - 1 ))); do
      _s_card=${_available_ids[$_s]}
      cp -r ${_A5_ST_DIR}/build ${_A5_ST_DIR}/build_card${_s_card}_p0
    done
    echo "[PARALLEL] ${_SLOTS} slots (devices: ${_available_ids[*]})"

    # Build case list (auto_mode-only cases gated by IS_AUTO_MODE)
    _cases=()
    _cases+=("-t tconcatdstidx -g TCONCATTest.case_int16_16x32_16x16_16x16_8x16_8x16")
    _cases+=("-t tpartargmax -g TPARTARGMAXTest.case_fp32_64x64_64x64_64x64")
    _cases+=("-t tpartargmin -g TPARTARGMINTest.case_fp32_64x64_64x64_64x64")
    _cases+=("-t tconcatidx -g TCONCATTest.case_int16_16x32_16x16_16x16_8x16_8x16")
    _cases+=("-t taxpy -g TAXPYTest.case1")
    _cases+=("-t tconcat -g TCONCATTest.case_half_16x128_16x64_16x64_16x63_16x64")
    _cases+=("-t tfmods -g TFMODSTest.case1")
    _cases+=("-t tfmod -g TFMODTest.case1")
    _cases+=("-t tsubs -g TSUBSTest.case1")
    _cases+=("-t tmaxs -g TMAXSTest.case_float_64x64_32x32_32x32")
    _cases+=("-t trems -g TREMSTest.case1")
    _cases+=("-t tlrelu -g TLRELUTest.case1")
    _cases+=("-t tadd -g TADDTest.case_float_64x64_64x64_64x64_64x64")
    _cases+=("-t tadds -g TADDSTest.case1")
    _cases+=("-t tand -g TANDTest.case1")
    _cases+=("-t tands -g TANDSTest.case1")
    _cases+=("-t tors -g TORSTest.case1")
    _cases+=("-t tshls -g TSHLSTest.case_int16_64x64_64x64_64x64")
    _cases+=("-t tcmps -g TCMPSTest.case_float_8x64_8x64")
    _cases+=("-t tcolexpandadd -g TColExpandAddTest.case_fp32_16_128_1_128")
    _cases+=("-t tcolexpandmax -g TColExpandMaxTest.case_fp32_32_32_1_32")
    _cases+=("-t tcolexpandmin -g TColExpandMinTest.case_fp16_4_256_1_256")
    _cases+=("-t tcolmax -g TCOLMAXTest.case01")
    _cases+=("-t tcolmin -g TCOLMINTest.case01")
    _cases+=("-t tdivs -g TDIVSTest.case4")
    _cases+=("-t tdivs -g TDIVSTest.case5")
    _cases+=("-t texp -g TEXPTest.case1")
    _cases+=("-t tlog -g TLOGTest.case_float_64x64_64x64_64x64")
    _cases+=("-t texpands -g TEXPANDSTest.case_float_64x64_64x64_64x64_PAD_VALUE_NULL")
    _cases+=("-t tload -g TLOADTest.case_float_GT_2_2_2_256_60_VT_256_64_BLK8_PADMAX")
    _cases+=("-t tload_mx_NZ -g TLOADSCALETest.4_3_3_16_2_4_10_5_16_2_192_10_scale_ZZ2ZZ")
    _cases+=("-t tload_mx_NZ -g TLOADSCALETest.7_5_3_16_2_7_7_11_16_2_12_560_scale_NN2NN")
    _cases+=("-t tmax -g TMAXTest.case_float_64x64_64x64_64x64_64x64")
    _cases+=("-t tmin -g TMINTest.case_float_64x64_64x64_64x64_64x64")
    _cases+=("-t tmins -g TMINSTest.case_float_60x128_64x64_60x60")
    _cases+=("-t tmins -g TMINSTest.case_float_16x200_20x512_16x200")
    _cases+=("-t tmins -g TMINSTest.case_float_1x3600_2x4096_1x3600")
    _cases+=("-t tmov_vect -g TMOVTest.vect_copy_case1")
    _cases+=("-t tmov_mx -g TMOVMXTest.case1")
    _cases+=("-t tmov_nd2zn -g TMovNd2ZnTest.case_half_32x32")
    _cases+=("-t tmrgsort -g TMRGSORTTest.case_topk1")
    _cases+=("-t tmul -g TMULTest.case_float_64x64_64x64_64x64_64x64")
    _cases+=("-t tmadd -g TMADDTest.case_float_32x128_32x192_32x256_32x127")
    _cases+=("-t tmuls -g TMULSTest.case1")
    _cases+=("-t tor -g TORTest.case2")
    _cases+=("-t tpow -g TPOWTest.case1")
    _cases+=("-t tpows -g TPOWSTest.case1")
    _cases+=("-t tprelu -g TPRELUTest.case1")
    _cases+=("-t trem -g TREMTest.case1")
    _cases+=("-t trowexpand -g TROWEXPANDTest.case5_float_16_8_16_127")
    _cases+=("-t trowexpanddiv -g TRowExpandDivTest.case_fp32_40_64")
    _cases+=("-t trowmax -g TROWMAXTest.case1")
    _cases+=("-t trowmin -g TROWMINTest.case1")
    _cases+=("-t trowargmax -g TROWARGMAXTest.case_uint32_float_64x1_32x128_32x128")
    _cases+=("-t trowargmin -g TROWARGMINTest.case_uint32_float_64x1_32x128_32x128")
    _cases+=("-t trowsum -g TROWSUMTest.case1")
    _cases+=("-t trowprod -g TROWPRODTest.case1")
    _cases+=("-t trsqrt -g TRSQRTTest.case1")
    _cases+=("-t tsel -g TSELTest.case1")
    _cases+=("-t tpartsel -g TPARTSELTest.float_tail")
    _cases+=("-t tsels -g TSELSTest.case_uint8_uint8_2x32_2x32_2x32_2x32")
    _cases+=("-t tsels -g TSELSTest.case_float_uint16_2x8_2x16_2x8_2x8")
    _cases+=("-t tshl -g TSHLTest.case1")
    _cases+=("-t tshr -g TSHRTest.case2")
    _cases+=("-t tsort32 -g TSort32Test.case1")
    _cases+=("-t tsqrt -g TSQRTTest.case1")
    _cases+=("-t tstore -g TStoreTest.case1")
    _cases+=("-t tstore_acc2gm -g TStoreAcc2gmTest.case7")
    _cases+=("-t tadd_tdiv -g TADD_TDIVTest.case_float_64x64_64x64")
    _cases+=("-t tmul_tadds -g TMUL_TADDSTest.case_float_64x64_64x64")
    _cases+=("-t tsub_texp -g TSUB_TEXPTest.case_float_64x64_64x64")
    _cases+=("-t tmuls_trowsum -g TMULS_TROWSUMTest.case_float_64x64_64x64")
    _cases+=("-t trowexpand_tsqrt -g TROWEXPAND_TSQRTTest.case_float_64x64_64x64")
    _cases+=("-t trowexpand_trowsum -g TROWEXPAND_TROWSUMTest.case_float_64x64_64x64")
    _cases+=("-t trowsum_trowexpand -g TROWSUM_TROWEXPANDTest.case_float_64x64_64x64")
    _cases+=("-t trowexpand_tdiv -g TROWEXPAND_TDIVTest.case_float_64x64_64x64")
    _cases+=("-t tmov_ub2l1 -g TMovUb2l1Test.case1")
    _cases+=("-t tneg -g TNEGTest.case_float_64x64_64x64")
    _cases+=("-t tcolexpand -g TCOLEXPANDTest.case_float_1_8_128_63")
    _cases+=("-t ttri -g TTRITest.case_float_128x128_upper_diag_n3")
    _cases+=("-t tnot -g TNOTTest.case_int16_64x64_64x64_64x64")
    _cases+=("-t trelu -g TRELUTest.case_int32_64x64_64x64_64x64")
    _cases+=("-t tmov_acc2mat -g TMOVTest.case_nz2nz_insert")
    _cases+=("-t mgather -g MGATHERTest.case_row_half_16x64_64rows")
    _cases+=("-t mscatter -g MSCATTERTest.case_row_uint8_random_8x32_32rows")
    _cases+=("-t mscatter -g MSCATTERTest.case_row_int32_clamp_8x16_8rows")
    _cases+=("-t mscatter -g MSCATTERTest.case_elem2d_float_3072x8_default_24576size")
    _cases+=("-t mscatter -g MSCATTERTest.case_elem2d_float_3072x8_last_256size")
    _cases+=("-t ttri -g TTRITest.case_float_128x128_lower_diag_n3")
    _cases+=("-t ttri -g TTRITest.case_float_128x128_upper_diag_0")
    _cases+=("-t textract_vec -g TExtractVecTest.case_nd_aligned_6")
    _cases+=("-t textract_vec -g TExtractVecTest.case_nd_unaligned_validcol_2")
    _cases+=("-t textract_vec -g TExtractVecTest.case_nd_unaligned_indexcol_2")
    _cases+=("-t textract_vec -g TExtractVecTest.case_nd_unaligned_validcol_3")
    _cases+=("-t textract_vec -g TExtractVecTest.case_nd_aligned_hif8")
    _cases+=("-t textract_vec -g TExtractVecTest.case_nd_scalar_2")
    _cases+=("-t textract_vec -g TExtractVecTest.case_nz_2")
    _cases+=("-t textract_vec -g TExtractVecTest.case_nz_6")
    _cases+=("-t textract_vec -g TExtractVecTest.case_nz_indexcol_nonzero")
    _cases+=("-t textract_vec -g TExtractVecTest.case_nz_multi_fractal_dst")
    _cases+=("-t textract_vec -g TExtractVecTest.case_nz_hif8")
    _cases+=("-t textract_vec -g TExtractVecTest.case_nz_scalar_1")
    _cases+=("-t textract_vec -g TExtractVecTest.case_nz_scalar_fp4_e2m1")
    _cases+=("-t textract_vec -g TExtractVecTest.case_nz_scalar_fp4_e1m2")
    _cases+=("-t textract_ndto2xnz -g TExtractNd2xNzTest.case_half_aligned")
    _cases+=("-t textract_ndto2xnz -g TExtractNd2xNzTest.case_float_unaligned")
    _cases+=("-t textract_ndto2xnz -g TExtractNd2xNzTest.case_hif8_aligned")
    _cases+=("-t textract_ndto2xnz -g TExtractNd2xNzTest.case_fp8e5m2_aligned")
    _cases+=("-t textract_ndto2xnz -g TExtractNd2xNzTest.case_half_1x1")
    _cases+=("-t tinterleave -g TINTERLEAVETest.case_float_64x64_64x64")
    _cases+=("-t tinterleave -g TINTERLEAVETest.case_int32_64x64_64x64")
    _cases+=("-t tinterleave -g TINTERLEAVETest.case_int16_64x64_64x64")
    _cases+=("-t tinterleave -g TINTERLEAVETest.case_half_16x256_16x256")
    _cases+=("-t tinterleave -g TINTERLEAVETest.case_float_16x32_16x32")
    _cases+=("-t tinterleave -g TINTERLEAVETest.case_int32_16x32_16x32")
    _cases+=("-t tinterleave -g TINTERLEAVETest.case_int8_32x256_32x256")
    _cases+=("-t tinterleave -g TINTERLEAVETest.case_uint8_32x256_32x256")
    _cases+=("-t tdeinterleave -g TDEINTERLEAVETest.case_float_64x64_64x64")
    _cases+=("-t tdeinterleave -g TDEINTERLEAVETest.case_int32_64x64_64x64")
    _cases+=("-t tdeinterleave -g TDEINTERLEAVETest.case_int16_64x64_64x64")
    _cases+=("-t tdeinterleave -g TDEINTERLEAVETest.case_half_16x256_16x256")
    _cases+=("-t tdeinterleave -g TDEINTERLEAVETest.case_int32_16x32_16x32")
    _cases+=("-t tdeinterleave -g TDEINTERLEAVETest.case_int8_32x256_32x256")
    _cases+=("-t tdeinterleave -g TDEINTERLEAVETest.case_float_single_src_16x128_16x128")
    _cases+=("-t tdeinterleave -g TDEINTERLEAVETest.case_int32_single_src_16x128_16x128")
    _cases+=("-t tdeinterleave -g TDEINTERLEAVETest.case_half_single_src_16x256_16x256")
    _cases+=("-t tdeinterleave -g TDEINTERLEAVETest.case_int8_single_src_8x512_8x512")
    if [ "$IS_AUTO_MODE" = "false" ]; then
      _cases+=("-t texpands_mat -g TEXPANDSTest.case1")
      _cases+=("-t tpushpop_cv -g TPushPopCVTest.case1_half_single_tile")
      _cases+=("-t tpushpop_vc -g TPushPopVCTest.case1_int8_single_k_tile")
      _cases+=("-t tpushpop_cv_nosplit -g TPushPopCVNoSplitTest.case1_half_single_tile")
      _cases+=("-t tpushpop_vc_nosplit -g TPushPopVCNSTest.case1_int8_single_k_tile")
      _cases+=("-t tpushpop_dir_both -g TPushPopDirBothTest.case1_float_dir_both")
      _cases+=("-t tpushpop_subtile -g TPushTpopSubtileTest.case1_half_128x512")
      _cases+=("-t tpushpop_subblock_dispatch -g TPushPopSubBlockDispatchTest.case1_v2c_nosplit_implicit_id")
      _cases+=("-t thistogram -g THISTOGRAMTest.case_8x128_b1")
      _cases+=("-t thistogram -g THISTOGRAMTest.case_u32_6x912_b1_k64")
      _cases+=("-t tcolgather -g TCOLGATHERTest.case_mask_half_16x64_16x64_P1111")
      _cases+=("-t tcolscatter -g TCOLSCATTERTest.case_mask_half_16x64_16x64_P1111")
      _cases+=("-t tshrs -g TSHRSTest.case_int16_64x64_64x64_64x64")
      _cases+=("-t tdequant -g TDEQUANTTest.case1")
      _cases+=("-t tci -g TCITest.case5")
      _cases+=("-t tcolsum -g TCOLSUMTest.case01")
      _cases+=("-t tcolprod -g TCOLPRODTest.case01")
      _cases+=("-t tcvt -g TCVTTest.case_fp16_fp32_2x64")
      _cases+=("-t tcvt_fp4_lut")
      _cases+=("-t tcvt_fp4_lut_v2c")
      _cases+=("-t textract -g TEXTRACTTest.case1")
      _cases+=("-t textract_acc2vec -g TMOVTest.case_nz2nd_sc_quant_1")
      _cases+=("-t tfillpad -g TFILLPADTest.case_float_GT_128_127_VT_128_128_BLK1_PADMAX_PADMAX")
      _cases+=("-t tfillpad -g TFILLPADTest.case_u16_GT_259_7_VT_260_32_BLK1_PADMIN_PADMAX_EXPAND -d")
      _cases+=("-t tgather -g TGATHERTest.case1_float_32x1024_16x64")
      _cases+=("-t tgatherb -g TGATHERBTest.case_float_2x128_2x16_2x128")
      _cases+=("-t tload_mix -g TLOADMIXTest.1_1_1_59_119_1_1_1_64_128_64_128_int8_t_ND2NZ")
      _cases+=("-t tload_mx_ND_DN -g TLOADMXTest.1_1_1_64_128_uint8_AND2ZZ")
      _cases+=("-t tload_mx_gmtensor -g TLOADMXTest.1_1_1_64_128_uint8_ADN2ZZ")
      _cases+=("-t tload_shape2d -g TLOADSHAPE2DTest.1_1_1_59_119_1_1_1_64_128_64_128_int8_t_ND2NZ")
      _cases+=("-t tmov -g TMOVTest.case_bias1")
      _cases+=("-t tpartadd -g TPARTADDTest.case_float_64x64_64x64_64x64")
      _cases+=("-t tpartmul -g TPARTMULTest.case_float_64x64_64x64_64x64")
      _cases+=("-t tpartmax -g TPARTMAXTest.case_fp32_64x64_64x64_64x64")
      _cases+=("-t tpartmin -g TPARTMINTest.case_fp32_64x64_64x64_64x64")
      _cases+=("-t ttrans -g TTRANSTest.case_float_8x8_2x8_2x8")
      _cases+=("-t ttrans -g TTRANSTest.case_hif8_32x32_32x32_32x32")
      _cases+=("-t ttrans -g TTRANSTest.case_hif8_64x64_64x64_22x63")
      _cases+=("-t ttrans_conv -g TTRANSConvTest.uint8_11_2_7_7_32")
      _cases+=("-t tscatter -g TSCATTERTest.case_mask_float_16x64_16x64_P1111")
      _cases+=("-t tquant -g TQUANTTEST.case_int8_sym_fp32_128x128_nd")
      _cases+=("-t tquant -g TQUANTTEST.case_int8_asym_fp32_128x128_nd")
      _cases+=("-t tinsert -g TInsertTest.case_acc2mat_1")
      _cases+=("-t tinsert -g TInsertTest.case_nz_1")
      _cases+=("-t tinsert -g TInsertTest.case_nd_1")
      _cases+=("-t tinsert -g TInsertTest.case_nz_2")
      _cases+=("-t tinsert -g TInsertTest.case_nz_6")
      _cases+=("-t tinsert_acc2vec -g TMOVTest.case_nz2nd_fb_quant_1")
      _cases+=("-t tinsert -g TInsertTest.case_nd_vec_16")
      _cases+=("-t tinsert -g TInsertTest.case_nz_split_1")
      _cases+=("-t tinsert -g TInsertTest.case_nz_split_4")
      _cases+=("-t tinsert -g TInsertTest.case_nz_hif8_3")
      _cases+=("-t tinsert -g TInsertTest.case_nz_twoinput_bf16_1")
      _cases+=("-t tinsert -g TInsertTest.case_nz_twoinput_fp8e5_1")
      _cases+=("-t tinsert -g TInsertTest.case_nz_twoinput_fp4e1m2_2")
      _cases+=("-t tinsert -g TInsertTest.case_nz_dblinput_fp4e2m1_1")
      _cases+=("-t tinsert -g TInsertTest.case_nz_twoinput_fp4e2m1_1")
      _cases+=("-t tinsert -g TInsertTest.case_nz_dblinput_fp8e4_2")
      _cases+=("-t tinsert_zn -g TInsertZNTest.case_zn_offset_1")
      _cases+=("-t tinsert_zn -g TInsertZNTest.case_zn_offset_4")
      _cases+=("-t tinsert_zn -g TInsertZNTest.case_zn_offset_9")
      _cases+=("-t tinsert_zn -g TInsertZNTest.case_zn_offset_11")
      _cases+=("-t tinsert_zn -g TInsertZNTest.case_zn_offset_12")
      _cases+=("-t tinsert_zn -g TInsertZNTest.case_zn_offset_13")
      _cases+=("-t tmatmul -g TMATMULTest.case1")
      _cases+=("-t tmatmul_mx -g TMATMULMXTest.case1")
      _cases+=("-t tmov_acc2vec -g TMOVTest.case_nz2nd_1")
      _cases+=("-t txor -g TXORTest.case1")
      _cases+=("-t txors -g TXORSTest.case1")
      _cases+=("-t mgather_gm2l1 -g MGATHERGM2L1Test.case_row_int32_16x8_32rows")
      _cases+=("-t mgather_gm2l1 -g MGATHERGM2L1Test.case_elem_int32_16x8_128size")
      _cases+=("-t mgather_gm2l1 -g MGATHERGM2L1Test.case_elem_simt_bfloat16_16x16_256size")
      _cases+=("-t mgather_gm2l1 -g MGATHERGM2L1Test.case_elem_simt_half_zero_16x16_64size")
      _cases+=("-t mgather_gm2l1 -g MGATHERGM2L1Test.case_row_float_16x16_64rows")
      _cases+=("-t mgather_gm2l1 -g MGATHERGM2L1Test.case_elem_bfloat16_16x16_256size")
      _cases+=("-t mgather_gm2l1 -g MGATHERGM2L1Test.case_elem_simt_int8_16x32_512size")
      _cases+=("-t mgather_gm2l1 -g MGATHERGM2L1Test.case_elem_simt_int16_16x16_256size")
      _cases+=("-t mgather_gm2l1 -g MGATHERGM2L1Test.case_elem_simt_float_clamp_16x16_64size")
      _cases+=("-t tcmp -g TCMPTest.case_half_32x32_32x32")
    fi

    # ---- BATCH DISPATCH: for testcases whose main.cpp was refactored to
    # lift aclInit into a gtest Environment (once per process instead of once
    # per TEST_F). For these, dispatch ONE process per testcase that runs ALL
    # its cases in a single aclInit->aclFinalize lifecycle via gtest_filter=A:B:C.
    # gen_data.py already generates goldens for every case in one invocation, so
    # batching also eliminates the N-fold redundant golden regeneration that the
    # per-case loop would do (N = number of cases in the testcase).
    #
    # Refactored testcases (17): tneg, tinterleave, tdeinterleave, tinsert_zn,
    # mgather_gm2l1, mscatter, tload_mx_NZ, tmins, ttrans, tsels, tdivs,
    # thistogram, textract_ndto2xnz, textract_vec, ttri, tquant, tinsert.
    _batch_tcs="tneg tinterleave tdeinterleave tinsert_zn mgather_gm2l1 mscatter tload_mx_NZ tmins ttrans tsels tdivs thistogram textract_ndto2xnz textract_vec ttri tquant tinsert"

    # _batch_run <testcase> <slot> <gtest_filter>
    # slot -> device = _available_ids[slot] (see build dir setup above).
    _batch_run() {
      _tc="$1"; _slot="$2"; _filter="$3"
      _card=${_available_ids[$_slot]}
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 \
        PTO_BUILD_DIR=build_card${_card}_p0 ASCEND_DEVICE_ID=${_card} \
        python3 tests/script/run_st.py $ARGS -w -v a5 -t ${_tc} -g "${_filter}"
    }

    # Dispatch batch testcases across all slots (slot = _batch_idx % _SLOTS).
    # Throttle: only wait once a full slot-batch is in flight, i.e. one batch
    # process per slot runs concurrently.
    _batch_throttle() {
      if [ $(( _batch_idx % _SLOTS )) -eq 0 ]; then
        wait
        _collect_results
      fi
    }
    _batch_idx=0
    # tneg: 1 case
    _batch_run tneg $((_batch_idx % _SLOTS)) "TNEGTest.case_float_64x64_64x64" &
    _batch_idx=$((_batch_idx + 1))
    # tinterleave: 8 cases
    _batch_run tinterleave $((_batch_idx % _SLOTS)) \
      "TINTERLEAVETest.case_float_64x64_64x64:TINTERLEAVETest.case_int32_64x64_64x64:TINTERLEAVETest.case_int16_64x64_64x64:TINTERLEAVETest.case_half_16x256_16x256:TINTERLEAVETest.case_float_16x32_16x32:TINTERLEAVETest.case_int32_16x32_16x32:TINTERLEAVETest.case_int8_32x256_32x256:TINTERLEAVETest.case_uint8_32x256_32x256" &
    _batch_idx=$((_batch_idx + 1))
    # tdeinterleave: 9 cases (mix of dual-src and single-src)
    _batch_run tdeinterleave $((_batch_idx % _SLOTS)) \
      "TDEINTERLEAVETest.case_float_64x64_64x64:TDEINTERLEAVETest.case_int32_64x64_64x64:TDEINTERLEAVETest.case_int16_64x64_64x64:TDEINTERLEAVETest.case_half_16x256_16x256:TDEINTERLEAVETest.case_int32_16x32_16x32:TDEINTERLEAVETest.case_int8_32x256_32x256:TDEINTERLEAVETest.case_float_single_src_16x128_16x128:TDEINTERLEAVETest.case_int32_single_src_16x128_16x128:TDEINTERLEAVETest.case_half_single_src_16x256_16x256:TDEINTERLEAVETest.case_int8_single_src_8x512_8x512" &
    _batch_idx=$((_batch_idx + 1))
    # tinsert_zn: 6 cases
    _batch_run tinsert_zn $((_batch_idx % _SLOTS)) \
      "TInsertZNTest.case_zn_offset_1:TInsertZNTest.case_zn_offset_4:TInsertZNTest.case_zn_offset_9:TInsertZNTest.case_zn_offset_11:TInsertZNTest.case_zn_offset_12:TInsertZNTest.case_zn_offset_13" &
    _batch_idx=$((_batch_idx + 1))
    _batch_throttle

    # mgather_gm2l1: 9 cases
    _batch_run mgather_gm2l1 $((_batch_idx % _SLOTS)) \
      "MGATHERGM2L1Test.case_row_int32_16x8_32rows:MGATHERGM2L1Test.case_elem_int32_16x8_128size:MGATHERGM2L1Test.case_elem_simt_bfloat16_16x16_256size:MGATHERGM2L1Test.case_elem_simt_half_zero_16x16_64size:MGATHERGM2L1Test.case_row_float_16x16_64rows:MGATHERGM2L1Test.case_elem_bfloat16_16x16_256size:MGATHERGM2L1Test.case_elem_simt_int8_16x32_512size:MGATHERGM2L1Test.case_elem_simt_int16_16x16_256size:MGATHERGM2L1Test.case_elem_simt_float_clamp_16x16_64size" &
    _batch_idx=$((_batch_idx + 1))
    # mscatter: 4 cases
    _batch_run mscatter $((_batch_idx % _SLOTS)) \
      "MSCATTERTest.case_row_uint8_random_8x32_32rows:MSCATTERTest.case_row_int32_clamp_8x16_8rows:MSCATTERTest.case_elem2d_float_3072x8_default_24576size:MSCATTERTest.case_elem2d_float_3072x8_last_256size" &
    _batch_idx=$((_batch_idx + 1))
    # tload_mx_NZ: 2 cases
    _batch_run tload_mx_NZ $((_batch_idx % _SLOTS)) \
      "TLOADSCALETest.4_3_3_16_2_4_10_5_16_2_192_10_scale_ZZ2ZZ:TLOADSCALETest.7_5_3_16_2_7_7_11_16_2_12_560_scale_NN2NN" &
    _batch_idx=$((_batch_idx + 1))
    # tmins: 3 cases
    _batch_run tmins $((_batch_idx % _SLOTS)) \
      "TMINSTest.case_float_60x128_64x64_60x60:TMINSTest.case_float_16x200_20x512_16x200:TMINSTest.case_float_1x3600_2x4096_1x3600" &
    _batch_idx=$((_batch_idx + 1))
    _batch_throttle

    # ttrans: 3 cases
    _batch_run ttrans $((_batch_idx % _SLOTS)) \
      "TTRANSTest.case_float_8x8_2x8_2x8:TTRANSTest.case_hif8_32x32_32x32_32x32:TTRANSTest.case_hif8_64x64_64x64_22x63" &
    _batch_idx=$((_batch_idx + 1))
    # tsels: 2 cases
    _batch_run tsels $((_batch_idx % _SLOTS)) \
      "TSELSTest.case_uint8_uint8_2x32_2x32_2x32_2x32:TSELSTest.case_float_uint16_2x8_2x16_2x8_2x8" &
    _batch_idx=$((_batch_idx + 1))
    # tdivs: 2 cases
    _batch_run tdivs $((_batch_idx % _SLOTS)) \
      "TDIVSTest.case4:TDIVSTest.case5" &
    _batch_idx=$((_batch_idx + 1))
    # thistogram: 2 cases
    _batch_run thistogram $((_batch_idx % _SLOTS)) \
      "THISTOGRAMTest.case_8x128_b1:THISTOGRAMTest.case_u32_6x912_b1_k64" &
    _batch_idx=$((_batch_idx + 1))
    _batch_throttle

    # textract_ndto2xnz: 5 cases
    _batch_run textract_ndto2xnz $((_batch_idx % _SLOTS)) \
      "TExtractNd2xNzTest.case_half_aligned:TExtractNd2xNzTest.case_float_unaligned:TExtractNd2xNzTest.case_hif8_aligned:TExtractNd2xNzTest.case_fp8e5m2_aligned:TExtractNd2xNzTest.case_half_1x1" &
    _batch_idx=$((_batch_idx + 1))
    # textract_vec: 15 cases
    _batch_run textract_vec $((_batch_idx % _SLOTS)) \
      "TExtractVecTest.case_nd_aligned_6:TExtractVecTest.case_nd_unaligned_validcol_2:TExtractVecTest.case_nd_unaligned_indexcol_2:TExtractVecTest.case_nd_unaligned_validcol_3:TExtractVecTest.case_nd_aligned_hif8:TExtractVecTest.case_nd_aligned_fp8_e4m3:TExtractVecTest.case_nd_scalar_2:TExtractVecTest.case_nz_2:TExtractVecTest.case_nz_6:TExtractVecTest.case_nz_indexcol_nonzero:TExtractVecTest.case_nz_multi_fractal_dst:TExtractVecTest.case_nz_hif8:TExtractVecTest.case_nz_scalar_1:TExtractVecTest.case_nz_scalar_fp4_e2m1:TExtractVecTest.case_nz_scalar_fp4_e1m2" &
    _batch_idx=$((_batch_idx + 1))
    # ttri: 3 cases
    _batch_run ttri $((_batch_idx % _SLOTS)) \
      "TTRITest.case_float_128x128_upper_diag_n3:TTRITest.case_float_128x128_lower_diag_n3:TTRITest.case_float_128x128_upper_diag_0" &
    _batch_idx=$((_batch_idx + 1))
    # tquant: 2 cases
    _batch_run tquant $((_batch_idx % _SLOTS)) \
      "TQUANTTEST.case_int8_sym_fp32_128x128_nd:TQUANTTEST.case_int8_asym_fp32_128x128_nd" &
    _batch_idx=$((_batch_idx + 1))
    _batch_throttle

    # tinsert: 15 cases
    _batch_run tinsert $((_batch_idx % _SLOTS)) \
      "TInsertTest.case_acc2mat_1:TInsertTest.case_nz_1:TInsertTest.case_nd_1:TInsertTest.case_nz_2:TInsertTest.case_nz_6:TInsertTest.case_nd_vec_16:TInsertTest.case_nz_split_1:TInsertTest.case_nz_split_4:TInsertTest.case_nz_hif8_3:TInsertTest.case_nz_twoinput_bf16_1:TInsertTest.case_nz_twoinput_fp8e5_1:TInsertTest.case_nz_twoinput_fp4e1m2_2:TInsertTest.case_nz_dblinput_fp4e2m1_1:TInsertTest.case_nz_twoinput_fp4e2m1_1:TInsertTest.case_nz_dblinput_fp8e4_2" &
    _batch_idx=$((_batch_idx + 1))
    wait
    _collect_results

    # Run remaining (non-batch) cases in parallel across all slots (round-robin).
    # Pin each process to single-threaded BLAS: gen_data.py uses numpy which
    # defaults to spawning cpu_count() BLAS threads per process. With 4 parallel
    # processes that is 4*cores threads fighting for cores, ballooning per-case
    # time from 8.3s (serial) to 14.8s (4-way). Setting OMP/BLAS threads=1 makes
    # each gen_data use 1 core, so 4 processes use 4 cores total with no
    # contention. NPU execution is unaffected (it runs on the card, not CPU).
    _idx=0
    for _c in "${_cases[@]}"; do
      # Skip batch testcases — already handled by batch dispatch above
      _skip=0
      for _btc in $_batch_tcs; do
        if [[ "$_c" == *"-t $_btc "* ]]; then _skip=1; break; fi
      done
      [ $_skip -eq 1 ] && continue
      _slot=$(( _idx % _SLOTS ))
      _card=${_available_ids[$_slot]}
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 \
        PTO_BUILD_DIR=build_card${_card}_p0 ASCEND_DEVICE_ID=${_card} \
        python3 tests/script/run_st.py $ARGS -w -v a5 $_c &
      _idx=$((_idx + 1))
      if [ $(( _idx % _SLOTS )) -eq 0 ]; then wait; _collect_results; fi
    done
    wait
    _collect_results
  elif [ "$ENABLE_ALL" = "true" ]; then            # 所有用例
    python3 tests/script/build_st.py $ARGS -v a5 -t all
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tconcatdstidx
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tpartargmax
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tpartargmin
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tconcatidx
    python3 tests/script/run_st.py $ARGS -w -v a5 -t taxpy
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tfmod
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tfmods
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tsubs
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmaxs
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trems
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tlrelu
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tadd
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tadds
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tand
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tands
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tors
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tshls
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tcmps
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tcolexpandadd
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tcolexpandmax
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tcolexpandmin
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tcolmax
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tcolmin
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tdivs
    python3 tests/script/run_st.py $ARGS -w -v a5 -t texp
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tlog
    python3 tests/script/run_st.py $ARGS -w -v a5 -t texpands
    if [ "$IS_AUTO_MODE" = "false" ]; then
      # this testcase has to directly call CCE intrinsics now, which won't compile for auto mode;
      # besides, auto-sync doesn't work with CCE intrinsics
      python3 tests/script/run_st.py $ARGS -w -v a5 -t texpands_mat
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tpushpop_cv
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tpushpop_vc
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tpushpop_cv_nosplit
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tpushpop_vc_nosplit
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tpushpop_dir_both
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tpushpop_subtile
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tpushpop_subblock_dispatch
      python3 tests/script/run_st.py $ARGS -w -v a5 -t thistogram
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tcolgather
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tcolscatter
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tshrs
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tdequant
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tci
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tcolargmax
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tcolargmin
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tcolsum
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tcolprod
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tcvt
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tcvt_fp4_lut
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tcvt_fp4_lut_v2c
      python3 tests/script/run_st.py $ARGS -w -v a5 -t textract
      python3 tests/script/run_st.py $ARGS -w -v a5 -t textract_acc2vec
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tfillpad
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tgather
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tgatherb
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tload_mix
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tload_mx_ND_DN
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tload_mx_gmtensor
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tload_shape2d
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tmov
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tpartadd
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tpartmul
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tpartmax
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tpartmin
      python3 tests/script/run_st.py $ARGS -w -v a5 -t ttrans
      python3 tests/script/run_st.py $ARGS -w -v a5 -t ttrans_conv
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tscatter
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tquant
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert_zn
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert_acc2vec
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tmov_zz
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tmatmul
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tmatmul_mx
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tmov_acc2vec
      python3 tests/script/run_st.py $ARGS -w -v a5 -t txor
      python3 tests/script/run_st.py $ARGS -w -v a5 -t txors
      python3 tests/script/run_st.py $ARGS -w -v a5 -t mgather_gm2l1
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tcmp
    fi
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tload
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tload_mx_NZ
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmax
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmin
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmins
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmov_vect
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmov_mx
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmrgsort
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmul
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmadd
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmuls
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tor
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tpow
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tpows
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tprelu
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trem
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowexpand
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowexpanddiv
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowmax
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowmin
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowargmax
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowargmin
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowsum
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowprod
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trsqrt
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tsel
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tsels
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tpartsel
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tshl
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tshr
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tsort32
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tsqrt
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tstore
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tstore_acc2gm
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tadd_tdiv
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmul_tadds
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tsub_texp
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmuls_trowsum
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowexpand_tsqrt
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowexpand_trowsum
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowsum_trowexpand
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowexpand_tdiv
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmov_ub2l1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tneg
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tcolexpand
    python3 tests/script/run_st.py $ARGS -w -v a5 -t ttri
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tnot
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trelu
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmov_acc2mat
    python3 tests/script/run_st.py $ARGS -w -v a5 -t mgather
    python3 tests/script/run_st.py $ARGS -w -v a5 -t mscatter
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tconcat
    python3 tests/script/run_st.py $ARGS -w -v a5 -t textract_vec
    python3 tests/script/run_st.py $ARGS -w -v a5 -t textract_ndto2xnz
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tinterleave
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tdeinterleave
  fi
fi

if [ "$ENABLE_KIRIN9030" = "true" ]; then
  python3 tests/script/build_st.py $ARGS -v kirin9030 -t all
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t taxpy
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tfmod
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tfmods
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tsubs
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tmaxs
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tci
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tmatmul
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tload
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tmrgsort
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t ttrans
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tstore
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t textract
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t textract_compact
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trowsum
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trowprod
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trowmax
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trowargmax
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trowmin
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trowargmin
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tcolsum
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tcolprod
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tcolmax
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tcolargmax
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tcolmin
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tcolargmin
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tsort32
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trowexpand
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trowexpandadd
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trowexpanddiv
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trowexpandmax
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trowexpandmin
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trowexpandmul
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trowexpandsub
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trowexpandexpdif
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tgather
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tcvt
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tsub
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tprelu
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tand
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tor
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tands
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tors
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t txor
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t txors
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tshl
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tshls
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tshr
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tshrs
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tadd
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tpartadd
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tpartmul
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tfillpad
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tmin
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tmax
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tmins
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tmov
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tmul
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tlrelu
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tdiv
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tcmps
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tsel
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tsels
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trsqrt
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tsqrt
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t texp
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tabs
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tlog
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trecip
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tdivs
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tmuls
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tadds
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tpartargmin
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tpartargmax
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tmov_vect
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t texpands
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tmov_acc2mat
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tassign
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tcmp
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tadd_tdiv
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tmul_tadds
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tsub_texp
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trems
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tscatter
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tneg
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tcolexpandadd
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tcolexpandmax
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tcolexpandmin
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tcolexpanddiv
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tcolexpandmul
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tcolexpandsub
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tcolexpandexpdif
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trowexpandbrcb
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t ttri
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tcolexpand
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tnot
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t trelu
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tquant
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tdequant
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tpow
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tpows
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tconcatdstidx
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tconcatidx
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tconcat
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t thistogram
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t textract_nd_vec
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t textract_vec
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t texpands_mat
fi

if [ "$ENABLE_KIRINX90" = "true" ]; then
  python3 tests/script/build_st.py $ARGS -v kirinX90 -t all
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tsubs
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tmaxs
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tci
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tmatmul
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tload_mix
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tload_shape2d
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tload
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tmrgsort
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t ttrans
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t ttrans_conv
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tstore
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t textract
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t textract_compact
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trowsum
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trowprod
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trowmax
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trowargmax
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trowmin
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trowargmin
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tcolsum
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tcolprod
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tcolmax
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tcolargmax
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tcolmin
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tcolargmin
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tsort32
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trowexpand
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trowexpandadd
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trowexpanddiv
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trowexpandmax
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trowexpandmin
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trowexpandmul
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trowexpandsub
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trowexpandexpdif
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tgather
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tcvt
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tsub
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trem
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tfmod
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tprelu
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tand
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tor
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tands
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tors
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t txor
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t txors
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tshl
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tshls
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tshr
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tshrs
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tadd
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tpartadd
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tpartmul
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tfillpad
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tmin
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tmax
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tmins
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tmov
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tmul
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tlrelu
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tdiv
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tcmps
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tmov_acc2vec
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tsel
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tsels
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trsqrt
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tsqrt
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t texp
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tabs
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tlog
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trecip
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tstore_acc2gm
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tdivs
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tmuls
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tadds
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t taxpy
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tgatherb
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tpartmin
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tpartmax
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tpartargmin
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tpartargmax
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tmov_vect
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t texpands
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tmov_acc2mat
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tassign
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tmov_ub2l1
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tcmp
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tadd_tdiv
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tmul_tadds
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tsub_texp
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tmuls_trowsum
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trowexpand_tsqrt
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trowexpand_trowsum
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trowsum_trowexpand
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trowexpand_tdiv
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trems
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tfmods
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tscatter
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tneg
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tcolexpandadd
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tcolexpandmax
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tcolexpandmin
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tcolexpanddiv
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tcolexpandmul
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tcolexpandsub
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tcolexpandexpdif
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trowexpandbrcb
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t ttri
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tcolexpand
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tnot
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t trelu
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tquant
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tdequant
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tpow
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tpows
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tconcatdstidx
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tconcatidx
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tconcat
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tinsert
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tinsert_acc2vec
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t textract_nd_vec
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t textract_vec
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tmov_zz
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t tmov_nd2nz
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t textract_acc2vec
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t texpands_mat
  python3 tests/script/run_st.py $ARGS -w -v kirinX90 -t timg2col
fi

if [ "$ENABLE_KIRIN9050" = "true" ]; then
  python3 tests/script/build_st.py $ARGS -v kirin9050 -t all
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tsubs
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tmaxs
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tci
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tmatmul
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tload_shape2d
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tload
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tload_dn2nz
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tmrgsort
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t ttrans
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t ttrans_conv
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tstore
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t textract
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t textract_compact
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trowsum
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trowprod
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trowmax
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trowargmax
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trowmin
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trowargmin
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tcolsum
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tcolprod
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tcolmax
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tcolargmax
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tcolmin
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tcolargmin
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tsort32
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trowexpand
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trowexpandadd
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trowexpanddiv
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trowexpandmax
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trowexpandmin
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trowexpandmul
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trowexpandsub
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trowexpandexpdif
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tgather
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tcvt
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tsub
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trem
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tfmod
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tprelu
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tand
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tor
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tands
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tors
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t txor
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t txors
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tshl
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tshls
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tshr
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tshrs
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tadd
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tpartadd
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tpartmul
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tfillpad
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tmin
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tmax
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tmins
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tmov
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tmul
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tlrelu
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tdiv
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tcmps
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tmov_acc2vec
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tsel
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tsels
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trsqrt
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tsqrt
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t texp
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tabs
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tlog
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trecip
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tdivs
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tmuls
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tadds
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t taxpy
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tgatherb
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tpartmin
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tpartmax
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tpartargmin
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tpartargmax
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tmov_vect
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t texpands
  # python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tmov_acc2mat
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tassign
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tmov_ub2l1
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tcmp
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tadd_tdiv
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tmul_tadds
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tsub_texp
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tmuls_trowsum
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trowexpand_tsqrt
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trowexpand_trowsum
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trowsum_trowexpand
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trowexpand_tdiv
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trems
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tfmods
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tscatter
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tneg
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tcolexpandadd
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tcolexpandmax
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tcolexpandmin
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tcolexpanddiv
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tcolexpandmul
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tcolexpandsub
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tcolexpandexpdif
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trowexpandbrcb
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t ttri
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tcolexpand
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tnot
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t trelu
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tquant
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tdequant
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tpow
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tpows
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tconcat
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tinsert
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tinsert_acc2vec
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t textract_nd_vec
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t textract_vec
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tmov_zz
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tmov_nd2nz
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t textract_acc2vec
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tmatmul_new
  python3 tests/script/run_st.py $ARGS -w -v kirin9050 -t tmatmul_new_full
fi

if [ "$ENABLE_COMM" = "true" ]; then
  if [ -z "$ST_PART" ] || [ "$ST_PART" = "3" ]; then
    if [ "$ENABLE_A3" = "true" ]; then
      python3 tests/script/run_st.py $ARGS -v a3 -t comm/tput
      python3 tests/script/run_st.py $ARGS -v a3 -t comm/tget
      python3 tests/script/run_st.py $ARGS -v a3 -t comm/tnotify
      python3 tests/script/run_st.py $ARGS -v a3 -t comm/twait
      python3 tests/script/run_st.py $ARGS -v a3 -t comm/ttest
      python3 tests/script/run_st.py $ARGS -v a3 -t comm/tgather
      python3 tests/script/run_st.py $ARGS -v a3 -t comm/tscatter
      python3 tests/script/run_st.py $ARGS -v a3 -t comm/treduce
      python3 tests/script/run_st.py $ARGS -v a3 -t comm/tbroadcast
      python3 tests/script/run_st.py $ARGS -v a3 -t comm/tput_async
      python3 tests/script/run_st.py $ARGS -v a3 -t comm/tget_async
    fi
  fi
fi


# Collect any result files left by calls that ran in a subshell but were not
# followed by a wait. The A3 / Kirin / comm paths run serially and never call
# _collect_results, so without this line their results would be dropped and the
# summary block below would be skipped entirely (fail-open).
_collect_results
if [ "$TOTAL_CASES" -gt 0 ]; then
  echo "============================================================"
  echo "[SUMMARY] total=${TOTAL_CASES} passed=$((TOTAL_CASES - FAILED_CASES)) failed=${FAILED_CASES}"
  # Run the count check BEFORE the failure exit: when a case fails, its
  # [CASES] line is still the evidence that the guard saw the right number of
  # cases, and pre_smoke.sh needs that line to distinguish "guard ran and
  # agreed" from "guard never ran".
  _counts_ok=1
  _check_case_counts || _counts_ok=0
  if [ "$FAILED_CASES" -gt 0 ]; then
    _cleanup_results_dir
    echo "[SUMMARY] ${FAILED_CASES} case(s) failed" >&2
    exit 1
  fi
  if [ "$_counts_ok" -eq 0 ]; then
    _cleanup_results_dir
    echo "[SUMMARY] case count guard failed" >&2
    exit 1
  fi
  echo "[SUMMARY] all cases passed"
fi
_cleanup_results_dir
if [ -n "$CASE_COUNT_FILE" ]; then rm -f "$CASE_COUNT_FILE"; fi
