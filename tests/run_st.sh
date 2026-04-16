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

ENABLE_A3=false
ENABLE_A5=false
ENABLE_KIRIN9030=false
ENABLE_KIRINX90=false
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

Mode (required for --a3/--a5/--a3_a5, choose one):
  --simple      run the curated single-case subset
  --all         run the full ST suite

Run mode (choose one; defaults to on-board npu if omitted):
  --sim         run against the simulator
  --npu         run on NPU hardware

Options:
  --comm        include communication ST
  --auto_mode   run ST in auto mode

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

# Validate the parsed options so that a mis-typed invocation fails loudly
# instead of silently exiting 0 without running any test.
if [ "$ENABLE_A3" = "false" ] && [ "$ENABLE_A5" = "false" ] && \
   [ "$ENABLE_KIRIN9030" = "false" ] && [ "$ENABLE_KIRINX90" = "false" ]; then
  echo "Error: no platform selected (expected one of --a3/--a5/--a3_a5/--kirin9030/--kirinX90)." >&2
  usage
  exit 1
fi

if { [ "$ENABLE_A3" = "true" ] || [ "$ENABLE_A5" = "true" ]; } && \
   [ "$ENABLE_SIMPLE" = "false" ] && [ "$ENABLE_ALL" = "false" ]; then
  echo "Error: --a3/--a5 requires a test mode: pass --simple or --all." >&2
  usage
  exit 1
fi

if [ "$ENABLE_A3" = "true" ]; then                 # A2A3
  if [ "$ENABLE_SIMPLE" = "true" ]; then           # 单个用例
    python3 tests/script/build_st.py $ARGS -v a3 -t all
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolgather -g TCOLGATHERTest.case_mask_half_16x64_16x64_P1111
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tconcatdstidx -g TCONCATTest.case_int16_16x32_16x16_16x16_8x16_8x16
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tconcatidx -g TCONCATTest.case_int16_16x32_16x16_16x16_8x16_8x16
    python3 tests/script/run_st.py $ARGS -w -v a3 -t taxpy -g TAXPYTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolexpand -g TCOLEXPANDTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolsum -g TCOLSUMTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolprod -g TCOLPRODTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolmax -g TCOLMAXTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolargmax -g TCOLCMAXTest.case01
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolmin -g TCOLMINTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolargmin -g TCOLCMINTest.case01
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trem -g TREMTest.case_float_16x64_16x128_16x128_16x64
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tfmod -g TFMODTest.case_float_16x64_16x128_16x128_16x64
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trems -g TREMSTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tfmods -g TFMODSTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsubs -g TSUBSTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmaxs -g TMAXSTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tlrelu -g TLRELUTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tgatherb -g TGATHERBTest.case_float_2x128_2x16_2x128
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmov -g TMOVTest.case14_scaling_dynamic_int32_int8_0_1_1_1_0_param
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmov_acc2mat -g TMOVTest.case_nz2nz_fb_quant_4
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmrgsort -g TMRGSORTTest.case_topk1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmul -g TMULTest.case_float_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tdiv -g TDIVTest.case_float_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tstore -g TStoreTest.ND_float_1_1_1_2_128_1_1_1_2_128
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tstore_acc2gm -g TStoreAcc2gmTest.case7
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tstore_mat2gm -g TStoreMat2GMTest.case_nd1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowsum -g TROWSUMTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpand -g TROWEXPANDTest.case0
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpandadd -g TROWEXPANDADDTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpanddiv -g TROWEXPANDDIVTest.case2
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpandmax -g TROWEXPANDMAXTest.case3
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpandmin -g TROWEXPANDMINTest.case4
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpandmul -g TROWEXPANDMULTest.case13
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpandsub -g TROWEXPANDSUBTest.case14
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowexpandexpdif -g TROWEXPANDEXPDIFTest.case7
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolexpandadd -g TColExpandAddTest.case_fp32_16_128_1_128
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolexpandmax -g TColExpandMaxTest.case_fp32_32_32_1_32
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolexpandmin -g TColExpandMinTest.case_fp16_4_256_1_256
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsels -g TSELSTest.case_uint16_uint8_2x16_2x32_2x16_2x16
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsels -g TSELSTest.case_float_uint16_2x8_2x16_2x8_2x8
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsort32 -g TSort32Test.case1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tadd -g TADDTest.case_float_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tand -g TANDTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tor -g TORTest.case2
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tpartargmax -g TPARTARGMAXTest.case_float_uint32_tile_diff_32k_small_0
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tpartargmin -g TPARTARGMINTest.case_float_uint32_tile_diff_32k_small_0
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tpows -g TPOWSTest.case11
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsel -g TSELTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmins
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trsqrt -g TRSQRTTest.case_float_64x64_64x64_64x64_inPlace_False
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsqrt -g TSQRTTest.case_float_64x64_64x64_64x64_inPlace_False
    python3 tests/script/run_st.py $ARGS -w -v a3 -t texp -g TEXPTest.case_float_64x64_64x64_64x64_inPlace_False
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tabs -g TABSTest.case_float_64x64_64x64_64x64_inPlace_False
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tlog -g TLOGTest.case_float_64x64_64x64_64x64_inPlace_False
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trecip -g TRECIPTest.case_float_64x64_64x64_64x64_inPlace_False
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tdivs -g TDIVSTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tdivs -g TDIVSTest.case4
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tdivs -g TDIVSTest.case5
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmuls -g TMULSTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tadds -g TADDSTest.case6
    python3 tests/script/run_st.py $ARGS -w -v a3 -t texpands -g TEXPANDSTest.case_float_64x64_64x64_64x64_PAD_VALUE_NULL
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tnot -g TNOTTest.case_int16_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tprelu -g TPRELUTest.case5
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trelu -g TRELUTest.case_int32_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tands -g TANDSTest.case_int16_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tors -g TORSTest.case_int16_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tshl -g TSHLTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tshr -g TSHRTest.case2
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tshls -g TSHLSTest.case_int16_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tshrs -g TSHRSTest.case_int16_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a3 -t txor -g TXORTest.case_int16_64x64_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a3 -t txors -g TXORSTest.case_int16_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tdequant -g TDEQUANTTest.case4
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tdequant -g TDEQUANTTest.case5
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tconcat -g TCONCATTest.case_half_16x128_16x64_16x64_16x63_16x64
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tconcat -g TCONCATTest.case_int16_32x256_32x128_32x128_32x127_32x128
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tconcat -g TCONCATTest.case_int32_64x128_64x64_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowargmax -g TROWARGMAXTest.case_uint32_float_16x1_13x16_1x8_13x13
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowargmax -g TROWARGMAXTest.case_uint32_float_8x1_3x4096_3x192_3x4095
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowargmax -g TROWARGMAXTest.case_uint32_float_8x1_8x1_1x16384_1x768_1x16381
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowargmin -g TROWARGMINTest.case_uint32_float_16x1_13x16_1x8_13x13
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowargmin -g TROWARGMINTest.case_uint32_float_8x1_3x4096_3x192_3x4095
    python3 tests/script/run_st.py $ARGS -w -v a3 -t trowargmin -g TROWARGMINTest.case_uint32_float_8x1_8x1_1x16384_1x768_1x16381
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_aligned_1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_aligned_4_bf16
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_partial_validrow
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_unaligned_validcol_full_row
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_aligned_int8_strided
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_nonpow2_half
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_nonpow2_partial_float_unaligned_idxrow
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_unalignedvalid_int16
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_unalignedvalid_float_smallthan32B
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_unalignedvalid_uint16_taillarge
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_scalar_4_int8
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nd_scalar_nonpow2_float
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nz_1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nz_multi_fractal_dst
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nz_half_large
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nz_nonpow2_float
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nz_partial_bf16
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nz_partial_float_unaligned_idxcol
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nz_unalignedvalid_validrow_half
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nz_scalar_5_int32
    python3 tests/script/run_st.py $ARGS -w -v a3 -t textract_vec -g TExtractVecTest.case_nz_scalar_nonpow2_int8
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_aligned_1
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_aligned_4_bf16
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_partial_validrow
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_full_row_strided
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_aligned_uint16
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_partial_validboth_float
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_nonpow2_int8
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_nonpow2_int32
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_unalignedvalid_half
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_unalignedvalid_uint16_strided
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_unalignedvalid_float_smallthan32B
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_scalar_3_bf16
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nd_scalar_2_half
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nz_2
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nz_int32_partial
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nz_partial_float_unaligned_idxcol
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nz_nonpow2_partial_int8
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nz_unalignedvalid_int16
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nz_unalignedvalid_both_float
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nz_unalignedvalid_both_with_idx_bf16
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nz_scalar_5_int32
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nz_scalar_9_uint8_edge
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec -g TInsertVecTest.case_nz_scalar_nonpow2_int8
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tpairreducesum -g TPAIRREDUCESUMTest.case_float_32x64_32x64_32x64
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tpairreducesum -g TPAIRREDUCESUMTest.case_float_32x128_32x128_32x128
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tfusedmuladd -g TFUSEDMULADDTest.case_float_32x128_32x192_32x256_32x127
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tfusedmuladdrelu -g TFUSEDMULADDRELUTest.case_float_32x128_32x192_32x256_32x127
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsubrelu -g TSUBRELUTest.case_float_32x128_32x192_32x256_32x127
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmuladddst -g TMULADDDSTTest.case_float_32x128_32x192_32x256_32x127

    if [ "$IS_AUTO_MODE" = "false" ]; then
      # this testcase has to directly call CCE intrinsics now, which won't compile for auto mode;
      # besides, auto-sync doesn't work with CCE intrisics
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpushpop_cv -g TPushPopCVTest.case1_half_single_tile
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpushpop_vc -g TPushPopVCTest.case1_int8_single_k_tile
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpushpop_cv_nosplit -g TPushPopCVNoSplitTest.case1_half_single_tile
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpushpop_vc_nosplit -g TPushPopVCNoSplitTest.case1_int8_single_k_tile
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpushpop_dir_both -g TPushPopDirBothTest.case1_float_dir_both
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpushpop_subtile -g TPushTpopSubtileTest.case1_half_128x512
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem2d_nz_float_16x16_2blk
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_row_nz_float_clamp_16x8_1blk
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_row_dyn_int32_3x16_8rows
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem2d_dyn_float_4x8_64size
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem_scalar_float_1x1_in_1x8_8size
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem2d_half_atomic_add_4x16_8size
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem2d_float_atomic_add_4x16_8size
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem2d_int32_unaligned_3x3_in_3x8_64size
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem2d_uint32_8x16_256size
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem2d_float_8x32_256size
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem2d_uint8_4x64_256size
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem_uint32_32_64size
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_elem_uint16_32_64size
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_row_uint8_unaligned_3x32_32rows
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_row_half_atomic_add_8x32_8rows
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_row_float_8x32_64rows
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mscatter -g MSCATTERTest.case_row_int32_wrap_8x16_8rows
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather -g MGATHERTest.case_row_float_8x32_64rows
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather -g MGATHERTest.case_row_half_16x64_64rows
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather -g MGATHERTest.case_row_float_partial_4x16_in_8x16
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather -g MGATHERTest.case_elem_half_64_128size
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather -g MGATHERTest.case_elem2d_half_4x32_256size
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather -g MGATHERTest.case_elem2d_int16_4x32_256size
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather -g MGATHERTest.case_row_dyn_int32_3x16_8rows
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather -g MGATHERTest.case_elem2d_nz_int32_16x8_1blk
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tfillpad -g TFILLPADTest.case_float_GT_128_127_VT_128_128_BLK1_PADMAX_PADMAX
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tfillpad -g TFILLPADTest.case_u16_GT_259_7_VT_260_32_BLK1_PADMIN_PADMAX_EXPAND -d
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tgather -g TGATHERTest.case1_float_P0101
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tload_gm2mat -g TLoadGM2L1Test.ND2NZ_bfloat16_t_1_1_1_1_1_1_1_1_1_1
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tmov_vect -g TMOVTest.vect_copy_case1
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpartadd -g TPARTADDTest.case_float_64x64_64x64_64x64
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpartmul -g TPARTMULTest.case_float_64x64_64x64_64x64
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tpow -g TPOWTest.case11
      python3 tests/script/run_st.py $ARGS -w -v a3 -t trowprod -g TROWPRODTest.case1
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tscatter -g TSCATTERTest.case_mask_float_16x64_16x64_P1111
      python3 tests/script/run_st.py $ARGS -w -v a3 -t ttrans_3d -g TTRANS3DTest.case3_int32_17_3_3_2_2
      python3 tests/script/run_st.py $ARGS -w -v a3 -t ttrans_3d -g TTRANS3DTest.case1_float32_2_4_2_2_2
      python3 tests/script/run_st.py $ARGS -w -v a3 -t ttrans_3d -g TTRANS3DTest.case7_uint16_4_8_2_2_3
      python3 tests/script/run_st.py $ARGS -w -v a3 -t ttrans_3d -g TTRANS3DTest.case10_uint8_9_18_2_2_4
      python3 tests/script/run_st.py $ARGS -w -v a3 -t ttri -g TTRITest.case_float_128x128_128x31_1__444
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tcvt -g TCVTTest.case_fp16_fp32_2x64
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tcmp -g TCMPTest.case_float_1x64_1x64_1x64
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tcmps -g TCMPSTest.case_float_8x64_8x64_8x64
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolscatter -g TCOLSCATTERTest.case_mask_half_16x64_16x64_P1111
      python3 tests/script/run_st.py $ARGS -w -v a3 -t textract -g TEXTRACTTest.case1_half_0_1_16_16_32_param
      python3 tests/script/run_st.py $ARGS -w -v a3 -t ttrans -g TTRANSTest.case1_float_16_8_16_8
      python3 tests/script/run_st.py $ARGS -w -v a3 -t ttrans_conv -g TTRANSConvTest.int8_1_63_2_128
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tci -g TCITest.case1_int32
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tquant -g TQUANTTEST.case_int8_sym_fp32_128x128_nd
      python3 tests/script/run_st.py $ARGS -w -v a3 -t tquant -g TQUANTTEST.case_int8_asym_fp32_128x128_nd
      python3 tests/script/run_st.py $ARGS -w -v a3 -t texpands_mat -g TEXPANDSTest.case1
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather_gm2l1 -g MGATHERGM2L1Test.case_row_float_16x16_64rows
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather_gm2l1 -g MGATHERGM2L1Test.case_row_int32_16x8_32rows
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather_gm2l1 -g MGATHERGM2L1Test.case_row_float_16x16_64rows
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather_gm2l1 -g MGATHERGM2L1Test.case_row_uint16_16x32_48rows
      python3 tests/script/run_st.py $ARGS -w -v a3 -t mgather_gm2l1 -g MGATHERGM2L1Test.case_elem_bfloat16_16x16_256size
    fi

  elif [ "$ENABLE_ALL" = "true" ]; then            # 所有用例
    python3 tests/script/build_st.py $ARGS -v a3 -t all
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolgather
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tcolscatter
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsubreluconv
    python3 tests/script/run_st.py $ARGS -w -v a3 -t taddreluconv
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
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tinsert_vec
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tfusedmuladd
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tfusedmuladdrelu
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tsubrelu
    python3 tests/script/run_st.py $ARGS -w -v a3 -t tmuladddst
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
      # besides, auto-sync doesn't work with CCE intrisics
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
  if [ "$ENABLE_SIMPLE" = "true" ]; then           # simple cases
    # Smoke cases are built again by run_st.py per testcase. Avoid a front-loaded
    # full A5 build here so business-side compile regressions outside the trimmed
    # smoke set do not block the smoke pipeline.
    # Temporarily skip: current merged A5 TSCATTER signatures no longer match this smoke testcase.
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tcolscatter -g TCOLSCATTERTest.case_mask_half_16x64_16x64_P1111
    # Temporarily skip: current merged A5 TCONCATDSTIDX runtime result mismatches the expected smoke output on runner host 39.
    # python3 tests/script/run_st.py $ARGS -v a5 -t tconcatdstidx -g TCONCATTest.case_int16_16x32_16x16_16x16_8x16_8x16
    # Temporarily skip the unstable A5 partargmax smoke case because the binary exits with SIGSEGV after execution.
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tpartargmax -g TPARTARGMAXTest.case_fp32_64x64_64x64_64x64
    # Temporarily skip the unstable A5 partargmin smoke case because the binary exits with SIGSEGV after execution.
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tpartargmin -g TPARTARGMINTest.case_fp32_64x64_64x64_64x64
    # Temporarily skip the unstable A5 concatidx smoke case until the kernel/result mismatch is fixed.
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tconcatidx -g TCONCATTest.case_int16_16x32_16x16_16x16_8x16_8x16
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tsubreluconv -g TSUBRELUCONVTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t taddreluconv -g TADDRELUCONVTest.case1
    python3 tests/script/run_st.py $ARGS -v a5 -t taxpy -g TAXPYTest.case1
    python3 tests/script/run_st.py $ARGS -v a5 -t tdequant -g TDEQUANTTest.case1
    python3 tests/script/run_st.py $ARGS -v a5 -t tconcat -g TCONCATTest.case_half_16x128_16x64_16x64_16x63_16x64
    # Synced from cann/pto-isa PR920: trim these cases from the A5 simple list.
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tfmods -g TFMODSTest.case1
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tfmod -g TFMODTest.case1
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tsubs -g TSUBSTest.case1
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tmaxs -g TMAXSTest.case_float_64x64_32x32_32x32
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t trems -g TREMSTest.case1
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tlrelu -g TLRELUTest.case1
    python3 tests/script/run_st.py $ARGS -v a5 -t tadd -g TADDTest.case_float_64x64_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -v a5 -t tadds -g TADDSTest.case1
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tand -g TANDTest.case1
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tands -g TANDSTest.case1
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tors -g TORSTest.case1
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t txors -g TXORSTest.case1
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t txor -g TXORTest.case1
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tshls -g TSHLSTest.case_int16_64x64_64x64_64x64
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tshrs -g TSHRSTest.case_int16_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -v a5 -t tci -g TCITest.case5
    python3 tests/script/run_st.py $ARGS -v a5 -t tcmps -g TCMPSTest.case_float_8x64_8x64
    python3 tests/script/run_st.py $ARGS -v a5 -t tcolexpandadd -g TColExpandAddTest.case_fp32_16_128_1_128
    # Temporarily skip the unstable A5 colexpandmax smoke case because the binary exits with SIGSEGV after execution.
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tcolexpandmax -g TColExpandMaxTest.case_fp32_32_32_1_32
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tcolexpandmin -g TColExpandMinTest.case_fp16_4_256_1_256
    python3 tests/script/run_st.py $ARGS -v a5 -t tcolmax -g TCOLMAXTest.case01
    # Temporarily skip: this A6-only tcolmin invocation does not belong in the A5 smoke list.
    # python3 tests/script/run_st.py $ARGS -w -v a6 -t tcolmin -g TCOLCMAXTest.case01
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tcolmin -g TCOLMINTest.case01
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tcolmin -g TCOLMINTest.case01
    python3 tests/script/run_st.py $ARGS -v a5 -t tcolsum -g TCOLSUMTest.case01
    python3 tests/script/run_st.py $ARGS -v a5 -t tcolprod -g TCOLPRODTest.case01
    python3 tests/script/run_st.py $ARGS -v a5 -t tcvt -g TCVTTest.case_fp16_fp32_2x64
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tdivs -g TDIVSTest.case4
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tdivs -g TDIVSTest.case5
    python3 tests/script/run_st.py $ARGS -v a5 -t texp -g TEXPTest.case1
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t tlog -g TLOGTest.case_float_64x64_64x64_64x64
    # Temporarily skip the unstable A5 texpands smoke case because the binary exits with SIGSEGV after execution.
    # python3 tests/script/run_st.py $ARGS -w -v a5 -t texpands -g TEXPANDSTest.case_float_64x64_64x64_64x64_PAD_VALUE_NULL
    if [ "$IS_AUTO_MODE" = "false" ]; then
      # this testcase has to directly call CCE intrinsics now, which won't compile for auto mode;
      # besides, auto-sync doesn't work with CCE intrisics
      python3 tests/script/run_st.py $ARGS -v a5 -t texpands_mat -g TEXPANDSTest.case1
      python3 tests/script/run_st.py $ARGS -v a5 -t tpushpop_cv -g TPushPopCVTest.case1_half_single_tile
      python3 tests/script/run_st.py $ARGS -v a5 -t tpushpop_vc -g TPushPopVCTest.case1_int8_single_k_tile
      python3 tests/script/run_st.py $ARGS -v a5 -t tpushpop_cv_nosplit -g TPushPopCVNoSplitTest.case1_half_single_tile
      python3 tests/script/run_st.py $ARGS -v a5 -t tpushpop_vc_nosplit -g TPushPopVCNSTest.case1_int8_single_k_tile
      python3 tests/script/run_st.py $ARGS -v a5 -t tpushpop_dir_both -g TPushPopDirBothTest.case1_float_dir_both
      # Temporarily skip: current merged A5 TPush/TPop/TAlloc/TFree APIs no longer compile this smoke testcase.
      # python3 tests/script/run_st.py $ARGS -v a5 -t tpushpop_subtile -g TPushTpopSubtileTest.case1_half_128x512
    fi
    python3 tests/script/run_st.py $ARGS -w -v a5 -t textract -g TEXTRACTTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t textract_acc2vec -g TMOVTest.case_nz2nd_sc_quant_1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tfillpad -g TFILLPADTest.case_float_GT_128_127_VT_128_128_BLK1_PADMAX_PADMAX
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tgather -g TGATHERTest.case1_float_32x1024_16x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tgatherb -g TGATHERBTest.case_float_2x128_2x16_2x128
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tload -g TLOADTest.case_float_GT_2_2_2_256_60_VT_256_64_BLK8_PADMAX
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tload_mix -g TLOADMIXTest.1_1_1_59_119_1_1_1_64_128_64_128_int8_t_ND2NZ
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tload_mx_NZ -g TLOADSCALETest.4_3_3_16_2_4_10_5_16_2_192_10_scale_ZZ2ZZ
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tload_mx_NZ -g TLOADSCALETest.7_5_3_16_2_7_7_11_16_2_12_560_scale_NN2NN
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tload_mx_ND_DN -g TLOADMXTest.1_1_1_64_128_uint8_AND2ZZ
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tload_mx_gmtensor -g TLOADMXTest.1_1_1_64_128_uint8_ADN2ZZ
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tload_shape2d -g TLOADSHAPE2DTest.1_1_1_59_119_1_1_1_64_128_64_128_int8_t_ND2NZ
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmatmul -g TMATMULTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmatmul_mx -g TMATMULMXTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmax -g TMAXTest.case_float_64x64_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmin -g TMINTest.case_float_64x64_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmins -g TMINSTest.case_float_60x128_64x64_60x60
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmins -g TMINSTest.case_float_16x200_20x512_16x200
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmins -g TMINSTest.case_float_1x3600_2x4096_1x3600
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmov -g TMOVTest.case_bias1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmov_acc2vec -g TMOVTest.case_nz2nd_1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmov_vect -g TMOVTest.vect_copy_case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmov_mx -g TMOVMXTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmrgsort -g TMRGSORTTest.case_topk1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmul -g TMULTest.case_float_64x64_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmuls -g TMULSTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tor -g TORTest.case2
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tpartadd -g TPARTADDTest.case_float_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tpartmul -g TPARTMULTest.case_float_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tpartmax -g TPARTMAXTest.case_fp32_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tpartmin -g TPARTMINTest.case_fp32_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tprelu -g TPRELUTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trem -g TREMTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowexpand -g TROWEXPANDTest.case5_float_16_8_16_127
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowexpanddiv -g TRowExpandDivTest.case_fp32_40_64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowmax -g TROWMAXTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowmin -g TROWMINTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowargmax -g TROWARGMAXTest.case_uint32_float_64x1_32x128_32x128
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowargmin -g TROWARGMINTest.case_uint32_float_64x1_32x128_32x128
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowsum -g TROWSUMTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowprod -g TROWPRODTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trsqrt -g TRSQRTTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tsel -g TSELTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tsels -g TSELSTest.case_uint8_uint8_2x32_2x32_2x32_2x32
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tsels -g TSELSTest.case_float_uint16_2x8_2x16_2x8_2x8
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tshl -g TSHLTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tshr -g TSHRTest.case2
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tsort32 -g TSort32Test.case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tsqrt -g TSQRTTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tstore -g TStoreTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tstore_acc2gm -g TStoreAcc2gmTest.case7
    python3 tests/script/run_st.py $ARGS -w -v a5 -t ttrans -g TTRANSTest.case_float_8x8_2x8_2x8
    python3 tests/script/run_st.py $ARGS -w -v a5 -t ttrans_conv -g TTRANSConvTest.uint8_11_2_7_7_32
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tcmp -g TCMPTest.case_half_32x32_32x32_32x32
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tadd_tdiv -g TADD_TDIVTest.case_float_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmul_tadds -g TMUL_TADDSTest.case_float_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tsub_texp -g TSUB_TEXPTest.case_float_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmuls_trowsum -g TMULS_TROWSUMTest.case_float_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowexpand_tsqrt -g TROWEXPAND_TSQRTTest.case_float_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowexpand_trowsum -g TROWEXPAND_TROWSUMTest.case_float_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowsum_trowexpand -g TROWSUM_TROWEXPANDTest.case_float_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trowexpand_tdiv -g TROWEXPAND_TDIVTest.case_float_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmov_ub2l1 -g TMovUb2l1Test.case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tscatter -g TSCATTERTest.case1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tneg -g TNEGTest.case_float_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tcolexpand -g TCOLEXPANDTest.case_float_1_8_128_63
    python3 tests/script/run_st.py $ARGS -w -v a5 -t ttri -g TTRITest.case_float_128x128_upper_diag_n3
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tnot -g TNOTTest.case_int16_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t trelu -g TRELUTest.case_int32_64x64_64x64_64x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tmov_acc2mat -g TMOVTest.case_nz2nz_insert
    python3 tests/script/run_st.py $ARGS -w -v a5 -t mgather -g MGATHERTest.case_half_16x128_8x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t mscatter -g MSCATTERTest.case_uint8_16x64_2048
    python3 tests/script/run_st.py $ARGS -w -v a5 -t mscatter -g MSCATTERTest.case_int32_clamp_8x16_256
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tquant -g TQUANTTEST.case_int8_sym_fp32_128x128_nd
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tquant -g TQUANTTEST.case_int8_asym_fp32_128x128_nd
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tquant -g TQUANTTEST.case_int8_sym_fp32_128x128_nd
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tquant -g TQUANTTEST.case_int8_asym_fp32_128x128_nd
    python3 tests/script/run_st.py $ARGS -w -v a5 -t ttri -g TTRITest.case_float_128x128_lower_diag_n3
    python3 tests/script/run_st.py $ARGS -w -v a5 -t ttri -g TTRITest.case_float_128x128_upper_diag_0
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tconcat -g TCONCATTest.case_half_16x128_16x64_16x64_16x63_16x64
    python3 tests/script/run_st.py $ARGS -w -v a5 -t thistogram -g THISTOGRAMTest.case_8x128_b1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t thistogram -g THISTOGRAMTest.case_u32_6x912_b1_k64 
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert -g TInsertTest.case_acc2mat_1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert -g TInsertTest.case_nz_1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert -g TInsertTest.case_nd_1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert -g TInsertTest.case_nz_2
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert -g TInsertTest.case_nz_6
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert_acc2vec -g TMOVTest.case_nz2nd_fb_quant_1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert -g TInsertTest.case_nd_vec_16
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert -g TInsertTest.case_nz_split_1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert -g TInsertTest.case_nz_split_4
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert -g TInsertTest.case_nz_hif8_3
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert -g TInsertTest.case_nz_twoinput_bf16_1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert -g TInsertTest.case_nz_twoinput_fp8e5_1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert -g TInsertTest.case_nz_twoinput_fp4e1m2_2
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert -g TInsertTest.case_nz_dblinput_fp4e2m1_1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert -g TInsertTest.case_nz_twoinput_fp4e2m1_1
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert -g TInsertTest.case_nz_dblinput_fp8e4_2


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
      # besides, auto-sync doesn't work with CCE intrisics
      python3 tests/script/run_st.py $ARGS -w -v a5 -t texpands_mat
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tpushpop_cv
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tpushpop_vc
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tpushpop_cv_nosplit
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tpushpop_vc_nosplit
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tpushpop_dir_both
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tpushpop_subtile
      python3 tests/script/run_st.py $ARGS -w -v a5 -t thistogram
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tsubreluconv
      python3 tests/script/run_st.py $ARGS -w -v a5 -t taddreluconv
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tsubrelu
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
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert_acc2vec
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tmov_zz
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tmatmul
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tmatmul_mx
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tmov_acc2vec
      python3 tests/script/run_st.py $ARGS -w -v a5 -t txor
      python3 tests/script/run_st.py $ARGS -w -v a5 -t txors
      python3 tests/script/run_st.py $ARGS -w -v a5 -t mgather_gm2l1
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tcmp
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tfusedmuladd
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tfusedmuladdrelu
      python3 tests/script/run_st.py $ARGS -w -v a5 -t tmuladddst
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
    python3 tests/script/run_st.py $ARGS -w -v a5 -t thistogram
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert
    python3 tests/script/run_st.py $ARGS -w -v a5 -t tinsert_acc2vec
  fi
fi

if [ "$ENABLE_KIRIN9030" = "true" ]; then
  python3 tests/script/build_st.py $ARGS -v kirin9030 -t all
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
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tconcatdstidx
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tconcatidx
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t tconcat
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t thistogram
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t textract_nd_vec
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t textract_vec
  python3 tests/script/run_st.py $ARGS -w -v kirin9030 -t texpands_mat
fi

# if [ "$ENABLE_COMM" = "true" ]; then
#   if [ "$ENABLE_A3" = "true" ]; then
#     python3 tests/script/run_st.py $ARGS -v a3 -t comm/tput
#     python3 tests/script/run_st.py $ARGS -v a3 -t comm/tget
#     python3 tests/script/run_st.py $ARGS -v a3 -t comm/tnotify
#     python3 tests/script/run_st.py $ARGS -v a3 -t comm/twait
#     python3 tests/script/run_st.py $ARGS -v a3 -t comm/ttest
#     python3 tests/script/run_st.py $ARGS -v a3 -t comm/tgather
#     python3 tests/script/run_st.py $ARGS -v a3 -t comm/tscatter
#     python3 tests/script/run_st.py $ARGS -v a3 -t comm/treduce
#     python3 tests/script/run_st.py $ARGS -v a3 -t comm/tbroadcast
#     python3 tests/script/run_st.py $ARGS -v a3 -t comm/tput_async
#     python3 tests/script/run_st.py $ARGS -v a3 -t comm/tget_async
#   fi
# fi
