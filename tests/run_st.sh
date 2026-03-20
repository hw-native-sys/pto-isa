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
ENABLE_SIM=false
ENABLE_NPU=false
RUN_TYPE=sim
RUN_MODE="run_st.py -w"

if [ "$1" = "a3" ]; then
  ENABLE_A3=true
elif [ "$1" = "a5" ]; then
  ENABLE_A5=true
elif [ "$1" = "a3_a5" ]; then
  ENABLE_A3=true
  ENABLE_A5=true
elif [ "$1" = "kirin9030" ]; then
  ENABLE_KIRIN9030=true
elif [ "$1" = "kirinX90" ]; then
  ENABLE_KIRINX90=true
fi

if [ "$2" = "sim" ]; then
  RUN_TYPE=sim
elif [ "$2" = "npu" ]; then
  RUN_TYPE=npu
fi

if [ "$3" = "simple" ]; then
  ENABLE_SIMPLE=true
elif [ "$3" = "all" ]; then
  ENABLE_ALL=true
fi

if [ "$4" = "build_only" ]; then
  RUN_MODE=build_st.py
fi

if [ "$ENABLE_A3" = "true" ]; then                 # A2A3
  if [ "$ENABLE_SIMPLE" = "true" ]; then           # 单个用例
    python3 tests/script/build_st.py -r $RUN_TYPE -v a3 -t all
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t taxpy -g TAXPYTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcolexpand -g TCOLEXPANDTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcolsum -g TCOLSUMTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcolprod -g TCOLPRODTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trowprod -g TROWPRODTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcolmax -g TCOLMAXTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcolmin -g TCOLMINTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trem -g TREMTest.case_half_16x64_16x128_16x128_16x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tfmod -g TFMODTest.case_half_16x64_16x128_16x128_16x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trems -g TREMSTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tfmods -g TFMODSTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tsubs -g TSUBSTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tmaxs -g TMAXSTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tlrelu -g TLRELUTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tgatherb -g TGATHERBTest.case_float_2x128_2x16_2x128
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tci -g TCITest.case1_int32
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcvt -g TCVTTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tmov -g TMOVTest.case14_scaling_dynamic_int32_int8_0_1_1_1_0_param
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tmov_acc2mat -g TMOVTest.case_nz2nz_fb_quant_4
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t textract -g TEXTRACTTest.case1_half_0_1_param
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tmrgsort -g TMRGSORTTest.case_topk1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tmul -g TMULTest.case_float_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tdiv -g TDIVTest.case_float_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tstore -g TStoreTest.ND_float_1_1_1_2_128_1_1_1_2_128
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tstore_acc2gm -g TStoreAcc2gmTest.case7
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tstore_mat2gm -g TStoreMat2GMTest.case_nd1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcmps -g TCMPSTest.case_float_8x64_8x64_8x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trowsum -g TROWSUMTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trowexpand -g TROWEXPANDTest.case0
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trowexpandadd -g TROWEXPANDADDTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trowexpanddiv -g TROWEXPANDDIVTest.case2
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trowexpandmax -g TROWEXPANDMAXTest.case3
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trowexpandmin -g TROWEXPANDMINTest.case4
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trowexpandmul -g TROWEXPANDMULTest.case13
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trowexpandsub -g TROWEXPANDSUBTest.case14
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trowexpandexpdif -g TROWEXPANDEXPDIFTest.case7
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcolexpandadd -g TColExpandAddTest.case_fp32_16_128_1_128
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcolexpandmax -g TColExpandMaxTest.case_fp32_32_32_1_32
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcolexpandmin -g TColExpandMinTest.case_fp16_4_256_1_256
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tgather -g TGATHERTest.case1_float_P0101
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t ttrans -g TTRANSTest.case1_float_16_8_16_8
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tsels -g TSELSTest.case_uint16_uint8_2x16_2x32_2x16_2x16
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tsels -g TSELSTest.case_float_uint16_2x8_2x16_2x8_2x8
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tsort32 -g TSort32Test.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tadd -g TADDTest.case_float_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tand -g TANDTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tor -g TANDTest.case2
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tpartadd -g TPARTADDTest.case_float_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tpartmul -g TPARTMULTest.case_float_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tsel -g TSELTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tfillpad -g TFILLPADTest.case_float_GT_128_127_VT_128_128_BLK1_PADMAX_PADMAX
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tmins
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tload_gm2mat -g TLoadGM2L1Test.ND2NZ_bfloat16_t_1_1_1_1_1_1_1_1_1_1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trsqrt -g TRSQRTTest.case_float_64x64_64x64_64x64_inPlace_False
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tsqrt -g TSQRTTest.case_float_64x64_64x64_64x64_inPlace_False
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t texp -g TEXPTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tabs -g TABSTest.case_float_64x64_64x64_64x64_inPlace_False
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tlog -g TLOGTest.case_float_64x64_64x64_64x64_inPlace_False
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trecip -g TRECIPTest.case_float_64x64_64x64_64x64_inPlace_False
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tdivs -g TDIVSTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tdivs -g TDIVSTest.case4
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tdivs -g TDIVSTest.case5
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tmuls -g TMULSTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tadds -g TADDSTest.case6
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t texpands -g TEXPANDSTest.case_float_64x64_64x64_64x64_PAD_VALUE_NULL
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t texpands_mat -g TEXPANDSTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcmp -g TCMPTest.case_float_1x64_1x64_1x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tscatter -g TSCATTERTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t ttri -g TTRITest.case_float_128x128_128x31_1__444
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tnot -g TNOTTest.case_int16_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tprelu -g TPRELUTest.case5
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trelu -g TRELUTest.case_int32_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tands -g TANDSTest.case_int16_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tors -g TORSTest.case_int16_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tshl -g TSHLTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tshr -g TSHRTest.case2
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tshls -g TSHLSTest.case_int16_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tshrs -g TSHRSTest.case_int16_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t txor -g TXORSTest.case_int16_64x64_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t txors -g TXORSTest.case_int16_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tquant -g TQUANTTest.case_int8_sym_fp32_128x128_nd
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tquant -g TQUANTTest.case_int8_asym_fp32_128x128_nd
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tdequant -g TDEQUANTTest.case4
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tdequant -g TDEQUANTTest.case5
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tconcat -g TCONCATTest.case_half_16x128_16x64_16x64_16x63_16x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tconcat -g TCONCATTest.case_int16_32x256_32x128_32x128_32x127_32x128
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tconcat -g TCONCATTest.case_int32_64x128_64x64_64x64_64x64_64x64

  elif [ "$ENABLE_ALL" = "true" ]; then            # 所有用例
    python3 tests/script/build_st.py -r $RUN_TYPE -v a3 -t all
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t taxpy
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcolexpand
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcolsum
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcolprod
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trowprod
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcolmax
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcolmin
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trem
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trems
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tfmod
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tfmods
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tsubs
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tmaxs
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tlrelu
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcvt
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tmatmul
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tmov
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tmov_acc2mat
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t textract
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tmrgsort
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tstore
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tstore_acc2gm
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tstore_mat2gm
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trowsum
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trowexpand
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trowexpandadd
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trowexpanddiv
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trowexpandmax
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trowexpandmin
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trowexpandmul
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trowexpandsub
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trowexpandexpdif
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcolexpandadd
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcolexpandmax
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcolexpandmin
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tgather
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t ttrans
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tsort32
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tpartadd
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tpartmul
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tsel
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tload_gm2mat
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tload
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tadd
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tand
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tor
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tsels
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tmins
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tsub
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tci
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tgatherb
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t texp
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trsqrt
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tsqrt
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tabs
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tlog
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trecip
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t texpands
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t texpands_mat
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tcmp
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tscatter
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t ttri
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tnot
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tprelu
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t trelu
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tands
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tors
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tshl
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tshr
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tshls
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tshrs
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t txor
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t txors
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tquant
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tdequant
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a3 -t tconcat
    tconcat
  fi
fi

if [ "$ENABLE_A5" = "true" ]; then
  if [ "$ENABLE_SIMPLE" = "true" ]; then           # 单个用例
    python3 tests/script/build_st.py -r $RUN_TYPE -v a5 -t all
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t taxpy -g TAXPYTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tdequant -g TDEQUANTTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tconcat -g TCONCATTest.case_half_16x128_16x64_16x64_16x63_16x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tfmods -g TFMODSTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tfmod -g TFMODTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tsubs -g TSUBSTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmaxs -g TMAXSTest.case_float_64x64_32x32_32x32
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trems -g TREMSTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tlrelu -g TLRELUTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tadd -g TADDTest.case_float_64x64_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tadds -g TADDSTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tand -g TANDTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tands -g TANDSTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tors -g TORSTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t txors -g TXORSTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t txor -g TXORTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tshls -g TSHLSTest.case_int16_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tshrs -g TSHRSTest.case_int16_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tci -g TCITest.case5
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcmps -g TCMPSTest.case_float_8x64_8x64_8x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcolexpandadd -g TColExpandAddTest.case_fp32_16_128_1_128
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcolexpandmax -g TColExpandMaxTest.case_fp32_32_32_1_32
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcolexpandmin -g TColExpandMinTest.case_fp16_4_256_1_256
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcolmax -g TCOLMAXTest.case01
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcolmin -g TCOLMINTest.case01
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcolsum -g TCOLSUMTest.case01
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcolprod -g TCOLPRODTest.case01
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcvt -g TCVTTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tdivs -g TDIVSTest.case4
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tdivs -g TDIVSTest.case5
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t texp -g TEXPTest.case_float_64x64_64x64_64x64_inPlace_False
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t texpands -g TEXPANDSTest.case_float_64x64_64x64_64x64_PAD_VALUE_NULL
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t texpands_mat -g TEXPANDSTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t textract -g TEXTRACTTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tfillpad -g TFILLPADTest.case_float_GT_128_127_VT_128_128_BLK1_PADMAX_PADMAX
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tgather -g TGATHERTest.case1_float_32x1024_16x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tgatherb -g TGATHERBTest.case_float_2x128_2x16_2x128
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tload -g TLOADTest.case_float_GT_2_2_2_256_60_VT_256_64_BLK8_PADMAX
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tload_mix -g TLOADMIXTest.1_1_1_59_119_1_1_1_64_128_64_128_int8_t_ND2NZ
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tload_mx_NZ -g TLOADSCALETest.4_3_3_16_2_4_10_5_16_2_192_10_scale_ZZ2ZZ
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tload_mx_NZ -g TLOADSCALETest.7_5_3_16_2_7_7_11_16_2_12_560_scale_NN2NN
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tload_mx_ND_DN -g TLOADMXTest.1_1_1_64_128_uint8_AND2ZZ
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tload_mx_gmtensor -g TLOADMXTest.1_1_1_64_128_uint8_ADN2ZZ
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tload_shape2d -g TLOADSHAPE2DTest.1_1_1_59_119_1_1_1_64_128_64_128_int8_t_ND2NZ
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmatmul -g TMATMULTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmatmul_mx -g TMATMULMXTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmax -g TMAXTest.case_float_64x64_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmin -g TMINTest.case_float_64x64_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmins -g TMINSTest.case_float_60x128_64x64_60x60
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmins -g TMINSTest.case_float_16x200_20x512_16x200
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmins -g TMINSTest.case_float_1x3600_2x4096_1x3600
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmov -g TMOVTest.case_bias1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmov_acc2vec -g TMOVTest.case_nz2nd_1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmov_vect -g TMOVTest.vect_copy_case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmov_mx -g TMOVMXTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmrgsort -g TMRGSORTTest.case_topk1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmul -g TMULTest.case_float_64x64_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmuls -g TMULSTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tor -g TANDTest.case2
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tpartadd -g TPARTADDTest.case_float_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tpartmul -g TPARTMULTest.case_float_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tpartmax -g TPARTMAXTest.case_fp32_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tpartmin -g TPARTMINTest.case_fp32_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tprelu -g TPRELUTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trem -g TREMTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trowexpand -g TROWEXPANDTest.case5_float_16_8_16_127
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trowexpanddiv -g TRowExpandDivTest.case_fp32_40_64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trowmax -g TROWMAXTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trowmin -g TROWMINTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trowsum -g TROWSUMTest.test1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trowprod -g TROWPRODTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trsqrt -g TRSQRTTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tsel -g TSELTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tsels -g TSELSTest.case_uint8_uint8_2x32_2x32_2x32_2x32
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tsels -g TSELSTest.case_float_uint16_2x8_2x16_2x8_2x8
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tshl -g TSHLTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tshr -g TSHRTest.case2
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tsort32 -g TSort32Test.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tsqrt -g TSQRTTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tstore -g TStoreTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tstore_acc2gm -g TStoreAcc2gmTest.case7
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t ttrans -g TTRANSTest.case_float_8x8_2x8_2x8
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcmp -g TCMPTest.case_half_32x32_32x32_32x32
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tadd_tdiv -g TADD_TDIVTest.case_float_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmul_tadds -g TMUL_TADDSTest.case_float_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tsub_texp -g TSUB_TEXPTest.case_float_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmuls_trowsum -g TMULS_TROWSUMTest.case_float_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trowexpand_tsqrt -g TROWEXPAND_TSQRTTest.case_float_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trowexpand_trowsum -g TROWEXPAND_TROWSUMTest.case_float_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trowsum_trowexpand -g TROWSUM_TROWEXPANDTest.case_float_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trowexpand_tdiv -g TROWEXPAND_TDIVTest.case_float_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmov_ub2l1 -g TMovUb2l1Test.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tscatter -g TSCATTERTest.case1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tneg -g TNEGTest.case_float_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcolexpand -g TCOLEXPANDTest.case_float_1_8_128_63
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t ttri -g TTRITest.case_float_128x128_upper_diag_n3
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tnot -g TNOTTest.case_int16_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trelu -g TNOTTest.case_int32_64x64_64x64_64x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmov_acc2mat -g TMOVTest.case_nz2nz_insert
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t mgather -g MGATHERTest.case_half_16x128_8x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t mscatter -g MSCATTERTest.case_uint8_16x64_2048
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tquant -g TQUANTTest.case_mxfp8_fp32_128x128_nd
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tquant -g TQUANTTest.case_mxfp8_fp32_128x128_nz
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tquant -g TQUANTTest.case_int8_sym_fp32_128x128_nd
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tquant -g TQUANTTest.case_int8_asym_fp32_128x128_nd
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t ttri -g TTRITest.case_float_128x128_lower_diag_n3
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t ttri -g TTRITest.case_float_128x128_upper_diag_0
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tpack -g TPACKTEST.case_fp32_fp16_128x128
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tpack -g TPACKTEST.case_fp32_fp8_128x128
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tpack -g TPACKTEST.case_fp16_fp8_128x128
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tconcat -g TCONCATTest.case_half_16x128_16x64_16x64_16x63_16x64
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t thistogram -g THISTOGRAMTest.case_8x128_LSB_k104
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tinsert -g TInsertTest.case_acc2mat_1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tinsert -g TInsertTest.case_nz_1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tinsert -g TInsertTest.case_nd_1


  elif [ "$ENABLE_ALL" = "true" ]; then            # 所有用例
    python3 tests/script/build_st.py -r $RUN_TYPE -v a5 -t all
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t taxpy
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tdequant
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tfmod
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tfmods
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tsubs
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmaxs
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trems
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tlrelu
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tadd
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tadds
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tand
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tands
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tors
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t txors
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t txor
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tshls
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tshrs
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tci
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcmps
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcolexpandadd
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcolexpandmax
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcolexpandmin
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcolmax
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcolmin
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcolsum
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcolprod
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcvt
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tdivs
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t texp
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t texpands
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t texpands_mat
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t textract
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tfillpad
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tgather
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tgatherb
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tload
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tload_mix
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tload_mx_NZ
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tload_mx_ND_DN
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tload_mx_gmtensor
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tload_shape2d
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmatmul
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmatmul_mx
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmax
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmin
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmins
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmov
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmov_acc2vec
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmov_vect
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmov_mx
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmrgsort
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmul
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmuls
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tor
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tpartadd
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tpartmul
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tpartmax
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tpartmin
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tprelu
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trem
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trowexpand
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trowexpanddiv
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trowmax
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trowmin
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trowsum
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trowprod
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trsqrt
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tsel
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tsels
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tshl
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tshr
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tsort32
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tsqrt
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tstore
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tstore_acc2gm
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t ttrans
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcmp
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tadd_tdiv
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmul_tadds
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tsub_texp
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmuls_trowsum
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trowexpand_tsqrt
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trowexpand_trowsum
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trowsum_trowexpand
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trowexpand_tdiv
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmov_ub2l1
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tscatter
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tneg
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tcolexpand
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t ttri
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tnot
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t trelu
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tmov_acc2mat
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t mgather
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t mscatter
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tquant
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tdequant
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tconcat
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tpack
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t thistogram
    python3 tests/script/$RUN_MODE -r $RUN_TYPE -v a5 -t tinsert
  fi
fi

if [ "$ENABLE_KIRIN9030" = "true" ]; then
  python3 tests/script/build_st.py -r $RUN_TYPE -v kirin9030 -t all
  python3 tests/script/$RUN_MODE -r $RUN_TYPE -v kirin9030 -t textract
  python3 tests/script/$RUN_MODE -r $RUN_TYPE -v kirin9030 -t tmov
  python3 tests/script/$RUN_MODE -r $RUN_TYPE -v kirin9030 -t tadd
  python3 tests/script/$RUN_MODE -r $RUN_TYPE -v kirin9030 -t tcolsum
  python3 tests/script/$RUN_MODE -r $RUN_TYPE -v kirin9030 -t tpartadd
  python3 tests/script/$RUN_MODE -r $RUN_TYPE -v kirin9030 -t trowsum
  python3 tests/script/$RUN_MODE -r $RUN_TYPE -v kirin9030 -t tsort32
  python3 tests/script/$RUN_MODE -r $RUN_TYPE -v kirin9030 -t tcvt
  python3 tests/script/$RUN_MODE -r $RUN_TYPE -v kirin9030 -t tmrgsort
  python3 tests/script/$RUN_MODE -r $RUN_TYPE -v kirin9030 -t tgather
  python3 tests/script/$RUN_MODE -r $RUN_TYPE -v kirin9030 -t tsub
  python3 tests/script/$RUN_MODE -r $RUN_TYPE -v kirin9030 -t tmatmul
  python3 tests/script/$RUN_MODE -r $RUN_TYPE -v kirin9030 -t tload
  python3 tests/script/$RUN_MODE -r $RUN_TYPE -v kirin9030 -t ttrans
  python3 tests/script/$RUN_MODE -r $RUN_TYPE -v kirin9030 -t tstore
  python3 tests/script/$RUN_MODE -r $RUN_TYPE -v kirin9030 -t trowexpand
fi