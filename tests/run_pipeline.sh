#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

set -e

# Validate test case names before running pipeline
echo "Validating test case names..."
python3 tests/validate_testcase_names.py

python3 tests/script/build_st.py -r npu -v a3 -t tcvt -g TCVTTest.case_fp16_fp32_2x64
python3 tests/script/build_st.py -r npu -v a3 -t tmatmul -g TMATMULTest.case1
python3 tests/script/build_st.py -r npu -v a3 -t textract -g TEXTRACTTest.case31_float_1_1_29_29_36_param
python3 tests/script/build_st.py -r npu -v a3 -t tmov -g TMOVTest.case4_bias_dynamic_half_half_0_1_1_0_0_param
python3 tests/script/build_st.py -r npu -v a3 -t tmov -g TMOVTest.case11_scaling_static_int32_int8_0_1_0_1_0_param
python3 tests/script/build_st.py -r npu -v a3 -t tmov_acc2mat -g TMOVTest.case_nz2nz_fb_quant_4
python3 tests/script/build_st.py -r npu -v a3 -t tmrgsort -g TMRGSORTTest.case_topk1
python3 tests/script/run_st.py -r npu -v a3 -t tstore -g TStoreTest.ND_int16_t_1_2_1_23_121_3_2_2_35_125
python3 tests/script/run_st.py -r npu -v a3 -t tstore_acc2gm -g TStoreAcc2gmTest.case7
python3 tests/script/run_st.py -r npu -v a3 -t tstore_mat2gm -g TStoreMat2GMTest.case_nd1
python3 tests/script/build_st.py -r npu -v a3 -t trowsum -g TROWSUMTest.case1
python3 tests/script/build_st.py -r npu -v a3 -t tgather -g TGATHERTest.case1_float_P0101
python3 tests/script/build_st.py -r npu -v a3 -t tsort32 -g TSort32Test.case1
python3 tests/script/build_st.py -r npu -v a3 -t tadd -g TADDTest.case_float_64x64_64x64
python3 tests/script/build_st.py -r npu -v a3 -t tpartadd -g TPARTADDTest.case_float_64x64_64x64_64x64
python3 tests/script/build_st.py -r npu -v a3 -t tsub -g TSUBTest.case_float_64x64_64x64_64x64
python3 tests/script/build_st.py -r npu -v a3 -t tci -g TCITest.case1_int32
python3 tests/script/build_st.py -r npu -v a3 -t tcvt -g TCVTTest.case_fp16_fp32_2x64
python3 tests/script/build_st.py -r npu -v a3 -t tpartmin -g TPARTMINTest.test0
python3 tests/script/build_st.py -r npu -v a3 -t tpartmax -g TPARTMAXTest.test0
python3 tests/script/build_st.py -r npu -v a3 -t ttrans -g TTRANSTest.case1_float_16_8_16_8
python3 tests/script/build_st.py -r npu -v a3 -t ttrans_conv -g TTRANSConvTest.uint16_45_3_2_7_16
python3 tests/script/build_st.py -r npu -v a3 -t texpands_mat -g TEXPANDSTest.case1
python3 tests/script/build_st.py -r npu -v a3 -t tpushpop_cv -g TPushPopCvTest.case1_half_single_tile
python3 tests/script/build_st.py -r npu -v a3 -t tpushpop_vc -g TPushPopVcTest.case1_int8_single_k_tile
python3 tests/script/build_st.py -r npu -v a3 -t tpushpop_cv_nosplit -g TPushPopCvNoSplitTest.case1_half_single_tile
python3 tests/script/build_st.py -r npu -v a3 -t tpushpop_vc_nosplit -g TPushPopVcNoSplitTest.case1_int8_single_k_tile
python3 tests/script/build_st.py -r npu -v a3 -t tpushpop_dir_both -g TPushPopDirBothTest.case1_float_dir_both

python3 tests/script/build_st.py -r npu -v a5 -t tcvt -g TCVTTest.case_fp16_fp32_2x64
python3 tests/script/build_st.py -r npu -v a5 -t tmatmul -g TMATMULTest.case1
python3 tests/script/build_st.py -r npu -v a5 -t tmatmul_mx -g TMATMULMXTest.case1
python3 tests/script/build_st.py -r npu -v a5 -t textract -g TEXTRACTTest.case1
python3 tests/script/build_st.py -r npu -v a5 -t tinsert_acc2vec -g TMOVTest.case_nz2nd_sc_quant_1
python3 tests/script/build_st.py -r npu -v a5 -t textract_acc2vec -g TMOVTest.case_nz2nd_fb_quant_1
python3 tests/script/build_st.py -r npu -v a5 -t tmrgsort -g TMRGSORTTest.case_topk1
python3 tests/script/build_st.py -r npu -v a5 -t tmov_acc2vec -g TMOVTest.case_nz2nd_1
python3 tests/script/build_st.py -r npu -v a5 -t tmov_acc2vec -g TMOVTest.case_nz2nz_fb_quant_1
python3 tests/script/build_st.py -r npu -v a5 -t tmov_acc2mat -g TMOVTest.case_nz2nz_1
python3 tests/script/build_st.py -r npu -v a5 -t tmov_mx -g TMOVMXTest.case16
python3 tests/script/build_st.py -r npu -v a5 -t tstore -g TStoreTest.case1
python3 tests/script/build_st.py -r npu -v a5 -t trowsum -g TROWSUMTest.case1
python3 tests/script/build_st.py -r npu -v a5 -t tcolsum -g TCOLSUMTest.case01
python3 tests/script/build_st.py -r npu -v a5 -t tcolmax -g TCOLMAXTest.case01
python3 tests/script/build_st.py -r npu -v a5 -t tcolmin -g TCOLMINTest.case01
python3 tests/script/build_st.py -r npu -v a5 -t trowexpand -g TROWEXPANDTest.case0_half_16_16_16_512
python3 tests/script/build_st.py -r npu -v a5 -t tgather -g TGATHERTest.case1_float_32x1024_16x64
python3 tests/script/build_st.py -r npu -v a5 -t ttrans -g TTRANSTest.case_float_66x88_9x16_7x15
python3 tests/script/build_st.py -r npu -v a5 -t ttrans_conv -g TTRANSConvTest.float32_1_32_6_56
python3 tests/script/build_st.py -r npu -v a5 -t tsort32 -g TSort32Test.case1
python3 tests/script/build_st.py -r npu -v a5 -t tload -g TLOADTest.case_float_GT_2_2_2_256_60_VT_256_64_BLK8_PADMAX
python3 tests/script/build_st.py -r npu -v a5 -t tload_mix -g TLOADMIXTest.1_1_1_59_119_1_1_1_64_128_64_128_int8_t_ND2NZ
python3 tests/script/build_st.py -r npu -v a5 -t tadd -g TADDTest.case_float_64x64_64x64_64x64_64x64
python3 tests/script/build_st.py -r npu -v a5 -t tpartadd -g TPARTADDTest.case_float_64x64_64x64_64x64
python3 tests/script/build_st.py -r npu -v a5 -t tsub -g TSUBTest.case_float_64x64_64x64_64x64_64x64
python3 tests/script/build_st.py -r npu -v a5 -t tmul -g TMULTest.case_float_64x64_64x64_64x64_64x64
python3 tests/script/build_st.py -r npu -v a5 -t tdiv -g TDIVTest.case_float_64x64_64x64_64x64_64x64
python3 tests/script/build_st.py -r npu -v a5 -t tmax -g TMAXTest.case_float_64x64_64x64_64x64_64x64
python3 tests/script/build_st.py -r npu -v a5 -t tmin -g TMINTest.case_float_64x64_64x64_64x64_64x64
python3 tests/script/build_st.py -r npu -v a5 -t tmov -g TMOVTest.case_bias_dynamic8
python3 tests/script/build_st.py -r npu -v a5 -t tmov -g TMOVTest.case_fixpipe1
python3 tests/script/build_st.py -r npu -v a5 -t tstore_acc2gm -g TStoreAcc2gmTest.case1
python3 tests/script/build_st.py -r npu -v a5 -t tstore_acc2gm -g TStoreAcc2gmTest.case17
python3 tests/script/build_st.py -r npu -v a5 -t texpands_mat -g TEXPANDSTest.case1
python3 tests/script/build_st.py -r npu -v a5 -t tpushpop_cv -g TPushPopCvTest.case1_half_single_tile
python3 tests/script/build_st.py -r npu -v a5 -t tpushpop_vc -g TPushPopVcTest.case1_int8_single_k_tile
python3 tests/script/build_st.py -r npu -v a5 -t tpushpop_cv_nosplit -g TPushPopCvNoSplitTest.case1_half_single_tile
python3 tests/script/build_st.py -r npu -v a5 -t tpushpop_vc_nosplit -g TPushPopVcNoSplitTest.case1_int8_single_k_tile
python3 tests/script/build_st.py -r npu -v a5 -t tpushpop_dir_both -g TPushPopDirBothTest.case1_float_dir_both
