/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <pto/pto-inst.hpp>
#include "test_common.h"
#include <gtest/gtest.h>
#include <pto/common/constants.hpp>

using namespace std;
using namespace pto;
using namespace PtoTestCommon;
using namespace pto;

template <typename ST, typename DT, size_t rows, size_t cols, size_t validRows, size_t validCols, uint16_t idxRow,
          uint16_t idxCol, uint16_t srcLayout, uint16_t dstLayout>
AICORE inline void runTEXTRACT(__gm__ DT *out, __gm__ ST *src)
{
    constexpr int validRowsDst = validRows - idxRow;
    constexpr int validColsDst = validCols - idxCol;

    using GlobalDataSrc = GlobalTensor<
        ST, pto::Shape<1, 1, 1, validRows, validCols>,
        pto::Stride<1 * validRows * validCols, 1 * validRows * validCols, validRows * validCols, validCols, 1>>;
    using GlobalDataDst = GlobalTensor<DT, pto::Shape<1, 1, 1, validRowsDst, validColsDst>,
                                       pto::Stride<1 * validRowsDst * validColsDst, 1 * validRowsDst * validColsDst,
                                                   validRowsDst * validColsDst, validColsDst, 1>>;

    GlobalDataSrc srcGlobal(src);
    GlobalDataDst dstGlobal(out);

    constexpr BLayout srcBL = srcLayout > 0 ? BLayout::ColMajor : BLayout::RowMajor;
    constexpr SLayout srcSL = srcLayout < 2 ? SLayout::NoneBox : SLayout::RowMajor;
    constexpr BLayout dstBL = dstLayout > 0 ? BLayout::ColMajor : BLayout::RowMajor;
    constexpr SLayout dstSL = dstLayout < 2 ? SLayout::NoneBox : SLayout::RowMajor;

    Tile<TileType::Mat, ST, rows, cols, srcBL, validRows, validCols, srcSL, 512> srcTile;
    Tile<TileType::Mat, DT, rows, cols, dstBL, validRowsDst, validColsDst, dstSL, 512> dstTile;

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    std::fill(dstTile.data(), dstTile.data() + rows * cols, 0);

    /*************************************TLOAD****************************************/
    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TMOV && TEXTRACT**********************************/
    TEXTRACT(dstTile, srcTile, idxRow, idxCol);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

class TEXTRACTTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename ST, typename DT, size_t rows, size_t cols, size_t validRows, size_t validCols, size_t idxRow,
          size_t idxCol, uint16_t srcLayout, uint16_t dstLayout>
void textract_test()
{
    size_t srcFileSize = validRows * validCols * sizeof(ST);
    size_t dstFileSize = (validRows - idxRow) * (validCols - idxCol) * sizeof(DT);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *srcHost;
    uint8_t *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t inputSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", inputSize, srcHost, srcFileSize));

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    runTEXTRACT<ST, DT, rows, cols, validRows, validCols, idxRow, idxCol, srcLayout, dstLayout>((DT *)dstDevice,
                                                                                                (ST *)srcDevice);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    std::vector<DT> golden(dstFileSize / sizeof(DT));
    size_t goldenSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", goldenSize, golden.data(), dstFileSize));

    bool ret = ResultCmp(golden, (DT *)dstHost, 0);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    EXPECT_TRUE(ret);
}

TEST_F(TEXTRACTTest, case_half_half_32_32_32_32_IDX_0_0_L_0_0)
{
    textract_test<half, half, 32, 32, 32, 32, 0, 0, 0, 0>();
}

TEST_F(TEXTRACTTest, case_half_float_32_32_32_32_IDX_0_0_L_0_0)
{
    textract_test<half, float, 32, 32, 32, 32, 0, 0, 0, 0>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_128_96_IDX_0_0_L_0_0)
{
    textract_test<float, float, 128, 96, 128, 96, 0, 0, 0, 0>();
}

TEST_F(TEXTRACTTest, case_int32_t_float_128_96_128_96_IDX_0_0_L_0_0)
{
    textract_test<int32_t, float, 128, 96, 128, 96, 0, 0, 0, 0>();
}

TEST_F(TEXTRACTTest, case_int8_t_int32_t_128_64_128_64_IDX_0_0_L_0_0)
{
    textract_test<int8_t, int32_t, 128, 64, 128, 64, 0, 0, 0, 0>();
}

TEST_F(TEXTRACTTest, case_half_half_32_32_32_32_IDX_8_16_L_0_0)
{
    textract_test<half, half, 32, 32, 32, 32, 8, 16, 0, 0>();
}

TEST_F(TEXTRACTTest, case_half_float_32_32_32_32_IDX_8_16_L_0_0)
{
    textract_test<half, float, 32, 32, 32, 32, 8, 16, 0, 0>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_128_96_IDX_8_16_L_0_0)
{
    textract_test<float, float, 128, 96, 128, 96, 8, 16, 0, 0>();
}

TEST_F(TEXTRACTTest, case_int32_t_float_128_96_128_96_IDX_8_16_L_0_0)
{
    textract_test<int32_t, float, 128, 96, 128, 96, 8, 16, 0, 0>();
}

TEST_F(TEXTRACTTest, case_int8_t_int32_t_128_64_128_64_IDX_8_16_L_0_0)
{
    textract_test<int8_t, int32_t, 128, 64, 128, 64, 8, 16, 0, 0>();
}

TEST_F(TEXTRACTTest, case_half_half_32_32_31_31_IDX_8_16_L_0_0)
{
    textract_test<half, half, 32, 32, 31, 31, 8, 16, 0, 0>();
}

TEST_F(TEXTRACTTest, case_half_float_32_32_31_31_IDX_8_16_L_0_0)
{
    textract_test<half, float, 32, 32, 31, 31, 8, 16, 0, 0>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_125_93_IDX_8_16_L_0_0)
{
    textract_test<float, float, 128, 96, 125, 93, 8, 16, 0, 0>();
}

TEST_F(TEXTRACTTest, case_int32_t_float_128_96_125_93_IDX_8_16_L_0_0)
{
    textract_test<int32_t, float, 128, 96, 125, 93, 8, 16, 0, 0>();
}

TEST_F(TEXTRACTTest, case_int8_t_int32_t_128_64_125_61_IDX_8_16_L_0_0)
{
    textract_test<int8_t, int32_t, 128, 64, 125, 61, 8, 16, 0, 0>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_125_93_IDX_8_16_L_0_1)
{
    textract_test<float, float, 128, 96, 125, 93, 8, 16, 0, 1>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_125_93_IDX_8_16_L_0_2)
{
    textract_test<float, float, 128, 96, 125, 93, 8, 16, 0, 2>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_125_93_IDX_8_16_L_1_0)
{
    textract_test<float, float, 128, 96, 125, 93, 8, 16, 1, 0>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_125_93_IDX_8_16_L_1_1)
{
    textract_test<float, float, 128, 96, 125, 93, 8, 16, 1, 1>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_125_93_IDX_8_16_L_1_2)
{
    textract_test<float, float, 128, 96, 125, 93, 8, 16, 1, 2>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_125_93_IDX_8_16_L_2_0)
{
    textract_test<float, float, 128, 96, 125, 93, 8, 16, 2, 0>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_125_93_IDX_8_16_L_2_1)
{
    textract_test<float, float, 128, 96, 125, 93, 8, 16, 2, 1>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_125_93_IDX_8_16_L_2_2)
{
    textract_test<float, float, 128, 96, 125, 93, 8, 16, 2, 2>();
}

#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TEXTRACTTest, case_bfloat16_t_bfloat16_t_32_32_32_32_IDX_0_0_L_0_0)
{
    textract_test<bfloat16_t, bfloat16_t, 32, 32, 32, 32, 0, 0, 0, 0>();
}

TEST_F(TEXTRACTTest, case_bfloat16_t_float_32_32_32_32_IDX_8_16_L_0_0)
{
    textract_test<bfloat16_t, float, 32, 32, 32, 32, 8, 16, 0, 0>();
}

TEST_F(TEXTRACTTest, case_bfloat16_t_bfloat16_t_32_32_31_31_IDX_8_16_L_0_0)
{
    textract_test<bfloat16_t, bfloat16_t, 32, 32, 31, 31, 8, 16, 0, 0>();
}
#endif
