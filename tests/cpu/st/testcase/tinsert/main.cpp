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

template <typename ST, typename DT, TileType locSrc, TileType locDst, size_t rows, size_t cols, size_t srcValidRows,
          size_t srcValidCols, size_t dstValidRows, size_t dstValidCols, uint16_t srcLayout, uint16_t dstLayout>
AICORE inline void runTINSERT(__gm__ DT *out, __gm__ ST *src)
{
    constexpr int idxRow = dstValidRows - srcValidRows;
    constexpr int idxCol = dstValidCols - srcValidCols;

    using GlobalDataSrc = GlobalTensor<ST, pto::Shape<1, 1, 1, srcValidRows, srcValidCols>,
                                       pto::Stride<1 * srcValidRows * srcValidCols, 1 * srcValidRows * srcValidCols,
                                                   srcValidRows * srcValidCols, srcValidCols, 1>>;
    using GlobalDataDst = GlobalTensor<DT, pto::Shape<1, 1, 1, dstValidRows, dstValidCols>,
                                       pto::Stride<1 * dstValidRows * dstValidCols, 1 * dstValidRows * dstValidCols,
                                                   dstValidRows * dstValidCols, dstValidCols, 1>>;

    GlobalDataSrc srcGlobal(src);
    GlobalDataDst dstGlobal(out);

    constexpr BLayout srcBL = srcLayout > 0 ? BLayout::ColMajor : BLayout::RowMajor;
    constexpr SLayout srcSL = srcLayout < 2 ? SLayout::NoneBox : SLayout::RowMajor;
    constexpr BLayout dstBL = dstLayout > 0 ? BLayout::ColMajor : BLayout::RowMajor;
    constexpr SLayout dstSL = dstLayout < 2 ? SLayout::NoneBox : SLayout::RowMajor;

    Tile<locSrc, ST, rows, cols, srcBL, srcValidRows, srcValidCols, srcSL, 512> srcTile;
    Tile<locDst, DT, rows, cols, dstBL, dstValidRows, dstValidCols, dstSL, 512> dstTile;

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    std::fill(dstTile.data(), dstTile.data() + rows * cols, 0);

    /*************************************TLOAD****************************************/
    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /********************************** TINSERT**********************************/
    TINSERT(dstTile, srcTile, idxRow, idxCol);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

class TINSERTTest : public testing::Test {
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

template <typename ST, typename DT, TileType locSrc, TileType locDst, size_t rows, size_t cols, size_t srcValidRows,
          size_t srcValidCols, size_t dstValidRows, size_t dstValidCols, uint16_t srcLayout, uint16_t dstLayout>
void tinsert_test()
{
    size_t srcFileSize = srcValidRows * srcValidCols * sizeof(ST);
    size_t dstFileSize = dstValidRows * dstValidCols * sizeof(DT);

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
    runTINSERT<ST, DT, locSrc, locDst, rows, cols, srcValidRows, srcValidCols, dstValidRows, dstValidCols, srcLayout,
               dstLayout>((DT *)dstDevice, (ST *)srcDevice);

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

TEST_F(TINSERTTest, case_half_half_Mat_Mat_32_32_32_32_DST_32_32_L_0_0)
{
    tinsert_test<half, half, TileType::Mat, TileType::Mat, 32, 32, 32, 32, 32, 32, 0, 0>();
}

TEST_F(TINSERTTest, case_half_float_Mat_Mat_32_32_32_32_DST_32_32_L_0_0)
{
    tinsert_test<half, float, TileType::Mat, TileType::Mat, 32, 32, 32, 32, 32, 32, 0, 0>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_128_96_DST_128_96_L_0_0)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 128, 96, 128, 96, 0, 0>();
}

TEST_F(TINSERTTest, case_int32_t_float_Mat_Mat_128_96_128_96_DST_128_96_L_0_0)
{
    tinsert_test<int32_t, float, TileType::Mat, TileType::Mat, 128, 96, 128, 96, 128, 96, 0, 0>();
}

TEST_F(TINSERTTest, case_int8_t_int32_t_Mat_Mat_128_64_128_64_DST_128_64_L_0_0)
{
    tinsert_test<int8_t, int32_t, TileType::Mat, TileType::Mat, 128, 64, 128, 64, 128, 64, 0, 0>();
}

TEST_F(TINSERTTest, case_half_half_Mat_Mat_32_32_24_16_DST_24_16_L_0_0)
{
    tinsert_test<half, half, TileType::Mat, TileType::Mat, 32, 32, 24, 16, 32, 32, 0, 0>();
}

TEST_F(TINSERTTest, case_half_float_Mat_Mat_32_32_24_16_DST_24_16_L_0_0)
{
    tinsert_test<half, float, TileType::Mat, TileType::Mat, 32, 32, 24, 16, 32, 32, 0, 0>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_24_16_DST_24_16_L_0_0)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 24, 16, 128, 96, 0, 0>();
}

TEST_F(TINSERTTest, case_int32_t_float_Mat_Mat_128_96_24_16_DST_24_16_L_0_0)
{
    tinsert_test<int32_t, float, TileType::Mat, TileType::Mat, 128, 96, 24, 16, 128, 96, 0, 0>();
}

TEST_F(TINSERTTest, case_int8_t_int32_t_Mat_Mat_128_64_24_16_DST_24_16_L_0_0)
{
    tinsert_test<int8_t, int32_t, TileType::Mat, TileType::Mat, 128, 64, 24, 16, 128, 64, 0, 0>();
}

TEST_F(TINSERTTest, case_half_half_Mat_Mat_32_32_23_16_DST_23_16_L_0_0)
{
    tinsert_test<half, half, TileType::Mat, TileType::Mat, 32, 32, 23, 16, 31, 31, 0, 0>();
}

TEST_F(TINSERTTest, case_half_float_Mat_Mat_32_32_23_16_DST_23_16_L_0_0)
{
    tinsert_test<half, float, TileType::Mat, TileType::Mat, 32, 32, 23, 16, 31, 31, 0, 0>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_23_16_DST_23_16_L_0_0)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 23, 16, 125, 93, 0, 0>();
}

TEST_F(TINSERTTest, case_int32_t_float_Mat_Mat_128_96_23_16_DST_23_16_L_0_0)
{
    tinsert_test<int32_t, float, TileType::Mat, TileType::Mat, 128, 96, 23, 16, 125, 93, 0, 0>();
}

TEST_F(TINSERTTest, case_int8_t_int32_t_Mat_Mat_128_64_23_16_DST_23_16_L_0_0)
{
    tinsert_test<int8_t, int32_t, TileType::Mat, TileType::Mat, 128, 64, 23, 16, 125, 61, 0, 0>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_18_16_DST_18_16_L_0_1)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 18, 16, 125, 93, 0, 1>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_18_16_DST_18_16_L_0_2)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 18, 16, 125, 93, 0, 2>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_18_16_DST_18_16_L_1_0)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 18, 16, 125, 93, 1, 0>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_18_16_DST_18_16_L_1_1)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 18, 16, 125, 93, 1, 1>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_18_16_DST_18_16_L_1_2)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 18, 16, 125, 93, 1, 2>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_18_16_DST_18_16_L_2_0)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 18, 16, 125, 93, 2, 0>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_18_16_DST_18_16_L_2_1)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 18, 16, 125, 93, 2, 1>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Mat_128_96_18_16_DST_18_16_L_2_2)
{
    tinsert_test<float, float, TileType::Mat, TileType::Mat, 128, 96, 18, 16, 125, 93, 2, 2>();
}

TEST_F(TINSERTTest, case_half_half_Mat_Vec_32_32_8_16_DST_8_16_L_0_0)
{
    tinsert_test<half, half, TileType::Mat, TileType::Vec, 32, 32, 8, 16, 32, 32, 0, 0>();
}

TEST_F(TINSERTTest, case_half_float_Mat_Vec_32_32_8_16_DST_8_16_L_0_0)
{
    tinsert_test<half, float, TileType::Mat, TileType::Vec, 32, 32, 8, 16, 32, 32, 0, 0>();
}

TEST_F(TINSERTTest, case_float_float_Mat_Vec_128_96_8_16_DST_8_16_L_0_0)
{
    tinsert_test<float, float, TileType::Mat, TileType::Vec, 128, 96, 8, 16, 128, 96, 0, 0>();
}

TEST_F(TINSERTTest, case_int32_t_float_Mat_Vec_128_96_8_16_DST_8_16_L_0_0)
{
    tinsert_test<int32_t, float, TileType::Mat, TileType::Vec, 128, 96, 8, 16, 128, 96, 0, 0>();
}

TEST_F(TINSERTTest, case_int8_t_int32_t_Mat_Vec_128_64_8_16_DST_8_16_L_0_0)
{
    tinsert_test<int8_t, int32_t, TileType::Mat, TileType::Vec, 128, 64, 8, 16, 128, 64, 0, 0>();
}

TEST_F(TINSERTTest, case_half_half_Vec_Vec_32_32_8_16_DST_8_16_L_0_0)
{
    tinsert_test<half, half, TileType::Vec, TileType::Vec, 32, 32, 8, 16, 32, 32, 0, 0>();
}

TEST_F(TINSERTTest, case_half_float_Vec_Vec_32_32_8_16_DST_8_16_L_0_0)
{
    tinsert_test<half, float, TileType::Vec, TileType::Vec, 32, 32, 8, 16, 32, 32, 0, 0>();
}

TEST_F(TINSERTTest, case_float_float_Vec_Vec_128_96_8_16_DST_8_16_L_0_0)
{
    tinsert_test<float, float, TileType::Vec, TileType::Vec, 128, 96, 8, 16, 128, 96, 0, 0>();
}

TEST_F(TINSERTTest, case_int32_t_float_Vec_Vec_128_96_8_16_DST_8_16_L_0_0)
{
    tinsert_test<int32_t, float, TileType::Vec, TileType::Vec, 128, 96, 8, 16, 128, 96, 0, 0>();
}

TEST_F(TINSERTTest, case_int8_t_int32_t_Vec_Vec_128_64_8_16_DST_8_16_L_0_0)
{
    tinsert_test<int8_t, int32_t, TileType::Vec, TileType::Vec, 128, 64, 8, 16, 128, 64, 0, 0>();
}
