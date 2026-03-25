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
#include <cmath>

using namespace std;
using namespace pto;
using namespace PtoTestCommon;
using namespace pto;

template <typename ST, typename DT, size_t rows, size_t cols, size_t validRows, size_t validCols, uint16_t idxRow,
          uint16_t idxCol, uint16_t srcLayout, uint16_t dstLayout, float profiling, float accuracy>
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
    if constexpr (srcLayout == 1) {
        // DN tile: use DN GlobalTensor (ColMajor stride, Layout::DN)
        using GlobalDataSrcDN = GlobalTensor<
            ST, pto::Shape<1, 1, 1, validRows, validCols>,
            pto::Stride<1 * validRows * validCols, 1 * validRows * validCols, validRows * validCols, 1, validRows>,
            pto::Layout::DN>;
        GlobalDataSrcDN srcGlobalDN(src);
        TLOAD(srcTile, srcGlobalDN);
    } else {
        TLOAD(srcTile, srcGlobal);
    }

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    /**********************************TMOV && TEXTRACT**********************************/
    TEXTRACT(dstTile, srcTile, idxRow, idxCol);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    /****************************************TSTORE*****************************************/
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();

    float costResult = dstTile.GetCycle();
    float precision = 1 - fabs(profiling - costResult) / profiling;
    bool ret = precision >= accuracy;
    EXPECT_TRUE(ret);
}

class TEXTRACTTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

template <typename ST, typename DT, size_t rows, size_t cols, size_t validRows, size_t validCols, size_t idxRow,
          size_t idxCol, uint16_t srcLayout, uint16_t dstLayout, float profiling, float accuracy>
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

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    runTEXTRACT<ST, DT, rows, cols, validRows, validCols, idxRow, idxCol, srcLayout, dstLayout, profiling, accuracy>(
        (DT *)dstDevice, (ST *)srcDevice);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

TEST_F(TEXTRACTTest, case_half_half_32_32_32_32_IDX_0_0_L_0_0)
{
    textract_test<half, half, 32, 32, 32, 32, 0, 0, 0, 0, 64.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_half_float_32_32_32_32_IDX_0_0_L_0_0)
{
    textract_test<half, float, 32, 32, 32, 32, 0, 0, 0, 0, 128.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_128_96_IDX_0_0_L_0_0)
{
    textract_test<float, float, 128, 96, 128, 96, 0, 0, 0, 0, 1536.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_int32_t_float_128_96_128_96_IDX_0_0_L_0_0)
{
    textract_test<int32_t, float, 128, 96, 128, 96, 0, 0, 0, 0, 1536.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_int8_t_int32_t_128_64_128_64_IDX_0_0_L_0_0)
{
    textract_test<int8_t, int32_t, 128, 64, 128, 64, 0, 0, 0, 0, 1024.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_half_half_32_32_32_32_IDX_8_16_L_0_0)
{
    textract_test<half, half, 32, 32, 32, 32, 8, 16, 0, 0, 64.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_half_float_32_32_32_32_IDX_8_16_L_0_0)
{
    textract_test<half, float, 32, 32, 32, 32, 8, 16, 0, 0, 128.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_128_96_IDX_8_16_L_0_0)
{
    textract_test<float, float, 128, 96, 128, 96, 8, 16, 0, 0, 1536.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_int32_t_float_128_96_128_96_IDX_8_16_L_0_0)
{
    textract_test<int32_t, float, 128, 96, 128, 96, 8, 16, 0, 0, 1536.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_int8_t_int32_t_128_64_128_64_IDX_8_16_L_0_0)
{
    textract_test<int8_t, int32_t, 128, 64, 128, 64, 8, 16, 0, 0, 1024.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_half_half_32_32_31_31_IDX_8_16_L_0_0)
{
    textract_test<half, half, 32, 32, 31, 31, 8, 16, 0, 0, 60.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_half_float_32_32_31_31_IDX_8_16_L_0_0)
{
    textract_test<half, float, 32, 32, 31, 31, 8, 16, 0, 0, 120.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_125_93_IDX_8_16_L_0_0)
{
    textract_test<float, float, 128, 96, 125, 93, 8, 16, 0, 0, 1453.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_int32_t_float_128_96_125_93_IDX_8_16_L_0_0)
{
    textract_test<int32_t, float, 128, 96, 125, 93, 8, 16, 0, 0, 1453.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_int8_t_int32_t_128_64_125_61_IDX_8_16_L_0_0)
{
    textract_test<int8_t, int32_t, 128, 64, 125, 61, 8, 16, 0, 0, 953.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_125_93_IDX_8_16_L_0_1)
{
    textract_test<float, float, 128, 96, 125, 93, 8, 16, 0, 1, 1453.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_125_93_IDX_8_16_L_0_2)
{
    textract_test<float, float, 128, 96, 125, 93, 8, 16, 0, 2, 1453.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_125_93_IDX_8_16_L_1_0)
{
    textract_test<float, float, 128, 96, 125, 93, 8, 16, 1, 0, 1453.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_125_93_IDX_8_16_L_1_1)
{
    textract_test<float, float, 128, 96, 125, 93, 8, 16, 1, 1, 1453.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_125_93_IDX_8_16_L_1_2)
{
    textract_test<float, float, 128, 96, 125, 93, 8, 16, 1, 2, 1453.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_125_93_IDX_8_16_L_2_0)
{
    textract_test<float, float, 128, 96, 125, 93, 8, 16, 2, 0, 1453.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_125_93_IDX_8_16_L_2_1)
{
    textract_test<float, float, 128, 96, 125, 93, 8, 16, 2, 1, 1453.0f, 1.0f>();
}

TEST_F(TEXTRACTTest, case_float_float_128_96_125_93_IDX_8_16_L_2_2)
{
    textract_test<float, float, 128, 96, 125, 93, 8, 16, 2, 2, 1453.0f, 1.0f>();
}
