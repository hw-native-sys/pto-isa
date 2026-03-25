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
#include <functional>
#include <cmath>

using namespace std;
using namespace pto;
using namespace PtoTestCommon;

template <typename T, int rows, int cols, int validRow, int validCol, float profiling, float accuracy, TileType srcLoc,
          BLayout srcBL, SLayout srcSL, TileType dstLoc, BLayout dstBL, SLayout dstSL>
void testMov()
{
    Tile<srcLoc, T, rows, cols, srcBL, validRow, validCol, srcSL> src;
    Tile<dstLoc, T, rows, cols, dstBL, -1, -1, dstSL> dst(validRow, validCol);

    std::fill(src.data(), src.data() + rows * cols, 0);
    std::fill(dst.data(), dst.data() + rows * cols, 0);

    std::vector<T> srcData(validCol * validRow, 0);
    std::vector<T> dstData(validCol * validRow, 0);

    for (int i = 0; i < srcData.size(); i++) {
        srcData[i] = std::rand() / 1000.0;
    }

    using TensorType = GlobalTensor<T, Shape<1, 1, 1, validRow, validCol>,
                                    Stride<validRow * validCol, validRow * validCol, validRow, validCol, 1>>;
    TensorType srcTensor(srcData.data());
    TensorType dstTensor(dstData.data());

    if constexpr (srcBL == BLayout::RowMajor && srcSL == SLayout::NoneBox) {
        // ND tile: ND2ND TLOAD
        TLOAD(src, srcTensor);
    } else if constexpr (srcBL == BLayout::ColMajor && srcSL == SLayout::NoneBox) {
        // DN tile: DN2DN TLOAD
        using DnTensorType =
            GlobalTensor<T, Shape<1, 1, 1, validRow, validCol>,
                         Stride<validRow * validCol, validRow * validCol, validRow * validCol, 1, validRow>,
                         Layout::DN>;
        DnTensorType srcTensorDN(srcData.data());
        TLOAD(src, srcTensorDN);
    }
    // For NZ and other tile layouts, skip TLOAD (dst.GetCycle() measures TMOV, not TLOAD)
    TMOV(dst, src);
    TSTORE(dstTensor, dst);

    float costResult = dst.GetCycle();
    float precision = 1 - fabs(profiling - costResult) / profiling;
    bool ret = precision >= accuracy;
    EXPECT_TRUE(ret);
}

class TMOVTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

#define TMOV_TEST(T, rows, cols, validRow, validCol, profiling, accuracy, srcLoc, srcBL, srcSL, dstLoc, dstBL, dstSL) \
    TEST_F(                                                                                                           \
        TMOVTest,                                                                                                     \
        T##_##rows##_##cols##_##validRow##_##validCol##_##srcLoc##_##srcBL##_##srcSL##_##dstLoc##_##dstBL##_##dstSL)  \
    {                                                                                                                 \
        testMov<T, rows, cols, validRow, validCol, profiling, accuracy, TileType::srcLoc, BLayout::srcBL,             \
                SLayout::srcSL, TileType::dstLoc, BLayout::dstBL, SLayout::dstSL>();                                  \
    }

TMOV_TEST(float, 64, 128, 64, 128, 256.0f, 1.0f, Vec, RowMajor, NoneBox, Vec, RowMajor, NoneBox)
TMOV_TEST(float, 64, 128, 64, 128, 256.0f, 1.0f, Vec, RowMajor, NoneBox, Vec, ColMajor, NoneBox)
TMOV_TEST(float, 64, 128, 64, 128, 256.0f, 1.0f, Vec, ColMajor, NoneBox, Vec, ColMajor, NoneBox)
TMOV_TEST(float, 64, 128, 64, 128, 256.0f, 1.0f, Vec, ColMajor, NoneBox, Vec, RowMajor, NoneBox)
TMOV_TEST(float, 64, 128, 64, 128, 256.0f, 1.0f, Vec, RowMajor, NoneBox, Vec, ColMajor, RowMajor)
TMOV_TEST(float, 64, 128, 64, 128, 256.0f, 1.0f, Vec, ColMajor, RowMajor, Vec, RowMajor, NoneBox)
TMOV_TEST(float, 64, 128, 64, 128, 256.0f, 1.0f, Vec, ColMajor, NoneBox, Vec, ColMajor, RowMajor)
TMOV_TEST(float, 64, 128, 64, 128, 256.0f, 1.0f, Vec, ColMajor, RowMajor, Vec, ColMajor, NoneBox)

TMOV_TEST(float, 16, 24, 15, 23, 10.0f, 1.0f, Vec, RowMajor, NoneBox, Vec, RowMajor, NoneBox)
TMOV_TEST(float, 64, 128, 63, 125, 246.0f, 1.0f, Vec, RowMajor, NoneBox, Vec, ColMajor, NoneBox)
TMOV_TEST(float, 64, 128, 63, 125, 246.0f, 1.0f, Vec, ColMajor, NoneBox, Vec, ColMajor, NoneBox)
TMOV_TEST(float, 64, 128, 63, 125, 246.0f, 1.0f, Vec, ColMajor, NoneBox, Vec, RowMajor, NoneBox)
TMOV_TEST(float, 64, 128, 63, 125, 246.0f, 1.0f, Vec, RowMajor, NoneBox, Vec, ColMajor, RowMajor)
TMOV_TEST(float, 64, 128, 63, 125, 246.0f, 1.0f, Vec, ColMajor, RowMajor, Vec, RowMajor, NoneBox)
TMOV_TEST(float, 64, 128, 63, 125, 246.0f, 1.0f, Vec, ColMajor, NoneBox, Vec, ColMajor, RowMajor)
TMOV_TEST(float, 64, 128, 63, 125, 246.0f, 1.0f, Vec, ColMajor, RowMajor, Vec, ColMajor, NoneBox)