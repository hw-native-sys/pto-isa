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

using namespace std;
using namespace pto;
using namespace PtoTestCommon;

template <
    typename T, int rows, int cols, int validRow, int validCol, TileType srcLoc, BLayout srcBL, SLayout srcSL,
    TileType dstLoc, BLayout dstBL, SLayout dstSL>
void testMov()
{
    Tile<srcLoc, T, rows, cols, srcBL, validRow, validCol, srcSL> src;
    Tile<dstLoc, T, rows, cols, dstBL, -1, -1, dstSL> dst(validRow, validCol);

    TASSIGN(src, 0);
    TASSIGN(dst, rows * cols * sizeof(T));

    std::fill(src.data(), src.data() + rows * cols, 0);
    std::fill(dst.data(), dst.data() + rows * cols, 0);

    std::vector<T> srcData(validCol * validRow, 0);
    std::vector<T> dstData(validCol * validRow, 0);

    for (int i = 0; i < srcData.size(); i++) {
        srcData[i] = static_cast<T>(std::rand() / 1000.0);
    }

    using TensorType = GlobalTensor<
        T, Shape<1, 1, 1, validRow, validCol>, Stride<validRow * validCol, validRow * validCol, validRow, validCol, 1>>;
    TensorType srcTensor(srcData.data());
    TensorType dstTensor(dstData.data());

    TLOAD(src, srcTensor);
    TMOV(dst, src);
    TSTORE(dstTensor, dst);

    EXPECT_TRUE(ResultCmp(srcData, dstData.data(), 0));
}

class TMOVTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

#define TMOV_TEST(T, rows, cols, validRow, validCol, srcLoc, srcBL, srcSL, dstLoc, dstBL, dstSL)                     \
    TEST_F(                                                                                                          \
        TMOVTest,                                                                                                    \
        T##_##rows##_##cols##_##validRow##_##validCol##_##srcLoc##_##srcBL##_##srcSL##_##dstLoc##_##dstBL##_##dstSL) \
    {                                                                                                                \
        testMov<                                                                                                     \
            T, rows, cols, validRow, validCol, TileType::srcLoc, BLayout::srcBL, SLayout::srcSL, TileType::dstLoc,   \
            BLayout::dstBL, SLayout::dstSL>();                                                                       \
    }

TMOV_TEST(float, 64, 128, 64, 128, Vec, RowMajor, NoneBox, Vec, RowMajor, NoneBox)
TMOV_TEST(float, 64, 128, 64, 128, Vec, RowMajor, NoneBox, Vec, ColMajor, NoneBox)
TMOV_TEST(float, 64, 128, 64, 128, Vec, ColMajor, NoneBox, Vec, ColMajor, NoneBox)
TMOV_TEST(float, 64, 128, 64, 128, Vec, ColMajor, NoneBox, Vec, RowMajor, NoneBox)
TMOV_TEST(float, 64, 128, 64, 128, Vec, RowMajor, NoneBox, Vec, ColMajor, RowMajor)
TMOV_TEST(float, 64, 128, 64, 128, Vec, ColMajor, RowMajor, Vec, RowMajor, NoneBox)
TMOV_TEST(float, 64, 128, 64, 128, Vec, ColMajor, NoneBox, Vec, ColMajor, RowMajor)
TMOV_TEST(float, 64, 128, 64, 128, Vec, ColMajor, RowMajor, Vec, ColMajor, NoneBox)

TMOV_TEST(float, 16, 24, 15, 23, Vec, RowMajor, NoneBox, Vec, RowMajor, NoneBox)
TMOV_TEST(float, 64, 128, 63, 125, Vec, RowMajor, NoneBox, Vec, ColMajor, NoneBox)
TMOV_TEST(float, 64, 128, 63, 125, Vec, ColMajor, NoneBox, Vec, ColMajor, NoneBox)
TMOV_TEST(float, 64, 128, 63, 125, Vec, ColMajor, NoneBox, Vec, RowMajor, NoneBox)
TMOV_TEST(float, 64, 128, 63, 125, Vec, RowMajor, NoneBox, Vec, ColMajor, RowMajor)
TMOV_TEST(float, 64, 128, 63, 125, Vec, ColMajor, RowMajor, Vec, RowMajor, NoneBox)
TMOV_TEST(float, 64, 128, 63, 125, Vec, ColMajor, NoneBox, Vec, ColMajor, RowMajor)
TMOV_TEST(float, 64, 128, 63, 125, Vec, ColMajor, RowMajor, Vec, ColMajor, NoneBox)

TMOV_TEST(float, 64, 128, 64, 128, Acc, RowMajor, NoneBox, Vec, RowMajor, NoneBox)
TMOV_TEST(float, 64, 128, 64, 128, Acc, RowMajor, NoneBox, Vec, ColMajor, NoneBox)
TMOV_TEST(float, 64, 128, 64, 128, Acc, ColMajor, NoneBox, Vec, ColMajor, NoneBox)
TMOV_TEST(float, 64, 128, 64, 128, Acc, ColMajor, NoneBox, Vec, RowMajor, NoneBox)
TMOV_TEST(float, 64, 128, 64, 128, Acc, RowMajor, NoneBox, Vec, ColMajor, RowMajor)
TMOV_TEST(float, 64, 128, 64, 128, Acc, ColMajor, RowMajor, Vec, RowMajor, ColMajor)
TMOV_TEST(float, 64, 128, 64, 128, Acc, ColMajor, ColMajor, Vec, RowMajor, RowMajor)
TMOV_TEST(float, 64, 128, 64, 128, Acc, RowMajor, RowMajor, Vec, ColMajor, ColMajor)
TMOV_TEST(float, 64, 128, 64, 128, Acc, RowMajor, ColMajor, Vec, ColMajor, RowMajor)

TMOV_TEST(float, 16, 24, 15, 23, Acc, RowMajor, NoneBox, Vec, RowMajor, NoneBox)
TMOV_TEST(float, 64, 128, 64, 125, Acc, RowMajor, NoneBox, Vec, ColMajor, NoneBox)
TMOV_TEST(float, 64, 128, 64, 125, Acc, ColMajor, NoneBox, Vec, ColMajor, NoneBox)
TMOV_TEST(float, 64, 128, 64, 125, Acc, ColMajor, NoneBox, Vec, RowMajor, NoneBox)
TMOV_TEST(float, 64, 128, 64, 125, Acc, RowMajor, NoneBox, Vec, ColMajor, RowMajor)
TMOV_TEST(float, 64, 128, 64, 125, Acc, ColMajor, RowMajor, Vec, RowMajor, ColMajor)
TMOV_TEST(float, 64, 128, 64, 125, Acc, ColMajor, ColMajor, Vec, RowMajor, RowMajor)
TMOV_TEST(float, 64, 128, 64, 125, Acc, RowMajor, RowMajor, Vec, ColMajor, ColMajor)
TMOV_TEST(float, 64, 128, 64, 125, Acc, RowMajor, ColMajor, Vec, ColMajor, RowMajor)

TMOV_TEST(half, 32, 48, 15, 23, Vec, RowMajor, NoneBox, Vec, RowMajor, NoneBox)
TMOV_TEST(half, 64, 128, 63, 125, Vec, RowMajor, NoneBox, Vec, ColMajor, NoneBox)
TMOV_TEST(half, 64, 128, 63, 125, Vec, ColMajor, NoneBox, Vec, ColMajor, NoneBox)
TMOV_TEST(half, 64, 128, 63, 125, Vec, ColMajor, NoneBox, Vec, RowMajor, NoneBox)
TMOV_TEST(half, 64, 128, 63, 125, Vec, RowMajor, NoneBox, Vec, ColMajor, RowMajor)
TMOV_TEST(half, 64, 128, 63, 125, Vec, ColMajor, RowMajor, Vec, RowMajor, NoneBox)
TMOV_TEST(half, 64, 128, 63, 125, Vec, ColMajor, NoneBox, Vec, ColMajor, RowMajor)
TMOV_TEST(half, 64, 128, 63, 125, Vec, ColMajor, RowMajor, Vec, ColMajor, NoneBox)

TMOV_TEST(half, 64, 128, 64, 128, Acc, RowMajor, NoneBox, Vec, RowMajor, NoneBox)
TMOV_TEST(half, 64, 128, 64, 128, Acc, RowMajor, NoneBox, Vec, ColMajor, NoneBox)
TMOV_TEST(half, 64, 128, 64, 128, Acc, ColMajor, NoneBox, Vec, ColMajor, NoneBox)
TMOV_TEST(half, 64, 128, 64, 128, Acc, ColMajor, NoneBox, Vec, RowMajor, NoneBox)
TMOV_TEST(half, 64, 128, 64, 128, Acc, RowMajor, NoneBox, Vec, ColMajor, RowMajor)
TMOV_TEST(half, 64, 128, 64, 128, Acc, ColMajor, RowMajor, Vec, RowMajor, ColMajor)
TMOV_TEST(half, 64, 128, 64, 128, Acc, ColMajor, ColMajor, Vec, RowMajor, RowMajor)
TMOV_TEST(half, 64, 128, 64, 128, Acc, RowMajor, RowMajor, Vec, ColMajor, ColMajor)
TMOV_TEST(half, 64, 128, 64, 128, Acc, RowMajor, ColMajor, Vec, ColMajor, RowMajor)

#if defined(PTO_CPU_SIM_ENABLE_BF16)
TMOV_TEST(bfloat16_t, 32, 48, 15, 23, Vec, RowMajor, NoneBox, Vec, RowMajor, NoneBox)
TMOV_TEST(bfloat16_t, 64, 128, 63, 125, Vec, RowMajor, NoneBox, Vec, ColMajor, NoneBox)
TMOV_TEST(bfloat16_t, 64, 128, 63, 125, Vec, ColMajor, NoneBox, Vec, ColMajor, NoneBox)
TMOV_TEST(bfloat16_t, 64, 128, 63, 125, Vec, ColMajor, NoneBox, Vec, RowMajor, NoneBox)
TMOV_TEST(bfloat16_t, 64, 128, 63, 125, Vec, RowMajor, NoneBox, Vec, ColMajor, RowMajor)
TMOV_TEST(bfloat16_t, 64, 128, 63, 125, Vec, ColMajor, RowMajor, Vec, RowMajor, NoneBox)
TMOV_TEST(bfloat16_t, 64, 128, 63, 125, Vec, ColMajor, NoneBox, Vec, ColMajor, RowMajor)
TMOV_TEST(bfloat16_t, 64, 128, 63, 125, Vec, ColMajor, RowMajor, Vec, ColMajor, NoneBox)

TMOV_TEST(bfloat16_t, 64, 128, 64, 128, Acc, RowMajor, NoneBox, Vec, RowMajor, NoneBox)
TMOV_TEST(bfloat16_t, 64, 128, 64, 128, Acc, RowMajor, NoneBox, Vec, ColMajor, NoneBox)
TMOV_TEST(bfloat16_t, 64, 128, 64, 128, Acc, ColMajor, NoneBox, Vec, ColMajor, NoneBox)
TMOV_TEST(bfloat16_t, 64, 128, 64, 128, Acc, ColMajor, NoneBox, Vec, RowMajor, NoneBox)
TMOV_TEST(bfloat16_t, 64, 128, 64, 128, Acc, RowMajor, NoneBox, Vec, ColMajor, RowMajor)
TMOV_TEST(bfloat16_t, 64, 128, 64, 128, Acc, ColMajor, RowMajor, Vec, RowMajor, ColMajor)
TMOV_TEST(bfloat16_t, 64, 128, 64, 128, Acc, ColMajor, ColMajor, Vec, RowMajor, RowMajor)
TMOV_TEST(bfloat16_t, 64, 128, 64, 128, Acc, RowMajor, RowMajor, Vec, ColMajor, ColMajor)
TMOV_TEST(bfloat16_t, 64, 128, 64, 128, Acc, RowMajor, ColMajor, Vec, ColMajor, RowMajor)
#endif