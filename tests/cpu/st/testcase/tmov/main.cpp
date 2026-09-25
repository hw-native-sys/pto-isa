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
#include <pto/common/constants.hpp>
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

template <typename T, int rows, int cols>
void testMovNdToZn()
{
    constexpr int k0 = BLOCK_BYTE_SIZE / sizeof(T);

    using SrcShape = Shape<1, 1, 1, rows, cols>;
    using SrcStride = pto::Stride<1, 1, 1, cols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;
    using DstShape = Shape<1, 1, 1, rows, cols>;
    using DstStride = pto::Stride<1, 1, 1, cols, 1>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride>;

    using SrcTile = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, -1, -1>;
    using ZnTile = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, -1, -1, SLayout::ColMajor>;
    using NdTile = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, -1, -1>;

    SrcTile srcTile(rows, cols);
    ZnTile znTile(rows, cols);
    NdTile ndTile(rows, cols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(znTile, 0x10000);
    TASSIGN(ndTile, 0x10000);

    std::vector<T> srcData(rows * cols);
    std::vector<T> dstData(rows * cols, 0);

    for (size_t i = 0; i < srcData.size(); i++) {
        srcData[i] = static_cast<T>(std::rand() / 1000.0);
    }

    SrcGlobal srcGlobal(srcData.data());
    DstGlobal dstGlobal(dstData.data());

    TLOAD(srcTile, srcGlobal);
    TMOV(znTile, srcTile);
    TSTORE(dstGlobal, ndTile);

    std::vector<T> expected(rows * cols);

    for (int br = 0; br < rows / k0; ++br) {
        for (int bc = 0; bc < cols / 16; ++bc) {
            for (int j = 0; j < 16; ++j) {
                for (int i = 0; i < k0; ++i) {
                    const size_t srcIndex = br * k0 * cols + i * cols + bc * 16 + j;
                    const size_t dstIndex = br * k0 * cols + bc * 16 * k0 + j * k0 + i;
                    expected[dstIndex] = srcData[srcIndex];
                }
            }
        }
    }

    EXPECT_TRUE(ResultCmp(expected, dstData.data(), 0));
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

#define TMOV_ND_TO_ZN_TEST(T, rows, cols) \
    TEST_F(TMOVTest, T##_##rows##_##cols##_ND_To_ZN) { testMovNdToZn<T, rows, cols>(); }

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

TMOV_ND_TO_ZN_TEST(float, 16, 32)
TMOV_ND_TO_ZN_TEST(float, 32, 32)
TMOV_ND_TO_ZN_TEST(float, 32, 64)
TMOV_ND_TO_ZN_TEST(float, 64, 64)
TMOV_ND_TO_ZN_TEST(float, 128, 128)
