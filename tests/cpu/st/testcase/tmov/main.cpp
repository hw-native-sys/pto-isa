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

// Compare encoded bytes: E8M0 has no numerical zero, but native X->ZZ pads with byte zero.
template <typename T, bool dn = false, bool tagged = false, bool shaped = false, bool wideDst = false>
void testMovMxScale(unsigned validRows)
{
    constexpr unsigned rows = 32;
    constexpr unsigned cols = shaped ? 32 : 8;
    constexpr unsigned dstCols = wideDst ? cols * 2 : cols;
    constexpr unsigned srcRows = dn ? cols : (shaped ? rows : 1);
    constexpr unsigned srcCols = dn ? rows : (shaped ? cols : rows * cols);
    using Src = Tile<TileType::Vec, T, srcRows, srcCols, BLayout::RowMajor, -1, -1>;
    using Dst = Tile<TileType::Vec, T, rows, dstCols, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 32>;
    using Tmp = Tile<TileType::Vec, T, 1, 128>;
    Src src(dn ? cols : (shaped ? validRows : 1), dn ? validRows : (shaped ? cols : validRows * cols));
    Dst dst(validRows, cols);
    Tmp tmp;
    TASSIGN(src, 0);
    TASSIGN(dst, 16384);
    TASSIGN(tmp, 32768);
    auto* input = reinterpret_cast<uint8_t*>(src.data());
    auto* output = reinterpret_cast<uint8_t*>(dst.data());
    std::fill(input, input + rows * cols, uint8_t{239});
    std::fill(output, output + rows * dstCols, uint8_t{167});
    for (unsigned r = 0; r < validRows; ++r) {
        for (unsigned c = 0; c < cols; ++c) {
            input[dn ? c * validRows + r : r * cols + c] = static_cast<uint8_t>((r * cols + c) * 17 + 11);
        }
    }
    if constexpr (dn) {
        TMOV<0>(dst, src, tmp);
    } else if constexpr (tagged) {
        TMOV<1>(dst, src, tmp);
    } else {
        TMOV(dst, src, tmp);
    }
    const unsigned paddedRows = (validRows + 15) / 16 * 16;
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned c = 0; c < cols; ++c) {
            const unsigned offset = (r / 16) * cols * 16 + (c / 2) * 32 + (r % 16) * 2 + c % 2;
            const uint8_t expected = r < validRows  ? static_cast<uint8_t>((r * cols + c) * 17 + 11) :
                                     r < paddedRows ? uint8_t{0} :
                                                      uint8_t{167};
            EXPECT_EQ(output[offset], expected) << "row=" << r << " col=" << c;
        }
    }
    for (unsigned i = rows * cols; i < rows * dstCols; ++i) {
        EXPECT_EQ(output[i], uint8_t{167});
    }
    if constexpr (!dn) {
        for (unsigned i = validRows * cols; i < paddedRows * cols; ++i) {
            EXPECT_EQ(input[i], uint8_t{0});
        }
    }
}

TEST_F(TMOVTest, mx_scale_flat_nd_to_zz)
{
    for (unsigned rows : {1U, 12U, 15U, 16U, 17U, 32U}) {
        testMovMxScale<uint8_t>(rows);
        testMovMxScale<hifloat8_t>(rows);
        testMovMxScale<float8_e8m0_t>(rows);
    }
}

TEST_F(TMOVTest, mx_scale_tagged_and_dense_nd_to_zz)
{
    testMovMxScale<uint8_t, false, true>(17);
    testMovMxScale<uint8_t, false, false, false, true>(17);
    testMovMxScale<float8_e8m0_t, false, true, true>(32);
    testMovMxScale<float8_e8m0_t, false, true, true>(17);
}

TEST_F(TMOVTest, mx_scale_dn_to_zz)
{
    testMovMxScale<uint8_t, true>(32);
    testMovMxScale<uint8_t, true>(16);
    testMovMxScale<hifloat8_t, true>(32);
    testMovMxScale<float8_e8m0_t, true>(32);
}

TEST_F(TMOVTest, non_mx_three_argument_fallback)
{
    using Data = Tile<TileType::Vec, float, 2, 8>;
    using Fp = Tile<TileType::Vec, uint64_t, 1, 8>;
    Data src, dst;
    Fp fp;
    TASSIGN(src, 0);
    TASSIGN(dst, 16384);
    TASSIGN(fp, 32768);
    std::fill(fp.data(), fp.data() + 8, uint64_t{0});
    for (unsigned i = 0; i < 16; ++i) {
        src.data()[i] = (static_cast<float>(i) - 7) / 4;
    }
    TMOV(dst, src, fp);
    for (unsigned i = 0; i < 16; ++i) {
        EXPECT_EQ(dst.data()[i], src.data()[i]);
    }
}
