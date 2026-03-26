/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <pto/pto-inst.hpp>
#include <pto/cpu/tile_offsets.hpp>
#include <sstream>
#include <string>
#include <vector>
#include <gtest/gtest.h>

using namespace pto;

namespace {

template <typename TileData>
void FillLinear(TileData &tile, typename TileData::DType start = typename TileData::DType(1))
{
    auto value = start;
    for (int r = 0; r < tile.GetValidRow(); ++r) {
        for (int c = 0; c < tile.GetValidCol(); ++c) {
            tile.data()[GetTileElementOffset<TileData>(r, c)] = value++;
        }
    }
}

template <typename TileData>
void FillAll(TileData &tile, typename TileData::DType value)
{
    std::fill(tile.data(), tile.data() + TileData::Numel, value);
}

template <typename TileData>
auto GetValue(const TileData &tile, int r, int c) -> typename TileData::DType
{
    return tile.data()[GetTileElementOffset<TileData>(r, c)];
}

template <typename TileData>
void SetValue(TileData &tile, int r, int c, typename TileData::DType value)
{
    tile.data()[GetTileElementOffset<TileData>(r, c)] = value;
}

template <typename AccTile, typename LeftTile, typename RightTile>
std::vector<typename AccTile::DType> ComputeMatmulExpected(const LeftTile &lhs, const RightTile &rhs,
                                                           const AccTile *acc = nullptr, const float *bias = nullptr)
{
    std::vector<typename AccTile::DType> expected(AccTile::Numel, typename AccTile::DType(0));
    for (int r = 0; r < lhs.GetValidRow(); ++r) {
        for (int c = 0; c < rhs.GetValidCol(); ++c) {
            typename AccTile::DType value = acc ? GetValue(*acc, r, c) : typename AccTile::DType(0);
            for (int k = 0; k < lhs.GetValidCol(); ++k) {
                value += static_cast<typename AccTile::DType>(GetValue(lhs, r, k)) *
                         static_cast<typename AccTile::DType>(GetValue(rhs, k, c));
            }
            if (bias != nullptr) {
                value += static_cast<typename AccTile::DType>(bias[c]);
            }
            expected[GetTileElementOffset<AccTile>(r, c)] = value;
        }
    }
    return expected;
}

template <typename TileData>
void ExpectTileEqualsVector(const TileData &tile, const std::vector<typename TileData::DType> &expected)
{
    ASSERT_EQ(expected.size(), static_cast<size_t>(TileData::Numel));
    for (int i = 0; i < TileData::Numel; ++i) {
        if constexpr (std::is_floating_point_v<typename TileData::DType>) {
            EXPECT_FLOAT_EQ(tile.data()[i], expected[i]);
        } else {
            EXPECT_EQ(tile.data()[i], expected[i]);
        }
    }
}

class CpuSimMissingOpsTest : public testing::Test {
};

TEST_F(CpuSimMissingOpsTest, TaxpyAccumulatesScaledSource)
{
    using TileData = Tile<TileType::Vec, float, 2, 8>;
    TileData dst;
    TileData src;

    FillLinear(dst, 10.0f);
    FillLinear(src, 1.0f);
    TAXPY(dst, src, 0.5f);

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            const float originalDst = 10.0f + static_cast<float>(r * dst.GetValidCol() + c);
            const float srcValue = 1.0f + static_cast<float>(r * src.GetValidCol() + c);
            EXPECT_FLOAT_EQ(GetValue(dst, r, c), originalDst + srcValue * 0.5f);
        }
    }
}

TEST_F(CpuSimMissingOpsTest, TfmodAndTfmodsUseFloatingPointRemainder)
{
    using TileData = Tile<TileType::Vec, float, 2, 8>;
    TileData dstVec;
    TileData src0;
    TileData src1;
    TileData dstScalar;

    FillAll(dstVec, 0.0f);
    FillAll(dstScalar, 0.0f);
    for (int r = 0; r < src0.GetValidRow(); ++r) {
        for (int c = 0; c < src0.GetValidCol(); ++c) {
            SetValue(src0, r, c, 10.0f + static_cast<float>(r + c));
            SetValue(src1, r, c, 3.0f + static_cast<float>((r + c) % 3));
        }
    }

    TFMOD(dstVec, src0, src1);
    TFMODS(dstScalar, src0, 4.0f);

    for (int r = 0; r < src0.GetValidRow(); ++r) {
        for (int c = 0; c < src0.GetValidCol(); ++c) {
            EXPECT_FLOAT_EQ(GetValue(dstVec, r, c), std::fmod(GetValue(src0, r, c), GetValue(src1, r, c)));
            EXPECT_FLOAT_EQ(GetValue(dstScalar, r, c), std::fmod(GetValue(src0, r, c), 4.0f));
        }
    }
}

TEST_F(CpuSimMissingOpsTest, TfillpadInplaceAndExpandPadRemainingElements)
{
    using InplaceDst = Tile<TileType::Vec, int16_t, 4, 16, BLayout::RowMajor, 4, 16, SLayout::NoneBox,
                            TileConfig::fractalABSize, PadValue::Max>;
    using SrcTile = Tile<TileType::Vec, int16_t, 4, 16, BLayout::RowMajor, 3, 8>;
    using ExpandDst = Tile<TileType::Vec, int16_t, 5, 16, BLayout::RowMajor, 5, 16, SLayout::NoneBox,
                           TileConfig::fractalABSize, PadValue::Max>;

    InplaceDst inplaceDst;
    ExpandDst expandDst;
    SrcTile src;
    FillAll(inplaceDst, 0);
    FillAll(expandDst, 0);
    FillAll(src, 0);
    FillLinear(src, static_cast<int16_t>(1));

    TFILLPAD_INPLACE(inplaceDst, src);
    TFILLPAD_EXPAND(expandDst, src);

    const int16_t pad = std::numeric_limits<int16_t>::max();
    for (int r = 0; r < inplaceDst.GetValidRow(); ++r) {
        for (int c = 0; c < inplaceDst.GetValidCol(); ++c) {
            const bool copied = r < src.GetValidRow() && c < src.GetValidCol();
            EXPECT_EQ(GetValue(inplaceDst, r, c), copied ? GetValue(src, r, c) : pad);
        }
    }
    for (int r = 0; r < expandDst.GetValidRow(); ++r) {
        for (int c = 0; c < expandDst.GetValidCol(); ++c) {
            const bool copied = r < src.GetValidRow() && c < src.GetValidCol();
            EXPECT_EQ(GetValue(expandDst, r, c), copied ? GetValue(src, r, c) : pad);
        }
    }
}

TEST_F(CpuSimMissingOpsTest, TshlsAndTshrsApplyScalarShift)
{
    using TileData = Tile<TileType::Vec, int32_t, 2, 8>;
    TileData left;
    TileData right;
    TileData src;

    FillLinear(src, 1);
    TSHLS(left, src, 2);
    TSHRS(right, src, 1);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            const int32_t value = GetValue(src, r, c);
            EXPECT_EQ(GetValue(left, r, c), value << 2);
            EXPECT_EQ(GetValue(right, r, c), value >> 1);
        }
    }
}

TEST_F(CpuSimMissingOpsTest, TsubviewExtractsRequestedWindow)
{
    using SrcTile = Tile<TileType::Vec, float, 4, 8>;
    using DstTile = Tile<TileType::Vec, float, 3, 8, BLayout::RowMajor, 3, 6>;
    SrcTile src;
    DstTile dst;

    FillLinear(src, 1.0f);
    TSUBVIEW(dst, src, 1, 2);

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            EXPECT_FLOAT_EQ(GetValue(dst, r, c), GetValue(src, r + 1, c + 2));
        }
    }
}

TEST_F(CpuSimMissingOpsTest, TconcatAppendsColumns)
{
    using Src0Tile = Tile<TileType::Vec, int32_t, 2, 8>;
    using Src1Tile = Tile<TileType::Vec, int32_t, 2, 8, BLayout::RowMajor, 2, 4>;
    using DstTile = Tile<TileType::Vec, int32_t, 2, 16, BLayout::RowMajor, 2, 12>;
    Src0Tile src0;
    Src1Tile src1;
    DstTile dst;

    FillLinear(src0, 1);
    FillLinear(src1, 101);
    FillAll(dst, 0);
    TCONCAT(dst, src0, src1);

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < src0.GetValidCol(); ++c) {
            EXPECT_EQ(GetValue(dst, r, c), GetValue(src0, r, c));
        }
        for (int c = 0; c < src1.GetValidCol(); ++c) {
            EXPECT_EQ(GetValue(dst, r, src0.GetValidCol() + c), GetValue(src1, r, c));
        }
    }
}

TEST_F(CpuSimMissingOpsTest, TinsertCopiesIntoDestinationAtOffset)
{
    using DstTile = Tile<TileType::Vec, float, 4, 16>;
    using SrcTile = Tile<TileType::Vec, float, 2, 8>;
    using FpTile = Tile<TileType::Vec, float, 1, 8>;
    DstTile plainDst;
    DstTile scalarDst;
    DstTile fpDst;
    SrcTile src;
    FpTile fp;

    FillAll(plainDst, 0.0f);
    FillAll(scalarDst, 0.0f);
    FillAll(fpDst, 0.0f);
    FillLinear(src, 1.0f);
    FillAll(fp, 2.0f);

    TINSERT(plainDst, src, 1, 4);
    TINSERT<DstTile, SrcTile>(scalarDst, src, static_cast<uint64_t>(7), 1, 4);
    TINSERT_FP<DstTile, SrcTile, FpTile>(fpDst, src, fp, 1, 4);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            const float value = GetValue(src, r, c);
            EXPECT_FLOAT_EQ(GetValue(plainDst, r + 1, c + 4), value);
            EXPECT_FLOAT_EQ(GetValue(scalarDst, r + 1, c + 4), value);
            EXPECT_FLOAT_EQ(GetValue(fpDst, r + 1, c + 4), value);
        }
    }
}

TEST_F(CpuSimMissingOpsTest, TpartmulMultipliesOverlapAndCopiesRemainder)
{
    using DstTile = Tile<TileType::Vec, int32_t, 2, 8>;
    using Src0Tile = Tile<TileType::Vec, int32_t, 2, 8>;
    using Src1Tile = Tile<TileType::Vec, int32_t, 2, 8, BLayout::RowMajor, 1, 4>;
    DstTile dst;
    Src0Tile src0;
    Src1Tile src1;

    FillLinear(src0, 1);
    FillLinear(src1, 10);
    TPARTMUL(dst, src0, src1);

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            if (r < src1.GetValidRow() && c < src1.GetValidCol()) {
                EXPECT_EQ(GetValue(dst, r, c), GetValue(src0, r, c) * GetValue(src1, r, c));
            } else {
                EXPECT_EQ(GetValue(dst, r, c), GetValue(src0, r, c));
            }
        }
    }
}

TEST_F(CpuSimMissingOpsTest, TprintWritesReadableMatrix)
{
    using TileData = Tile<TileType::Vec, int32_t, 2, 8, BLayout::RowMajor, 2, 4>;
    TileData src;
    FillLinear(src, 1);

    std::ostringstream captured;
    auto *old = std::cout.rdbuf(captured.rdbuf());
    TPRINT(src);
    std::cout.rdbuf(old);

    EXPECT_EQ(captured.str(), std::string("TPRINT 2x4\n1 2 3 4\n5 6 7 8\n"));
}

TEST_F(CpuSimMissingOpsTest, TgetScaleAddrAliasesSourceStorage)
{
    using TileData = Tile<TileType::Vec, float, 2, 8>;
    TileData src;
    TileData dst;
    FillLinear(src, 1.0f);

    TGET_SCALE_ADDR(dst, src);

    ASSERT_EQ(dst.data(), src.data());
    src.data()[3] = 42.0f;
    EXPECT_FLOAT_EQ(dst.data()[3], 42.0f);
}

TEST_F(CpuSimMissingOpsTest, TpackCopiesValidValues)
{
    using SrcTile = Tile<TileType::Vec, int32_t, 2, 8>;
    using DstTile = Tile<TileType::Vec, int16_t, 2, 16, BLayout::RowMajor, 2, 8>;
    SrcTile src;
    DstTile dst;
    FillLinear(src, 1);
    FillAll(dst, 0);

    TPACK_IMPL(dst, src);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            EXPECT_EQ(GetValue(dst, r, c), static_cast<int16_t>(GetValue(src, r, c)));
        }
    }
}

TEST_F(CpuSimMissingOpsTest, TcolprodAndTrowprodReduceProducts)
{
    using SrcTile = Tile<TileType::Vec, float, 3, 8>;
    using ColDst = Tile<TileType::Vec, float, 1, 8>;
    using RowDst = Tile<TileType::Vec, float, 3, 8, BLayout::RowMajor, 3, 1>;
    using TmpTile = Tile<TileType::Vec, float, 3, 8>;
    SrcTile src;
    ColDst colDst;
    RowDst rowDst;
    TmpTile tmp;

    FillLinear(src, 1.0f);
    TCOLPROD(colDst, src);
    TROWPROD(rowDst, src, tmp);

    for (int c = 0; c < src.GetValidCol(); ++c) {
        float expected = 1.0f;
        for (int r = 0; r < src.GetValidRow(); ++r) {
            expected *= GetValue(src, r, c);
        }
        EXPECT_FLOAT_EQ(GetValue(colDst, 0, c), expected);
    }
    for (int r = 0; r < src.GetValidRow(); ++r) {
        float expected = 1.0f;
        for (int c = 0; c < src.GetValidCol(); ++c) {
            expected *= GetValue(src, r, c);
        }
        EXPECT_FLOAT_EQ(GetValue(rowDst, r, 0), expected);
    }
}

TEST_F(CpuSimMissingOpsTest, TrowArgmaxAndTrowArgminReturnIndices)
{
    using SrcTile = Tile<TileType::Vec, float, 3, 8>;
    using DstTile = Tile<TileType::Vec, int32_t, 3, 8, BLayout::RowMajor, 3, 1>;
    using TmpTile = Tile<TileType::Vec, float, 3, 8>;
    SrcTile src;
    DstTile argmax;
    DstTile argmin;
    TmpTile tmp;

    FillAll(src, 0.0f);
    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            SetValue(src, r, c, static_cast<float>((r + 1) * 10 + c));
        }
    }
    SetValue(src, 0, 5, 100.0f);
    SetValue(src, 1, 2, -7.0f);
    SetValue(src, 2, 7, 99.0f);

    TROWARGMAX(argmax, src, tmp);
    TROWARGMIN(argmin, src, tmp);

    EXPECT_EQ(GetValue(argmax, 0, 0), 5);
    EXPECT_EQ(GetValue(argmax, 1, 0), 7);
    EXPECT_EQ(GetValue(argmax, 2, 0), 7);
    EXPECT_EQ(GetValue(argmin, 0, 0), 0);
    EXPECT_EQ(GetValue(argmin, 1, 0), 2);
    EXPECT_EQ(GetValue(argmin, 2, 0), 0);
}

TEST_F(CpuSimMissingOpsTest, TdequantAppliesScaleAndOffset)
{
    using DstTile = Tile<TileType::Vec, float, 2, 8>;
    using SrcTile = Tile<TileType::Vec, int32_t, 2, 8>;
    using ParaTile = Tile<TileType::Vec, float, 2, 8>;
    DstTile dst;
    SrcTile src;
    ParaTile scale;
    ParaTile offset;

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            SetValue(src, r, c, (r + 1) * 10 + c);
            SetValue(scale, r, c, 0.5f + static_cast<float>(c) * 0.25f);
            SetValue(offset, r, c, static_cast<float>(r + c));
        }
    }

    TDEQUANT(dst, src, scale, offset);

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            const float expected = (static_cast<float>(GetValue(src, r, c)) - GetValue(offset, r, c)) *
                                   GetValue(scale, r, c);
            EXPECT_FLOAT_EQ(GetValue(dst, r, c), expected);
        }
    }
}

TEST_F(CpuSimMissingOpsTest, TgemvAndMxVariantsMatchCpuMatmulSemantics)
{
    using LeftTile = TileLeft<float, 16, 16>;
    using RightTile = TileRight<float, 16, 16>;
    using AccTile = TileAcc<float, 16, 16>;
    using BiasTile = Tile<TileType::Bias, float, 1, 16>;
    using LeftScaleTile = TileLeftScale<float, 16, 2>;
    using RightScaleTile = TileRightScale<float, 16, 2>;

    LeftTile lhs;
    RightTile rhs;
    AccTile gemv;
    AccTile gemvAcc;
    AccTile gemvMx;
    AccTile matmulMx;
    AccTile accIn;
    BiasTile bias;
    LeftScaleTile lhsScale;
    RightScaleTile rhsScale;

    FillAll(lhs, 0.0f);
    FillAll(rhs, 0.0f);
    FillAll(accIn, 1.0f);
    FillAll(lhsScale, 2.0f);
    FillAll(rhsScale, 3.0f);
    for (int r = 0; r < lhs.GetValidRow(); ++r) {
        for (int c = 0; c < lhs.GetValidCol(); ++c) {
            SetValue(lhs, r, c, static_cast<float>(r + c + 1));
            SetValue(rhs, r, c, static_cast<float>((r == c) ? 2 : 1));
        }
    }
    for (int c = 0; c < bias.GetValidCol(); ++c) {
        SetValue(bias, 0, c, static_cast<float>(c));
    }

    TGEMV(gemv, lhs, rhs);
    TGEMV_ACC(gemvAcc, accIn, lhs, rhs);
    TGEMV_MX_IMPL(gemvMx, lhs, lhsScale, rhs, rhsScale);
    TMATMUL_MX_IMPL(matmulMx, lhs, lhsScale, rhs, rhsScale);

    const auto expectedGemv = ComputeMatmulExpected<AccTile>(lhs, rhs);
    const auto expectedGemvAcc = ComputeMatmulExpected<AccTile>(lhs, rhs, &accIn);
    ExpectTileEqualsVector(gemv, expectedGemv);
    ExpectTileEqualsVector(gemvAcc, expectedGemvAcc);
    ExpectTileEqualsVector(gemvMx, expectedGemv);
    ExpectTileEqualsVector(matmulMx, expectedGemv);

    AccTile gemvBias;
    TGEMV_BIAS(gemvBias, lhs, rhs, bias);
    std::vector<float> biasValues(bias.GetValidCol());
    for (int c = 0; c < bias.GetValidCol(); ++c) {
        biasValues[c] = GetValue(bias, 0, c);
    }
    const auto expectedGemvBias = ComputeMatmulExpected<AccTile>(lhs, rhs, nullptr, biasValues.data());
    ExpectTileEqualsVector(gemvBias, expectedGemvBias);
}

} // namespace
