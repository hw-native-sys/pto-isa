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
#include <bit>
#include <cstdint>
#include <tuple>

#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace {
constexpr int LEGACY = -1;
using Scaling = Tile<TileType::Scaling, uint64_t, 1, 64, BLayout::RowMajor>;

uint64_t Scale(float value) { return std::bit_cast<uint32_t>(value); }

float SourceValue(int r, int c) { return static_cast<float>((r * 3 + c) % 9 - 4); }

// Kind: plain, plain+ReLU, scalar, scalar+ReLU, vector, vector+ReLU.
template <int Phase, int Kind, typename D>
void CheckMove()
{
    constexpr bool relu = Kind % 2 != 0;
    constexpr auto reluMode = relu ? ReluPreMode::NormalRelu : ReluPreMode::NoRelu;
    using S = std::conditional_t<(Kind < 2), float, int32_t>;
    using Src = TileAcc<S, 32, 64, -1, -1>;
    using Dst = Tile<TileType::Mat, D, 32, 64, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    Src src(29, 9);
    Dst dst(17, 7);
    Scaling fp;
    TASSIGN(src, 0);
    TASSIGN(dst, 0);
    TASSIGN(fp, 0);
    for (int r = 0; r < 32; ++r) {
        for (int c = 0; c < 64; ++c) {
            src.SetElement(r, c, static_cast<S>(SourceValue(r, c)));
            dst.SetElement(r, c, static_cast<D>(-7));
        }
    }
    for (int c = 0; c < 64; ++c) {
        fp.SetElement(0, c, Scale(static_cast<float>(c % 3 + 1)) | (uint64_t{1} << 46));
    }
    const uint64_t scalar = Scale(2.0f) | (uint64_t{1} << 46);
    if constexpr (Kind < 2) {
        if constexpr (Phase == LEGACY && !relu) {
            TMOV(dst, src);
        } else if constexpr (Phase == LEGACY) {
            TMOV<Dst, Src, reluMode>(dst, src);
        } else {
            TMOV<static_cast<STPhase>(Phase), Dst, Src, reluMode>(dst, src);
        }
    } else if constexpr (Kind < 4) {
        if constexpr (Phase == LEGACY) {
            TMOV<Dst, Src, reluMode>(dst, src, scalar);
        } else {
            TMOV<static_cast<STPhase>(Phase), Dst, Src, reluMode>(dst, src, scalar);
        }
    } else {
        if constexpr (Phase == LEGACY) {
            TMOV<Dst, Src, Scaling, reluMode>(dst, src, fp);
        } else {
            TMOV<static_cast<STPhase>(Phase), Dst, Src, Scaling, reluMode>(dst, src, fp);
        }
    }
    // Nine valid columns require 16 half columns or 32 int8 columns. Rows 29..31
    // and the column padding contain nonzero input, so a valid-only copy fails.
    constexpr int copiedCols = sizeof(D) == 2 ? 16 : 32;
    for (int r = 0; r < 32; ++r) {
        for (int c = 0; c < 64; ++c) {
            float expected = -7;
            if (c < copiedCols) {
                const float scale = Kind < 2 ? 1 : (Kind < 4 ? 2 : c % 3 + 1);
                expected = SourceValue(r, c) * scale;
                if constexpr (relu) {
                    expected = std::max(expected, 0.0f);
                }
            }
            ASSERT_EQ(static_cast<float>(dst.GetElement(r, c)), expected) << "row=" << r << " col=" << c;
        }
    }
    EXPECT_EQ(src.GetValidRow(), 29U);
    EXPECT_EQ(src.GetValidCol(), 9U);
    EXPECT_EQ(dst.GetValidRow(), 17U);
    EXPECT_EQ(dst.GetValidCol(), 7U);
}

template <int Phase, int Kind>
void CheckMoveTypes()
{
    CheckMove<Phase, Kind, half>();
    if constexpr (Kind >= 2) {
        CheckMove<Phase, Kind, int8_t>();
    }
}

template <int Phase>
void CheckMoveKind(int kind)
{
    switch (kind) {
        case 0:
            CheckMoveTypes<Phase, 0>();
            break;
        case 1:
            CheckMoveTypes<Phase, 1>();
            break;
        case 2:
            CheckMoveTypes<Phase, 2>();
            break;
        case 3:
            CheckMoveTypes<Phase, 3>();
            break;
        case 4:
            CheckMoveTypes<Phase, 4>();
            break;
        case 5:
            CheckMoveTypes<Phase, 5>();
            break;
    }
}

class AccToMatFootprint : public testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(AccToMatFootprint, CopiesAlignedPhysicalRegionOnly)
{
    const auto [phase, kind] = GetParam();
    switch (phase) {
        case LEGACY:
            CheckMoveKind<LEGACY>(kind);
            break;
        case int(STPhase::Unspecified):
            CheckMoveKind<int(STPhase::Unspecified)>(kind);
            break;
        case int(STPhase::Partial):
            CheckMoveKind<int(STPhase::Partial)>(kind);
            break;
        case int(STPhase::Final):
            CheckMoveKind<int(STPhase::Final)>(kind);
            break;
    }
}

INSTANTIATE_TEST_SUITE_P(
    LegacyAndPhases, AccToMatFootprint,
    testing::Combine(
        testing::Values(LEGACY, int(STPhase::Unspecified), int(STPhase::Partial), int(STPhase::Final)),
        testing::Range(0, 6)));

template <int Phase>
void CheckExtract(bool vector)
{
    using Src = TileAcc<int32_t, 32, 64>;
    using Dst = Tile<TileType::Mat, half, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor>;
    Src src;
    Dst dst;
    Scaling fp;
    TASSIGN(src, 0);
    TASSIGN(dst, 0);
    TASSIGN(fp, 0);
    for (int r = 0; r < 32; ++r) {
        for (int c = 0; c < 64; ++c) {
            src.SetElement(r, c, static_cast<int32_t>(SourceValue(r, c)));
        }
    }
    for (int c = 0; c < 64; ++c) {
        fp.SetElement(0, c, Scale(static_cast<float>(c % 5 + 1)));
    }
    if (vector) {
        if constexpr (Phase == LEGACY) {
            TEXTRACT(dst, src, fp, 16, 48);
        } else {
            TEXTRACT<static_cast<STPhase>(Phase)>(dst, src, fp, 16, 48);
        }
    } else {
        if constexpr (Phase == LEGACY) {
            TEXTRACT(dst, src, Scale(2.0f), 16, 48);
        } else {
            TEXTRACT<static_cast<STPhase>(Phase)>(dst, src, Scale(2.0f), 16, 48);
        }
    }
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 16; ++c) {
            const float scale = vector ? (c + 48) % 5 + 1 : 2;
            ASSERT_EQ(static_cast<float>(dst.GetElement(r, c)), SourceValue(r + 16, c + 48) * scale);
        }
    }
}

class ExtractQuantOffset : public testing::TestWithParam<std::tuple<int, bool>> {};

TEST_P(ExtractQuantOffset, SelectsScalarOrSourceColumnScale)
{
    const auto [phase, vector] = GetParam();
    switch (phase) {
        case LEGACY:
            CheckExtract<LEGACY>(vector);
            break;
        case int(STPhase::Unspecified):
            CheckExtract<int(STPhase::Unspecified)>(vector);
            break;
        case int(STPhase::Partial):
            CheckExtract<int(STPhase::Partial)>(vector);
            break;
        case int(STPhase::Final):
            CheckExtract<int(STPhase::Final)>(vector);
            break;
    }
}

INSTANTIATE_TEST_SUITE_P(
    LegacyAndPhases, ExtractQuantOffset,
    testing::Combine(
        testing::Values(LEGACY, int(STPhase::Unspecified), int(STPhase::Partial), int(STPhase::Final)),
        testing::Bool()));

TEST(AccToMatBoundsDeathTest, RejectsDestinationSmallerThanPhysicalTransfer)
{
    TileAcc<float, 32, 32, 16, 16> src;
    Tile<TileType::Mat, half, 16, 32, BLayout::ColMajor, 16, 16, SLayout::RowMajor> dst;
    TASSIGN(src, 0);
    TASSIGN(dst, 0);
    ASSERT_DEATH(TMOV(dst, src), "SrcTileData::Rows <= DstTileData::Rows");
}

TEST(AccToMatBoundsDeathTest, RejectsMissingScalesForAlignedColumns)
{
    TileAcc<int32_t, 32, 32, 29, 9> src;
    Tile<TileType::Mat, half, 32, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor> dst;
    Tile<TileType::Scaling, uint64_t, 1, 32, BLayout::RowMajor, 1, 9> fp;
    TASSIGN(src, 0);
    TASSIGN(dst, 0);
    TASSIGN(fp, 0);
    std::fill_n(src.data(), src.Numel, 3);
    std::fill_n(fp.data(), fp.Numel, Scale(2.0f));
    ASSERT_DEATH(TMOV(dst, src, fp), "c < fp.GetValidCol");
}

TEST(AccToMatBoundsDeathTest, PreservesExtractValidRegionCheck)
{
    TileAcc<int32_t, 32, 32, 29, 9> src;
    Tile<TileType::Mat, half, 16, 16, BLayout::ColMajor, 16, 16, SLayout::RowMajor> dst;
    TASSIGN(src, 0);
    TASSIGN(dst, 0);
    ASSERT_DEATH(TEXTRACT(dst, src, Scale(2.0f), 0, 0), "dst.GetValidCol.*src.GetValidCol");
}

TEST(ExtractQuantOffset, ScalarAccToVecModeAtNonzeroColumn)
{
    using Src = TileAcc<int32_t, 16, 32>;
    using Dst = Tile<TileType::Vec, half, 16, 16>;
    Src src;
    Dst dst;
    TASSIGN(src, 0);
    TASSIGN(dst, 0);
    std::fill_n(src.data(), Src::Numel, 3);
    TEXTRACT<Dst, Src, AccToVecMode::SingleModeVec0>(dst, src, Scale(2.0f), 0, 16);
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 16; ++c) {
            ASSERT_EQ(static_cast<float>(dst.GetElement(r, c)), 6.0f);
        }
    }
}

TEST(AccToMatBoundsDeathTest, PreservesOtherMoveValidShapeCheck)
{
    Tile<TileType::Vec, float, 8, 8, BLayout::RowMajor, 4, 8> src;
    Tile<TileType::Vec, float, 8, 8> dst;
    TASSIGN(src, 0);
    TASSIGN(dst, 1024);
    ASSERT_DEATH(TMOV(dst, src), "src.GetValidRow.*dst.GetValidRow");
}
} // namespace
