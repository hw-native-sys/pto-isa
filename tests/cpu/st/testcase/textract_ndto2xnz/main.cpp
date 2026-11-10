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
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

#include "cpu_tile_test_utils.h"

using namespace pto;

class TExtractNdTo2xNzTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Verify one NZ window of an ND->2xNZ extraction both logically (GetValue) and
// physically (the NZ fractal placement inside data()). The physical formula is the
// same Nz layout used by pto::GetTileElementOffset for a ColMajor / RowMajor
// fractal tile, i.e. outer blocks column-major, inner boxes row-major.
template <typename DstTile, typename SrcTile>
void CheckNzWindow(const DstTile& dst, const SrcTile& src, uint16_t ir, uint16_t ic)
{
    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            const auto expected = CpuTileTestUtils::GetValue(src, r + ir, c + ic);
            CpuTileTestUtils::ExpectValueEquals(CpuTileTestUtils::GetValue(dst, r, c), expected);
            const int subTileR = r / DstTile::InnerRows;
            const int innerR = r % DstTile::InnerRows;
            const int subTileC = c / DstTile::InnerCols;
            const int innerC = c % DstTile::InnerCols;
            const size_t nzOffset = static_cast<size_t>(subTileC) * DstTile::Rows * DstTile::InnerCols +
                                    static_cast<size_t>(subTileR) * DstTile::InnerNumel +
                                    static_cast<size_t>(innerR) * DstTile::InnerCols + static_cast<size_t>(innerC);
            CpuTileTestUtils::ExpectValueEquals(dst.data()[nzOffset], expected);
        }
    }
}

// Native compact pitch for a RowPlusOne (NZ+1) tile: one extra 32-byte row per
// column panel, i.e. ceil(Rows / InnerRows) * InnerRows + 1 rows. This is the pitch
// StorageNumel reserves and pto::GetTileElementOffset uses for non-padded tiles.
template <typename DstTile>
constexpr size_t RowPlusOnePanelPitch()
{
    constexpr size_t innerRows = static_cast<size_t>(DstTile::InnerRows);
    return (static_cast<size_t>(DstTile::Rows) + innerRows - 1) / innerRows * innerRows + 1;
}

// Verify an ND->2xNZ window stored into a RowPlusOne destination both logically
// (GetValue) and physically. Regression for the CPU bug where SetElement used the
// plain NZ pitch (Rows) and placed column block b at b * Rows * InnerCols instead of
// b * pitch * InnerCols. For a 32x64 float tile (Rows 32, valid 32) the native pitch
// is 33, so element (0,8) must land at 1 * 33 * 8 = 264, not the plain-NZ 1 * 32 * 8
// = 256.
template <typename DstTile, typename SrcTile>
void CheckNzWindowRowPlusOne(const DstTile& dst, const SrcTile& src, uint16_t ir, uint16_t ic)
{
    static_assert(DstTile::kRowPlusOneNZ, "RowPlusOne regression requires a RowPlusOne NZ tile");
    static_assert(!DstTile::kRowPlusOneRowsPadded, "RowPlusOne regression must use a native compact (non-padded) tile");
    constexpr size_t pitch = RowPlusOnePanelPitch<DstTile>();
    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            const auto expected = CpuTileTestUtils::GetValue(src, r + ir, c + ic);
            CpuTileTestUtils::ExpectValueEquals(CpuTileTestUtils::GetValue(dst, r, c), expected);
            const int subTileR = r / DstTile::InnerRows;
            const int innerR = r % DstTile::InnerRows;
            const int subTileC = c / DstTile::InnerCols;
            const int innerC = c % DstTile::InnerCols;
            const size_t offset = static_cast<size_t>(subTileC) * pitch * DstTile::InnerCols +
                                  static_cast<size_t>(subTileR) * DstTile::InnerNumel +
                                  static_cast<size_t>(innerR) * DstTile::InnerCols + static_cast<size_t>(innerC);
            CpuTileTestUtils::ExpectValueEquals(dst.data()[offset], expected);
        }
    }
}

// Dynamic-shape variant of CheckNzWindowRowPlusOne: the panel pitch derives from the
// runtime valid row (matching TINSERT SPLIT2/SPLIT4 and A5 TSTORE), not from the
// declared Rows or the padded allocation envelope. For a 33-row declaration with a
// runtime valid row of 32 the pitch is 33, so element (0,8) must land at 1 * 33 * 8
// = 264, not at the allocation pitch 1 * 49 * 8 = 392. Regression for the CPU bug
// that addressed dynamic RowPlusOne tiles by their storage capacity, breaking the
// TEXTRACT -> TINSERT path.
template <typename DstTile, typename SrcTile>
void CheckNzWindowRowPlusOneDynamic(DstTile& dst, const SrcTile& src, uint16_t ir, uint16_t ic)
{
    static_assert(DstTile::kRowPlusOneNZ, "RowPlusOne regression requires a RowPlusOne NZ tile");
    static_assert(DstTile::ValidRow == DYNAMIC, "dynamic-shape checker requires a dynamic valid row");
    const size_t pitch =
        (static_cast<size_t>(dst.GetValidRow()) + DstTile::InnerRows - 1) / DstTile::InnerRows * DstTile::InnerRows + 1;
    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            const auto expected = CpuTileTestUtils::GetValue(src, r + ir, c + ic);
            CpuTileTestUtils::ExpectValueEquals(dst.GetElement(r, c), expected);
            const int subTileR = r / DstTile::InnerRows;
            const int innerR = r % DstTile::InnerRows;
            const int subTileC = c / DstTile::InnerCols;
            const int innerC = c % DstTile::InnerCols;
            const size_t offset = static_cast<size_t>(subTileC) * pitch * DstTile::InnerCols +
                                  static_cast<size_t>(subTileR) * DstTile::InnerNumel +
                                  static_cast<size_t>(innerR) * DstTile::InnerCols + static_cast<size_t>(innerC);
            CpuTileTestUtils::ExpectValueEquals(dst.data()[offset], expected);
        }
    }
}

// Two-window ND -> 2xNZ extraction. Mirrors the A5 ST reference
// (tests/npu/a5/src/st/testcase/textract_ndto2xnz): a single ND source and two
// independent windows, each stored into its own NZ destination via the 2-destination
// TEXTRACT(dst0, dst1, src, ir0, ic0, ir1, ic1) interface (TEXTRACT_ND2XNZ_IMPL).
template <
    typename T, int SrcRows, int SrcCols, int DstRows, int DstCols, int V0Rows, int V0Cols, int V1Rows, int V1Cols>
void TestNdTo2xNz(uint16_t ir0, uint16_t ic0, uint16_t ir1, uint16_t ic1)
{
    using SrcTile = Tile<TileType::Vec, T, SrcRows, SrcCols, BLayout::RowMajor, SrcRows, SrcCols, SLayout::NoneBox>;
    using Dst0Tile = Tile<TileType::Vec, T, DstRows, DstCols, BLayout::ColMajor, V0Rows, V0Cols, SLayout::RowMajor>;
    using Dst1Tile = Tile<TileType::Vec, T, DstRows, DstCols, BLayout::ColMajor, V1Rows, V1Cols, SLayout::RowMajor>;

    static_assert(Dst0Tile::InnerCols > 0 && Dst0Tile::InnerRows > 0, "NZ destination must be boxed");
    static_assert(Dst1Tile::InnerCols > 0 && Dst1Tile::InnerRows > 0, "NZ destination must be boxed");

    SrcTile src;
    Dst0Tile dst0;
    Dst1Tile dst1;
    size_t addr = 0;
    CpuTileTestUtils::AssignTileStorage(addr, src, dst0, dst1);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            CpuTileTestUtils::SetValue(src, r, c, T((r * 31 + c + 1) % 97));
        }
    }
    // Padding must stay zero; matches the ZeroDst semantics of the A5 kernel.
    CpuTileTestUtils::FillAll(dst0, T(0));
    CpuTileTestUtils::FillAll(dst1, T(0));

    TEXTRACT(dst0, dst1, src, ir0, ic0, ir1, ic1);

    CheckNzWindow(dst0, src, ir0, ic0);
    CheckNzWindow(dst1, src, ir1, ic1);
}

TEST_F(TExtractNdTo2xNzTest, HalfAligned) { TestNdTo2xNz<half, 64, 128, 32, 64, 32, 64, 32, 64>(0, 0, 32, 64); }

TEST_F(TExtractNdTo2xNzTest, HalfUnaligned) { TestNdTo2xNz<half, 64, 128, 32, 64, 32, 64, 32, 64>(0, 0, 16, 8); }

TEST_F(TExtractNdTo2xNzTest, HalfOneByOne)
{
    constexpr int c0 = BLOCK_BYTE_SIZE / sizeof(half);
    TestNdTo2xNz<half, 64, 128, FRACTAL_NZ_ROW, c0, 1, 1, 1, 1>(0, 0, 5, 7);
}

TEST_F(TExtractNdTo2xNzTest, FloatAligned) { TestNdTo2xNz<float, 64, 128, 32, 64, 32, 64, 32, 64>(0, 0, 32, 64); }

TEST_F(TExtractNdTo2xNzTest, FloatUnaligned) { TestNdTo2xNz<float, 64, 128, 32, 64, 32, 64, 32, 64>(0, 0, 16, 4); }

TEST_F(TExtractNdTo2xNzTest, FloatOneByOne)
{
    constexpr int c0 = BLOCK_BYTE_SIZE / sizeof(float);
    TestNdTo2xNz<float, 64, 128, FRACTAL_NZ_ROW, c0, 1, 1, 1, 1>(0, 0, 10, 3);
}

TEST_F(TExtractNdTo2xNzTest, Int8Aligned) { TestNdTo2xNz<int8_t, 64, 128, 32, 64, 32, 64, 32, 64>(0, 0, 32, 64); }

TEST_F(TExtractNdTo2xNzTest, Int8Unaligned) { TestNdTo2xNz<int8_t, 64, 128, 32, 64, 32, 64, 32, 64>(0, 0, 16, 16); }

TEST_F(TExtractNdTo2xNzTest, Int8OneByOne)
{
    constexpr int c0 = BLOCK_BYTE_SIZE / sizeof(int8_t);
    TestNdTo2xNz<int8_t, 64, 128, FRACTAL_NZ_ROW, c0, 1, 1, 1, 1>(0, 0, 20, 17);
}

// Partial-valid windows (valid cols 63/61 with full 32x64 NZ tiles), mirrors the
// A5 case_int8_tail_unaligned. The tail columns keep their zero padding.
TEST_F(TExtractNdTo2xNzTest, Int8TailUnaligned) { TestNdTo2xNz<int8_t, 64, 128, 32, 64, 32, 63, 32, 61>(0, 0, 0, 2); }

TEST_F(TExtractNdTo2xNzTest, Int32Aligned) { TestNdTo2xNz<int32_t, 64, 128, 32, 64, 32, 64, 32, 64>(0, 0, 32, 64); }

// NZ+1 (RowPlusOne) destinations: logical 32x64 float with a 32x64 valid region.
// Cross-column-block physical offsets must honor the extra row between panels
// (element (0,8) at 264, not 256). Mirrors the A5 reference RowPlusOne variant.
TEST_F(TExtractNdTo2xNzTest, FloatRowPlusOneAcrossColumnBlocks)
{
    using SrcTile = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, 64, 128, SLayout::NoneBox>;
    using Dst0Tile = Tile<
        TileType::Vec, float, 32, 64, BLayout::ColMajor, 32, 64, SLayout::RowMajor, TileConfig::fractalABSize,
        PadValue::Null, CompactMode::RowPlusOne>;
    using Dst1Tile = Dst0Tile;

    static_assert(Dst0Tile::kRowPlusOneNZ && !Dst0Tile::kRowPlusOneRowsPadded, "fixture must be native compact");
    static_assert(RowPlusOnePanelPitch<Dst0Tile>() == 33, "32x64 float NZ+1 panel pitch must be 33 rows");

    SrcTile src;
    Dst0Tile dst0;
    Dst1Tile dst1;
    size_t addr = 0;
    CpuTileTestUtils::AssignOneTileStorage(src, addr);
    CpuTileTestUtils::AssignOneTileStorage(dst0, addr);
    CpuTileTestUtils::AssignOneTileStorage(dst1, addr);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            CpuTileTestUtils::SetValue(src, r, c, float((r * 31 + c + 1) % 97));
        }
    }
    // Zero the whole physical envelope so gap rows and padding stay deterministic.
    std::fill(dst0.data(), dst0.data() + Dst0Tile::StorageNumel, 0.0f);
    std::fill(dst1.data(), dst1.data() + Dst1Tile::StorageNumel, 0.0f);

    TEXTRACT(dst0, dst1, src, 0, 0, 32, 64);

    CheckNzWindowRowPlusOne(dst0, src, 0, 0);
    CheckNzWindowRowPlusOne(dst1, src, 32, 64);
}

// Dynamic-shape NZ+1 destination: declared Rows 33 with a runtime valid row of 32
// (ValidRow == DYNAMIC). The panel pitch must come from the runtime valid row
// (ceil(32/16)*16+1 = 33, what TINSERT SPLIT reads), not from the declared Rows
// through the allocation envelope (ceil(33/16)*16+1 = 49). Regression for the CPU
// bug that made TEXTRACT write at pitch 49 while TINSERT<SPLIT2/SPLIT4> reads at
// pitch 33, silently corrupting the TEXTRACT -> TINSERT path for this shape.
TEST_F(TExtractNdTo2xNzTest, FloatRowPlusOneDynamicShape)
{
    using SrcTile = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, 64, 128, SLayout::NoneBox>;
    using Dst0Tile = Tile<
        TileType::Vec, float, 33, 64, BLayout::ColMajor, DYNAMIC, 64, SLayout::RowMajor, TileConfig::fractalABSize,
        PadValue::Null, CompactMode::RowPlusOne>;
    using Dst1Tile = Dst0Tile;

    static_assert(Dst0Tile::kRowPlusOneNZ && Dst0Tile::ValidRow == DYNAMIC, "fixture must be dynamic-shape NZ+1");
    static_assert(Dst0Tile::StorageNumel == 49 * 64, "33-row dynamic NZ+1 reserves a 49-row panel envelope");

    SrcTile src;
    Dst0Tile dst0(32);
    Dst1Tile dst1(32);
    size_t addr = 0;
    CpuTileTestUtils::AssignOneTileStorage(src, addr);
    CpuTileTestUtils::AssignOneTileStorage(dst0, addr);
    CpuTileTestUtils::AssignOneTileStorage(dst1, addr);

    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            CpuTileTestUtils::SetValue(src, r, c, float((r * 31 + c + 1) % 97));
        }
    }
    std::fill(dst0.data(), dst0.data() + Dst0Tile::StorageNumel, 0.0f);
    std::fill(dst1.data(), dst1.data() + Dst1Tile::StorageNumel, 0.0f);

    TEXTRACT(dst0, dst1, src, 0, 0, 32, 64);

    CheckNzWindowRowPlusOneDynamic(dst0, src, 0, 0);
    CheckNzWindowRowPlusOneDynamic(dst1, src, 32, 64);
}

#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TExtractNdTo2xNzTest, Bfloat16Aligned)
{
    TestNdTo2xNz<bfloat16_t, 64, 128, 32, 64, 32, 64, 32, 64>(0, 0, 32, 64);
}

TEST_F(TExtractNdTo2xNzTest, Bfloat16OneByOne)
{
    constexpr int c0 = BLOCK_BYTE_SIZE / sizeof(bfloat16_t);
    TestNdTo2xNz<bfloat16_t, 64, 128, FRACTAL_NZ_ROW, c0, 1, 1, 1, 1>(0, 0, 5, 7);
}
#endif
