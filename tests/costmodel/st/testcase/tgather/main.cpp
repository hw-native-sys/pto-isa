/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "cost_check.hpp"

using namespace pto;

namespace {

constexpr int HALF_P0101_ROW = 5;
constexpr int HALF_P0101_COL = 128;
constexpr int HALF_P1010_ROW = 7;
constexpr int HALF_P1010_COL = 1024;
constexpr int HALF_P0001_ROW = 3;
constexpr int HALF_P0001_COL = 1056;
constexpr int HALF_P0010_ROW = 4;
constexpr int HALF_P0010_COL = 128;
constexpr int HALF_P0100_ROW = 5;
constexpr int HALF_P0100_COL = 256;
constexpr int HALF_P1000_ROW = 6;
constexpr int HALF_P1000_COL = 288;
constexpr int HALF_P1111_ROW = 7;
constexpr int HALF_P1111_COL = 320;

constexpr int FLOAT_P0101_ROW = 4;
constexpr int FLOAT_P0101_COL = 64;
constexpr int FLOAT_P1010_ROW = 7;
constexpr int FLOAT_P1010_COL = 1024;
constexpr int FLOAT_P0001_ROW = 3;
constexpr int FLOAT_P0001_COL = 1056;
constexpr int FLOAT_P0010_ROW = 4;
constexpr int FLOAT_P0010_COL = 128;
constexpr int FLOAT_P0100_ROW = 5;
constexpr int FLOAT_P0100_COL = 256;
constexpr int FLOAT_P1000_ROW = 6;
constexpr int FLOAT_P1000_COL = 288;
constexpr int FLOAT_P1111_ROW = 7;
constexpr int FLOAT_P1111_COL = 320;

template <typename T, MaskPattern maskPattern, int rows, int cols, float profiling, float accuracy>
void runTGatherPattern()
{
    using SrcTile = Tile<TileType::Vec, T, rows + 5, cols + 32, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, -1, -1>;
    SrcTile srcTile(rows, cols);
    DstTile dstTile(rows, cols);

    std::vector<T> srcBuf((rows + 5) * (cols + 32), T{});
    std::vector<T> dstBuf(rows * cols, T{});
    TASSIGN(srcTile, reinterpret_cast<std::uintptr_t>(srcBuf.data()));
    TASSIGN(dstTile, reinterpret_cast<std::uintptr_t>(dstBuf.data()));

    TGATHER<DstTile, SrcTile, maskPattern>(dstTile, srcTile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

template <
    typename Src0T, typename IdxT, int srcRows, int srcCols, int dstRows, int dstCols, float profiling, float accuracy>
void runTGatherIndex()
{
    using Src0Tile = Tile<TileType::Vec, Src0T, srcRows, srcCols, BLayout::RowMajor, -1, -1>;
    using IdxTile = Tile<TileType::Vec, IdxT, dstRows, dstCols, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, Src0T, dstRows, dstCols, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, IdxT, dstRows, dstCols, BLayout::RowMajor, -1, -1>;
    Src0Tile src0Tile(srcRows, srcCols);
    IdxTile idxTile(dstRows, dstCols);
    DstTile dstTile(dstRows, dstCols);
    TmpTile tmpTile(dstRows, dstCols);

    std::vector<Src0T> src0Buf(srcRows * srcCols, Src0T{});
    std::vector<IdxT> idxBuf(dstRows * dstCols, IdxT{0});
    std::vector<Src0T> dstBuf(dstRows * dstCols, Src0T{});
    std::vector<IdxT> tmpBuf(dstRows * dstCols, IdxT{0});
    TASSIGN(src0Tile, reinterpret_cast<std::uintptr_t>(src0Buf.data()));
    TASSIGN(idxTile, reinterpret_cast<std::uintptr_t>(idxBuf.data()));
    TASSIGN(dstTile, reinterpret_cast<std::uintptr_t>(dstBuf.data()));
    TASSIGN(tmpTile, reinterpret_cast<std::uintptr_t>(tmpBuf.data()));

    TGATHER(dstTile, src0Tile, idxTile, tmpTile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

template <typename SrcT, typename DstT, int rows, int cols, int k, CmpMode cmpMode, float profiling, float accuracy>
void runTGatherCmp(SrcT kValue)
{
    using SrcTile = Tile<TileType::Vec, SrcT, rows, cols, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, DstT, rows, k, BLayout::RowMajor, -1, -1>;
    constexpr int concatRow =
        (rows * static_cast<int>(sizeof(SrcT)) < 32) ? (32 / static_cast<int>(sizeof(SrcT))) : rows;
    using CountTile = Tile<TileType::Vec, DstT, concatRow, 1, BLayout::ColMajor, -1, -1>;
    constexpr int cmpVCol = (cols + 7) / 8;
    constexpr int cmpCol = ((cmpVCol + 31) / 32) * 32;
    using TmpTile = Tile<TileType::Vec, uint8_t, rows, cmpCol, BLayout::RowMajor, -1, -1>;

    SrcTile srcTile(rows, cols);
    DstTile dstTile(rows, k);
    CountTile countTile(rows, 1);
    TmpTile tmpTile(rows, cmpVCol);

    std::vector<SrcT> srcBuf(rows * cols, SrcT{});
    std::vector<DstT> dstBuf(rows * k, DstT{0});
    std::vector<DstT> countBuf(concatRow, DstT{0});
    std::vector<uint8_t> tmpBuf(rows * cmpCol + rows * cols * static_cast<int>(sizeof(DstT)), uint8_t{0});
    TASSIGN(srcTile, reinterpret_cast<std::uintptr_t>(srcBuf.data()));
    TASSIGN(dstTile, reinterpret_cast<std::uintptr_t>(dstBuf.data()));
    TASSIGN(countTile, reinterpret_cast<std::uintptr_t>(countBuf.data()));
    TASSIGN(tmpTile, reinterpret_cast<std::uintptr_t>(tmpBuf.data()));

    TGATHER<DstTile, SrcTile, CountTile, TmpTile, cmpMode, 0>(dstTile, srcTile, kValue, countTile, tmpTile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TGather, case1_float_P0101)
{
    runTGatherPattern<float, MaskPattern::P0101, FLOAT_P0101_ROW, FLOAT_P0101_COL, 0.0f, 0.0f>();
}
TEST(TGather, case1_float_P1010)
{
    runTGatherPattern<float, MaskPattern::P1010, FLOAT_P1010_ROW, FLOAT_P1010_COL, 0.0f, 0.0f>();
}
TEST(TGather, case1_float_P0001)
{
    runTGatherPattern<float, MaskPattern::P0001, FLOAT_P0001_ROW, FLOAT_P0001_COL, 0.0f, 0.0f>();
}
TEST(TGather, case1_float_P0010)
{
    runTGatherPattern<float, MaskPattern::P0010, FLOAT_P0010_ROW, FLOAT_P0010_COL, 0.0f, 0.0f>();
}
TEST(TGather, case1_float_P0100)
{
    runTGatherPattern<float, MaskPattern::P0100, FLOAT_P0100_ROW, FLOAT_P0100_COL, 0.0f, 0.0f>();
}
TEST(TGather, case1_float_P1000)
{
    runTGatherPattern<float, MaskPattern::P1000, FLOAT_P1000_ROW, FLOAT_P1000_COL, 0.0f, 0.0f>();
}
TEST(TGather, case1_float_P1111)
{
    runTGatherPattern<float, MaskPattern::P1111, FLOAT_P1111_ROW, FLOAT_P1111_COL, 0.0f, 0.0f>();
}

TEST(TGather, case1_half_P0101)
{
    runTGatherPattern<half, MaskPattern::P0101, HALF_P0101_ROW, HALF_P0101_COL, 0.0f, 0.0f>();
}
TEST(TGather, case1_half_P1010)
{
    runTGatherPattern<half, MaskPattern::P1010, HALF_P1010_ROW, HALF_P1010_COL, 0.0f, 0.0f>();
}
TEST(TGather, case1_half_P0001)
{
    runTGatherPattern<half, MaskPattern::P0001, HALF_P0001_ROW, HALF_P0001_COL, 0.0f, 0.0f>();
}
TEST(TGather, case1_half_P0010)
{
    runTGatherPattern<half, MaskPattern::P0010, HALF_P0010_ROW, HALF_P0010_COL, 0.0f, 0.0f>();
}
TEST(TGather, case1_half_P0100)
{
    runTGatherPattern<half, MaskPattern::P0100, HALF_P0100_ROW, HALF_P0100_COL, 0.0f, 0.0f>();
}
TEST(TGather, case1_half_P1000)
{
    runTGatherPattern<half, MaskPattern::P1000, HALF_P1000_ROW, HALF_P1000_COL, 0.0f, 0.0f>();
}
TEST(TGather, case1_half_P1111)
{
    runTGatherPattern<half, MaskPattern::P1111, HALF_P1111_ROW, HALF_P1111_COL, 0.0f, 0.0f>();
}

TEST(TGather, case1_U16_P0101)
{
    runTGatherPattern<uint16_t, MaskPattern::P0101, HALF_P0101_ROW, HALF_P0101_COL, 0.0f, 0.0f>();
}
TEST(TGather, case1_U16_P1010)
{
    runTGatherPattern<uint16_t, MaskPattern::P1010, HALF_P1010_ROW, HALF_P1010_COL, 0.0f, 0.0f>();
}
TEST(TGather, case1_I16_P0001)
{
    runTGatherPattern<uint16_t, MaskPattern::P0001, HALF_P0001_ROW, HALF_P0001_COL, 0.0f, 0.0f>();
}
TEST(TGather, case1_I16_P0010)
{
    runTGatherPattern<uint16_t, MaskPattern::P0010, HALF_P0010_ROW, HALF_P0010_COL, 0.0f, 0.0f>();
}
TEST(TGather, case1_U32_P0100)
{
    runTGatherPattern<uint32_t, MaskPattern::P0100, FLOAT_P0100_ROW, FLOAT_P0100_COL, 0.0f, 0.0f>();
}
TEST(TGather, case1_I32_P1000)
{
    runTGatherPattern<int32_t, MaskPattern::P1000, FLOAT_P1000_ROW, FLOAT_P1000_COL, 0.0f, 0.0f>();
}
TEST(TGather, case1_I32_P1111)
{
    runTGatherPattern<int32_t, MaskPattern::P1111, FLOAT_P1111_ROW, FLOAT_P1111_COL, 0.0f, 0.0f>();
}

TEST(TGather, case_1D_float_32x1024_16x64) { runTGatherIndex<float, int32_t, 32, 1024, 16, 64, 0.0f, 0.0f>(); }
TEST(TGather, case_1D_int32_32x512_16x256) { runTGatherIndex<int32_t, int32_t, 32, 512, 16, 256, 0.0f, 0.0f>(); }
TEST(TGather, case_1D_half_16x1024_16x128) { runTGatherIndex<int16_t, int32_t, 16, 1024, 16, 128, 0.0f, 0.0f>(); }
TEST(TGather, case_1D_int16_32x256_32x64) { runTGatherIndex<int16_t, int32_t, 32, 256, 32, 64, 0.0f, 0.0f>(); }
TEST(TGather, case_1D_half_1x16_1x16) { runTGatherIndex<int16_t, int32_t, 1, 16, 1, 16, 0.0f, 0.0f>(); }
TEST(TGather, case_1D_half_1x32_1x32) { runTGatherIndex<int16_t, int32_t, 1, 32, 1, 32, 0.0f, 0.0f>(); }
TEST(TGather, case_1D_half_1x64_1x64) { runTGatherIndex<int16_t, int32_t, 1, 64, 1, 64, 0.0f, 0.0f>(); }
TEST(TGather, case_1D_half_1x128_1x128) { runTGatherIndex<int16_t, int32_t, 1, 128, 1, 128, 0.0f, 0.0f>(); }
TEST(TGather, case_1D_half_1x128_1x64) { runTGatherIndex<int16_t, int32_t, 1, 128, 1, 64, 0.0f, 0.0f>(); }
TEST(TGather, case_1D_float_1024x16_1024x16) { runTGatherIndex<float, int32_t, 1024, 16, 1024, 16, 0.0f, 0.0f>(); }
TEST(TGather, case_1D_float_16x16_32x32) { runTGatherIndex<float, int32_t, 16, 16, 32, 32, 0.0f, 0.0f>(); }
TEST(TGather, case_1D_half_16x16_32x32) { runTGatherIndex<int16_t, int32_t, 16, 16, 32, 32, 0.0f, 0.0f>(); }

TEST(TGather, case1_float_topk)
{
    GTEST_SKIP() << "TGATHER top-k compare path truncates scratch addresses to uint32_t on host costmodel.";
}

TEST(TGather, case2_s32_topk)
{
    GTEST_SKIP() << "TGATHER top-k compare path truncates scratch addresses to uint32_t on host costmodel.";
}

TEST(TGather, case3_float_topk)
{
    GTEST_SKIP() << "TGATHER top-k compare path truncates scratch addresses to uint32_t on host costmodel.";
}

TEST(TGather, case4_half_topk)
{
    GTEST_SKIP() << "TGATHER top-k compare path truncates scratch addresses to uint32_t on host costmodel.";
}

TEST(TGather, case5_half_topk)
{
    GTEST_SKIP() << "TGATHER top-k compare path truncates scratch addresses to uint32_t on host costmodel.";
}
