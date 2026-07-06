/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// V2C (vector -> cube) split TPUSH/TPOP coverage.
//
// Two AIV sub-cores each push a half of a tile (TILE_UP_DOWN splits the rows,
// TILE_LEFT_RIGHT splits the columns); the single AIC pops the full combined
// tile. This is the reverse of the C2V split case (1 producer / 2 consumers)
// and was previously untested in CPU-sim -- the sibling testcase only covers
// tpushpop_vc_nosplit. The wraparound variants drive more iterations than the
// ring has slots (NB > SLOT_NUM) so that slots are reused; that reuse path is
// exactly where the V2C split consumer used to hand the same slot to two
// overlapping pops and return corrupted data.

#include <algorithm>
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>
#include <vector>
#include "test_common.h"

using namespace pto;
using namespace PtoTestCommon;

namespace {
constexpr uint32_t kMatConsumerBase = 0x20000;
constexpr uint8_t kPipeFlagId = 12;

// Per-(iter, lane, element) distinct values so any slot mix-up shows up.
template <typename T>
T cellValue(int iter, int lane, int elem)
{
    return static_cast<T>(iter * 100000 + lane * 1000 + elem + 1);
}

template <typename TileData>
void fillHalfTile(TileData &tile, int iter, int lane)
{
    using T = typename TileData::DType;
    for (int i = 0; i < tile.Numel; ++i) {
        tile.data()[i] = cellValue<T>(iter, lane, i);
    }
}

// Expected full [rows, cols] combined tile.
template <typename T, int rows, int cols, TileSplitAxis Split>
std::vector<T> makeExpected(int iter)
{
    std::vector<T> expected(static_cast<std::size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int lane;
            int localElem;
            if constexpr (Split == TileSplitAxis::TILE_UP_DOWN) {
                constexpr int halfRows = rows / 2;
                lane = r / halfRows;
                localElem = (r % halfRows) * cols + c; // producer tile is [rows/2, cols]
            } else {                                   // TILE_LEFT_RIGHT
                constexpr int halfCols = cols / 2;
                lane = c / halfCols;
                localElem = r * halfCols + (c % halfCols); // producer tile is [rows, cols/2]
            }
            expected[static_cast<std::size_t>(r) * cols + c] = cellValue<T>(iter, lane, localElem);
        }
    }
    return expected;
}

// One AIV lane pushes its half tile into the shared slot.
template <typename T, int rows, int cols, TileSplitAxis Split, typename Pipe>
void pushLane(Pipe &pipe, int iter, int lane)
{
    constexpr int prodRows = (Split == TileSplitAxis::TILE_UP_DOWN) ? rows / 2 : rows;
    constexpr int prodCols = (Split == TileSplitAxis::TILE_LEFT_RIGHT) ? cols / 2 : cols;
    using VecTile = Tile<TileType::Vec, T, prodRows, prodCols>;
    cpu_sim::ScopedExecutionContext vecCtx(0, lane, 2); // (blockIdx, subblockId, subblockDim)
    VecTile vecTile;
    TASSIGN(vecTile, 0x0);
    fillHalfTile(vecTile, iter, lane);
    TPUSH<Pipe, VecTile, Split>(pipe, vecTile);
}

template <typename T, int rows, int cols, TileSplitAxis Split, typename Pipe>
void popTile(Pipe &pipe, std::vector<T> &out)
{
    using MatTile = Tile<TileType::Mat, T, rows, cols>;
    cpu_sim::ScopedExecutionContext cubeCtx(0, 0, 1);
    MatTile matTile;
    TASSIGN(matTile, 0x4000);
    TPOP<Pipe, MatTile, Split>(pipe, matTile);
    out.assign(matTile.data(), matTile.data() + matTile.Numel);
    TFREE<Pipe, Split>(pipe);
}

// TPOP without the matching TFREE, so multiple tiles can be in flight at once
// (the cube pipelines: it pops tile k+1 before freeing tile k). Reading the
// wrong slot here is exactly the V2C-split-reuse bug.
template <typename T, int rows, int cols, TileSplitAxis Split, typename Pipe>
void popOnly(Pipe &pipe, std::vector<T> &out)
{
    using MatTile = Tile<TileType::Mat, T, rows, cols>;
    cpu_sim::ScopedExecutionContext cubeCtx(0, 0, 1);
    MatTile matTile;
    TASSIGN(matTile, 0x4000);
    TPOP<Pipe, MatTile, Split>(pipe, matTile);
    out.assign(matTile.data(), matTile.data() + matTile.Numel);
}

template <TileSplitAxis Split, typename Pipe>
void freeOne(Pipe &pipe)
{
    cpu_sim::ScopedExecutionContext cubeCtx(0, 0, 1);
    TFREE<Pipe, Split>(pipe);
}

template <typename T, int rows, int cols, TileSplitAxis Split>
void runSplitSingleTile()
{
    constexpr int kFifoDepth = 2;
    using MatTile = Tile<TileType::Mat, T, rows, cols>;
    using Pipe = TPipe<kPipeFlagId, Direction::DIR_V2C, sizeof(T) * MatTile::Numel, kFifoDepth>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    Pipe pipe((__gm__ void *)nullptr, 0x0, kMatConsumerBase);

    pushLane<T, rows, cols, Split>(pipe, 0, 0);
    pushLane<T, rows, cols, Split>(pipe, 0, 1);
    std::vector<T> actual;
    popTile<T, rows, cols, Split>(pipe, actual);

    const auto expected = makeExpected<T, rows, cols, Split>(0);
    EXPECT_TRUE(ResultCmp(expected, actual, 0));
}

// Drive kIterations > kFifoDepth so the ring wraps and slots get reused.
template <typename T, int rows, int cols, TileSplitAxis Split>
void runSplitWraparound()
{
    constexpr int kFifoDepth = 2;
    constexpr int kIterations = 5;
    using MatTile = Tile<TileType::Mat, T, rows, cols>;
    using Pipe = TPipe<kPipeFlagId + 1, Direction::DIR_V2C, sizeof(T) * MatTile::Numel, kFifoDepth>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    Pipe pipe((__gm__ void *)nullptr, 0x0, kMatConsumerBase);

    std::vector<std::vector<T>> actual(kIterations);

    auto pushBoth = [&](int iter) {
        pushLane<T, rows, cols, Split>(pipe, iter, 0);
        pushLane<T, rows, cols, Split>(pipe, iter, 1);
    };

    // Keep the FIFO full and pop with depth-2 in flight: TPOP(k+1) is issued
    // before TFREE(k), so two slots are outstanding at once -- the pipelined
    // cube behaviour that exposes V2C-split slot reuse.
    for (int iter = 0; iter < std::min(kFifoDepth, kIterations); ++iter) {
        pushBoth(iter);
    }
    popOnly<T, rows, cols, Split>(pipe, actual[0]); // first tile in flight
    for (int iter = 0; iter < kIterations; ++iter) {
        const int nextIter = iter + 1;
        if (nextIter < kIterations) {
            popOnly<T, rows, cols, Split>(pipe, actual[nextIter]); // 2 in flight
        }
        freeOne<Split>(pipe);                                      // retire tile `iter`
        const int producerIter = iter + kFifoDepth;
        if (producerIter < kIterations) {
            pushBoth(producerIter);
        }
    }

    for (int iter = 0; iter < kIterations; ++iter) {
        const auto expected = makeExpected<T, rows, cols, Split>(iter);
        EXPECT_TRUE(ResultCmp(expected, actual[iter], 0));
    }
}

// Same depth-2-in-flight pop pattern, but on a DIR_BOTH pipe. A bidirectional pipe
// carries V2C (vector->cube) and C2V (cube->vector) traffic on one ring, so the
// cube's V2C consumer is NOT "is_v2c && !is_c2v" and, before the fix, fell back to
// a naive consumer that did not advance the slot cursor per pop -- two outstanding
// pops then aliased the same slot and the two frees corrupted the ring accounting.
// This is the hc_head linear-kernel deadlock in miniature (cube finishes, both
// vector lanes hang at TPOP). The fix routes DIR_BOTH V2C consumers through the
// pending-slot FIFO just like the pure-V2C path, so the pops read distinct slots.
template <typename T, int rows, int cols, TileSplitAxis Split>
void runSplitWraparoundBothDir()
{
    constexpr int kFifoDepth = 4;
    constexpr int kIterations = 6;
    using MatTile = Tile<TileType::Mat, T, rows, cols>;
    using Pipe = TPipe<kPipeFlagId + 2, Direction::DIR_BOTH, sizeof(T) * MatTile::Numel, kFifoDepth, kFifoDepth, false>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    Pipe pipe((__gm__ void *)nullptr, 0x0, kMatConsumerBase);

    std::vector<std::vector<T>> actual(kIterations);

    auto pushBoth = [&](int iter) {
        pushLane<T, rows, cols, Split>(pipe, iter, 0);
        pushLane<T, rows, cols, Split>(pipe, iter, 1);
    };

    for (int iter = 0; iter < std::min(kFifoDepth, kIterations); ++iter) {
        pushBoth(iter);
    }
    popOnly<T, rows, cols, Split>(pipe, actual[0]); // first tile in flight
    for (int iter = 0; iter < kIterations; ++iter) {
        const int nextIter = iter + 1;
        if (nextIter < kIterations) {
            popOnly<T, rows, cols, Split>(pipe, actual[nextIter]); // 2 in flight
        }
        freeOne<Split>(pipe);                                      // retire tile `iter`
        const int producerIter = iter + kFifoDepth;
        if (producerIter < kIterations) {
            pushBoth(producerIter);
        }
    }

    for (int iter = 0; iter < kIterations; ++iter) {
        const auto expected = makeExpected<T, rows, cols, Split>(iter);
        EXPECT_TRUE(ResultCmp(expected, actual[iter], 0));
    }

    auto &sharedState = Pipe::GetSharedState();
    std::lock_guard<std::mutex> lock(sharedState.mutex);
    EXPECT_EQ(sharedState.occupied, 0);
}

template <typename TileData>
void fillFullTile(TileData &tile, int base)
{
    using T = typename TileData::DType;
    for (int r = 0; r < tile.GetValidRow(); ++r) {
        for (int c = 0; c < tile.GetValidCol(); ++c) {
            tile.data()[GetTileElementOffset<TileData>(r, c)] = static_cast<T>(base + r * tile.GetValidCol() + c + 1);
        }
    }
}

template <typename TileData>
std::vector<typename TileData::DType> readFullTile(TileData &tile)
{
    std::vector<typename TileData::DType> out;
    out.reserve(static_cast<std::size_t>(tile.GetValidRow()) * tile.GetValidCol());
    for (int r = 0; r < tile.GetValidRow(); ++r) {
        for (int c = 0; c < tile.GetValidCol(); ++c) {
            out.push_back(tile.data()[GetTileElementOffset<TileData>(r, c)]);
        }
    }
    return out;
}

template <typename T, int rows, int cols>
std::vector<T> makeLinearExpected(int base, int rowStart, int rowCount)
{
    std::vector<T> expected;
    expected.reserve(static_cast<std::size_t>(rowCount) * cols);
    for (int r = rowStart; r < rowStart + rowCount; ++r) {
        for (int c = 0; c < cols; ++c) {
            expected.push_back(static_cast<T>(base + r * cols + c + 1));
        }
    }
    return expected;
}

void runDirBothGmFifoKeepsC2VOrderAcrossV2CPrefetch()
{
    constexpr int kRows = 16;
    constexpr int kCols = 128;
    constexpr int kLaneRows = kRows / 2;
    constexpr int kSlotBytes = 8192;
    constexpr int kFifoDepth = 4;
    using Pipe = TPipe<30, Direction::DIR_BOTH, kSlotBytes, kFifoDepth, 4, false>;
    using CubeProdTile = Tile<TileType::Acc, float, kRows, kCols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
    using VecTile = Tile<TileType::Vec, float, kLaneRows, kCols, BLayout::RowMajor, kLaneRows, kCols>;
    using CubeConsTile = Tile<TileType::Mat, float, kRows, kCols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    std::vector<uint8_t> gmStorage(static_cast<std::size_t>(Pipe::RingFiFo::SLOT_SIZE) * Pipe::RingFiFo::SLOT_NUM);
    Pipe pipe(reinterpret_cast<__gm__ void *>(gmStorage.data()), 0x0, 0x0);

    auto pushC2V = [&](int base) {
        cpu_sim::ScopedExecutionContext cubeCtx(0, 0, 1);
        CubeProdTile cubeTile(kRows, kCols);
        TASSIGN(cubeTile, 0x0);
        fillFullTile(cubeTile, base);
        TPUSH<Pipe, CubeProdTile, TileSplitAxis::TILE_UP_DOWN>(pipe, cubeTile);
    };

    auto popC2V = [&](int base) {
        for (int lane = 0; lane < 2; ++lane) {
            cpu_sim::ScopedExecutionContext laneCtx(0, lane, 2);
            VecTile vecTile;
            TASSIGN(vecTile, 0x4000 + lane * 0x4000);
            TPOP<Pipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(pipe, vecTile);
            const auto actual = readFullTile(vecTile);
            const auto expected = makeLinearExpected<float, kRows, kCols>(base, lane * kLaneRows, kLaneRows);
            EXPECT_TRUE(ResultCmp(expected, actual, 0));
            TFREE<Pipe, TileSplitAxis::TILE_UP_DOWN>(pipe);
        }
    };

    auto pushV2C = [&]() {
        for (int lane = 0; lane < 2; ++lane) {
            cpu_sim::ScopedExecutionContext laneCtx(0, lane, 2);
            VecTile vecTile;
            TASSIGN(vecTile, 0x4000 + lane * 0x4000);
            fillFullTile(vecTile, 7000 + lane * 1000);
            TPUSH<Pipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(pipe, vecTile);
        }
    };

    auto popV2C = [&]() {
        cpu_sim::ScopedExecutionContext cubeCtx(0, 0, 1);
        CubeConsTile cubeTile(kRows, kCols);
        TASSIGN(cubeTile, 0x0);
        TPOP<Pipe, CubeConsTile, TileSplitAxis::TILE_UP_DOWN>(pipe, cubeTile);
        const auto actual = readFullTile(cubeTile);
        std::vector<float> expected;
        const auto upper = makeLinearExpected<float, kLaneRows, kCols>(7000, 0, kLaneRows);
        const auto lower = makeLinearExpected<float, kLaneRows, kCols>(8000, 0, kLaneRows);
        expected.insert(expected.end(), upper.begin(), upper.end());
        expected.insert(expected.end(), lower.begin(), lower.end());
        EXPECT_TRUE(ResultCmp(expected, actual, 0));
        TFREE<Pipe, TileSplitAxis::TILE_UP_DOWN>(pipe);
    };

    pushC2V(1000);
    pushC2V(2000);
    popC2V(1000);
    pushV2C();
    popV2C();
    pushC2V(3000);
    popC2V(2000);
    popC2V(3000);

    auto &sharedState = Pipe::GetSharedState();
    std::lock_guard<std::mutex> lock(sharedState.mutex);
    EXPECT_EQ(sharedState.occupied, 0);
}
} // namespace

class TPushPopVCSplitTest : public testing::Test {
protected:
    void SetUp() override
    {
        NPU_MEMORY_INIT(NPUArch::A5);
        NPU_MEMORY_CLEAR();
    }

    void TearDown() override
    {
        cpu_sim::reset_execution_context();
        NPU_MEMORY_CLEAR();
    }
};

TEST_F(TPushPopVCSplitTest, up_down_single_tile_float_16x32)
{
    runSplitSingleTile<float, 16, 32, TileSplitAxis::TILE_UP_DOWN>();
}

TEST_F(TPushPopVCSplitTest, up_down_fifo_wraparound_float_16x32)
{
    runSplitWraparound<float, 16, 32, TileSplitAxis::TILE_UP_DOWN>();
}

TEST_F(TPushPopVCSplitTest, left_right_single_tile_float_16x32)
{
    runSplitSingleTile<float, 16, 32, TileSplitAxis::TILE_LEFT_RIGHT>();
}

TEST_F(TPushPopVCSplitTest, left_right_fifo_wraparound_float_16x32)
{
    runSplitWraparound<float, 16, 32, TileSplitAxis::TILE_LEFT_RIGHT>();
}

TEST_F(TPushPopVCSplitTest, both_dir_up_down_overlapping_pops_read_distinct_slots_float_16x32)
{
    runSplitWraparoundBothDir<float, 16, 32, TileSplitAxis::TILE_UP_DOWN>();
}

TEST_F(TPushPopVCSplitTest, both_dir_left_right_overlapping_pops_read_distinct_slots_float_16x32)
{
    runSplitWraparoundBothDir<float, 16, 32, TileSplitAxis::TILE_LEFT_RIGHT>();
}

TEST_F(TPushPopVCSplitTest, both_dir_up_down_gm_fifo_keeps_c2v_order_across_v2c_prefetch)
{
    runDirBothGmFifoKeepsC2VOrderAcrossV2CPrefetch();
}
