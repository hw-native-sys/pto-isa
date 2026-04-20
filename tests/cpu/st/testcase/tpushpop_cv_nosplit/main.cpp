/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include <gtest/gtest.h>
#include <mutex>
#include <pto/pto-inst.hpp>
#include <vector>
#include "test_common.h"

using namespace pto;
using namespace PtoTestCommon;

namespace {
constexpr uint32_t kVecConsumerBase = 0x10000;
constexpr uint8_t kPipeFlagId = 10;

template <typename T, int rows, int cols>
void fillCubeTile(auto &tile, int iter)
{
    for (int i = 0; i < tile.Numel; ++i) {
        tile.data()[i] = static_cast<T>(iter * 1000 + i + 1);
    }
}

template <typename T, int rows, int cols>
std::vector<T> makeExpected(int iter)
{
    using TileT = Tile<TileType::Mat, T, rows, cols>;
    std::vector<T> expected(TileT::Numel);
    for (int i = 0; i < TileT::Numel; ++i) {
        expected[i] = static_cast<T>(iter * 1000 + i + 1);
    }
    return expected;
}

template <typename T, int rows, int cols>
void runCubeToVectorNoSplitSingleTile()
{
    constexpr int kFifoDepth = 2;
    using CubeTile = Tile<TileType::Mat, T, rows, cols>;
    using VecTile = Tile<TileType::Vec, T, rows, cols>;
    using Pipe = TPipe<kPipeFlagId, Direction::DIR_C2V, sizeof(T) * VecTile::Numel, kFifoDepth>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    Pipe pipe((__gm__ void *)nullptr, kVecConsumerBase, 0x0);
    CubeTile cubeTile;
    VecTile vecTile;
    TASSIGN(cubeTile, 0x0);
    TASSIGN(vecTile, 0x4000);

    cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
    fillCubeTile<T, rows, cols>(cubeTile, 0);
    TPUSH<Pipe, CubeTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, cubeTile);
    TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
    TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);

    const auto expected = makeExpected<T, rows, cols>(0);
    EXPECT_TRUE(ResultCmp(expected, vecTile.data(), 0));
}

template <typename T, int rows, int cols>
void runCubeToVectorNoSplitWraparound()
{
    constexpr int kFifoDepth = 2;
    constexpr int kIterations = 5;
    using CubeTile = Tile<TileType::Mat, T, rows, cols>;
    using VecTile = Tile<TileType::Vec, T, rows, cols>;
    using Pipe = TPipe<kPipeFlagId + 1, Direction::DIR_C2V, sizeof(T) * VecTile::Numel, kFifoDepth>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    Pipe pipe((__gm__ void *)nullptr, kVecConsumerBase, 0x0);
    std::vector<std::vector<T>> actual(kIterations);

    auto pushIter = [&](int iter) {
        CubeTile cubeTile;
        TASSIGN(cubeTile, 0x0);
        fillCubeTile<T, rows, cols>(cubeTile, iter);
        TPUSH<Pipe, CubeTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, cubeTile);
    };

    auto popIter = [&](int iter) {
        VecTile vecTile;
        TASSIGN(vecTile, 0x4000);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        actual[iter].assign(vecTile.data(), vecTile.data() + vecTile.Numel);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    };

    cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
    for (int iter = 0; iter < std::min(kFifoDepth, kIterations); ++iter) {
        pushIter(iter);
    }
    for (int iter = 0; iter < kIterations; ++iter) {
        popIter(iter);
        const int nextIter = iter + kFifoDepth;
        if (nextIter < kIterations) {
            pushIter(nextIter);
        }
    }

    for (int iter = 0; iter < kIterations; ++iter) {
        const auto expected = makeExpected<T, rows, cols>(iter);
        EXPECT_TRUE(ResultCmp(expected, actual[iter], 0));
    }
}

template <typename T, int rows, int cols>
void runCubeToVectorNoSplitInactiveLaneNoOp()
{
    constexpr int kFifoDepth = 2;
    using CubeTile = Tile<TileType::Mat, T, rows, cols>;
    using VecTile = Tile<TileType::Vec, T, rows, cols>;
    using Pipe = TPipe<kPipeFlagId + 2, Direction::DIR_C2V, sizeof(T) * VecTile::Numel, kFifoDepth>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    Pipe pipe((__gm__ void *)nullptr, kVecConsumerBase, 0x0);

    {
        cpu_sim::ScopedExecutionContext producerCtx(0, 0, 1);
        CubeTile cubeTile;
        TASSIGN(cubeTile, 0x0);
        fillCubeTile<T, rows, cols>(cubeTile, 0);
        TPUSH<Pipe, CubeTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, cubeTile);
    }

    {
        cpu_sim::ScopedExecutionContext inactiveVecCtx(0, 1, 2);
        VecTile vecTile;
        TASSIGN(vecTile, 0x4000);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }

    {
        auto &sharedState = Pipe::GetSharedState();
        std::lock_guard<std::mutex> lock(sharedState.mutex);
        EXPECT_EQ(sharedState.occupied, 1);
        EXPECT_EQ(sharedState.next_consumer_slot, 0);
    }

    VecTile vecTile;
    TASSIGN(vecTile, 0x4000);
    {
        cpu_sim::ScopedExecutionContext activeVecCtx(0, 0, 2);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }

    const auto expected = makeExpected<T, rows, cols>(0);
    EXPECT_TRUE(ResultCmp(expected, vecTile.data(), 0));

    {
        auto &sharedState = Pipe::GetSharedState();
        std::lock_guard<std::mutex> lock(sharedState.mutex);
        EXPECT_EQ(sharedState.occupied, 0);
    }
}
} // namespace

class TPushPopCVNoSplitTest : public testing::Test {
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

TEST_F(TPushPopCVNoSplitTest, cube_to_vector_single_tile_float_16x32)
{
    runCubeToVectorNoSplitSingleTile<float, 16, 32>();
}

TEST_F(TPushPopCVNoSplitTest, cube_to_vector_fifo_wraparound_float_16x32)
{
    runCubeToVectorNoSplitWraparound<float, 16, 32>();
}

TEST_F(TPushPopCVNoSplitTest, cube_to_vector_inactive_aiv1_is_noop_in_no_split_mode)
{
    runCubeToVectorNoSplitInactiveLaneNoOp<float, 16, 32>();
}
