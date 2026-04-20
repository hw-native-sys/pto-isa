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
constexpr uint32_t kMatConsumerBase = 0x20000;
constexpr uint8_t kPipeFlagId = 12;

template <typename T, int rows, int cols>
void fillVectorTile(auto &tile, int iter)
{
    for (int i = 0; i < tile.Numel; ++i) {
        tile.data()[i] = static_cast<T>(iter * 1000 + i + 1);
    }
}

template <typename T, int rows, int cols>
std::vector<T> makeExpected(int iter)
{
    using TileT = Tile<TileType::Vec, T, rows, cols>;
    std::vector<T> expected(TileT::Numel);
    for (int i = 0; i < TileT::Numel; ++i) {
        expected[i] = static_cast<T>(iter * 1000 + i + 1);
    }
    return expected;
}

template <typename T, int rows, int cols>
void runVectorToCubeNoSplitSingleTile()
{
    constexpr int kFifoDepth = 2;
    using VecTile = Tile<TileType::Vec, T, rows, cols>;
    using MatTile = Tile<TileType::Mat, T, rows, cols>;
    using Pipe = TPipe<kPipeFlagId, Direction::DIR_V2C, sizeof(T) * MatTile::Numel, kFifoDepth>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    Pipe pipe((__gm__ void *)nullptr, 0x0, kMatConsumerBase);
    VecTile vecTile;
    MatTile matTile;
    TASSIGN(vecTile, 0x0);
    TASSIGN(matTile, 0x4000);

    cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
    fillVectorTile<T, rows, cols>(vecTile, 0);
    TPUSH<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
    TPOP<Pipe, MatTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, matTile);
    TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);

    const auto expected = makeExpected<T, rows, cols>(0);
    EXPECT_TRUE(ResultCmp(expected, matTile.data(), 0));
}

template <typename T, int rows, int cols>
void runVectorToCubeNoSplitWraparound()
{
    constexpr int kFifoDepth = 2;
    constexpr int kIterations = 5;
    using VecTile = Tile<TileType::Vec, T, rows, cols>;
    using MatTile = Tile<TileType::Mat, T, rows, cols>;
    using Pipe = TPipe<kPipeFlagId + 1, Direction::DIR_V2C, sizeof(T) * MatTile::Numel, kFifoDepth>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    Pipe pipe((__gm__ void *)nullptr, 0x0, kMatConsumerBase);
    std::vector<std::vector<T>> actual(kIterations);

    auto pushIter = [&](int iter) {
        VecTile vecTile;
        TASSIGN(vecTile, 0x0);
        fillVectorTile<T, rows, cols>(vecTile, iter);
        TPUSH<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
    };

    auto popIter = [&](int iter) {
        MatTile matTile;
        TASSIGN(matTile, 0x4000);
        TPOP<Pipe, MatTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, matTile);
        actual[iter].assign(matTile.data(), matTile.data() + matTile.Numel);
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
void runVectorToCubeNoSplitInactiveLaneNoOp()
{
    constexpr int kFifoDepth = 2;
    using VecTile = Tile<TileType::Vec, T, rows, cols>;
    using MatTile = Tile<TileType::Mat, T, rows, cols>;
    using Pipe = TPipe<kPipeFlagId + 2, Direction::DIR_V2C, sizeof(T) * MatTile::Numel, kFifoDepth>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    Pipe pipe((__gm__ void *)nullptr, 0x0, kMatConsumerBase);

    {
        cpu_sim::ScopedExecutionContext inactiveVecCtx(0, 1, 2);
        VecTile inactiveVecTile;
        TASSIGN(inactiveVecTile, 0x0);
        fillVectorTile<T, rows, cols>(inactiveVecTile, 7);
        TPUSH<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, inactiveVecTile);
    }

    {
        auto &sharedState = Pipe::GetSharedState();
        std::lock_guard<std::mutex> lock(sharedState.mutex);
        EXPECT_EQ(sharedState.occupied, 0);
        EXPECT_EQ(sharedState.next_producer_slot, 0);
    }

    VecTile vecTile;
    MatTile matTile;
    TASSIGN(vecTile, 0x0);
    TASSIGN(matTile, 0x4000);

    {
        cpu_sim::ScopedExecutionContext activeVecCtx(0, 0, 2);
        fillVectorTile<T, rows, cols>(vecTile, 0);
        TPUSH<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
    }

    {
        cpu_sim::ScopedExecutionContext cubeCtx(0, 0, 1);
        TPOP<Pipe, MatTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, matTile);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }

    const auto expected = makeExpected<T, rows, cols>(0);
    EXPECT_TRUE(ResultCmp(expected, matTile.data(), 0));

    {
        auto &sharedState = Pipe::GetSharedState();
        std::lock_guard<std::mutex> lock(sharedState.mutex);
        EXPECT_EQ(sharedState.occupied, 0);
    }
}
} // namespace

class TPushPopVCNoSplitTest : public testing::Test {
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

TEST_F(TPushPopVCNoSplitTest, vector_to_cube_single_tile_float_16x32)
{
    runVectorToCubeNoSplitSingleTile<float, 16, 32>();
}

TEST_F(TPushPopVCNoSplitTest, vector_to_cube_fifo_wraparound_float_16x32)
{
    runVectorToCubeNoSplitWraparound<float, 16, 32>();
}

TEST_F(TPushPopVCNoSplitTest, vector_to_cube_inactive_aiv1_is_noop_in_no_split_mode)
{
    runVectorToCubeNoSplitInactiveLaneNoOp<float, 16, 32>();
}
