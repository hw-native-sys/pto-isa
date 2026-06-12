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
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <mutex>
#include <pto/pto-inst.hpp>
#include <thread>
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
void runCubeToVectorNoSplitDelayedFree()
{
    constexpr int kFifoDepth = 2;
    using CubeTile = Tile<TileType::Mat, T, rows, cols>;
    using VecTile = Tile<TileType::Vec, T, rows, cols>;
    using Pipe = TPipe<kPipeFlagId + 2, Direction::DIR_C2V, sizeof(T) * VecTile::Numel, kFifoDepth>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    Pipe pipe((__gm__ void *)nullptr, kVecConsumerBase, 0x0);
    std::vector<std::vector<T>> actual(kFifoDepth);

    cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
    for (int iter = 0; iter < kFifoDepth; ++iter) {
        CubeTile cubeTile;
        TASSIGN(cubeTile, 0x0);
        fillCubeTile<T, rows, cols>(cubeTile, iter);
        TPUSH<Pipe, CubeTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, cubeTile);
    }

    for (int iter = 0; iter < kFifoDepth; ++iter) {
        VecTile vecTile;
        TASSIGN(vecTile, 0x4000);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        actual[iter].assign(vecTile.data(), vecTile.data() + vecTile.Numel);
    }
    for (int iter = 0; iter < kFifoDepth; ++iter) {
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }

    for (int iter = 0; iter < kFifoDepth; ++iter) {
        const auto expected = makeExpected<T, rows, cols>(iter);
        EXPECT_TRUE(ResultCmp(expected, actual[iter], 0));
    }

    auto &sharedState = Pipe::GetSharedState();
    std::lock_guard<std::mutex> lock(sharedState.mutex);
    EXPECT_EQ(sharedState.occupied, 0);
    EXPECT_EQ(sharedState.popped_not_freed, 0);
}

template <typename T, int rows, int cols>
void runCubeToVectorNoSplitInactiveLaneSkipsProtocol()
{
    constexpr int kFifoDepth = 2;
    using CubeTile = Tile<TileType::Mat, T, rows, cols>;
    using VecTile = Tile<TileType::Vec, T, rows, cols>;
    using Pipe = TPipe<kPipeFlagId + 3, Direction::DIR_C2V, sizeof(T) * VecTile::Numel, kFifoDepth>;

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

template <typename T, int rows, int cols>
void runCubeToVectorNoSplitDualLaneDelayedFree()
{
    constexpr int kFifoDepth = 2;
    using CubeTile = Tile<TileType::Mat, T, rows, cols>;
    using VecTile = Tile<TileType::Vec, T, rows, cols>;
    using Pipe = TPipe<kPipeFlagId + 4, Direction::DIR_C2V, sizeof(T) * VecTile::Numel, kFifoDepth, 2, true>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    Pipe pipe((__gm__ void *)nullptr, kVecConsumerBase, 0x0);
    std::vector<std::vector<T>> lane0Actual(kFifoDepth);
    std::vector<std::vector<T>> lane1Actual(kFifoDepth);

    {
        cpu_sim::ScopedExecutionContext producerCtx(0, 0, 1);
        for (int iter = 0; iter < kFifoDepth; ++iter) {
            CubeTile cubeTile;
            TASSIGN(cubeTile, 0x0);
            fillCubeTile<T, rows, cols>(cubeTile, iter);
            TPUSH<Pipe, CubeTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, cubeTile);
        }
    }

    {
        cpu_sim::ScopedExecutionContext lane0Ctx(0, 0, 2);
        for (int iter = 0; iter < kFifoDepth; ++iter) {
            VecTile vecTile;
            TASSIGN(vecTile, 0x4000);
            TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
            lane0Actual[iter].assign(vecTile.data(), vecTile.data() + vecTile.Numel);
        }
        for (int iter = 0; iter < kFifoDepth; ++iter) {
            TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
        }
    }

    {
        auto &sharedState = Pipe::GetSharedState();
        std::lock_guard<std::mutex> lock(sharedState.mutex);
        EXPECT_EQ(sharedState.occupied, kFifoDepth);
    }

    {
        cpu_sim::ScopedExecutionContext lane1Ctx(0, 1, 2);
        for (int iter = 0; iter < kFifoDepth; ++iter) {
            VecTile vecTile;
            TASSIGN(vecTile, 0x8000);
            TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
            lane1Actual[iter].assign(vecTile.data(), vecTile.data() + vecTile.Numel);
        }
        for (int iter = 0; iter < kFifoDepth; ++iter) {
            TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
        }
    }

    for (int iter = 0; iter < kFifoDepth; ++iter) {
        const auto expected = makeExpected<T, rows, cols>(iter);
        EXPECT_TRUE(ResultCmp(expected, lane0Actual[iter], 0));
        EXPECT_TRUE(std::all_of(lane1Actual[iter].begin(), lane1Actual[iter].end(),
                                [](T value) { return value == static_cast<T>(0); }));
    }

    auto &sharedState = Pipe::GetSharedState();
    std::lock_guard<std::mutex> lock(sharedState.mutex);
    EXPECT_EQ(sharedState.occupied, 0);
    EXPECT_EQ(sharedState.popped_not_freed_by_lane[0], 0);
    EXPECT_EQ(sharedState.popped_not_freed_by_lane[1], 0);
}

template <typename T, int rows, int cols>
void runCubeToVectorNoSplitDualLaneFastLaneWrapWaits()
{
    constexpr int kFifoDepth = 2;
    using CubeTile = Tile<TileType::Mat, T, rows, cols>;
    using VecTile = Tile<TileType::Vec, T, rows, cols>;
    using Pipe = TPipe<kPipeFlagId + 5, Direction::DIR_C2V, sizeof(T) * VecTile::Numel, kFifoDepth, 2, true>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    Pipe pipe((__gm__ void *)nullptr, kVecConsumerBase, 0x0);
    std::vector<std::vector<T>> lane0Actual(kFifoDepth + 1);
    std::vector<std::vector<T>> lane1Actual(kFifoDepth);

    {
        cpu_sim::ScopedExecutionContext producerCtx(0, 0, 1);
        for (int iter = 0; iter < kFifoDepth; ++iter) {
            CubeTile cubeTile;
            TASSIGN(cubeTile, 0x0);
            fillCubeTile<T, rows, cols>(cubeTile, iter);
            TPUSH<Pipe, CubeTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, cubeTile);
        }
    }

    {
        cpu_sim::ScopedExecutionContext lane0Ctx(0, 0, 2);
        for (int iter = 0; iter < kFifoDepth; ++iter) {
            VecTile vecTile;
            TASSIGN(vecTile, 0x4000);
            TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
            lane0Actual[iter].assign(vecTile.data(), vecTile.data() + vecTile.Numel);
        }
        for (int iter = 0; iter < kFifoDepth; ++iter) {
            TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
        }
    }

    std::atomic<bool> lane0ThirdPopDone{false};
    std::thread lane0ThirdPop([&]() {
        cpu_sim::ScopedExecutionContext lane0Ctx(0, 0, 2);
        VecTile vecTile;
        TASSIGN(vecTile, 0x4000);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        lane0Actual[kFifoDepth].assign(vecTile.data(), vecTile.data() + vecTile.Numel);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
        lane0ThirdPopDone.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(lane0ThirdPopDone.load(std::memory_order_acquire));

    {
        cpu_sim::ScopedExecutionContext lane1Ctx(0, 1, 2);
        for (int iter = 0; iter < kFifoDepth; ++iter) {
            VecTile vecTile;
            TASSIGN(vecTile, 0x8000);
            TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
            lane1Actual[iter].assign(vecTile.data(), vecTile.data() + vecTile.Numel);
            TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
        }
    }

    {
        cpu_sim::ScopedExecutionContext producerCtx(0, 0, 1);
        CubeTile cubeTile;
        TASSIGN(cubeTile, 0x0);
        fillCubeTile<T, rows, cols>(cubeTile, kFifoDepth);
        TPUSH<Pipe, CubeTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, cubeTile);
    }

    lane0ThirdPop.join();
    EXPECT_TRUE(lane0ThirdPopDone.load(std::memory_order_acquire));

    for (int iter = 0; iter < kFifoDepth + 1; ++iter) {
        const auto expected = makeExpected<T, rows, cols>(iter);
        EXPECT_TRUE(ResultCmp(expected, lane0Actual[iter], 0));
    }
    for (int iter = 0; iter < kFifoDepth; ++iter) {
        EXPECT_TRUE(std::all_of(lane1Actual[iter].begin(), lane1Actual[iter].end(),
                                [](T value) { return value == static_cast<T>(0); }));
    }

    auto &sharedState = Pipe::GetSharedState();
    std::lock_guard<std::mutex> lock(sharedState.mutex);
    EXPECT_EQ(sharedState.occupied, 1);
    EXPECT_EQ(sharedState.popped_not_freed_by_lane[0], 0);
    EXPECT_EQ(sharedState.popped_not_freed_by_lane[1], 0);
}

template <typename T, int rows, int cols, int validRows>
void runCubeToVectorNoSplitGmSlotUsesValidShape()
{
    constexpr int kFifoDepth = 2;
    using CubeTile = Tile<TileType::Acc, T, rows, cols, BLayout::ColMajor, validRows, cols, SLayout::RowMajor, 1024>;
    using VecTile = Tile<TileType::Vec, T, validRows, cols>;
    using Pipe = TPipe<kPipeFlagId + 6, Direction::DIR_C2V, sizeof(T) * VecTile::Numel, kFifoDepth, 2, true>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    std::vector<uint8_t> gmSlotStorage(Pipe::RingFiFo::SLOT_SIZE * Pipe::RingFiFo::SLOT_NUM);
    auto *gmSlotBuffer = reinterpret_cast<__gm__ void *>(gmSlotStorage.data());
    Pipe pipe(gmSlotBuffer, kVecConsumerBase, 0x0);
    std::vector<T> actual(VecTile::Numel);

    {
        cpu_sim::ScopedExecutionContext producerCtx(0, 0, 1);
        CubeTile cubeTile;
        TASSIGN(cubeTile, 0x0);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                cubeTile.data()[GetTileElementOffset<CubeTile>(r, c)] = static_cast<T>(r * cols + c + 1);
            }
        }
        TPUSH<Pipe, CubeTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, cubeTile);
    }

    {
        cpu_sim::ScopedExecutionContext lane0Ctx(0, 0, 2);
        VecTile vecTile;
        TASSIGN(vecTile, 0x4000);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        actual.assign(vecTile.data(), vecTile.data() + vecTile.Numel);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }

    {
        cpu_sim::ScopedExecutionContext lane1Ctx(0, 1, 2);
        VecTile vecTile;
        TASSIGN(vecTile, 0x8000);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }

    std::vector<T> expected(VecTile::Numel);
    for (int r = 0; r < validRows; ++r) {
        for (int c = 0; c < cols; ++c) {
            expected[r * cols + c] = static_cast<T>(r * cols + c + 1);
        }
    }
    EXPECT_TRUE(ResultCmp(expected, actual, 0));

    auto &sharedState = Pipe::GetSharedState();
    std::lock_guard<std::mutex> lock(sharedState.mutex);
    EXPECT_EQ(sharedState.occupied, 0);
}

template <typename T, int rows, int cols, int validRows, int validCols>
void runCubeToVectorNoSplitGmSlotUsesDynamicValidShape()
{
    constexpr int kFifoDepth = 2;
    using CubeTile = Tile<TileType::Acc, T, rows, cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
    using VecTile = Tile<TileType::Vec, T, validRows, validCols>;
    using Pipe = TPipe<kPipeFlagId + 7, Direction::DIR_BOTH, sizeof(T) * VecTile::Numel, kFifoDepth, 2, true>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    std::vector<uint8_t> gmSlotStorage(Pipe::RingFiFo::SLOT_SIZE * Pipe::RingFiFo::SLOT_NUM);
    auto *gmSlotBuffer = reinterpret_cast<__gm__ void *>(gmSlotStorage.data());
    Pipe pipe(gmSlotBuffer, kVecConsumerBase, 0x0);
    std::vector<T> actual(VecTile::Numel);

    {
        cpu_sim::ScopedExecutionContext producerCtx(0, 0, 1);
        CubeTile cubeTile(validRows, validCols);
        TASSIGN(cubeTile, 0x0);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                cubeTile.data()[GetTileElementOffset<CubeTile>(r, c)] = static_cast<T>(r * cols + c + 1);
            }
        }
        TPUSH<Pipe, CubeTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, cubeTile);
    }

    {
        cpu_sim::ScopedExecutionContext lane0Ctx(0, 0, 2);
        VecTile vecTile;
        TASSIGN(vecTile, 0x4000);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        actual.assign(vecTile.data(), vecTile.data() + vecTile.Numel);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }

    {
        cpu_sim::ScopedExecutionContext lane1Ctx(0, 1, 2);
        VecTile vecTile;
        TASSIGN(vecTile, 0x8000);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }

    std::vector<T> expected(VecTile::Numel);
    for (int r = 0; r < validRows; ++r) {
        for (int c = 0; c < validCols; ++c) {
            expected[r * validCols + c] = static_cast<T>(r * cols + c + 1);
        }
    }
    EXPECT_TRUE(ResultCmp(expected, actual, 0));

    auto &sharedState = Pipe::GetSharedState();
    std::lock_guard<std::mutex> lock(sharedState.mutex);
    EXPECT_EQ(sharedState.occupied, 0);
}

template <typename T, int rows, int cols>
void runBothDirectionNoSplitC2VThenV2C()
{
    constexpr int kFifoDepth = 4;
    using CubeProdTile = Tile<TileType::Acc, T, rows, cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
    using VecConsTile = Tile<TileType::Vec, T, rows, cols>;
    using VecProdTile = Tile<TileType::Vec, T, rows, cols>;
    using CubeConsTile = Tile<TileType::Mat, T, rows, cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    using Pipe = TPipe<kPipeFlagId + 8, Direction::DIR_BOTH, sizeof(T) * VecConsTile::Numel, kFifoDepth, 2, true>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    std::vector<uint8_t> gmSlotStorage(Pipe::RingFiFo::SLOT_SIZE * Pipe::RingFiFo::SLOT_NUM);
    auto *gmSlotBuffer = reinterpret_cast<__gm__ void *>(gmSlotStorage.data());
    Pipe pipe(gmSlotBuffer, kVecConsumerBase, kVecConsumerBase + 0x4000);
    std::vector<T> c2vActual(VecConsTile::Numel);
    std::vector<T> secondC2vActual(VecConsTile::Numel);
    std::vector<T> v2cActual(CubeConsTile::Numel);

    auto pushCube = [&](int iter) {
        cpu_sim::ScopedExecutionContext cubeCtx(0, 0, 1);
        CubeProdTile cubeTile(rows, cols);
        TASSIGN(cubeTile, 0x0);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                cubeTile.data()[GetTileElementOffset<CubeProdTile>(r, c)] =
                    static_cast<T>(iter * 1000 + r * cols + c + 1);
            }
        }
        TPUSH<Pipe, CubeProdTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, cubeTile);
    };

    pushCube(0);
    {
        cpu_sim::ScopedExecutionContext lane0Ctx(0, 0, 2);
        VecConsTile vecTile;
        TASSIGN(vecTile, 0x4000);
        TPOP<Pipe, VecConsTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        c2vActual.assign(vecTile.data(), vecTile.data() + vecTile.Numel);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }

    {
        cpu_sim::ScopedExecutionContext lane1Ctx(0, 1, 2);
        VecConsTile vecTile;
        TASSIGN(vecTile, 0x8000);
        TPOP<Pipe, VecConsTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }

    {
        cpu_sim::ScopedExecutionContext vecCtx(0, 0, 2);
        VecProdTile vecTile;
        TASSIGN(vecTile, 0x4000);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                vecTile.data()[GetTileElementOffset<VecProdTile>(r, c)] = static_cast<T>(1000 + r * cols + c + 1);
            }
        }
        TPUSH<Pipe, VecProdTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
    }

    pushCube(1);
    {
        cpu_sim::ScopedExecutionContext lane0Ctx(0, 0, 2);
        VecConsTile vecTile;
        TASSIGN(vecTile, 0x4000);
        TPOP<Pipe, VecConsTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        secondC2vActual.assign(vecTile.data(), vecTile.data() + vecTile.Numel);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }

    {
        cpu_sim::ScopedExecutionContext lane1Ctx(0, 1, 2);
        VecConsTile vecTile;
        TASSIGN(vecTile, 0x8000);
        TPOP<Pipe, VecConsTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }

    {
        cpu_sim::ScopedExecutionContext cubeCtx(0, 0, 1);
        CubeConsTile cubeTile(rows, cols);
        TASSIGN(cubeTile, 0x0);
        TPOP<Pipe, CubeConsTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, cubeTile);
        v2cActual.assign(cubeTile.data(), cubeTile.data() + cubeTile.Numel);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }

    std::vector<T> c2vExpected(VecConsTile::Numel);
    std::vector<T> secondC2vExpected(VecConsTile::Numel);
    std::vector<T> v2cExpected(CubeConsTile::Numel);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            c2vExpected[r * cols + c] = static_cast<T>(r * cols + c + 1);
            secondC2vExpected[r * cols + c] = static_cast<T>(1000 + r * cols + c + 1);
            v2cExpected[GetTileElementOffset<CubeConsTile>(r, c)] = static_cast<T>(1000 + r * cols + c + 1);
        }
    }
    EXPECT_TRUE(ResultCmp(c2vExpected, c2vActual, 0));
    EXPECT_TRUE(ResultCmp(secondC2vExpected, secondC2vActual, 0));
    EXPECT_TRUE(ResultCmp(v2cExpected, v2cActual, 0));

    auto &sharedState = Pipe::GetSharedState();
    std::lock_guard<std::mutex> lock(sharedState.mutex);
    EXPECT_EQ(sharedState.occupied, 0);
}

// Checks the DIR_BOTH producer does not reuse an out-of-order-freed ring slot.
//
// A leading V2C push consumed by the cube advances the producer cursor without
// advancing the vector consumer lane cursors, so the cube's subsequent C2V pushes
// land in a different slot order than the lanes release them. Scenario: push
// X->slot1, Y->slot0 (ring full); a third push Z must stay blocked while both
// lanes drain slot0(Y) and proceed only once slot1(X) is fully freed, so Z never
// overwrites X's still-unconsumed slot.
template <typename T, int rows, int cols>
void runBothDirectionC2VProducerWaitsForOutOfOrderSlot()
{
    constexpr int kFifoDepth = 2;
    using CubeProdTile = Tile<TileType::Acc, T, rows, cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;
    using VecConsTile = Tile<TileType::Vec, T, rows, cols>;
    using VecProdTile = Tile<TileType::Vec, T, rows, cols>;
    using CubeConsTile = Tile<TileType::Mat, T, rows, cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;
    using Pipe = TPipe<kPipeFlagId + 9, Direction::DIR_BOTH, sizeof(T) * VecConsTile::Numel, kFifoDepth, 2, true>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    std::vector<uint8_t> gmSlotStorage(Pipe::RingFiFo::SLOT_SIZE * Pipe::RingFiFo::SLOT_NUM);
    auto *gmSlotBuffer = reinterpret_cast<__gm__ void *>(gmSlotStorage.data());
    Pipe pipe(gmSlotBuffer, kVecConsumerBase, kVecConsumerBase + 0x4000);

    // base: X=1, Y=1001, Z=2001 so each C2V tile is distinguishable per slot.
    auto pushCube = [&](int base) {
        cpu_sim::ScopedExecutionContext cubeCtx(0, 0, 1);
        CubeProdTile cubeTile(rows, cols);
        TASSIGN(cubeTile, 0x0);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                cubeTile.data()[GetTileElementOffset<CubeProdTile>(r, c)] = static_cast<T>(base + r * cols + c);
            }
        }
        TPUSH<Pipe, CubeProdTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, cubeTile);
    };
    auto popLaneFree = [&](uint32_t subblock, uint32_t gmOffset, std::vector<T> *capture) {
        cpu_sim::ScopedExecutionContext laneCtx(0, subblock, 2);
        VecConsTile vecTile;
        TASSIGN(vecTile, gmOffset);
        TPOP<Pipe, VecConsTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
        if (capture != nullptr) {
            capture->assign(vecTile.data(), vecTile.data() + vecTile.Numel);
        }
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    };

    // Leading V2C push (vector lane 0) consumed by the cube: offsets the producer
    // cursor to slot 1 while the consumer lane cursors stay at slot 0.
    {
        cpu_sim::ScopedExecutionContext vecCtx(0, 0, 2);
        VecProdTile vecTile;
        TASSIGN(vecTile, 0x4000);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                vecTile.data()[GetTileElementOffset<VecProdTile>(r, c)] = static_cast<T>(5000 + r * cols + c);
            }
        }
        TPUSH<Pipe, VecProdTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
    }
    {
        cpu_sim::ScopedExecutionContext cubeCtx(0, 0, 1);
        CubeConsTile cubeTile(rows, cols);
        TASSIGN(cubeTile, 0x0);
        TPOP<Pipe, CubeConsTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, cubeTile);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }

    pushCube(1);    // X -> slot1
    pushCube(1001); // Y -> slot0 ; ring now full

    std::atomic<bool> zDone{false};
    std::thread zPush([&]() {
        pushCube(2001); // Z must wait until slot1 is free
        zDone.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    EXPECT_FALSE(zDone.load(std::memory_order_acquire));

    // Both lanes drain slot0 (Y). This frees slot0 and drops occupied below the
    // cap, but the cursor still points at slot1 (X), which no lane has consumed.
    std::vector<T> yActual;
    popLaneFree(0, 0x4000, &yActual);
    popLaneFree(1, 0x8000, nullptr);

    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    // slot1 (X) is still unconsumed, so Z must remain blocked. If Z has already
    // completed it reused slot1 and overwrote X; record the failure and stop
    // before the corrupted ring deadlocks the remaining pops (fail fast, no hang).
    const bool zStoleSlot = zDone.load(std::memory_order_acquire);
    EXPECT_FALSE(zStoleSlot) << "producer reused an out-of-order-freed slot (slot1/X)";
    if (zStoleSlot) {
        zPush.join();
        return;
    }

    // Both lanes drain slot1 (X). Only now is slot1 free and Z may proceed.
    std::vector<T> xActual;
    popLaneFree(0, 0x4000, &xActual);
    popLaneFree(1, 0x8000, nullptr);

    zPush.join();
    EXPECT_TRUE(zDone.load(std::memory_order_acquire));

    // Drain Z from slot1.
    std::vector<T> zActual;
    popLaneFree(0, 0x4000, &zActual);
    popLaneFree(1, 0x8000, nullptr);

    std::vector<T> xExpected(VecConsTile::Numel);
    std::vector<T> yExpected(VecConsTile::Numel);
    std::vector<T> zExpected(VecConsTile::Numel);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            xExpected[r * cols + c] = static_cast<T>(1 + r * cols + c);
            yExpected[r * cols + c] = static_cast<T>(1001 + r * cols + c);
            zExpected[r * cols + c] = static_cast<T>(2001 + r * cols + c);
        }
    }
    EXPECT_TRUE(ResultCmp(yExpected, yActual, 0)); // slot0, consumed first
    EXPECT_TRUE(ResultCmp(xExpected, xActual, 0)); // slot1, never corrupted by Z
    EXPECT_TRUE(ResultCmp(zExpected, zActual, 0));

    auto &sharedState = Pipe::GetSharedState();
    std::lock_guard<std::mutex> lock(sharedState.mutex);
    EXPECT_EQ(sharedState.occupied, 0);
}

// A dual-lane-capable C2V NO_SPLIT pipe (LocalSlotNum=2, IsNoSplit=true) consumed by a single
// vector subblock (subblock_dim == 1) must run as a single consumer: each free releases its slot
// immediately, instead of waiting for a second lane that the runtime never dispatched.
template <typename T, int rows, int cols>
void runCubeToVectorNoSplitSingleSubblockOnDualLanePipe()
{
    constexpr int kFifoDepth = 2;
    using CubeTile = Tile<TileType::Mat, T, rows, cols>;
    using VecTile = Tile<TileType::Vec, T, rows, cols>;
    using Pipe = TPipe<kPipeFlagId + 9, Direction::DIR_C2V, sizeof(T) * VecTile::Numel, kFifoDepth, 2, true>;

    NPU_MEMORY_CLEAR();
    Pipe::reset_for_cpu_sim();
    Pipe pipe((__gm__ void *)nullptr, kVecConsumerBase, 0x0);
    std::vector<std::vector<T>> actual(kFifoDepth);

    {
        cpu_sim::ScopedExecutionContext producerCtx(0, 0, 1);
        for (int iter = 0; iter < kFifoDepth; ++iter) {
            CubeTile cubeTile;
            TASSIGN(cubeTile, 0x0);
            fillCubeTile<T, rows, cols>(cubeTile, iter);
            TPUSH<Pipe, CubeTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, cubeTile);
        }
    }

    {
        cpu_sim::ScopedExecutionContext vecCtx(0, 0, 1);
        for (int iter = 0; iter < kFifoDepth; ++iter) {
            VecTile vecTile;
            TASSIGN(vecTile, 0x4000);
            TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
            actual[iter].assign(vecTile.data(), vecTile.data() + vecTile.Numel);
            TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);

            // The single consumer's free releases its slot right away (no second lane pending).
            auto &sharedState = Pipe::GetSharedState();
            std::lock_guard<std::mutex> lock(sharedState.mutex);
            EXPECT_EQ(sharedState.occupied, kFifoDepth - 1 - iter);
        }
    }

    for (int iter = 0; iter < kFifoDepth; ++iter) {
        const auto expected = makeExpected<T, rows, cols>(iter);
        EXPECT_TRUE(ResultCmp(expected, actual[iter], 0));
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

TEST_F(TPushPopCVNoSplitTest, cube_to_vector_delayed_free_float_16x32)
{
    runCubeToVectorNoSplitDelayedFree<float, 16, 32>();
}

TEST_F(TPushPopCVNoSplitTest, cube_to_vector_inactive_lane_skips_protocol_when_pipe_is_not_no_split)
{
    runCubeToVectorNoSplitInactiveLaneSkipsProtocol<float, 16, 32>();
}

TEST_F(TPushPopCVNoSplitTest, cube_to_vector_no_split_dual_lane_delayed_free_float_16x32)
{
    runCubeToVectorNoSplitDualLaneDelayedFree<float, 16, 32>();
}

TEST_F(TPushPopCVNoSplitTest, cube_to_vector_no_split_dual_lane_fast_lane_wrap_waits_float_16x32)
{
    runCubeToVectorNoSplitDualLaneFastLaneWrapWaits<float, 16, 32>();
}

TEST_F(TPushPopCVNoSplitTest, cube_to_vector_no_split_gm_slot_uses_valid_shape_int32_128x128)
{
    runCubeToVectorNoSplitGmSlotUsesValidShape<int32_t, 128, 128, 16>();
}

TEST_F(TPushPopCVNoSplitTest, cube_to_vector_no_split_gm_slot_uses_dynamic_valid_shape_float_16x32)
{
    runCubeToVectorNoSplitGmSlotUsesDynamicValidShape<float, 16, 32, 16, 32>();
}

TEST_F(TPushPopCVNoSplitTest, both_direction_no_split_c2v_then_v2c_float_16x32)
{
    runBothDirectionNoSplitC2VThenV2C<float, 16, 32>();
}

TEST_F(TPushPopCVNoSplitTest, both_direction_c2v_producer_waits_for_out_of_order_slot_float_16x32)
{
    runBothDirectionC2VProducerWaitsForOutOfOrderSlot<float, 16, 32>();
}

TEST_F(TPushPopCVNoSplitTest, cube_to_vector_no_split_single_subblock_on_dual_lane_pipe_float_16x32)
{
    runCubeToVectorNoSplitSingleSubblockOnDualLanePipe<float, 16, 32>();
}
