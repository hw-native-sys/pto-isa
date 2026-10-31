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
#include <atomic>
#include <barrier>
#include <chrono>
#include <exception>
#include <gtest/gtest.h>
#include <pto/common/fifo.hpp>
#include <thread>
#include <vector>
#include "test_common.h"

using namespace std;
using namespace pto;
using namespace PtoTestCommon;

namespace {
using HookTestPipe = TPipe<6, Direction::DIR_C2V, sizeof(float) * 16 * 16, 1>;
using HookedV2CPipe = TPipe<9, Direction::DIR_V2C, sizeof(float) * 16 * 16, 2>;

std::atomic<uint32_t> g_injected_subblock_id{0};
std::atomic<uint32_t> g_pipe_hook_call_count{0};
void* g_pipe_hook_storage = nullptr;
size_t g_pipe_hook_size = 0;
uint64_t g_pipe_hook_last_key = 0;

uint32_t MockSubblockIdHook() { return g_injected_subblock_id.load(std::memory_order_relaxed); }

void* MockPipeSharedStateHook(uint64_t pipeKey, size_t size)
{
    g_pipe_hook_last_key = pipeKey;
    g_pipe_hook_size = size;
    g_pipe_hook_call_count.fetch_add(1, std::memory_order_relaxed);
    return g_pipe_hook_storage;
}

struct ScopedCpuStubHooks {
    ScopedCpuStubHooks(void* subblockHook, void* pipeSharedStateHook)
    {
        pto::cpu_sim::register_hooks(subblockHook, pipeSharedStateHook);
    }

    ~ScopedCpuStubHooks()
    {
        pto::cpu_sim::register_hooks(nullptr, nullptr);
        cpu_sim::reset_execution_context();
        g_pipe_hook_storage = nullptr;
        g_pipe_hook_size = 0;
        g_pipe_hook_last_key = 0;
    }
};

template <TileSplitAxis SplitAxis>
using DirBothVecTile = Tile<
    TileType::Vec, float, (SplitAxis == TileSplitAxis::TILE_UP_DOWN) ? 8 : 16,
    (SplitAxis == TileSplitAxis::TILE_LEFT_RIGHT) ? 8 : 16, BLayout::RowMajor,
    (SplitAxis == TileSplitAxis::TILE_UP_DOWN) ? 8 : 16, (SplitAxis == TileSplitAxis::TILE_LEFT_RIGHT) ? 8 : 16>;

template <typename TileData>
void fillTileSequence(TileData& tile, float start)
{
    for (int i = 0; i < tile.Numel; ++i) {
        tile.data()[i] = start + static_cast<float>(i);
    }
}

template <TileSplitAxis SplitAxis>
void expectVecMatchesAccSplit(const auto& vec, const auto& acc, uint32_t laneId)
{
    for (int r = 0; r < vec.GetValidRow(); ++r) {
        for (int c = 0; c < vec.GetValidCol(); ++c) {
            uint32_t srcRow = r;
            uint32_t srcCol = c;
            if constexpr (SplitAxis == TileSplitAxis::TILE_UP_DOWN) {
                srcRow += laneId * vec.GetValidRow();
            } else {
                srcCol += laneId * vec.GetValidCol();
            }
            EXPECT_FLOAT_EQ(
                vec.data()[GetTileElementOffset<std::remove_cvref_t<decltype(vec)>>(r, c)],
                acc.data()[GetTileElementOffset<std::remove_cvref_t<decltype(acc)>>(
                    static_cast<int>(srcRow), static_cast<int>(srcCol))]);
        }
    }
}

template <TileSplitAxis SplitAxis, uint8_t FlagId>
void testDirBothConsumerWaitsForMatchingDirection()
{
    using VecTile = DirBothVecTile<SplitAxis>;
    using MatTile = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16>;
    using AccTile = TileAcc<float, 16, 16>;
    using Pipe = TPipe<FlagId, Direction::DIR_BOTH, sizeof(float) * MatTile::Numel, 2>;

    Pipe::reset_for_cpu_sim();
    Pipe vecProducer0((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe vecProducer1((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe cubePipe((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe vecConsumer0((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe vecConsumer1((__gm__ void*)nullptr, 0x0, 0x10000);

    VecTile src0;
    VecTile src1;
    MatTile poppedMat;
    AccTile accSrc;
    VecTile dst0;
    VecTile dst1;

    TASSIGN(src0, 0x0);
    TASSIGN(src1, VecTile::Numel * sizeof(float));
    TASSIGN(poppedMat, 2 * VecTile::Numel * sizeof(float));
    TASSIGN(accSrc, 2 * VecTile::Numel * sizeof(float) + MatTile::Numel * sizeof(float));
    TASSIGN(dst0, 2 * VecTile::Numel * sizeof(float) + MatTile::Numel * sizeof(float) + AccTile::Numel * sizeof(float));
    TASSIGN(
        dst1, 2 * VecTile::Numel * sizeof(float) + MatTile::Numel * sizeof(float) + AccTile::Numel * sizeof(float) +
                  VecTile::Numel * sizeof(float));

    fillTileSequence(src0, 1.0f);
    fillTileSequence(src1, 1001.0f);
    fillTileSequence(accSrc, 2001.0f);
    std::fill(poppedMat.data(), poppedMat.data() + poppedMat.Numel, 0.0f);
    std::fill(dst0.data(), dst0.data() + dst0.Numel, 0.0f);
    std::fill(dst1.data(), dst1.data() + dst1.Numel, 0.0f);

    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 2);
        TPUSH<Pipe, VecTile, SplitAxis>(vecProducer0, src0);
    }
    {
        cpu_sim::ScopedExecutionContext ctx(0, 1, 2);
        TPUSH<Pipe, VecTile, SplitAxis>(vecProducer1, src1);
    }

    std::atomic<bool> vec0Done{false};
    std::atomic<bool> vec1Done{false};
    std::thread consumerThread0([&]() {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 2);
        TPOP<Pipe, VecTile, SplitAxis>(vecConsumer0, dst0);
        vec0Done.store(true, std::memory_order_release);
    });
    std::thread consumerThread1([&]() {
        cpu_sim::ScopedExecutionContext ctx(0, 1, 2);
        TPOP<Pipe, VecTile, SplitAxis>(vecConsumer1, dst1);
        vec1Done.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    EXPECT_FALSE(vec0Done.load(std::memory_order_acquire));
    EXPECT_FALSE(vec1Done.load(std::memory_order_acquire));

    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
        TPOP<Pipe, MatTile, SplitAxis>(cubePipe, poppedMat);
        TFREE<Pipe, SplitAxis>(cubePipe);
    }

    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
        TPUSH<Pipe, AccTile, SplitAxis>(cubePipe, accSrc);
    }

    consumerThread0.join();
    consumerThread1.join();

    expectVecMatchesAccSplit<SplitAxis>(dst0, accSrc, 0);
    expectVecMatchesAccSplit<SplitAxis>(dst1, accSrc, 1);

    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 2);
        TFREE<Pipe, SplitAxis>(vecConsumer0);
    }
    {
        cpu_sim::ScopedExecutionContext ctx(0, 1, 2);
        TFREE<Pipe, SplitAxis>(vecConsumer1);
    }
}
} // namespace

template <typename T, int rows, int cols, TileType srcLoc>
void fillTile(auto& tile, int iter)
{
    for (int i = 0; i < tile.Numel; ++i) {
        tile.data()[i] = static_cast<T>(iter * 1000 + i + 1);
    }
}

template <typename T, int rows, int cols, TileType srcLoc>
std::vector<T> makeExpected(int iter)
{
    using PPTile = Tile<srcLoc, T, rows, cols>;
    std::vector<T> expected(PPTile::Numel);
    for (int i = 0; i < PPTile::Numel; ++i) {
        expected[i] = static_cast<T>(iter * 1000 + i + 1);
    }
    return expected;
}

template <TileSplitAxis SplitAxis, uint8_t FlagId>
void testGmFifoPreservesV2CSplitLayout()
{
    constexpr int vecRows = (SplitAxis == TileSplitAxis::TILE_UP_DOWN) ? 8 : 16;
    constexpr int vecCols = (SplitAxis == TileSplitAxis::TILE_LEFT_RIGHT) ? 8 : 16;
    using VecTile = Tile<TileType::Vec, float, vecRows, vecCols, BLayout::RowMajor, vecRows, vecCols>;
    using MatTile = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16>;
    using Pipe = TPipe<FlagId, Direction::DIR_V2C, sizeof(float) * MatTile::Numel, 2>;

    std::vector<float> fifoStorage(MatTile::Numel * Pipe::RingFiFo::SLOT_NUM, 0.0f);
    Pipe::reset_for_cpu_sim();
    Pipe producer0(fifoStorage.data(), 0x0, 0x10000);
    Pipe producer1(fifoStorage.data(), 0x0, 0x10000);
    Pipe consumer(fifoStorage.data(), 0x0, 0x10000);
    producer0.prod.setAllocateStatus(false);
    producer0.prod.setRecordStatus(false);
    producer1.prod.setAllocateStatus(false);
    producer1.prod.setRecordStatus(false);
    consumer.cons.setWaitStatus(false);
    consumer.cons.setFreeStatus(false);

    VecTile src0;
    VecTile src1;
    MatTile dst;
    TASSIGN(src0, 0x0);
    TASSIGN(src1, VecTile::GetSizeInBytes());
    TASSIGN(dst, 2 * VecTile::GetSizeInBytes());
    fillTile<float, vecRows, vecCols, TileType::Vec>(src0, 0);
    fillTile<float, vecRows, vecCols, TileType::Vec>(src1, 1);
    std::fill(dst.data(), dst.data() + dst.Numel, 0.0f);

    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 2);
        TPUSH<Pipe, VecTile, SplitAxis>(producer0, src0);
    }
    {
        cpu_sim::ScopedExecutionContext ctx(0, 1, 2);
        TPUSH<Pipe, VecTile, SplitAxis>(producer1, src1);
    }
    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
        TPOP<Pipe, MatTile, SplitAxis>(consumer, dst);
    }

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            const int lane = (SplitAxis == TileSplitAxis::TILE_UP_DOWN) ? r / VecTile::Rows : c / VecTile::Cols;
            const int laneRow = (SplitAxis == TileSplitAxis::TILE_UP_DOWN) ? r % VecTile::Rows : r;
            const int laneCol = (SplitAxis == TileSplitAxis::TILE_LEFT_RIGHT) ? c % VecTile::Cols : c;
            const auto& src = (lane == 0) ? src0 : src1;
            EXPECT_FLOAT_EQ(
                dst.data()[GetTileElementOffset<MatTile>(r, c)],
                src.data()[GetTileElementOffset<VecTile>(laneRow, laneCol)]);
        }
    }
}

template <TileSplitAxis SplitAxis, uint8_t FlagId>
void testDirBothSplitV2CFreePreservesC2VFifoOrder()
{
    using SplitVecTile = DirBothVecTile<SplitAxis>;
    using VecTile = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 16, 16>;
    using MatTile = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16>;
    using AccTile = TileAcc<float, 16, 16>;
    using Pipe = TPipe<FlagId, Direction::DIR_BOTH, sizeof(float) * MatTile::Numel, 3>;

    Pipe::reset_for_cpu_sim();
    Pipe c2vProducer0((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe v2cProducer0((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe v2cProducer1((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe c2vProducer1((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe cubeConsumer((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe vecConsumer((__gm__ void*)nullptr, 0x0, 0x10000);

    AccTile accSrc0;
    AccTile accSrc1;
    SplitVecTile splitSrc0;
    SplitVecTile splitSrc1;
    MatTile poppedMat;
    VecTile poppedVec0;
    VecTile poppedVec1;

    TASSIGN(accSrc0, 0);
    TASSIGN(accSrc1, AccTile::GetSizeInBytes());
    TASSIGN(splitSrc0, 2 * AccTile::GetSizeInBytes());
    TASSIGN(splitSrc1, 2 * AccTile::GetSizeInBytes() + SplitVecTile::GetSizeInBytes());
    TASSIGN(poppedMat, 2 * AccTile::GetSizeInBytes() + 2 * SplitVecTile::GetSizeInBytes());
    TASSIGN(poppedVec0, 2 * AccTile::GetSizeInBytes() + 2 * SplitVecTile::GetSizeInBytes() + MatTile::GetSizeInBytes());
    TASSIGN(
        poppedVec1, 2 * AccTile::GetSizeInBytes() + 2 * SplitVecTile::GetSizeInBytes() + MatTile::GetSizeInBytes() +
                        VecTile::GetSizeInBytes());

    fillTileSequence(accSrc0, 1.0f);
    fillTileSequence(accSrc1, 1001.0f);
    fillTileSequence(splitSrc0, 2001.0f);
    fillTileSequence(splitSrc1, 3001.0f);

    TPUSH<Pipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(c2vProducer0, accSrc0);
    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 2);
        TPUSH<Pipe, SplitVecTile, SplitAxis>(v2cProducer0, splitSrc0);
    }
    {
        cpu_sim::ScopedExecutionContext ctx(0, 1, 2);
        TPUSH<Pipe, SplitVecTile, SplitAxis>(v2cProducer1, splitSrc1);
    }
    TPUSH<Pipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(c2vProducer1, accSrc1);

    TPOP<Pipe, MatTile, SplitAxis>(cubeConsumer, poppedMat);
    TFREE<Pipe, SplitAxis>(cubeConsumer);

    TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(vecConsumer, poppedVec0);
    TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(vecConsumer);
    TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(vecConsumer, poppedVec1);
    TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(vecConsumer);

    for (int row = 0; row < VecTile::Rows; ++row) {
        for (int col = 0; col < VecTile::Cols; ++col) {
            EXPECT_FLOAT_EQ(poppedVec0.GetElement(row, col), accSrc0.GetElement(row, col));
            EXPECT_FLOAT_EQ(poppedVec1.GetElement(row, col), accSrc1.GetElement(row, col));
        }
    }
}

template <int FlagId>
void testDirBothOverlappingC2VPopsFreeInOrder()
{
    using VecTile = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 16, 16>;
    using AccTile = TileAcc<float, 16, 16>;
    using Pipe = TPipe<FlagId, Direction::DIR_BOTH, sizeof(float) * VecTile::Numel, 2>;

    Pipe::reset_for_cpu_sim();
    Pipe c2vProducer0((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe c2vProducer1((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe vecConsumer((__gm__ void*)nullptr, 0x0, 0x10000);

    AccTile accSrc0;
    AccTile accSrc1;
    VecTile poppedVec0;
    VecTile poppedVec1;

    TASSIGN(accSrc0, 0);
    TASSIGN(accSrc1, AccTile::GetSizeInBytes());
    TASSIGN(poppedVec0, 2 * AccTile::GetSizeInBytes());
    TASSIGN(poppedVec1, 2 * AccTile::GetSizeInBytes() + VecTile::GetSizeInBytes());

    fillTileSequence(accSrc0, 1.0f);
    fillTileSequence(accSrc1, 5001.0f);

    TPUSH<Pipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(c2vProducer0, accSrc0);
    TPUSH<Pipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(c2vProducer1, accSrc1);

    TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(vecConsumer, poppedVec0);
    TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(vecConsumer, poppedVec1);
    TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(vecConsumer);
    TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(vecConsumer);

    for (int row = 0; row < VecTile::Rows; ++row) {
        for (int col = 0; col < VecTile::Cols; ++col) {
            EXPECT_FLOAT_EQ(poppedVec0.GetElement(row, col), accSrc0.GetElement(row, col));
            EXPECT_FLOAT_EQ(poppedVec1.GetElement(row, col), accSrc1.GetElement(row, col));
        }
    }

    auto& state = Pipe::GetSharedState();
    EXPECT_EQ(state.occupied, 0);
    EXPECT_EQ(state.popped_not_freed, 0);
    for (std::size_t slot = 0; slot < 2; ++slot) {
        EXPECT_EQ(state.slot_busy[slot], 0);
        EXPECT_EQ(state.transfer_dirs[slot], cpu_pipe::TransferDir::None);
        EXPECT_EQ(state.remaining_consumers[slot], 0u);
    }

    AccTile accSrc2;
    VecTile poppedVec2;
    TASSIGN(accSrc2, 2 * AccTile::GetSizeInBytes() + 2 * VecTile::GetSizeInBytes());
    TASSIGN(poppedVec2, 2 * AccTile::GetSizeInBytes() + 3 * VecTile::GetSizeInBytes());
    fillTileSequence(accSrc2, 9001.0f);
    TPUSH<Pipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(c2vProducer0, accSrc2);
    TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(vecConsumer, poppedVec2);
    TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(vecConsumer);
    for (int row = 0; row < VecTile::Rows; ++row) {
        for (int col = 0; col < VecTile::Cols; ++col) {
            EXPECT_FLOAT_EQ(poppedVec2.GetElement(row, col), accSrc2.GetElement(row, col));
        }
    }
}

template <int FlagId>
void testDirBothMixedSingleAndOverlappingPopRounds()
{
    using VecTile = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 16, 16>;
    using AccTile = TileAcc<float, 16, 16>;
    using Pipe = TPipe<FlagId, Direction::DIR_BOTH, sizeof(float) * VecTile::Numel, 2>;

    Pipe::reset_for_cpu_sim();
    Pipe producerA((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe producerB((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe consumer((__gm__ void*)nullptr, 0x0, 0x10000);

    AccTile srcA;
    AccTile srcB;
    VecTile dstA;
    VecTile dstB;
    TASSIGN(srcA, 0);
    TASSIGN(srcB, AccTile::GetSizeInBytes());
    TASSIGN(dstA, 2 * AccTile::GetSizeInBytes());
    TASSIGN(dstB, 2 * AccTile::GetSizeInBytes() + VecTile::GetSizeInBytes());

    for (int round = 0; round < 8; ++round) {
        const float baseA = static_cast<float>(round * 1000 + 1);
        fillTileSequence(srcA, baseA);

        TPUSH<Pipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(producerA, srcA);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(consumer, dstA);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(consumer);
        for (int row = 0; row < VecTile::Rows; ++row) {
            for (int col = 0; col < VecTile::Cols; ++col) {
                ASSERT_FLOAT_EQ(dstA.GetElement(row, col), srcA.GetElement(row, col))
                    << "single-pop round " << round << " at (" << row << "," << col << ")";
            }
        }

        const float baseB = static_cast<float>(round * 1000 + 501);
        fillTileSequence(srcA, baseA);
        fillTileSequence(srcB, baseB);
        TPUSH<Pipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(producerA, srcA);
        TPUSH<Pipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(producerB, srcB);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(consumer, dstA);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(consumer, dstB);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(consumer);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(consumer);
        for (int row = 0; row < VecTile::Rows; ++row) {
            for (int col = 0; col < VecTile::Cols; ++col) {
                ASSERT_FLOAT_EQ(dstA.GetElement(row, col), srcA.GetElement(row, col))
                    << "overlapping round " << round << " first tile at (" << row << "," << col << ")";
                ASSERT_FLOAT_EQ(dstB.GetElement(row, col), srcB.GetElement(row, col))
                    << "overlapping round " << round << " second tile at (" << row << "," << col << ")";
            }
        }

        auto& state = Pipe::GetSharedState();
        ASSERT_EQ(state.occupied, 0) << "round " << round;
        ASSERT_EQ(state.popped_not_freed, 0) << "round " << round;
    }
}

template <int FlagId>
void testDirBothInterleavedV2CWithOverlappingC2VPops()
{
    using VecTile = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 16, 16>;
    using MatTile = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16>;
    using AccTile = TileAcc<float, 16, 16>;
    using Pipe = TPipe<FlagId, Direction::DIR_BOTH, sizeof(float) * MatTile::Numel, 2>;

    Pipe::reset_for_cpu_sim();
    Pipe cubeProducer((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe cubeConsumer((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe vecProducer((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe vecConsumer((__gm__ void*)nullptr, 0x0, 0x10000);

    AccTile accA;
    AccTile accB;
    AccTile accC;
    VecTile v2cA;
    VecTile v2cB;
    VecTile popped0;
    VecTile popped1;
    VecTile popped2;
    MatTile poppedMat0;
    MatTile poppedMat1;

    std::size_t off = 0;
    auto place = [&off](auto& tile) {
        TASSIGN(tile, off);
        off += std::remove_reference_t<decltype(tile)>::GetSizeInBytes();
    };
    place(accA);
    place(accB);
    place(accC);
    place(v2cA);
    place(v2cB);
    place(popped0);
    place(popped1);
    place(popped2);
    place(poppedMat0);
    place(poppedMat1);

    for (int round = 0; round < 4; ++round) {
        const float base = static_cast<float>(round * 10000 + 1);
        fillTileSequence(accA, base);
        fillTileSequence(v2cA, base + 1000.0f);
        fillTileSequence(v2cB, base + 2000.0f);
        fillTileSequence(accB, base + 3000.0f);
        fillTileSequence(accC, base + 4000.0f);

        TPUSH<Pipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(cubeProducer, accA);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(vecConsumer, popped0);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(vecConsumer);
        for (int i = 0; i < VecTile::Numel; ++i) {
            ASSERT_FLOAT_EQ(popped0.data()[i], accA.data()[i]) << "round " << round << " c2v single";
        }

        TPUSH<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(vecProducer, v2cA);
        TPUSH<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(vecProducer, v2cB);
        TPOP<Pipe, MatTile, TileSplitAxis::TILE_NO_SPLIT>(cubeConsumer, poppedMat0);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(cubeConsumer);
        TPOP<Pipe, MatTile, TileSplitAxis::TILE_NO_SPLIT>(cubeConsumer, poppedMat1);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(cubeConsumer);
        for (int r = 0; r < MatTile::Rows; ++r) {
            for (int c = 0; c < MatTile::Cols; ++c) {
                ASSERT_FLOAT_EQ(
                    poppedMat0.data()[GetTileElementOffset<MatTile>(r, c)],
                    v2cA.data()[GetTileElementOffset<VecTile>(r, c)])
                    << "round " << round << " v2c first at (" << r << "," << c << ")";
                ASSERT_FLOAT_EQ(
                    poppedMat1.data()[GetTileElementOffset<MatTile>(r, c)],
                    v2cB.data()[GetTileElementOffset<VecTile>(r, c)])
                    << "round " << round << " v2c second at (" << r << "," << c << ")";
            }
        }

        TPUSH<Pipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(cubeProducer, accB);
        TPUSH<Pipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(cubeProducer, accC);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(vecConsumer, popped1);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(vecConsumer, popped2);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(vecConsumer);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(vecConsumer);
        for (int i = 0; i < VecTile::Numel; ++i) {
            ASSERT_FLOAT_EQ(popped1.data()[i], accB.data()[i]) << "round " << round << " c2v overlap first";
            ASSERT_FLOAT_EQ(popped2.data()[i], accC.data()[i]) << "round " << round << " c2v overlap second";
        }

        auto& state = Pipe::GetSharedState();
        ASSERT_EQ(state.occupied, 0) << "round " << round;
        ASSERT_EQ(state.popped_not_freed, 0) << "round " << round;
    }
}

template <typename T, int rows, int cols, TileType srcLoc, TileType dstLoc>
void testPushPopSingleThread()
{
    constexpr int FiFoDepth = 8;
    constexpr int LocalDepth = 2;
    constexpr bool isC2V = (srcLoc == TileType::Acc || srcLoc == TileType::Mat) && dstLoc == TileType::Vec;
    constexpr bool isV2C = srcLoc == TileType::Vec && dstLoc == TileType::Mat;
    static_assert(isC2V || isV2C, "Only Acc/Mat->Vec and Vec->Mat modes are supported!");
    constexpr uint8_t PipeDirection = isC2V ? Direction::DIR_C2V : Direction::DIR_V2C;
    using PPTile = Tile<srcLoc, T, rows, cols>;
    using PPTile_dst = Tile<dstLoc, T, rows, cols>;
    using PPipe = TPipe<0, PipeDirection, sizeof(T) * PPTile::Numel, FiFoDepth, LocalDepth>;
    std::vector<T> fifoStorage(PPTile::Numel * FiFoDepth, static_cast<T>(0));
    PPipe::reset_for_cpu_sim();
    PPipe pipe(fifoStorage.data(), 0x0, 0x0);
    PPTile src;
    PPTile_dst dst;

    TASSIGN(src, 0);
    TASSIGN(dst, rows * cols * sizeof(T));

    fillTile<T, rows, cols, srcLoc>(src, 0);
    for (int i = 0; i < dst.Numel; ++i) {
        dst.data()[i] = static_cast<T>(0);
    }

    TPUSH(src, pipe);
    TPOP(dst, pipe);
    TFREE(pipe);

    const auto expected = makeExpected<T, rows, cols, srcLoc>(0);
    EXPECT_TRUE(ResultCmp(expected, dst.data(), 0));
}

template <typename T, int rows, int cols, TileType srcLoc, TileType dstLoc>
void testPushPopMultiCore()
{
    constexpr int FiFoDepth = 4;
    constexpr int LocalDepth = 2;
    constexpr bool isC2V = (srcLoc == TileType::Acc || srcLoc == TileType::Mat) && dstLoc == TileType::Vec;
    constexpr bool isV2C = srcLoc == TileType::Vec && dstLoc == TileType::Mat;
    static_assert(isC2V || isV2C, "Only Acc/Mat->Vec and Vec->Mat modes are supported!");
    constexpr uint8_t PipeDirection = isC2V ? Direction::DIR_C2V : Direction::DIR_V2C;
    using PPTile = Tile<srcLoc, T, rows, cols>;
    using PPTile_dst = Tile<dstLoc, T, rows, cols>;
    using PPipe = TPipe<1, PipeDirection, sizeof(T) * PPTile::Numel, FiFoDepth, LocalDepth>;

    constexpr int kIterations = 12;
    std::vector<T> fifoStorage(PPTile::Numel * FiFoDepth, static_cast<T>(0));
    std::vector<std::vector<T>> actual(kIterations);
    PPipe::reset_for_cpu_sim();
    PPipe pipe(fifoStorage.data(), 0x0, 0x0);

    std::thread producer([&]() {
        for (int iter = 0; iter < kIterations; ++iter) {
            PPTile src;
            TASSIGN(src, 0);
            fillTile<T, rows, cols, srcLoc>(src, iter);
            TPUSH(src, pipe);
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    std::thread consumer([&]() {
        for (int iter = 0; iter < kIterations; ++iter) {
            PPTile_dst dst;
            TASSIGN(dst, 0);
            for (int i = 0; i < dst.Numel; ++i) {
                dst.data()[i] = static_cast<T>(0);
            }
            TPOP(dst, pipe);
            TFREE(pipe);
            actual[iter].assign(dst.data(), dst.data() + dst.Numel);
        }
    });

    producer.join();
    consumer.join();

    for (int iter = 0; iter < kIterations; ++iter) {
        const auto expected = makeExpected<T, rows, cols, srcLoc>(iter);
        EXPECT_TRUE(ResultCmp(expected, actual[iter], 0));
    }
}

class TPushPopTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

#define TPUSHPOP_TEST(T, rows, cols, srcLoc, dstLoc)                                  \
    TEST_F(TPushPopTest, T##_##rows##_##cols##_##srcLoc)                              \
    {                                                                                 \
        testPushPopSingleThread<T, rows, cols, TileType::srcLoc, TileType::dstLoc>(); \
    }

TPUSHPOP_TEST(float, 64, 128, Vec, Mat)
TPUSHPOP_TEST(float, 128, 128, Vec, Mat)
TPUSHPOP_TEST(float, 64, 128, Acc, Vec)
TPUSHPOP_TEST(float, 128, 128, Acc, Vec)
TPUSHPOP_TEST(uint32_t, 64, 128, Vec, Mat)
TPUSHPOP_TEST(uint32_t, 128, 128, Vec, Mat)
TPUSHPOP_TEST(uint32_t, 64, 128, Acc, Vec)
TPUSHPOP_TEST(uint32_t, 128, 128, Acc, Vec)

TEST_F(TPushPopTest, multicore_float_64_128_Vec)
{
    testPushPopMultiCore<float, 64, 128, TileType::Vec, TileType::Mat>();
}

// A narrow view of a wider Vec tile: the payload must use the pushed window's width,
// otherwise the consumer reads every other row.
template <typename T, int parentCols, int windowCols, int rows, int colOffset>
void testPushPopNarrowVecView()
{
    constexpr int FiFoDepth = 4;
    constexpr int LocalDepth = 2;
    using ParentTile = Tile<TileType::Vec, T, rows, parentCols>;
    using ViewTile = Tile<TileType::Vec, T, rows, parentCols, BLayout::RowMajor, rows, windowCols>;
    using MatTile = Tile<TileType::Mat, T, rows, windowCols>;
    using PPipe = TPipe<3, Direction::DIR_V2C, sizeof(T) * rows * windowCols, FiFoDepth, LocalDepth>;

    std::vector<T> fifoStorage(static_cast<std::size_t>(rows) * windowCols * FiFoDepth, static_cast<T>(0));
    PPipe::reset_for_cpu_sim();
    PPipe pipe(fifoStorage.data(), 0x0, 0x0);

    ParentTile parent;
    TASSIGN(parent, 0);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < parentCols; ++c) {
            parent.data()[GetTileElementOffset<ParentTile>(r, c)] = static_cast<T>(r * parentCols + c);
        }
    }

    ViewTile view;
    TASSIGN(view, colOffset * static_cast<int>(sizeof(T)));

    MatTile dst;
    TASSIGN(dst, static_cast<int>(sizeof(T)) * rows * parentCols);
    for (int i = 0; i < dst.Numel; ++i) {
        dst.data()[i] = static_cast<T>(0);
    }

    TPUSH<PPipe, ViewTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, view);
    TPOP<PPipe, MatTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, dst);
    TFREE<PPipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < windowCols; ++c) {
            const auto expected = static_cast<T>(r * parentCols + colOffset + c);
            ASSERT_EQ(dst.data()[GetTileElementOffset<MatTile>(r, c)], expected)
                << "narrow view v2c at (" << r << "," << c << ")";
        }
    }
}

TEST_F(TPushPopTest, v2c_narrow_vec_view_keeps_rows)
{
    testPushPopNarrowVecView<float, 64, 32, 16, 0>();
    testPushPopNarrowVecView<float, 64, 32, 16, 32>();
    testPushPopNarrowVecView<float, 128, 32, 8, 64>();
}

TEST_F(TPushPopTest, v2c_gm_fifo_preserves_updown_and_leftright_layout)
{
    testGmFifoPreservesV2CSplitLayout<TileSplitAxis::TILE_UP_DOWN, 10>();
    testGmFifoPreservesV2CSplitLayout<TileSplitAxis::TILE_LEFT_RIGHT, 11>();
}

TEST_F(TPushPopTest, a5_style_c2v_local_split_push_pop)
{
    using AccTile = TileAcc<float, 16, 16>;
    using VecTile = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, 8, 16>;
    using Pipe = TPipe<2, Direction::DIR_C2V, sizeof(float) * VecTile::Numel, 2>;

    Pipe::reset_for_cpu_sim();
    Pipe pipe((__gm__ void*)nullptr, 0x0, 0x0);

    AccTile src;
    VecTile dst;
    TASSIGN(src, 0);
    TASSIGN(dst, AccTile::Rows * AccTile::Cols * sizeof(AccTile::DType));

    fillTile<float, 16, 16, TileType::Acc>(src, 0);
    std::fill(dst.data(), dst.data() + dst.Numel, 0.0f);

    EXPECT_EQ(get_subblockid(), 0u);
    EXPECT_EQ(get_subblockdim(), 1u);

    TPUSH<Pipe, AccTile, TileSplitAxis::TILE_UP_DOWN>(pipe, src);
    TPOP<Pipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(pipe, dst);
    TFREE<Pipe, TileSplitAxis::TILE_UP_DOWN>(pipe);

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            EXPECT_EQ(dst.data()[r * dst.Cols + c], src.data()[r * src.Cols + c]);
        }
    }
}

TEST_F(TPushPopTest, a5_style_c2v_dual_subblock_split_push_pop)
{
    using AccTile = TileAcc<float, 16, 16>;
    using VecTile = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, 8, 16>;
    using Pipe = TPipe<4, Direction::DIR_C2V, sizeof(float) * VecTile::Numel, 1>;

    Pipe::reset_for_cpu_sim();
    Pipe producer((__gm__ void*)nullptr, 0x0, 0x0);
    Pipe consumer0((__gm__ void*)nullptr, 0x0, 0x0);
    Pipe consumer1((__gm__ void*)nullptr, 0x0, 0x0);

    auto run_iteration = [&](int iter) {
        AccTile src;
        VecTile topHalf;
        VecTile bottomHalf;
        TASSIGN(src, 0);
        TASSIGN(topHalf, AccTile::GetSizeInBytes());
        TASSIGN(bottomHalf, AccTile::GetSizeInBytes() + VecTile::GetSizeInBytes());
        fillTile<float, 16, 16, TileType::Acc>(src, iter);
        std::fill(topHalf.data(), topHalf.data() + topHalf.Numel, 0.0f);
        std::fill(bottomHalf.data(), bottomHalf.data() + bottomHalf.Numel, 0.0f);

        {
            cpu_sim::ScopedExecutionContext producerCtx(0, 0, 1);
            TPUSH<Pipe, AccTile, TileSplitAxis::TILE_UP_DOWN>(producer, src);
        }
        {
            cpu_sim::ScopedExecutionContext consumerCtx(0, 0, 2);
            TPOP<Pipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(consumer0, topHalf);
        }
        {
            cpu_sim::ScopedExecutionContext consumerCtx(0, 1, 2);
            TPOP<Pipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(consumer1, bottomHalf);
        }

        for (int r = 0; r < topHalf.GetValidRow(); ++r) {
            for (int c = 0; c < topHalf.GetValidCol(); ++c) {
                EXPECT_EQ(
                    topHalf.data()[GetTileElementOffset<VecTile>(r, c)],
                    src.data()[GetTileElementOffset<AccTile>(r, c)]);
                EXPECT_EQ(
                    bottomHalf.data()[GetTileElementOffset<VecTile>(r, c)],
                    src.data()[GetTileElementOffset<AccTile>(r + topHalf.GetValidRow(), c)]);
            }
        }

        {
            cpu_sim::ScopedExecutionContext consumerCtx(0, 0, 2);
            TFREE<Pipe, TileSplitAxis::TILE_UP_DOWN>(consumer0);
        }
        {
            cpu_sim::ScopedExecutionContext consumerCtx(0, 1, 2);
            TFREE<Pipe, TileSplitAxis::TILE_UP_DOWN>(consumer1);
        }
    };

    run_iteration(0);
    run_iteration(1);
}

TEST_F(TPushPopTest, c2v_split_delayed_free_releases_each_popped_slot)
{
    using AccTile = TileAcc<float, 16, 16>;
    using VecTile = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, 8, 16>;
    using Pipe = TPipe<4, Direction::DIR_C2V, sizeof(float) * VecTile::Numel, 2>;

    Pipe::reset_for_cpu_sim();
    Pipe producer((__gm__ void*)nullptr, 0x0, 0x0);
    Pipe consumer0((__gm__ void*)nullptr, 0x0, 0x0);
    Pipe consumer1((__gm__ void*)nullptr, 0x0, 0x0);
    AccTile src0;
    AccTile src1;
    VecTile top0;
    VecTile top1;
    VecTile bottom0;
    VecTile bottom1;
    TASSIGN(src0, 0);
    TASSIGN(src1, AccTile::GetSizeInBytes());
    TASSIGN(top0, 2 * AccTile::GetSizeInBytes());
    TASSIGN(top1, 2 * AccTile::GetSizeInBytes() + VecTile::GetSizeInBytes());
    TASSIGN(bottom0, 2 * AccTile::GetSizeInBytes() + 2 * VecTile::GetSizeInBytes());
    TASSIGN(bottom1, 2 * AccTile::GetSizeInBytes() + 3 * VecTile::GetSizeInBytes());
    fillTileSequence(src0, 1.0f);
    fillTileSequence(src1, 1001.0f);

    {
        cpu_sim::ScopedExecutionContext producerCtx(0, 0, 1);
        TPUSH<Pipe, AccTile, TileSplitAxis::TILE_UP_DOWN>(producer, src0);
        TPUSH<Pipe, AccTile, TileSplitAxis::TILE_UP_DOWN>(producer, src1);
    }
    {
        cpu_sim::ScopedExecutionContext consumerCtx(0, 0, 2);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(consumer0, top0);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(consumer0, top1);
    }
    {
        cpu_sim::ScopedExecutionContext consumerCtx(0, 1, 2);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(consumer1, bottom0);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(consumer1, bottom1);
    }

    for (int r = 0; r < VecTile::Rows; ++r) {
        for (int c = 0; c < VecTile::Cols; ++c) {
            EXPECT_FLOAT_EQ(top0.GetElement(r, c), src0.GetElement(r, c));
            EXPECT_FLOAT_EQ(top1.GetElement(r, c), src1.GetElement(r, c));
            EXPECT_FLOAT_EQ(bottom0.GetElement(r, c), src0.GetElement(r + VecTile::Rows, c));
            EXPECT_FLOAT_EQ(bottom1.GetElement(r, c), src1.GetElement(r + VecTile::Rows, c));
        }
    }

    {
        cpu_sim::ScopedExecutionContext consumerCtx(0, 0, 2);
        TFREE<Pipe, TileSplitAxis::TILE_UP_DOWN>(consumer0);
        TFREE<Pipe, TileSplitAxis::TILE_UP_DOWN>(consumer0);
    }
    {
        cpu_sim::ScopedExecutionContext consumerCtx(0, 1, 2);
        TFREE<Pipe, TileSplitAxis::TILE_UP_DOWN>(consumer1);
        TFREE<Pipe, TileSplitAxis::TILE_UP_DOWN>(consumer1);
    }

    auto& state = Pipe::GetSharedState(cpu_pipe::TransferDir::C2V);
    EXPECT_EQ(state.occupied, 0);
    EXPECT_EQ(state.popped_not_freed_by_lane[0], 0);
    EXPECT_EQ(state.popped_not_freed_by_lane[1], 0);
    EXPECT_EQ(state.slot_busy[0], 0);
    EXPECT_EQ(state.slot_busy[1], 0);
}

TEST_F(TPushPopTest, cpu_stub_prefers_injected_hooks_for_subblock_and_pipe_state)
{
    HookTestPipe::SharedStateStorage storage{};
    g_injected_subblock_id.store(7, std::memory_order_relaxed);
    g_pipe_hook_call_count.store(0, std::memory_order_relaxed);
    g_pipe_hook_storage = &storage;
    g_pipe_hook_size = 0;
    g_pipe_hook_last_key = 0;

    ScopedCpuStubHooks hooks(
        reinterpret_cast<void*>(MockSubblockIdHook), reinterpret_cast<void*>(MockPipeSharedStateHook));
    cpu_sim::set_execution_context(0, 1, 2);

    EXPECT_EQ(get_subblockid(), 7u);

    auto& state = HookTestPipe::GetSharedState();
    state.next_producer_slot = 3;
    auto& stateAgain = HookTestPipe::GetSharedState();

    EXPECT_EQ(&state, &stateAgain);
    EXPECT_EQ(stateAgain.next_producer_slot, 3);
    EXPECT_GT(g_pipe_hook_call_count.load(std::memory_order_relaxed), 0u);
    EXPECT_EQ(g_pipe_hook_size, sizeof(HookTestPipe::SharedStateStorage));
    EXPECT_NE(g_pipe_hook_last_key, 0u);
}

TEST_F(TPushPopTest, v2c_split_with_injected_pipe_hook_waits_for_both_lanes_before_publish)
{
    using VecTile = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, 8, 16>;
    using MatTile = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16>;

    HookedV2CPipe::SharedStateStorage storage{};
    g_pipe_hook_call_count.store(0, std::memory_order_relaxed);
    g_pipe_hook_storage = &storage;
    g_pipe_hook_size = 0;
    g_pipe_hook_last_key = 0;

    ScopedCpuStubHooks hooks(nullptr, reinterpret_cast<void*>(MockPipeSharedStateHook));
    HookedV2CPipe::reset_for_cpu_sim();

    HookedV2CPipe producer0((__gm__ void*)nullptr, 0x0, 0x10000);
    HookedV2CPipe producer1((__gm__ void*)nullptr, 0x0, 0x10000);
    HookedV2CPipe consumer((__gm__ void*)nullptr, 0x0, 0x10000);
    VecTile topHalf;
    VecTile bottomHalf;
    MatTile dst;
    TASSIGN(topHalf, 0);
    TASSIGN(bottomHalf, VecTile::Numel * sizeof(VecTile::DType));
    TASSIGN(dst, 2 * VecTile::Numel * sizeof(VecTile::DType));
    fillTile<float, 8, 16, TileType::Vec>(topHalf, 0);
    fillTile<float, 8, 16, TileType::Vec>(bottomHalf, 1);
    std::fill(dst.data(), dst.data() + dst.Numel, 0.0f);

    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 2);
        TPUSH<HookedV2CPipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(producer0, topHalf);
    }

    auto& state = HookedV2CPipe::GetSharedState();
    EXPECT_EQ(state.occupied, 0);
    EXPECT_EQ(state.next_producer_slot, 0);
    EXPECT_EQ(state.producers_done[0], 0x1u);
    EXPECT_EQ(state.producers_allocated[0], 0x1u);

    {
        cpu_sim::ScopedExecutionContext ctx(0, 1, 2);
        TPUSH<HookedV2CPipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(producer1, bottomHalf);
    }

    EXPECT_EQ(state.occupied, 1);
    EXPECT_EQ(state.next_producer_slot, 1);
    EXPECT_EQ(state.producers_done[0], 0u);
    EXPECT_EQ(state.producers_allocated[0], 0u);

    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
        TPOP<HookedV2CPipe, MatTile, TileSplitAxis::TILE_UP_DOWN>(consumer, dst);
        TFREE<HookedV2CPipe, TileSplitAxis::TILE_UP_DOWN>(consumer);
    }

    for (int r = 0; r < topHalf.GetValidRow(); ++r) {
        for (int c = 0; c < topHalf.GetValidCol(); ++c) {
            EXPECT_EQ(
                dst.data()[GetTileElementOffset<MatTile>(r, c)], topHalf.data()[GetTileElementOffset<VecTile>(r, c)]);
            EXPECT_EQ(
                dst.data()[GetTileElementOffset<MatTile>(r + topHalf.GetValidRow(), c)],
                bottomHalf.data()[GetTileElementOffset<VecTile>(r, c)]);
        }
    }

    EXPECT_GT(g_pipe_hook_call_count.load(std::memory_order_relaxed), 0u);
    EXPECT_EQ(g_pipe_hook_size, sizeof(HookedV2CPipe::SharedStateStorage));
    EXPECT_NE(g_pipe_hook_last_key, 0u);
}

TEST_F(TPushPopTest, a5_style_dir_both_updown_waits_for_matching_direction)
{
    testDirBothConsumerWaitsForMatchingDirection<TileSplitAxis::TILE_UP_DOWN, 7>();
}

TEST_F(TPushPopTest, a5_style_dir_both_leftright_waits_for_matching_direction)
{
    testDirBothConsumerWaitsForMatchingDirection<TileSplitAxis::TILE_LEFT_RIGHT, 8>();
}

TEST_F(TPushPopTest, a5_style_dir_both_updown_split_v2c_free_preserves_c2v_fifo_order)
{
    testDirBothSplitV2CFreePreservesC2VFifoOrder<TileSplitAxis::TILE_UP_DOWN, 13>();
}

TEST_F(TPushPopTest, a5_style_dir_both_leftright_split_v2c_free_preserves_c2v_fifo_order)
{
    testDirBothSplitV2CFreePreservesC2VFifoOrder<TileSplitAxis::TILE_LEFT_RIGHT, 14>();
}

TEST_F(TPushPopTest, dir_both_overlapping_c2v_pops_free_in_order) { testDirBothOverlappingC2VPopsFreeInOrder<15>(); }

TEST_F(TPushPopTest, dir_both_mixed_single_and_overlapping_pop_rounds)
{
    testDirBothMixedSingleAndOverlappingPopRounds<16>();
}

template <uint8_t FlagId>
void testDirBothDualLaneC2VOrderAfterV2C()
{
    using AccTile = TileAcc<float, 16, 16>;
    using VecTile = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 16, 16>;
    using MatTile = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16>;
    using Pipe = TPipe<FlagId, Direction::DIR_BOTH, sizeof(float) * MatTile::Numel, 2, 2, true>;

    Pipe::reset_for_cpu_sim();
    Pipe cubeProducer((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe cubeConsumer((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe vecProducer((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe vecLane0((__gm__ void*)nullptr, 0x0, 0x10000);
    Pipe vecLane1((__gm__ void*)nullptr, 0x0, 0x10000);

    AccTile accA;
    AccTile accB;
    AccTile accC;
    VecTile v2cA;
    VecTile lane0First;
    VecTile lane0Second;
    VecTile lane1First;
    VecTile lane1Second;
    VecTile warmLane0;
    VecTile warmLane1;
    MatTile poppedMat;

    std::size_t off = 0;
    auto place = [&off](auto& tile) {
        TASSIGN(tile, off);
        off += std::remove_reference_t<decltype(tile)>::GetSizeInBytes();
    };
    place(accA);
    place(accB);
    place(accC);
    place(v2cA);
    place(lane0First);
    place(lane0Second);
    place(lane1First);
    place(lane1Second);
    place(warmLane0);
    place(warmLane1);
    place(poppedMat);

    for (int round = 0; round < 3; ++round) {
        const float base = static_cast<float>(round * 10000 + 1);
        fillTileSequence(accA, base);
        fillTileSequence(v2cA, base + 1000.0f);
        fillTileSequence(accB, base + 2000.0f);
        fillTileSequence(accC, base + 3000.0f);

        TPUSH<Pipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(cubeProducer, accA);
        {
            cpu_sim::ScopedExecutionContext lane(0, 0, 2);
            TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(vecLane0, warmLane0);
            TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(vecLane0);
        }
        {
            cpu_sim::ScopedExecutionContext lane(0, 1, 2);
            TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(vecLane1, warmLane1);
            TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(vecLane1);
        }

        TPUSH<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(vecProducer, v2cA);
        TPOP<Pipe, MatTile, TileSplitAxis::TILE_NO_SPLIT>(cubeConsumer, poppedMat);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(cubeConsumer);

        TPUSH<Pipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(cubeProducer, accB);
        TPUSH<Pipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(cubeProducer, accC);
        {
            cpu_sim::ScopedExecutionContext lane(0, 0, 2);
            TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(vecLane0, lane0First);
            TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(vecLane0, lane0Second);
        }
        {
            cpu_sim::ScopedExecutionContext lane(0, 1, 2);
            TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(vecLane1, lane1First);
            TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(vecLane1, lane1Second);
        }
        for (int i = 0; i < VecTile::Numel; ++i) {
            ASSERT_FLOAT_EQ(lane0First.data()[i], accB.data()[i]) << "round " << round << " lane0 first";
            ASSERT_FLOAT_EQ(lane0Second.data()[i], accC.data()[i]) << "round " << round << " lane0 second";
            ASSERT_FLOAT_EQ(lane1First.data()[i], accB.data()[i]) << "round " << round << " lane1 first";
            ASSERT_FLOAT_EQ(lane1Second.data()[i], accC.data()[i]) << "round " << round << " lane1 second";
        }
        {
            cpu_sim::ScopedExecutionContext lane(0, 0, 2);
            TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(vecLane0);
            TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(vecLane0);
        }
        {
            cpu_sim::ScopedExecutionContext lane(0, 1, 2);
            TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(vecLane1);
            TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(vecLane1);
        }

        auto& state = Pipe::GetSharedState();
        ASSERT_EQ(state.occupied, 0) << "round " << round;
        ASSERT_EQ(state.popped_not_freed, 0) << "round " << round;
    }
}

TEST_F(TPushPopTest, dir_both_dual_lane_c2v_order_after_v2c) { testDirBothDualLaneC2VOrderAfterV2C<19>(); }

TEST_F(TPushPopTest, dir_both_interleaved_v2c_with_overlapping_c2v_pops)
{
    testDirBothInterleavedV2CWithOverlappingC2VPops<17>();
}

TEST_F(TPushPopTest, tile_flow_keeps_non_null_gm_workspace_out_of_cpu_data_plane)
{
    using AccTile = TileAcc<float, 16, 16>;
    using VecTile = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 16, 16>;
    using MatTile = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16>;
    using Pipe = TPipe<12, Direction::DIR_BOTH, sizeof(float) * AccTile::Numel, 2, 2, true>;

    std::vector<uint8_t> gmWorkspace(Pipe::RingFiFo::SLOT_SIZE * Pipe::RingFiFo::SLOT_NUM, 0xa5);
    const auto expectedWorkspace = gmWorkspace;

    Pipe::reset_for_cpu_sim();
    Pipe cube(gmWorkspace.data(), 0, 0);
    Pipe vector(gmWorkspace.data(), 0, 0);

    AccTile accSrc;
    VecTile vecData;
    MatTile matDst;
    TASSIGN(accSrc, 0);
    TASSIGN(vecData, AccTile::GetSizeInBytes());
    TASSIGN(matDst, AccTile::GetSizeInBytes() + VecTile::GetSizeInBytes());

    fillTileSequence(accSrc, 1.0f);

    TPUSH<Pipe, AccTile, TileSplitAxis::TILE_NO_SPLIT>(cube, accSrc);
    TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(vector, vecData);
    TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(vector);

    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
            EXPECT_FLOAT_EQ(vecData.GetElement(row, col), accSrc.GetElement(row, col));
            vecData.SetElement(row, col, vecData.GetElement(row, col) + 1.0f);
        }
    }

    TPUSH<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(vector, vecData);
    TPOP<Pipe, MatTile, TileSplitAxis::TILE_NO_SPLIT>(cube, matDst);
    TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(cube);

    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
            EXPECT_FLOAT_EQ(matDst.GetElement(row, col), accSrc.GetElement(row, col) + 1.0f);
        }
    }
    EXPECT_EQ(gmWorkspace, expectedWorkspace);
}

TEST_F(TPushPopTest, host_slot_byte_storage_supports_unaligned_uint64_access)
{
    using MatTile = Tile<TileType::Mat, uint64_t, 1, 4, BLayout::RowMajor, 1, 4>;
    using VecTile = Tile<TileType::Vec, uint64_t, 1, 4, BLayout::RowMajor, 1, 4>;
    constexpr uint32_t kUnalignedSlotSize = MatTile::GetSizeInBytes() + 1;
    using Pipe = TPipe<15, Direction::DIR_C2V, kUnalignedSlotSize, 2>;

    Pipe::reset_for_cpu_sim();
    Pipe producer((__gm__ void*)nullptr, 0, 0);
    Pipe consumer((__gm__ void*)nullptr, 0, 0);

    MatTile src;
    VecTile dst;
    TASSIGN(src, 0);
    TASSIGN(dst, MatTile::GetSizeInBytes());

    for (uint64_t value : {0x0123456789abcdefULL, 0xfedcba9876543210ULL}) {
        for (int col = 0; col < MatTile::Cols; ++col) {
            src.SetElement(0, col, value + static_cast<uint64_t>(col));
            dst.SetElement(0, col, 0);
        }

        TPUSH<Pipe, MatTile, TileSplitAxis::TILE_NO_SPLIT>(producer, src);
        TPOP<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(consumer, dst);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(consumer);

        for (int col = 0; col < VecTile::Cols; ++col) {
            EXPECT_EQ(dst.GetElement(0, col), value + static_cast<uint64_t>(col));
        }
    }
}

TEST_F(TPushPopTest, host_slot_byte_storage_reports_logical_slot_overflow)
{
    using MatTile = Tile<TileType::Mat, uint64_t, 1, 4, BLayout::RowMajor, 1, 4>;
    constexpr uint32_t kUndersizedSlot = MatTile::GetSizeInBytes() - 1;
    using Pipe = TPipe<16, Direction::DIR_C2V, kUndersizedSlot, 1>;

    Pipe::reset_for_cpu_sim();
    Pipe producer((__gm__ void*)nullptr, 0, 0);
    MatTile src;
    TASSIGN(src, 0);
    for (int col = 0; col < MatTile::Cols; ++col) {
        src.SetElement(0, col, static_cast<uint64_t>(col + 1));
    }

    try {
        TPUSH<Pipe, MatTile, TileSplitAxis::TILE_NO_SPLIT>(producer, src);
        FAIL() << "Expected an undersized CPU TPipe slot to be rejected";
    } catch (const std::out_of_range& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("slot=0"), std::string::npos);
        EXPECT_NE(message.find("element_size=8"), std::string::npos);
        EXPECT_NE(message.find("region_byte_end=31"), std::string::npos);
    }

    Pipe::reset_for_cpu_sim();
}

TEST_F(TPushPopTest, dir_both_full_capacity_and_delayed_free)
{
    using AccTile = TileAcc<float, 16, 16>;
    using VecTile = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 16, 16>;
    using MatTile = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16>;
    using Pipe = TPipe<0, Direction::DIR_BOTH, sizeof(float) * 256, 2, 2, true>;
    constexpr auto split = TileSplitAxis::TILE_NO_SPLIT;
    cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
    Pipe::reset_for_cpu_sim();
    Pipe producer(nullptr, 0, 0);
    Pipe consumer(nullptr, 0, 0);
    AccTile acc;
    VecTile vec;
    VecTile forward;
    MatTile backward;
    TASSIGN(acc, 0);
    TASSIGN(vec, 0x1000);
    TASSIGN(forward, 0x2000);
    TASSIGN(backward, 0x3000);

    for (int round = 0; round < 8; ++round) {
        for (int i = 0; i < 2; ++i) {
            fillTileSequence(acc, float(round * 10000 + i * 1000));
            TPUSH<Pipe, AccTile, split>(producer, acc);
            fillTileSequence(vec, float(round * 10000 + i * 1000 + 500));
            TPUSH<Pipe, VecTile, split>(producer, vec);
        }
        // Both directions must retain two tiles without either consumer freeing a slot.
        for (int i = 0; i < 2; ++i) {
            TPOP<Pipe, VecTile, split>(consumer, forward);
            TPOP<Pipe, MatTile, split>(consumer, backward);
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                    const float expected = float(round * 10000 + i * 1000);
                    EXPECT_FLOAT_EQ(forward.GetElement(r, c), expected + GetTileElementOffset<AccTile>(r, c));
                    EXPECT_FLOAT_EQ(backward.GetElement(r, c), expected + 500 + r * 16 + c);
                }
            }
        }
        for (int i = 0; i < 4; ++i) {
            TFREE<Pipe, split>(consumer);
        }
        EXPECT_EQ(Pipe::GetSharedState(cpu_pipe::TransferDir::C2V).occupied, 0);
        EXPECT_EQ(Pipe::GetSharedState(cpu_pipe::TransferDir::V2C).occupied, 0);
    }
}

TEST_F(TPushPopTest, dir_both_four_push_pipeline_round_trip)
{
    using AccTile = TileAcc<float, 16, 16>;
    using VecTile = Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 16, 16>;
    using MatTile = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16>;
    using Pipe = TPipe<1, Direction::DIR_BOTH, sizeof(float) * 256, 2, 2, true>;
    constexpr auto split = TileSplitAxis::TILE_NO_SPLIT;
    Pipe::reset_for_cpu_sim();
    std::thread cube([] {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
        Pipe pipe(nullptr, 0, 0);
        AccTile input;
        MatTile output;
        TASSIGN(input, 0);
        TASSIGN(output, 0x1000);
        for (int round = 0; round < 8; ++round) {
            for (int i = 0; i < 4; ++i) {
                std::fill_n(input.data(), input.Numel, float(round * 4 + i + 1));
                TPUSH<Pipe, AccTile, split>(pipe, input);
            }
            for (int i = 0; i < 4; ++i) {
                TPOP<Pipe, MatTile, split>(pipe, output);
                for (int j = 0; j < output.Numel; ++j) {
                    EXPECT_FLOAT_EQ(output.data()[j], float(round * 4 + i + 1) * 0.5f);
                }
                TFREE<Pipe, split>(pipe);
            }
        }
    });
    std::thread vector([] {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
        Pipe pipe(nullptr, 0, 0);
        VecTile tile;
        TASSIGN(tile, 0x2000);
        for (int i = 0; i < 32; ++i) {
            TPOP<Pipe, VecTile, split>(pipe, tile);
            for (int j = 0; j < tile.Numel; ++j) {
                tile.data()[j] *= 0.5f;
            }
            TFREE<Pipe, split>(pipe);
            TPUSH<Pipe, VecTile, split>(pipe, tile);
        }
    });
    cube.join();
    vector.join();
    EXPECT_EQ(Pipe::GetSharedState(cpu_pipe::TransferDir::C2V).occupied, 0);
    EXPECT_EQ(Pipe::GetSharedState(cpu_pipe::TransferDir::V2C).occupied, 0);
}

TEST_F(TPushPopTest, dir_both_hook_storage_initializes_and_resets_both_directions)
{
    using Pipe = TPipe<2, Direction::DIR_BOTH, sizeof(float) * 256, 2, 2, true>;
    Pipe::SharedStateStorage storage{};
    g_pipe_hook_storage = &storage;
    ScopedCpuStubHooks hooks(nullptr, reinterpret_cast<void*>(&MockPipeSharedStateHook));
    auto& reverse = Pipe::GetSharedState(cpu_pipe::TransferDir::V2C);
    auto& forward = Pipe::GetSharedState(cpu_pipe::TransferDir::C2V);
    EXPECT_NE(&forward, &reverse);
    EXPECT_EQ(g_pipe_hook_size, sizeof(storage));
    forward.occupied = 1;
    reverse.occupied = 2;
    Pipe::reset_for_cpu_sim();
    EXPECT_EQ(forward.occupied, 0);
    EXPECT_EQ(reverse.occupied, 0);
    EXPECT_EQ(&reverse, &Pipe::GetSharedState(cpu_pipe::TransferDir::V2C));
}

namespace {
void testDirBothV2CNZLayout(uint32_t destinationBase)
{
    using VecTile = Tile<TileType::Vec, float, 16, 128, BLayout::ColMajor, 16, 128, SLayout::RowMajor, 512>;
    using MatTile = Tile<TileType::Mat, float, 16, 128, BLayout::ColMajor, 16, 128, SLayout::RowMajor, 512>;
    using Pipe = TPipe<3, Direction::DIR_BOTH, sizeof(float) * 16 * 128, 2, 2, true>;
    constexpr auto split = TileSplitAxis::TILE_NO_SPLIT;
    cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
    Pipe::reset_for_cpu_sim();
    Pipe pipe(nullptr, 0, 0);
    VecTile input;
    MatTile output;
    TASSIGN(input, 0x4000);
    TASSIGN(output, destinationBase);

    // NZ producer and consumer match the A5 V2C layout. Reuse both FIFO slots.
    for (int round = 0; round < 8; ++round) {
        SCOPED_TRACE(round);
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 128; ++c) {
                input.SetElement(r, c, float(round * 10000 + r * 128 + c + 1));
            }
        }
        TPUSH<Pipe, VecTile, split>(pipe, input);
        TPOP<Pipe, MatTile, split>(pipe, output);
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 128; ++c) {
                EXPECT_FLOAT_EQ(output.GetElement(r, c), float(round * 10000 + r * 128 + c + 1));
            }
        }
        TFREE<Pipe, split>(pipe);
    }
    EXPECT_EQ(Pipe::GetSharedState(cpu_pipe::TransferDir::V2C).occupied, 0);
}
} // namespace

TEST_F(TPushPopTest, dir_both_v2c_nz_layout_with_aliased_fifo_destination) { testDirBothV2CNZLayout(0); }

TEST_F(TPushPopTest, dir_both_v2c_nz_layout_with_separate_destination) { testDirBothV2CNZLayout(0x10000); }

namespace {
template <typename TileData>
void fillTileLogical(TileData& tile, int valueStride)
{
    for (int r = 0; r < tile.GetValidRow(); ++r) {
        for (int c = 0; c < tile.GetValidCol(); ++c) {
            tile.SetElement(r, c, static_cast<typename TileData::DType>(r * valueStride + c + 1));
        }
    }
}
} // namespace

// C2V: the cube produces an odd number of rows, so AIV0 takes ceil(rows/2) and AIV1 takes
// floor(rows/2). Each lane's window lives in its own local region.
TEST_F(TPushPopTest, c2v_updown_odd_split_push_pop)
{
    using AccTile = TileAcc<float, 16, 16, 15, 16>;
    using TopTile = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, 8, 16>;
    using BottomTile = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, 7, 16>;
    using Pipe = TPipe<20, Direction::DIR_C2V, sizeof(float) * TopTile::Numel, 1>;

    Pipe::reset_for_cpu_sim();
    Pipe producer((__gm__ void*)nullptr, 0x0, 0x0);
    Pipe consumer0((__gm__ void*)nullptr, 0x0, 0x0);
    Pipe consumer1((__gm__ void*)nullptr, 0x0, 0x0);

    AccTile src;
    TopTile top;
    BottomTile bottom;
    TASSIGN(src, 0);
    TASSIGN(top, AccTile::GetSizeInBytes());
    TASSIGN(bottom, AccTile::GetSizeInBytes() + TopTile::GetSizeInBytes());
    fillTileLogical(src, 16);

    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
        TPUSH<Pipe, AccTile, TileSplitAxis::TILE_UP_DOWN_ODD>(producer, src);
    }
    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 2);
        TPOP<Pipe, TopTile, TileSplitAxis::TILE_UP_DOWN_ODD>(consumer0, top);
    }
    {
        cpu_sim::ScopedExecutionContext ctx(0, 1, 2);
        TPOP<Pipe, BottomTile, TileSplitAxis::TILE_UP_DOWN_ODD>(consumer1, bottom);
    }

    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 16; ++c) {
            EXPECT_FLOAT_EQ(top.GetElement(r, c), static_cast<float>(r * 16 + c + 1));
        }
    }
    for (int r = 0; r < 7; ++r) {
        for (int c = 0; c < 16; ++c) {
            EXPECT_FLOAT_EQ(bottom.GetElement(r, c), static_cast<float>((r + 8) * 16 + c + 1));
        }
    }

    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 2);
        TFREE<Pipe, TileSplitAxis::TILE_UP_DOWN_ODD>(consumer0);
    }
    {
        cpu_sim::ScopedExecutionContext ctx(0, 1, 2);
        TFREE<Pipe, TileSplitAxis::TILE_UP_DOWN_ODD>(consumer1);
    }
}

// C2V: odd number of columns, so AIV0 takes ceil(cols/2) and AIV1 takes floor(cols/2).
TEST_F(TPushPopTest, c2v_leftright_odd_split_push_pop)
{
    using AccTile = TileAcc<float, 16, 16, 16, 15>;
    using LeftTile = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, 16, 8>;
    using RightTile = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, 16, 7>;
    using Pipe = TPipe<21, Direction::DIR_C2V, sizeof(float) * LeftTile::Numel, 1>;

    Pipe::reset_for_cpu_sim();
    Pipe producer((__gm__ void*)nullptr, 0x0, 0x0);
    Pipe consumer0((__gm__ void*)nullptr, 0x0, 0x0);
    Pipe consumer1((__gm__ void*)nullptr, 0x0, 0x0);

    AccTile src;
    LeftTile left;
    RightTile right;
    TASSIGN(src, 0);
    TASSIGN(left, AccTile::GetSizeInBytes());
    TASSIGN(right, AccTile::GetSizeInBytes() + LeftTile::GetSizeInBytes());
    fillTileLogical(src, 16);

    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
        TPUSH<Pipe, AccTile, TileSplitAxis::TILE_LEFT_RIGHT_ODD>(producer, src);
    }
    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 2);
        TPOP<Pipe, LeftTile, TileSplitAxis::TILE_LEFT_RIGHT_ODD>(consumer0, left);
    }
    {
        cpu_sim::ScopedExecutionContext ctx(0, 1, 2);
        TPOP<Pipe, RightTile, TileSplitAxis::TILE_LEFT_RIGHT_ODD>(consumer1, right);
    }

    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 8; ++c) {
            EXPECT_FLOAT_EQ(left.GetElement(r, c), static_cast<float>(r * 16 + c + 1));
        }
        for (int c = 0; c < 7; ++c) {
            EXPECT_FLOAT_EQ(right.GetElement(r, c), static_cast<float>(r * 16 + c + 8 + 1));
        }
    }

    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 2);
        TFREE<Pipe, TileSplitAxis::TILE_LEFT_RIGHT_ODD>(consumer0);
    }
    {
        cpu_sim::ScopedExecutionContext ctx(0, 1, 2);
        TFREE<Pipe, TileSplitAxis::TILE_LEFT_RIGHT_ODD>(consumer1);
    }
}

// V2C: two vector lanes each write their odd-split half into one combined L1 slot.
TEST_F(TPushPopTest, v2c_updown_odd_split_layout)
{
    using TopTile = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, 8, 16>;
    using BottomTile = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, 7, 16>;
    using MatTile = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 15, 16>;
    using Pipe = TPipe<22, Direction::DIR_V2C, sizeof(float) * MatTile::Numel, 2>;

    std::vector<float> fifoStorage(MatTile::Numel * Pipe::RingFiFo::SLOT_NUM, 0.0f);
    Pipe::reset_for_cpu_sim();
    Pipe producer0(fifoStorage.data(), 0x0, 0x10000);
    Pipe producer1(fifoStorage.data(), 0x0, 0x10000);
    Pipe consumer(fifoStorage.data(), 0x0, 0x10000);
    producer0.prod.setAllocateStatus(false);
    producer0.prod.setRecordStatus(false);
    producer1.prod.setAllocateStatus(false);
    producer1.prod.setRecordStatus(false);
    consumer.cons.setWaitStatus(false);
    consumer.cons.setFreeStatus(false);

    TopTile top;
    BottomTile bottom;
    MatTile dst;
    TASSIGN(top, 0);
    TASSIGN(bottom, TopTile::GetSizeInBytes());
    TASSIGN(dst, MatTile::GetSizeInBytes());
    fillTileLogical(top, 16);
    for (int r = 0; r < bottom.GetValidRow(); ++r) {
        for (int c = 0; c < bottom.GetValidCol(); ++c) {
            bottom.SetElement(r, c, static_cast<float>((r + 8) * 16 + c + 1));
        }
    }
    std::fill(dst.data(), dst.data() + dst.Numel, 0.0f);

    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 2);
        TPUSH<Pipe, TopTile, TileSplitAxis::TILE_UP_DOWN_ODD>(producer0, top);
    }
    {
        cpu_sim::ScopedExecutionContext ctx(0, 1, 2);
        TPUSH<Pipe, BottomTile, TileSplitAxis::TILE_UP_DOWN_ODD>(producer1, bottom);
    }
    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
        TPOP<Pipe, MatTile, TileSplitAxis::TILE_UP_DOWN_ODD>(consumer, dst);
    }

    for (int r = 0; r < 15; ++r) {
        for (int c = 0; c < 16; ++c) {
            ASSERT_FLOAT_EQ(dst.GetElement(r, c), static_cast<float>(r * 16 + c + 1))
                << "updown-odd v2c at (" << r << "," << c << ")";
        }
    }
}

TEST_F(TPushPopTest, v2c_leftright_odd_split_layout)
{
    using LeftTile = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, 16, 8>;
    using RightTile = Tile<TileType::Vec, float, 16, 8, BLayout::RowMajor, 16, 7>;
    using MatTile = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 15>;
    using Pipe = TPipe<23, Direction::DIR_V2C, sizeof(float) * MatTile::Numel, 2>;

    std::vector<float> fifoStorage(MatTile::Numel * Pipe::RingFiFo::SLOT_NUM, 0.0f);
    Pipe::reset_for_cpu_sim();
    Pipe producer0(fifoStorage.data(), 0x0, 0x10000);
    Pipe producer1(fifoStorage.data(), 0x0, 0x10000);
    Pipe consumer(fifoStorage.data(), 0x0, 0x10000);
    producer0.prod.setAllocateStatus(false);
    producer0.prod.setRecordStatus(false);
    producer1.prod.setAllocateStatus(false);
    producer1.prod.setRecordStatus(false);
    consumer.cons.setWaitStatus(false);
    consumer.cons.setFreeStatus(false);

    LeftTile left;
    RightTile right;
    MatTile dst;
    TASSIGN(left, 0);
    TASSIGN(right, LeftTile::GetSizeInBytes());
    TASSIGN(dst, MatTile::GetSizeInBytes());
    fillTileLogical(left, 16);
    for (int r = 0; r < right.GetValidRow(); ++r) {
        for (int c = 0; c < right.GetValidCol(); ++c) {
            right.SetElement(r, c, static_cast<float>(r * 16 + c + 8 + 1));
        }
    }
    std::fill(dst.data(), dst.data() + dst.Numel, 0.0f);

    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 2);
        TPUSH<Pipe, LeftTile, TileSplitAxis::TILE_LEFT_RIGHT_ODD>(producer0, left);
    }
    {
        cpu_sim::ScopedExecutionContext ctx(0, 1, 2);
        TPUSH<Pipe, RightTile, TileSplitAxis::TILE_LEFT_RIGHT_ODD>(producer1, right);
    }
    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
        TPOP<Pipe, MatTile, TileSplitAxis::TILE_LEFT_RIGHT_ODD>(consumer, dst);
    }

    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 15; ++c) {
            ASSERT_FLOAT_EQ(dst.GetElement(r, c), static_cast<float>(r * 16 + c + 1))
                << "leftright-odd v2c at (" << r << "," << c << ")";
        }
    }
}

class TPushPopPackedMxTest : public testing::Test {
protected:
    NPUArch previousArch;

    void SetUp() override
    {
        previousArch = NPUMemoryModel::Instance().GetArch();
        NPU_MEMORY_INIT(NPUArch::A5);
        NPU_MEMORY_CLEAR();
    }

    void TearDown() override
    {
        cpu_sim::reset_execution_context();
        NPU_MEMORY_CLEAR();
        NPU_MEMORY_INIT(previousArch);
    }
};

namespace {
template <int Rows, int Cols, bool Transpose, uint8_t DirectionType>
void checkPackedMxTransfer()
{
    using Vec = Tile<TileType::Vec, float8_e8m0_t, Rows, Cols, BLayout::RowMajor, Rows, Cols, SLayout::RowMajor, 32>;
    using Mat = Tile<
        TileType::Mat, float8_e8m0_t, Transpose ? Cols : Rows, Transpose ? Rows : Cols,
        Transpose ? BLayout::ColMajor : BLayout::RowMajor, Transpose ? Cols : Rows, Transpose ? Rows : Cols,
        Transpose ? SLayout::ColMajor : SLayout::RowMajor, 32>;
    using Pipe = TPipe<14, DirectionType, 4096, 2, 2, true>;
    Pipe::reset_for_cpu_sim();
    Pipe pipe(nullptr, 0, 0);
    Vec source;
    Mat result;
    TASSIGN(source, 0);
    TASSIGN(result, 0);
    cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
    for (int iter = 0; iter < 5; ++iter) {
        for (int r = 0; r < Rows; ++r) {
            for (int c = 0; c < Cols; ++c) {
                source.SetElement(r, c, float8_e8m0_t::FromRaw(static_cast<uint8_t>(r * Cols + c + iter)));
            }
        }
        TPUSH<Pipe, Vec, TileSplitAxis::TILE_NO_SPLIT>(pipe, source);
        TPOP<Pipe, Mat, TileSplitAxis::TILE_NO_SPLIT>(pipe, result);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
        for (int r = 0; r < Rows; ++r) {
            for (int c = 0; c < Cols; ++c) {
                EXPECT_EQ(
                    result.GetElement(Transpose ? c : r, Transpose ? r : c).RawData(),
                    source.GetElement(r, c).RawData());
            }
        }
        for (int i = 0; i < Rows * Cols; ++i) {
            EXPECT_EQ(result.data()[i].RawData(), source.data()[i].RawData());
        }
    }
}
} // namespace

TEST_F(TPushPopPackedMxTest, packed_mx_same_layout_and_transposed_layout_wraparound)
{
    checkPackedMxTransfer<16, 4, false, Direction::DIR_V2C>();
    checkPackedMxTransfer<128, 4, true, Direction::DIR_V2C>();
    checkPackedMxTransfer<256, 4, true, Direction::DIR_BOTH>();
    checkPackedMxTransfer<64, 8, false, Direction::DIR_BOTH>();
}

TEST_F(TPushPopPackedMxTest, packed_mx_entry_metadata_does_not_leak_to_reused_slot)
{
    using Packed = Tile<TileType::Vec, float8_e8m0_t, 16, 4, BLayout::RowMajor, 16, 4, SLayout::RowMajor, 32>;
    using Plain = Tile<TileType::Vec, float8_e8m0_t, 4, 32, BLayout::RowMajor, 4, 16>;
    using Mat = Tile<TileType::Mat, float8_e8m0_t, 4, 16, BLayout::ColMajor, 4, 16, SLayout::ColMajor, 32>;
    using Pipe = TPipe<15, Direction::DIR_V2C, 256, 1, 2, true>;
    Pipe::reset_for_cpu_sim();
    Pipe pipe(nullptr, 0, 0);
    Packed packed;
    Plain plain;
    Mat result;
    TASSIGN(packed, 0);
    TASSIGN(plain, 256);
    TASSIGN(result, 0);
    cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
    for (int iter = 0; iter < 3; ++iter) {
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 4; ++c) {
                packed.SetElement(r, c, float8_e8m0_t(1 << ((r + c + iter) % 6)));
                plain.SetElement(c, r, float8_e8m0_t(1 << ((2 * r + c + iter) % 6)));
            }
        }
        // Two independently typed entries share a slot and use nonzero offsets.
        pipe.prod.setRecordStatus(false);
        pipe.prod.setEntryOffset(32);
        TPUSH<Pipe, Packed, TileSplitAxis::TILE_NO_SPLIT>(pipe, packed);
        pipe.prod.setAllocateStatus(false);
        pipe.prod.setRecordStatus(true);
        pipe.prod.setEntryOffset(128);
        TPUSH<Pipe, Plain, TileSplitAxis::TILE_NO_SPLIT>(pipe, plain);
        pipe.prod.setAllocateStatus(true);
        pipe.cons.setentryOffset(32);
        TPOP<Pipe, Mat, TileSplitAxis::TILE_NO_SPLIT>(pipe, result);
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 16; ++c) {
                EXPECT_EQ(result.GetElement(r, c).RawData(), packed.GetElement(c, r).RawData());
            }
        }
        pipe.cons.setWaitStatus(false);
        pipe.cons.setentryOffset(128);
        TPOP<Pipe, Mat, TileSplitAxis::TILE_NO_SPLIT>(pipe, result);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
        pipe.cons.setWaitStatus(true);
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 16; ++c) {
                EXPECT_EQ(result.GetElement(r, c).RawData(), plain.GetElement(r, c).RawData());
            }
        }
        // Reuse the formerly packed entry with an ordinary logical transfer.
        pipe.prod.setEntryOffset(32);
        TPUSH<Pipe, Plain, TileSplitAxis::TILE_NO_SPLIT>(pipe, plain);
        pipe.cons.setentryOffset(32);
        TPOP<Pipe, Mat, TileSplitAxis::TILE_NO_SPLIT>(pipe, result);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 16; ++c) {
                EXPECT_EQ(result.GetElement(r, c).RawData(), plain.GetElement(r, c).RawData());
            }
        }
    }
}

TEST_F(TPushPopPackedMxTest, packed_mx_gm_transport_keeps_logical_row_major_payload)
{
    using Vec = Tile<TileType::Vec, float8_e8m0_t, 16, 4, BLayout::RowMajor, 16, 4, SLayout::RowMajor, 32>;
    using Mat = Tile<TileType::Mat, float8_e8m0_t, 4, 16, BLayout::ColMajor, 4, 16, SLayout::ColMajor, 32>;
    auto check = []<uint8_t DirType>(NPUArch arch) {
        NPU_MEMORY_INIT(arch);
        using Pipe = TPipe<16, DirType, 256, 2, 2, true>;
        Pipe::reset_for_cpu_sim();
        Pipe pipe(nullptr, 0, 0);
        Vec source;
        Mat result;
        TASSIGN(source, 0);
        TASSIGN(result, 0);
        cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 4; ++c) {
                source.SetElement(r, c, float8_e8m0_t(1 << ((r + c) % 6)));
            }
        }
        TPUSH<Pipe, Vec, TileSplitAxis::TILE_NO_SPLIT>(pipe, source);
        TPOP<Pipe, Mat, TileSplitAxis::TILE_NO_SPLIT>(pipe, result);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 16; ++c) {
                const int index = r * 16 + c;
                EXPECT_EQ(result.GetElement(r, c).RawData(), source.GetElement(index / 4, index % 4).RawData());
            }
        }
    };
    check.template operator()<Direction::DIR_V2C_GM>(NPUArch::A5);
    check.template operator()<Direction::DIR_V2C>(NPUArch::A2A3);
}

TEST_F(TPushPopPackedMxTest, packed_mx_many_entries_fill_slot_at_unaligned_offsets)
{
    using Vec = Tile<TileType::Vec, float8_e8m0_t, 16, 2, BLayout::RowMajor, 16, 2, SLayout::RowMajor, 32>;
    using Mat = Tile<TileType::Mat, float8_e8m0_t, 2, 16, BLayout::ColMajor, 2, 16, SLayout::ColMajor, 32>;
    using Pipe = TPipe<17, Direction::DIR_V2C, 257, 1, 2, true>;
    Pipe::reset_for_cpu_sim();
    Pipe pipe(nullptr, 0, 0);
    Vec source;
    Mat result;
    TASSIGN(source, 0);
    TASSIGN(result, 0);
    for (int iteration = 0; iteration < 3; ++iteration) {
        for (int entry = 0; entry < 8; ++entry) {
            for (int i = 0; i < 32; ++i) {
                source.data()[i] = float8_e8m0_t::FromRaw(static_cast<uint8_t>(entry * 32 + i + iteration));
            }
            pipe.prod.setEntryOffset(1 + entry * 32);
            pipe.prod.setAllocateStatus(entry == 0);
            pipe.prod.setRecordStatus(entry == 7);
            TPUSH<Pipe, Vec, TileSplitAxis::TILE_NO_SPLIT>(pipe, source);
        }
        for (int entry = 0; entry < 8; ++entry) {
            pipe.cons.setentryOffset(1 + entry * 32);
            pipe.cons.setWaitStatus(entry == 0);
            TPOP<Pipe, Mat, TileSplitAxis::TILE_NO_SPLIT>(pipe, result);
            for (int i = 0; i < 32; ++i) {
                EXPECT_EQ(result.data()[i].RawData(), static_cast<uint8_t>(entry * 32 + i + iteration));
            }
        }
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
    }
}

TEST_F(TPushPopPackedMxTest, packed_mx_overlapping_plain_entry_invalidates_old_shape)
{
    using Packed = Tile<TileType::Vec, float8_e8m0_t, 16, 4, BLayout::RowMajor, 16, 4, SLayout::RowMajor, 32>;
    using Plain = Tile<TileType::Vec, float8_e8m0_t, 4, 32, BLayout::RowMajor, 4, 32>;
    using Mat = Tile<TileType::Mat, float8_e8m0_t, 4, 16, BLayout::ColMajor, 4, 16, SLayout::ColMajor, 32>;
    using Pipe = TPipe<18, Direction::DIR_V2C, 256, 1, 2, true>;
    Pipe::reset_for_cpu_sim();
    Pipe pipe(nullptr, 0, 0);
    Packed packed;
    Plain plain;
    Mat result;
    TASSIGN(packed, 0);
    TASSIGN(plain, 256);
    TASSIGN(result, 0);
    for (int i = 0; i < 64; ++i) {
        packed.data()[i] = float8_e8m0_t::FromRaw(static_cast<uint8_t>(200 + i));
    }
    for (int i = 0; i < 128; ++i) {
        plain.data()[i] = float8_e8m0_t::FromRaw(static_cast<uint8_t>(i));
    }
    pipe.prod.setRecordStatus(false);
    pipe.prod.setEntryOffset(32);
    TPUSH<Pipe, Packed, TileSplitAxis::TILE_NO_SPLIT>(pipe, packed);
    pipe.prod.setAllocateStatus(false);
    pipe.prod.setEntryOffset(160);
    TPUSH<Pipe, Packed, TileSplitAxis::TILE_NO_SPLIT>(pipe, packed);
    pipe.prod.setRecordStatus(true);
    pipe.prod.setEntryOffset(0);
    TPUSH<Pipe, Plain, TileSplitAxis::TILE_NO_SPLIT>(pipe, plain);
    pipe.cons.setentryOffset(32);
    TPOP<Pipe, Mat, TileSplitAxis::TILE_NO_SPLIT>(pipe, result);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 16; ++c) {
            EXPECT_EQ(result.GetElement(r, c).RawData(), 32 + r * 16 + c);
        }
    }
    // An entry outside the overwritten byte range must retain its packed view.
    pipe.cons.setWaitStatus(false);
    pipe.cons.setentryOffset(160);
    TPOP<Pipe, Mat, TileSplitAxis::TILE_NO_SPLIT>(pipe, result);
    for (int i = 0; i < 64; ++i) {
        EXPECT_EQ(result.data()[i].RawData(), packed.data()[i].RawData());
    }
    TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
}

TEST_F(TPushPopPackedMxTest, packed_mx_split_overwrite_invalidates_old_shape)
{
    using Packed = Tile<TileType::Vec, float8_e8m0_t, 16, 16, BLayout::RowMajor, 16, 16, SLayout::RowMajor, 32>;
    using Plain = Tile<TileType::Vec, float8_e8m0_t, 4, 32, BLayout::RowMajor, 4, 32>;
    using Mat = Tile<TileType::Mat, float8_e8m0_t, 16, 16, BLayout::ColMajor, 16, 16, SLayout::ColMajor, 32>;
    using Pipe = TPipe<19, Direction::DIR_V2C, 256, 1>;
    auto check = []<TileSplitAxis Split>() {
        Pipe::reset_for_cpu_sim();
        Pipe pipe(nullptr, 0, 0);
        Packed packed;
        Plain plain;
        Mat result;
        TASSIGN(packed, 0);
        TASSIGN(plain, 256);
        TASSIGN(result, 0);
        std::fill_n(packed.data(), 256, float8_e8m0_t::FromRaw(0));
        pipe.prod.setRecordStatus(false);
        TPUSH<Pipe, Packed, TileSplitAxis::TILE_NO_SPLIT>(pipe, packed);
        pipe.prod.setAllocateStatus(false);
        pipe.prod.setRecordStatus(true);
        for (int lane = 0; lane < 2; ++lane) {
            cpu_sim::ScopedExecutionContext ctx(0, lane, 2);
            pipe.prod.setTileId(0, lane);
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 32; ++c) {
                    const int index =
                        Split == TileSplitAxis::TILE_UP_DOWN ? (lane * 4 + r) * 32 + c : r * 64 + lane * 32 + c;
                    plain.SetElement(r, c, float8_e8m0_t::FromRaw(static_cast<uint8_t>(index)));
                }
            }
            TPUSH<Pipe, Plain, Split>(pipe, plain);
        }
        TPOP<Pipe, Mat, Split>(pipe, result);
        TFREE<Pipe, Split>(pipe);
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < 16; ++c) {
                EXPECT_EQ(result.GetElement(r, c).RawData(), r * 16 + c);
            }
        }
    };
    check.template operator()<TileSplitAxis::TILE_UP_DOWN>();
    check.template operator()<TileSplitAxis::TILE_LEFT_RIGHT>();
}

TEST_F(TPushPopPackedMxTest, fp4_left_right_split_odd_valid_cols_preserves_nibbles)
{
    auto check = []<typename T>() {
        constexpr int rows = 32;
        constexpr int cols = 128;
        constexpr int validCols = 65;
        constexpr auto split = TileSplitAxis::TILE_LEFT_RIGHT;
        using Vec = Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, rows, validCols>;
        using Mat = Tile<TileType::Mat, T, rows, 2 * cols, BLayout::RowMajor, rows, 2 * cols>;
        using Pipe = TPipe<24, Direction::DIR_V2C, rows * cols + 2, 2>;
        Pipe::reset_for_cpu_sim();
        Pipe consumer(nullptr, 0, 0);
        consumer.cons.setentryOffset(1);
        const T guard = T::FromRaw(0xA5);
        std::array<std::vector<T>, 2> sources{
            std::vector<T>(Vec::GetSizeInUnits() + 2, guard), std::vector<T>(Vec::GetSizeInUnits() + 2, guard)};
        std::array<Vec, 2> sourceTiles;
        for (int lane = 0; lane < 2; ++lane) {
            sourceTiles[lane].data() = sources[lane].data() + 1;
        }
        std::vector<T> result(Mat::GetSizeInUnits() + 2, guard);
        Mat dst;
        dst.data() = result.data() + 1;
        auto& state = Pipe::GetSharedState();
        for (auto& slot : state.local_slot_storage) {
            slot.fill(0xA5);
        }
        for (int iteration = 0; iteration < 32; ++iteration) {
            auto rawValue = [iteration](int r, int c) { return (iteration * 7 + r * 3 + c) & 0xF; };
            for (int lane = 0; lane < 2; ++lane) {
                for (int r = 0; r < rows; ++r) {
                    for (int c = 0; c < validCols; ++c) {
                        sourceTiles[lane].SetElement(r, c, T::FromRaw(rawValue(r, lane * cols + c)));
                    }
                }
            }
            std::barrier ready(2);
            std::array<std::exception_ptr, 2> errors{};
            auto push = [&](int lane) {
                cpu_sim::ScopedExecutionContext ctx(0, lane, 2);
                Pipe producer(nullptr, 0, 0);
                producer.prod.setEntryOffset(1);
                ready.arrive_and_wait();
                try {
                    TPUSH<Pipe, Vec, split>(producer, sourceTiles[lane]);
                } catch (...) {
                    errors[lane] = std::current_exception();
                }
            };
            std::thread left(push, 0);
            std::thread right(push, 1);
            left.join();
            right.join();
            for (const auto& error : errors) {
                ASSERT_NO_THROW(if (error) { std::rethrow_exception(error); });
            }
            cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
            TPOP<Pipe, Mat, split>(consumer, dst);
            TFREE<Pipe, split>(consumer);
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < 2 * cols; ++c) {
                    // The physical width must be aligned. Odd valid columns
                    // leave the high nibble of each lane's final byte untouched.
                    const int expected = c % cols < validCols ? rawValue(r, c) : (c % 2 == 0 ? 0x5 : 0xA);
                    ASSERT_EQ(dst.GetElement(r, c).RawData(), expected)
                        << "iteration=" << iteration << ", row=" << r << ", col=" << c;
                }
            }
            for (const auto& source : sources) {
                EXPECT_EQ(source.front().RawData(), guard.RawData());
                EXPECT_EQ(source.back().RawData(), guard.RawData());
            }
            EXPECT_EQ(result.front().RawData(), guard.RawData());
            EXPECT_EQ(result.back().RawData(), guard.RawData());
            const auto& slot = state.local_slot_storage[iteration % Pipe::RingFiFo::SLOT_NUM];
            EXPECT_EQ(slot.front(), 0xA5);
            EXPECT_EQ(slot.back(), 0xA5);
        }
    };
    check.template operator()<float4_e2m1x2_t>();
    check.template operator()<float4_e1m2x2_t>();
}

TEST_F(TPushPopPackedMxTest, fp4_split_overwrite_invalidates_only_overlapping_mx_entries)
{
    using Packed = Tile<TileType::Vec, float8_e8m0_t, 16, 2, BLayout::RowMajor, 16, 2, SLayout::RowMajor, 32>;
    using Vec = Tile<TileType::Vec, float4_e2m1x2_t, 2, 128, BLayout::RowMajor, 2, 65>;
    using Pipe = TPipe<25, Direction::DIR_V2C, 320, 1>;
    // With entryOffset=1, lane 0 writes [1,34) and [129,162); lane 1 writes
    // [65,98) and [193,226). Each MX entry occupies 32 bytes. The last
    // nibble of each window must invalidate its overlapping entry, while
    // entries in the other lane, padding, and outside the payload survive.
    const std::array<int, 9> offsets{1, 33, 65, 97, 129, 161, 193, 225, 257};
    const std::array<std::array<bool, 9>, 2> preserved{
        {{false, false, true, true, false, false, true, true, true},
         {true, true, false, false, true, true, false, false, true}}};
    for (int lane = 0; lane < 2; ++lane) {
        Pipe::reset_for_cpu_sim();
        Pipe pipe(nullptr, 0, 0);
        Packed packed;
        Vec source;
        std::vector<float8_e8m0_t> packedData(Packed::GetSizeInUnits(), float8_e8m0_t::FromRaw(0x7F));
        std::vector<float4_e2m1x2_t> sourceData(Vec::GetSizeInUnits(), float4_e2m1x2_t::FromRaw(0x87));
        packed.data() = packedData.data();
        source.data() = sourceData.data();
        pipe.prod.setRecordStatus(false);
        for (std::size_t entry = 0; entry < offsets.size(); ++entry) {
            pipe.prod.setAllocateStatus(entry == 0);
            pipe.prod.setEntryOffset(offsets[entry]);
            TPUSH<Pipe, Packed, TileSplitAxis::TILE_NO_SPLIT>(pipe, packed);
        }
        auto& entries = Pipe::GetSharedState().packed_mx_entries[0];
        ASSERT_EQ(entries.count, offsets.size());
        pipe.prod.setAllocateStatus(false);
        pipe.prod.setEntryOffset(1);
        {
            cpu_sim::ScopedExecutionContext ctx(0, lane, 2);
            TPUSH<Pipe, Vec, TileSplitAxis::TILE_LEFT_RIGHT>(pipe, source);
        }
        for (std::size_t entry = 0; entry < offsets.size(); ++entry) {
            EXPECT_EQ(entries.find(offsets[entry]) != nullptr, preserved[lane][entry])
                << "lane=" << lane << ", offset=" << offsets[entry];
        }
    }
}

TEST_F(TPushPopPackedMxTest, packed_mx_partial_views_keep_logical_payload)
{
    using Vec = Tile<TileType::Vec, float8_e8m0_t, 16, 4, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 32>;
    using Mat = Tile<TileType::Mat, float8_e8m0_t, 4, 16, BLayout::ColMajor, -1, -1, SLayout::ColMajor, 32>;
    using Pipe = TPipe<21, Direction::DIR_V2C, 256, 1, 2, true>;
    for (bool partialProducer : {false, true}) {
        Pipe::reset_for_cpu_sim();
        Pipe pipe(nullptr, 0, 0);
        Vec source(partialProducer ? 12 : 16, 4);
        Mat result(3, 16);
        TASSIGN(source, 0);
        TASSIGN(result, 0);
        for (int r = 0; r < source.GetValidRow(); ++r) {
            for (int c = 0; c < 4; ++c) {
                source.SetElement(r, c, float8_e8m0_t::FromRaw(static_cast<uint8_t>(r * 4 + c)));
            }
        }
        TPUSH<Pipe, Vec, TileSplitAxis::TILE_NO_SPLIT>(pipe, source);
        TPOP<Pipe, Mat, TileSplitAxis::TILE_NO_SPLIT>(pipe, result);
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 16; ++c) {
                EXPECT_EQ(result.GetElement(r, c).RawData(), r * 16 + c);
            }
        }
    }
}

TEST_F(TPushPopPackedMxTest, packed_mx_invalid_offset_leaves_slot_unmodified)
{
    using Vec = Tile<TileType::Vec, float8_e8m0_t, 16, 2, BLayout::RowMajor, 16, 2, SLayout::RowMajor, 32>;
    using Pipe = TPipe<22, Direction::DIR_V2C, 32, 1, 2, true>;
    Pipe::reset_for_cpu_sim();
    Pipe pipe(nullptr, 0, 0);
    Vec source;
    TASSIGN(source, 0);
    for (int offset : {-1, 1, 32}) {
        pipe.prod.setEntryOffset(offset);
        auto& state = Pipe::GetSharedState(cpu_pipe::TransferDir::V2C);
        state.local_slot_storage[0].fill(0xa5);
        EXPECT_THROW((TPUSH<Pipe, Vec, TileSplitAxis::TILE_NO_SPLIT>(pipe, source)), std::out_of_range);
        for (auto byte : state.local_slot_storage[0]) {
            EXPECT_EQ(byte, 0xa5);
        }
        EXPECT_EQ(state.occupied, 0);
        Pipe::reset_for_cpu_sim();
    }
}
