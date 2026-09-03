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
#include <chrono>
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
