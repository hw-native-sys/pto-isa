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
void *g_pipe_hook_storage = nullptr;
size_t g_pipe_hook_size = 0;
uint64_t g_pipe_hook_last_key = 0;

uint32_t MockSubblockIdHook()
{
    return g_injected_subblock_id.load(std::memory_order_relaxed);
}

void *MockPipeSharedStateHook(uint64_t pipeKey, size_t size)
{
    g_pipe_hook_last_key = pipeKey;
    g_pipe_hook_size = size;
    g_pipe_hook_call_count.fetch_add(1, std::memory_order_relaxed);
    return g_pipe_hook_storage;
}

struct ScopedCpuStubHooks {
    ScopedCpuStubHooks(void *subblockHook, void *pipeSharedStateHook)
    {
        pto::cpu_sim::register_hooks(subblockHook, pipeSharedStateHook);
    }

    ~ScopedCpuStubHooks()
    {
        pto::cpu_sim::register_hooks(nullptr, nullptr);
        cpu_sim::reset_execution_context();
    }
};

template <TileSplitAxis SplitAxis>
using DirBothVecTile =
    Tile<TileType::Vec, float, (SplitAxis == TileSplitAxis::TILE_UP_DOWN) ? 8 : 16,
         (SplitAxis == TileSplitAxis::TILE_LEFT_RIGHT) ? 8 : 16, BLayout::RowMajor,
         (SplitAxis == TileSplitAxis::TILE_UP_DOWN) ? 8 : 16, (SplitAxis == TileSplitAxis::TILE_LEFT_RIGHT) ? 8 : 16>;

template <typename TileData>
void fillTileSequence(TileData &tile, float start)
{
    for (int i = 0; i < tile.Numel; ++i) {
        tile.data()[i] = start + static_cast<float>(i);
    }
}

template <TileSplitAxis SplitAxis>
void expectVecMatchesAccSplit(const auto &vec, const auto &acc, uint32_t laneId)
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
            EXPECT_FLOAT_EQ(vec.data()[GetTileElementOffset<std::remove_cvref_t<decltype(vec)>>(r, c)],
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
    Pipe vecProducer0((__gm__ void *)nullptr, 0x0, 0x10000);
    Pipe vecProducer1((__gm__ void *)nullptr, 0x0, 0x10000);
    Pipe cubePipe((__gm__ void *)nullptr, 0x0, 0x10000);
    Pipe vecConsumer0((__gm__ void *)nullptr, 0x0, 0x10000);
    Pipe vecConsumer1((__gm__ void *)nullptr, 0x0, 0x10000);

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
    TASSIGN(dst1, 2 * VecTile::Numel * sizeof(float) + MatTile::Numel * sizeof(float) + AccTile::Numel * sizeof(float) +
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
void fillTile(auto &tile, int iter)
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

template <typename T, int rows, int cols, TileType srcLoc>
void testPushPopSingleThread()
{
    constexpr int FiFoDepth = 8;
    constexpr int LocalDepth = 2;
    using PPTile = Tile<srcLoc, T, rows, cols>;
    using PPipe = TPipe<0, Direction::DIR_C2V, sizeof(T) * PPTile::Numel, FiFoDepth, LocalDepth>;
    std::vector<T> fifoStorage(PPTile::Numel * FiFoDepth, static_cast<T>(0));
    PPipe::reset_for_cpu_sim();
    PPipe pipe(fifoStorage.data(), 0x0, 0x0);
    PPTile src;
    PPTile dst;

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

template <typename T, int rows, int cols, TileType srcLoc>
void testPushPopMultiCore()
{
    constexpr int FiFoDepth = 4;
    constexpr int LocalDepth = 0;
    using PPTile = Tile<srcLoc, T, rows, cols>;
    using PPipe = TPipe<1, Direction::DIR_C2V, sizeof(T) * PPTile::Numel, FiFoDepth, LocalDepth>;

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
            PPTile dst;
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
    void SetUp() override
    {}
    void TearDown() override
    {}
};

#define TPUSHPOP_TEST(T, rows, cols, srcLoc)                        \
    TEST_F(TPushPopTest, T##_##rows##_##cols##_##srcLoc)            \
    {                                                               \
        testPushPopSingleThread<T, rows, cols, TileType::srcLoc>(); \
    }

TPUSHPOP_TEST(float, 64, 128, Vec)
TPUSHPOP_TEST(float, 128, 128, Vec)
TPUSHPOP_TEST(float, 64, 128, Mat)
TPUSHPOP_TEST(float, 128, 128, Mat)
TPUSHPOP_TEST(uint32_t, 64, 128, Vec)
TPUSHPOP_TEST(uint32_t, 128, 128, Vec)
TPUSHPOP_TEST(uint32_t, 64, 128, Mat)
TPUSHPOP_TEST(uint32_t, 128, 128, Mat)

TEST_F(TPushPopTest, multicore_float_64_128_Vec)
{
    testPushPopMultiCore<float, 64, 128, TileType::Vec>();
}

TEST_F(TPushPopTest, a5_style_c2v_local_split_push_pop)
{
    using AccTile = TileAcc<float, 16, 16>;
    using VecTile = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, 8, 16>;
    using Pipe = TPipe<2, Direction::DIR_C2V, sizeof(float) * VecTile::Numel, 2>;

    Pipe::reset_for_cpu_sim();
    Pipe pipe((__gm__ void *)nullptr, 0x0, 0x0);

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
    Pipe producer((__gm__ void *)nullptr, 0x0, 0x0);
    Pipe consumer0((__gm__ void *)nullptr, 0x0, 0x0);
    Pipe consumer1((__gm__ void *)nullptr, 0x0, 0x0);

    auto run_iteration = [&](int iter) {
        AccTile src;
        VecTile topHalf;
        VecTile bottomHalf;
        TASSIGN(src, 0);
        TASSIGN(topHalf, AccTile::Numel * sizeof(typename AccTile::DType));
        TASSIGN(bottomHalf,
                AccTile::Numel * sizeof(typename AccTile::DType) + VecTile::Numel * sizeof(typename VecTile::DType));
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
                EXPECT_EQ(topHalf.data()[GetTileElementOffset<VecTile>(r, c)],
                          src.data()[GetTileElementOffset<AccTile>(r, c)]);
                EXPECT_EQ(bottomHalf.data()[GetTileElementOffset<VecTile>(r, c)],
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

TEST_F(TPushPopTest, a5_style_v2c_local_split_push_pop)
{
    using VecTile = Tile<TileType::Vec, float, 8, 16, BLayout::RowMajor, 8, 16>;
    using MatTile = Tile<TileType::Mat, float, 16, 16, BLayout::RowMajor, 16, 16>;
    using Pipe = TPipe<3, Direction::DIR_V2C, sizeof(float) * MatTile::Numel, 2>;

    Pipe::reset_for_cpu_sim();
    Pipe pipe((__gm__ void *)nullptr, 0x0, 0x10000);

    VecTile src;
    MatTile dst;
    TASSIGN(src, 0);
    TASSIGN(dst, VecTile::Rows * VecTile::Cols * sizeof(VecTile::DType));

    fillTile<float, 8, 16, TileType::Vec>(src, 0);
    std::fill(dst.data(), dst.data() + dst.Numel, 0.0f);

    TPUSH<Pipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(pipe, src);
    TPOP<Pipe, MatTile, TileSplitAxis::TILE_UP_DOWN>(pipe, dst);
    TFREE<Pipe, TileSplitAxis::TILE_UP_DOWN>(pipe);

    for (int c = 0; c < dst.GetValidCol(); ++c) {
        EXPECT_EQ(dst.data()[GetTileElementOffset<MatTile>(0, c)], src.data()[GetTileElementOffset<VecTile>(0, c)]);
        for (int r = 0; r < src.GetValidRow(); ++r) {
            EXPECT_EQ(dst.data()[GetTileElementOffset<MatTile>(r, c)], src.data()[GetTileElementOffset<VecTile>(r, c)]);
        }
        for (int r = src.GetValidRow(); r < dst.GetValidRow(); ++r) {
            EXPECT_EQ(dst.data()[GetTileElementOffset<MatTile>(r, c)], 0.0f);
        }
    }
}

TEST_F(TPushPopTest, cpu_stub_prefers_injected_hooks_for_subblock_and_pipe_state)
{
    HookTestPipe::SharedStateStorage storage{};
    g_injected_subblock_id.store(7, std::memory_order_relaxed);
    g_pipe_hook_call_count.store(0, std::memory_order_relaxed);
    g_pipe_hook_storage = &storage;
    g_pipe_hook_size = 0;
    g_pipe_hook_last_key = 0;

    ScopedCpuStubHooks hooks(reinterpret_cast<void *>(MockSubblockIdHook),
                             reinterpret_cast<void *>(MockPipeSharedStateHook));
    cpu_sim::set_execution_context(0, 1, 2);

    EXPECT_EQ(get_subblockid(), 7u);

    auto &state = HookTestPipe::GetSharedState();
    state.next_producer_slot = 3;
    auto &stateAgain = HookTestPipe::GetSharedState();

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

    ScopedCpuStubHooks hooks(nullptr, reinterpret_cast<void *>(MockPipeSharedStateHook));
    HookedV2CPipe::reset_for_cpu_sim();

    HookedV2CPipe producer0((__gm__ void *)nullptr, 0x0, 0x10000);
    HookedV2CPipe producer1((__gm__ void *)nullptr, 0x0, 0x10000);
    HookedV2CPipe consumer((__gm__ void *)nullptr, 0x0, 0x10000);
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
        cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
        EXPECT_EQ(cpu_pipe::GetActiveSplitCount<TileSplitAxis::TILE_UP_DOWN>(), 2u);
        EXPECT_EQ(cpu_pipe::GetActiveSplitLaneMask<TileSplitAxis::TILE_UP_DOWN>(), 0x3u);
        TPUSH<HookedV2CPipe, VecTile, TileSplitAxis::TILE_UP_DOWN>(producer0, topHalf);
    }

    auto &state = HookedV2CPipe::GetSharedState();
    EXPECT_EQ(state.occupied, 0);
    EXPECT_EQ(state.next_producer_slot, 0);
    EXPECT_EQ(state.producers_done[0], 0x1u);
    EXPECT_EQ(state.producers_allocated[0], 0x1u);

    {
        cpu_sim::ScopedExecutionContext ctx(0, 1, 1);
        EXPECT_EQ(cpu_pipe::GetActiveSplitCount<TileSplitAxis::TILE_UP_DOWN>(), 2u);
        EXPECT_EQ(cpu_pipe::GetActiveSplitLaneMask<TileSplitAxis::TILE_UP_DOWN>(), 0x3u);
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
            EXPECT_EQ(dst.data()[GetTileElementOffset<MatTile>(r, c)],
                      topHalf.data()[GetTileElementOffset<VecTile>(r, c)]);
            EXPECT_EQ(dst.data()[GetTileElementOffset<MatTile>(r + topHalf.GetValidRow(), c)],
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
