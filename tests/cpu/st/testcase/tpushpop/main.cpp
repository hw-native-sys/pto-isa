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
#include <chrono>
#include <gtest/gtest.h>
#include <pto/common/fifo.hpp>
#include <thread>
#include <vector>
#include "test_common.h"

using namespace std;
using namespace pto;
using namespace PtoTestCommon;

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
        TASSIGN(topHalf, 8 * 16 * sizeof(float));
        TASSIGN(bottomHalf, 8 * 16 * sizeof(float) + 8 * 16 * sizeof(float));

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