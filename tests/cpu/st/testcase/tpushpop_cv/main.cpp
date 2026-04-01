/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <chrono>
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>
#include <pto/common/fifo.hpp>
#include <thread>
#include <vector>
#include "test_common.h"

using namespace pto;
using namespace PtoTestCommon;

class TPushPopCVTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

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
void runCubeToVectorSingleTile()
{
    constexpr int FiFoDepth = 4;
    using CubeTile = Tile<TileType::Mat, T, rows, cols>;
    using VecTile = Tile<TileType::Vec, T, rows, cols>;
    using Pipe = TPipe<5, Direction::DIR_C2V, sizeof(T) * VecTile::Numel, FiFoDepth>;

    std::vector<T> fifoStorage(VecTile::Numel * FiFoDepth, static_cast<T>(0));
    Pipe::reset_for_cpu_sim();
    Pipe pipe(fifoStorage.data(), 0x0, 0x0);
    CubeTile cubeTile;
    VecTile vecTile;
    TASSIGN(cubeTile, 0);
    TASSIGN(vecTile, 0);

    fillCubeTile<T, rows, cols>(cubeTile, 0);
    for (int i = 0; i < vecTile.Numel; ++i) {
        vecTile.data()[i] = static_cast<T>(0);
    }

    TPUSH(cubeTile, pipe);
    TPOP(vecTile, pipe);
    TFREE(pipe);

    const auto expected = makeExpected<T, rows, cols>(0);
    EXPECT_TRUE(ResultCmp(expected, vecTile.data(), 0));
}

template <typename T, int rows, int cols>
void runCubeToVectorMultiCoreStream()
{
    constexpr int FiFoDepth = 4;
    using CubeTile = Tile<TileType::Mat, T, rows, cols>;
    using VecTile = Tile<TileType::Vec, T, rows, cols>;
    using Pipe = TPipe<6, Direction::DIR_C2V, sizeof(T) * VecTile::Numel, FiFoDepth>;

    constexpr int kIterations = 10;
    std::vector<T> fifoStorage(VecTile::Numel * FiFoDepth, static_cast<T>(0));
    std::vector<std::vector<T>> actual(kIterations);
    Pipe::reset_for_cpu_sim();
    Pipe pipe(fifoStorage.data(), 0x0, 0x0);

    std::thread cubeCore([&]() {
        for (int iter = 0; iter < kIterations; ++iter) {
            CubeTile cubeTile;
            TASSIGN(cubeTile, 0);
            fillCubeTile<T, rows, cols>(cubeTile, iter);
            TPUSH(cubeTile, pipe);
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    std::thread vectorCore([&]() {
        for (int iter = 0; iter < kIterations; ++iter) {
            VecTile vecTile;
            TASSIGN(vecTile, 0);
            for (int i = 0; i < vecTile.Numel; ++i) {
                vecTile.data()[i] = static_cast<T>(0);
            }
            TPOP(vecTile, pipe);
            TFREE(pipe);
            actual[iter].assign(vecTile.data(), vecTile.data() + vecTile.Numel);
        }
    });

    cubeCore.join();
    vectorCore.join();

    for (int iter = 0; iter < kIterations; ++iter) {
        const auto expected = makeExpected<T, rows, cols>(iter);
        EXPECT_TRUE(ResultCmp(expected, actual[iter], 0));
    }
}

TEST_F(TPushPopCVTest, cube_to_vector_single_tile_float_64x128)
{
    runCubeToVectorSingleTile<float, 64, 128>();
}

TEST_F(TPushPopCVTest, cube_to_vector_multicore_stream_float_64x128)
{
    runCubeToVectorMultiCoreStream<float, 64, 128>();
}
