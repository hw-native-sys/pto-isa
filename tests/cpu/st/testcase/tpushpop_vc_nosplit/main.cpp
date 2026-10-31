/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
void fillVectorTile(auto& tile, int iter)
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
    Pipe pipe((__gm__ void*)nullptr, 0x0, kMatConsumerBase);
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
    Pipe pipe((__gm__ void*)nullptr, 0x0, kMatConsumerBase);
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
    Pipe pipe((__gm__ void*)nullptr, 0x0, kMatConsumerBase);

    {
        cpu_sim::ScopedExecutionContext inactiveVecCtx(0, 1, 2);
        VecTile inactiveVecTile;
        TASSIGN(inactiveVecTile, 0x0);
        fillVectorTile<T, rows, cols>(inactiveVecTile, 7);
        TPUSH<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, inactiveVecTile);
    }

    {
        auto& sharedState = Pipe::GetSharedState();
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
        auto& sharedState = Pipe::GetSharedState();
        std::lock_guard<std::mutex> lock(sharedState.mutex);
        EXPECT_EQ(sharedState.occupied, 0);
    }
}
// Exercise logical FP4 elements in packed storage, including signed zero (0x8).
// Both layouts are used by native kernels; compare bits instead of float values.
template <typename T, int rows, int slotBytes, bool fractal>
void runPackedFp4Wraparound()
{
    constexpr int cols = 128;
    constexpr int depth = 2;
    using VecTile = Tile<
        TileType::Vec, T, rows, cols, fractal ? BLayout::ColMajor : BLayout::RowMajor, rows, cols,
        fractal ? SLayout::RowMajor : SLayout::NoneBox>;
    using MatTile = Tile<TileType::Mat, T, rows, cols, BLayout::ColMajor, rows, cols, SLayout::RowMajor>;
    using Pipe = TPipe<kPipeFlagId + 3, Direction::DIR_V2C, slotBytes, depth, depth, true>;
    static_assert(VecTile::GetSizeInBytes() == rows * cols / 2);
    Pipe::reset_for_cpu_sim();
    Pipe pipe(nullptr, 0, kMatConsumerBase);
    const T guard = T::FromRaw(0xA5);
    std::vector<T> source(VecTile::GetSizeInUnits() + 2, guard);
    std::vector<T> result(MatTile::GetSizeInUnits() + 2, guard);
    VecTile vecTile;
    MatTile matTile;
    vecTile.data() = source.data() + 1;
    matTile.data() = result.data() + 1;
    cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
    auto rawValue = [](int iter, int r, int c) { return (iter * 7 + r * 3 + c) & 0xF; };
    auto push = [&](int iter) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                vecTile.SetElement(r, c, T::FromRaw(rawValue(iter, r, c)));
            }
        }
        TPUSH<Pipe, VecTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, vecTile);
    };
    push(0);
    push(1);
    for (int iter = 0; iter < 5; ++iter) {
        TPOP<Pipe, MatTile, TileSplitAxis::TILE_NO_SPLIT>(pipe, matTile);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                ASSERT_EQ(matTile.GetElement(r, c).RawData(), rawValue(iter, r, c))
                    << "iter=" << iter << ", row=" << r << ", col=" << c;
            }
        }
        TFREE<Pipe, TileSplitAxis::TILE_NO_SPLIT>(pipe);
        if (iter + depth < 5) {
            push(iter + depth);
        }
    }
    EXPECT_EQ(source.front().RawData(), guard.RawData());
    EXPECT_EQ(source.back().RawData(), guard.RawData());
    EXPECT_EQ(result.front().RawData(), guard.RawData());
    EXPECT_EQ(result.back().RawData(), guard.RawData());
}

template <typename T>
void checkPackedByteBounds()
{
    std::array<uint8_t, 4> storage{0xA5, 0xBC, 0xDE, 0xF1};
    // A nonzero entry offset with three logical values: preserve the unused high nibble.
    for (std::size_t i = 0; i < 3; ++i) {
        cpu_pipe::StoreByteStorageElement(storage, 0, 1, 3, i, T::FromRaw(i + 7), "test");
    }
    EXPECT_EQ(storage, (std::array<uint8_t, 4>{0xA5, 0x87, 0xD9, 0xF1}));
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(cpu_pipe::LoadByteStorageElement<T>(storage, 0, 1, 3, i, "test").RawData(), i + 7);
    }
    EXPECT_EQ(cpu_pipe::GetStorageByteCount<T>(3), 2u);
    EXPECT_THROW(cpu_pipe::StoreByteStorageElement(storage, 0, 1, 3, 4, T::FromRaw(0), "test"), std::out_of_range);
    EXPECT_THROW(cpu_pipe::LoadByteStorageElement<T>(storage, 0, 1, 3, 4, "test"), std::out_of_range);
    EXPECT_EQ(storage, (std::array<uint8_t, 4>{0xA5, 0x87, 0xD9, 0xF1}));
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

TEST_F(TPushPopVCNoSplitTest, fp4_e2m1_exact_capacity_wraparound)
{
    runPackedFp4Wraparound<float4_e2m1x2_t, 16, 1024, true>();
}

TEST_F(TPushPopVCNoSplitTest, fp4_e1m2_exact_capacity_wraparound)
{
    runPackedFp4Wraparound<float4_e1m2x2_t, 16, 1024, true>();
}

TEST_F(TPushPopVCNoSplitTest, fp4_e2m1_index_score_query_shape)
{
    runPackedFp4Wraparound<float4_e2m1x2_t, 192, 16384, true>();
}

TEST_F(TPushPopVCNoSplitTest, fp4_row_major_to_fractal) { runPackedFp4Wraparound<float4_e2m1x2_t, 16, 1024, false>(); }

TEST_F(TPushPopVCNoSplitTest, fp4_preserves_nibbles_and_checks_byte_bounds)
{
    checkPackedByteBounds<float4_e2m1x2_t>();
    checkPackedByteBounds<float4_e1m2x2_t>();
}
