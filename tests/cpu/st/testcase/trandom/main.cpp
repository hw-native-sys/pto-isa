/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <array>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace {
constexpr uint32_t kConst0 = 0xD2511F53u;
constexpr uint32_t kConst1 = 0xCD9E8D57u;
constexpr uint32_t kKeyAdd0 = 0x9E3779B9u;
constexpr uint32_t kKeyAdd1 = 0xBB67AE85u;

template <uint16_t Rounds>
std::array<uint32_t, 4> RunRounds(std::array<uint32_t, 4> counter, std::array<uint32_t, 2> key)
{
    for (uint16_t i = 0; i < Rounds; ++i) {
        const uint64_t mul0 = static_cast<uint64_t>(counter[0]) * kConst0;
        const uint64_t mul1 = static_cast<uint64_t>(counter[2]) * kConst1;
        const uint32_t hi0 = static_cast<uint32_t>(mul0 >> 32);
        const uint32_t lo0 = static_cast<uint32_t>(mul0);
        const uint32_t hi1 = static_cast<uint32_t>(mul1 >> 32);
        const uint32_t lo1 = static_cast<uint32_t>(mul1);
        counter = {hi1 ^ counter[1] ^ key[0], lo1, hi0 ^ counter[3] ^ key[1], lo0};
        key[0] += kKeyAdd0;
        key[1] += kKeyAdd1;
    }
    return counter;
}

template <uint16_t Rounds>
std::vector<uint32_t> BuildExpected(int rows, int cols, std::array<uint32_t, 2> key, std::array<uint32_t, 4> counter)
{
    constexpr uint32_t elementsPerRepeat = 256 / sizeof(uint32_t);
    constexpr uint32_t rowStrideChunks = 256 / 4;
    std::vector<uint32_t> out(rows * cols, 0);
    auto rowCounter = counter;
    for (int row = 0; row < rows; ++row) {
        for (uint32_t lane = 0; lane < elementsPerRepeat; ++lane) {
            auto laneCounter = rowCounter;
            cpu_random::IncrementCounter(laneCounter, lane);
            const auto values = RunRounds<Rounds>(laneCounter, key);
            for (uint32_t repeat = 0; repeat < 4; ++repeat) {
                out[row * cols + repeat * elementsPerRepeat + lane] = values[repeat];
            }
        }
        cpu_random::IncrementCounter(rowCounter, rowStrideChunks);
    }
    return out;
}
} // namespace

TEST(TRandomCpuSimTest, Rounds10MatchesExactReferenceFor4x256)
{
    using TileData = Tile<TileType::Vec, uint32_t, 4, 256>;
    TileData dst;
    TASSIGN(dst, 0);
    TRandomKey key = {0x12345678u, 0x9abcdef0u};
    TRandomCounter counter = {0x0u, 0x11111111u, 0x22222222u, 0x33333333u};

    TRANDOM<10>(dst, key, counter);

    const auto expected = BuildExpected<10>(4, 256, {key[0], key[1]}, {counter[0], counter[1], counter[2], counter[3]});
    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            EXPECT_EQ(dst.data()[GetTileElementOffset<TileData>(r, c)], expected[r * dst.GetValidCol() + c]);
        }
    }
}

TEST(TRandomCpuSimTest, Rounds7MatchesExactReference)
{
    using TileData = Tile<TileType::Vec, int32_t, 2, 256>;
    TileData dst;
    TASSIGN(dst, 0);
    TRandomKey key = {0x13579bdfu, 0x2468ace0u};
    TRandomCounter counter = {0x10u, 0x20u, 0x30u, 0x40u};

    TRANDOM<7>(dst, key, counter);

    const auto expected = BuildExpected<7>(2, 256, {key[0], key[1]}, {counter[0], counter[1], counter[2], counter[3]});
    for (int r = 0; r < dst.GetValidRow(); ++r) {
        for (int c = 0; c < dst.GetValidCol(); ++c) {
            EXPECT_EQ(static_cast<uint32_t>(dst.data()[GetTileElementOffset<TileData>(r, c)]),
                      expected[r * dst.GetValidCol() + c]);
        }
    }
}