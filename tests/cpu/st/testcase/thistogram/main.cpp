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
template <bool UseMsb>
std::array<uint32_t, 256> ReferenceHistogram(const std::vector<uint16_t> &values, uint8_t idx)
{
    std::array<uint32_t, 256> counts{};
    for (uint16_t value : values) {
        const uint8_t msb = static_cast<uint8_t>(value >> 8);
        const uint8_t lsb = static_cast<uint8_t>(value & 0xFFu);
        if constexpr (UseMsb) {
            ++counts[msb];
        } else if (msb == idx) {
            ++counts[lsb];
        }
    }
    uint32_t cumulative = 0;
    for (auto &count : counts) {
        cumulative += count;
        count = cumulative;
    }
    return counts;
}
} // namespace

TEST(THistogramCpuSimTest, MsbModeMatchesExactCumulativeHistogram)
{
    using SrcTile = Tile<TileType::Vec, uint16_t, 2, 16>;
    using DstTile = Tile<TileType::Vec, uint32_t, 2, 256>;
    using IdxTile = Tile<TileType::Vec, uint8_t, 32, 1, BLayout::ColMajor, 2, 1>;
    SrcTile src;
    DstTile dst;
    IdxTile idx;

    TASSIGN(src, 0);
    TASSIGN(dst, 2 * 16 * sizeof(uint16_t));
    TASSIGN(idx, 2 * 16 * sizeof(uint16_t) + 2 * 256 * sizeof(uint32_t));

    const std::vector<uint16_t> row0 = {0x1201u, 0x1202u, 0x13ffu, 0x1203u, 0x3400u, 0x0101u, 0x0202u, 0x0303u,
                                        0x0404u, 0x0505u, 0x0606u, 0x0707u, 0x0808u, 0x0909u, 0x0a0au, 0x0b0bu};
    const std::vector<uint16_t> row1 = {0xa001u, 0xa002u, 0xb001u, 0xa0ffu, 0xa010u, 0xb0ffu, 0xc001u, 0xc002u,
                                        0xc003u, 0xc004u, 0xc005u, 0xc006u, 0xc007u, 0xc008u, 0xc009u, 0xc00au};

    std::fill(dst.data(), dst.data() + DstTile::Numel, 0u);
    std::fill(src.data(), src.data() + SrcTile::Numel, 0u);
    idx.data()[1] = 0xa0u;
    idx.data()[0] = 0x12u;
    for (size_t i = 0; i < row0.size(); ++i) {
        src.data()[GetTileElementOffset<SrcTile>(0, static_cast<int>(i))] = row0[i];
    }
    for (size_t i = 0; i < row1.size(); ++i) {
        src.data()[GetTileElementOffset<SrcTile>(1, static_cast<int>(i))] = row1[i];
    }
    THISTOGRAM<true>(dst, src, idx);

    const auto expected0 = ReferenceHistogram<true>(row0, 0);
    const auto expected1 = ReferenceHistogram<true>(row1, 0);
    for (int bin = 0; bin < 256; ++bin) {
        EXPECT_EQ(dst.data()[GetTileElementOffset<DstTile>(0, bin)], expected0[bin]);
        EXPECT_EQ(dst.data()[GetTileElementOffset<DstTile>(1, bin)], expected1[bin]);
    }
}

TEST(THistogramCpuSimTest, LsbModeMatchesExactFilteredHistogram)
{
    using SrcTile = Tile<TileType::Vec, uint16_t, 2, 16>;
    using DstTile = Tile<TileType::Vec, uint32_t, 2, 256>;
    using IdxTile = Tile<TileType::Vec, uint8_t, 32, 1, BLayout::ColMajor, 2, 1>;
    SrcTile src;
    DstTile dst;
    IdxTile idx;

    TASSIGN(src, 0);
    TASSIGN(dst, 2 * 16 * sizeof(uint16_t));
    TASSIGN(idx, 2 * 16 * sizeof(uint16_t) + 2 * 256 * sizeof(uint32_t));

    const std::vector<uint16_t> row0 = {0x1201u, 0x1202u, 0x34ffu, 0x12ffu, 0x1210u};
    const std::vector<uint16_t> row1 = {0xa001u, 0xa002u, 0xa0ffu, 0xb003u, 0xa010u};

    std::fill(src.data(), src.data() + SrcTile::Numel, 0u);
    std::fill(dst.data(), dst.data() + DstTile::Numel, 0u);
    idx.data()[0] = 0x12u;
    idx.data()[1] = 0xa0u;
    for (size_t i = 0; i < row0.size(); ++i) {
        src.data()[GetTileElementOffset<SrcTile>(0, static_cast<int>(i))] = row0[i];
    }
    for (size_t i = 0; i < row1.size(); ++i) {
        src.data()[GetTileElementOffset<SrcTile>(1, static_cast<int>(i))] = row1[i];
    }

    THISTOGRAM<false>(dst, src, idx);

    const auto expected0 = ReferenceHistogram<false>(row0, 0x12u);
    const auto expected1 = ReferenceHistogram<false>(row1, 0xa0u);
    for (int bin = 0; bin < 256; ++bin) {
        EXPECT_EQ(dst.data()[GetTileElementOffset<DstTile>(0, bin)], expected0[bin]);
        EXPECT_EQ(dst.data()[GetTileElementOffset<DstTile>(1, bin)], expected1[bin]);
    }
}