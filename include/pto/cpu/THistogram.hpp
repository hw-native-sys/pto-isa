/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef THISTOGRAM_CPU_HPP
#define THISTOGRAM_CPU_HPP

#include <array>
#include <cstdint>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {
template <bool MSBorLSB = true, typename TileDst, typename TileSrc, typename TileIdx>
PTO_INTERNAL void THISTOGRAM_IMPL(TileDst &dst, TileSrc &src, TileIdx &idx)
{
    using SrcT = typename TileSrc::DType;
    using DstT = typename TileDst::DType;
    using IdxT = typename TileIdx::DType;
    static_assert(std::is_same_v<SrcT, uint16_t>, "Fix: THISTOGRAM source must be uint16_t.");
    static_assert(std::is_same_v<DstT, uint32_t>, "Fix: THISTOGRAM destination must be uint32_t.");
    static_assert(std::is_same_v<IdxT, uint8_t>, "Fix: THISTOGRAM index must be uint8_t.");

    PTO_CPU_ASSERT(dst.GetValidCol() >= 256, "Fix: THISTOGRAM destination must have at least 256 bins.");

    for (int row = 0; row < src.GetValidRow(); ++row) {
        std::array<uint32_t, 256> counts{};
        const uint8_t rowIdx = idx.data()[GetTileElementOffset<TileIdx>(row, 0)];
        for (int col = 0; col < src.GetValidCol(); ++col) {
            const uint16_t value = src.data()[GetTileElementOffset<TileSrc>(row, col)];
            const uint8_t msb = static_cast<uint8_t>(value >> 8);
            const uint8_t lsb = static_cast<uint8_t>(value & 0xFFu);
            if constexpr (MSBorLSB) {
                ++counts[msb];
            } else if (msb == rowIdx) {
                ++counts[lsb];
            }
        }

        uint32_t cumulative = 0;
        for (int bin = 0; bin < 256; ++bin) {
            cumulative += counts[bin];
            dst.data()[GetTileElementOffset<TileDst>(row, bin)] = cumulative;
        }
    }
}
} // namespace pto

#endif