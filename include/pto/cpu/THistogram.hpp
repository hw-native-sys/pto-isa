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

// ---- uint16 helper: histogram one byte of a 2-byte element ----
template <HistByte byte, typename TileDst, typename TileSrc, typename TileIdx>
PTO_INTERNAL std::enable_if_t<std::is_same_v<typename TileSrc::DType, uint16_t>> THISTOGRAM_IMPL(TileDst &dst,
                                                                                                 TileSrc &src,
                                                                                                 TileIdx &idx)
{
    using DstT = typename TileDst::DType;
    using IdxT = typename TileIdx::DType;
    static_assert(std::is_same_v<DstT, uint32_t>, "Fix: THISTOGRAM destination must be uint32_t.");
    static_assert(std::is_same_v<IdxT, uint8_t>, "Fix: THISTOGRAM index must be uint8_t.");
    static_assert(byte == HistByte::BYTE_0 || byte == HistByte::BYTE_1,
                  "Fix: THISTOGRAM with uint16 source only supports BYTE_0 (LSB) and BYTE_1 (MSB).");

    PTO_CPU_ASSERT(dst.GetValidCol() >= 256, "Fix: THISTOGRAM destination must have at least 256 bins.");

    for (int row = 0; row < src.GetValidRow(); ++row) {
        std::array<uint32_t, 256> counts{};
        const uint8_t rowIdx = idx.data()[GetTileElementOffset<TileIdx>(row, 0)];
        for (int col = 0; col < src.GetValidCol(); ++col) {
            const uint16_t value = src.data()[GetTileElementOffset<TileSrc>(row, col)];
            const uint8_t msb = static_cast<uint8_t>(value >> 8);
            const uint8_t lsb = static_cast<uint8_t>(value & 0xFFu);
            if constexpr (byte == HistByte::BYTE_1) {
                ++counts[msb];
            } else {
                if (msb == rowIdx) {
                    ++counts[lsb];
                }
            }
        }
        uint32_t cumulative = 0;
        for (int bin = 0; bin < 256; ++bin) {
            cumulative += counts[bin];
            dst.data()[GetTileElementOffset<TileDst>(row, bin)] = cumulative;
        }
    }
}

// ---- uint32 helper: extract one byte from a 4-byte element ----
inline uint8_t extractByte(uint32_t val, HistByte b)
{
    return static_cast<uint8_t>((val >> (static_cast<unsigned>(b) * 8)) & 0xFFu);
}

// ---- uint32: histogram one byte of a 4-byte element with cascaded filtering ----
// Radix sort processes MSB-first:
//   BYTE_3 → histogram of MSB (no filter)
//   BYTE_2 → filter by byte3 == idx[0,0]
//   BYTE_1 → filter by byte3 == idx[0,0] AND byte2 == idx[1,0]
//   BYTE_0 → filter by byte3/byte2/byte1 == idx rows 0/1/2
template <HistByte byte, typename TileDst, typename TileSrc, typename TileIdx>
PTO_INTERNAL std::enable_if_t<std::is_same_v<typename TileSrc::DType, uint32_t>> THISTOGRAM_IMPL(TileDst &dst,
                                                                                                 TileSrc &src,
                                                                                                 TileIdx &idx)
{
    using DstT = typename TileDst::DType;
    using IdxT = typename TileIdx::DType;
    static_assert(std::is_same_v<DstT, uint32_t>, "Fix: THISTOGRAM destination must be uint32_t.");
    static_assert(std::is_same_v<IdxT, uint8_t>, "Fix: THISTOGRAM index must be uint8_t.");

    PTO_CPU_ASSERT(dst.GetValidCol() >= 256, "Fix: THISTOGRAM destination must have at least 256 bins.");

    // Pre-load cascaded filter values from idx tile rows
    uint8_t filt3 = 0, filt2 = 0, filt1 = 0;
    if constexpr (byte <= HistByte::BYTE_2) {
        filt3 = idx.data()[GetTileElementOffset<TileIdx>(0, 0)];
    }
    if constexpr (byte <= HistByte::BYTE_1) {
        filt2 = idx.data()[GetTileElementOffset<TileIdx>(1, 0)];
    }
    if constexpr (byte <= HistByte::BYTE_0) {
        filt1 = idx.data()[GetTileElementOffset<TileIdx>(2, 0)];
    }

    for (int row = 0; row < src.GetValidRow(); ++row) {
        std::array<uint32_t, 256> counts{};
        for (int col = 0; col < src.GetValidCol(); ++col) {
            const uint32_t value = src.data()[GetTileElementOffset<TileSrc>(row, col)];
            bool pass = true;
            if constexpr (byte <= HistByte::BYTE_2) {
                pass = pass && (extractByte(value, HistByte::BYTE_3) == filt3);
            }
            if constexpr (byte <= HistByte::BYTE_1) {
                pass = pass && (extractByte(value, HistByte::BYTE_2) == filt2);
            }
            if constexpr (byte <= HistByte::BYTE_0) {
                pass = pass && (extractByte(value, HistByte::BYTE_1) == filt1);
            }
            if (pass) {
                ++counts[extractByte(value, byte)];
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
