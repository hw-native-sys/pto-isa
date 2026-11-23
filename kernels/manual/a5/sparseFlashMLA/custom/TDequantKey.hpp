/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SPARSE_FLASH_MLA_CUSTOM_TDEQUANTKEY_HPP_
#define SPARSE_FLASH_MLA_CUSTOM_TDEQUANTKEY_HPP_

#include "common.hpp"

#if defined(__CCE_AICORE__)
namespace flash_mla_bf16 {
namespace custom {
using RawKeyTile = Tile<TileType::Vec, uint8_t, K_CHUNK, RAW_STRIDE>;
using PairControlTile = Tile<TileType::Vec, uint8_t, 1, K_CHUNK / 2 * sizeof(uint64_t)>;
using DequantScaleTile = Tile<TileType::Vec, bfloat16_t, K_CHUNK, CACHE_SCALE_D>;
using DequantKeyTile =
    Tile<TileType::Vec, bfloat16_t, KV_NZ_ROWS, D, BLayout::ColMajor, K_CHUNK, D, SLayout::RowMajor, 512>;

namespace detail {
PTO_INTERNAL void dequantKeyImpl(
    __ubuf__ uint8_t* raw, __ubuf__ uint8_t* pairControls, __ubuf__ bfloat16_t* scaleWrite, __ubuf__ bfloat16_t* dst0)
{
    auto* rawScales = raw + D / 2;
    auto* scales = scaleWrite;
    auto* controls = pairControls + 7;
    auto* dst1 = dst0 + 16;
    constexpr uint32_t STORE_CONFIG = (KV_NZ_ROWS << 16) | (8 * KV_NZ_ROWS);
    __VEC_SCOPE__
    {
        const auto all8 = pset_b8(PAT_ALL);
        const auto all16 = pset_b16(PAT_ALL);
        const auto all32 = pset_b32(PAT_ALL);
        vector_u16 indices, rows;
        vci((vector_s16&)indices, 0, INC_ORDER);
        vshrs(rows, indices, 5, all16, MODE_ZEROING);
        vshls(rows, rows, 8, all16, MODE_ZEROING);
        vadd(indices, indices, rows, all16, MODE_ZEROING);
#pragma unroll 1
        for (uint16_t row = 0; row < K_CHUNK; row += 4) {
            vector_f8e4m3 packed;
            vector_f32 even, odd;
            vector_u32 low;
            vector_bf16 decoded;
            vgather2((vector_u16&)packed, rawScales, indices, all16);
            rawScales += 4 * RAW_STRIDE;
            vcvt(even, packed, all16, PART_P0);
            vcvt(odd, packed, all16, PART_P2);
            vshrs(low, (vector_u32&)even, 16, all32, MODE_ZEROING);
            vor((vector_u32&)decoded, low, (vector_u32&)odd, all32);
            vsts(decoded, scaleWrite, 128, NORM_B16, all16, POST_UPDATE);
        }
        mem_bar(VST_VLD);
#pragma unroll 1
        for (uint16_t pair = 0; pair < K_CHUNK / 2; ++pair) {
            MaskReg reverse;
            // Byte load into B16 lanes keeps the predicate aligned to BF16 data.
            vector_u8 byte;
            vlds(byte, controls, pair * 8, BRC_B8);
            vcmps_eq(reverse, byte, 0xc0u, all8);
#pragma unroll 1
            for (uint16_t chunk = 0; chunk < D / 128; ++chunk) {
                vector_u8 packed0, packed1;
                vector_bf16 first, second, scale0, scale1, restored0, restored1;
                vlds(scale0, scales, pair * 2 * CACHE_SCALE_D + chunk * 8, E2B_B16);
                vlds(scale1, scales, (pair * 2 + 1) * CACHE_SCALE_D + chunk * 8, E2B_B16);
                vlds(packed0, raw, pair * 2 * RAW_STRIDE + chunk * 64, UNPK4_B8);
                vlds(packed1, raw, (pair * 2 + 1) * RAW_STRIDE + chunk * 64, UNPK4_B8);
                vcvt(first, (vector_f4e2m1x2&)packed0, all8, PART_P0);
                vcvt(second, (vector_f4e2m1x2&)packed1, all8, PART_P0);
                vmul(first, first, scale0, all16, MODE_ZEROING);
                vmul(second, second, scale1, all16, MODE_ZEROING);
                vsel(restored0, second, first, reverse);
                vsel(restored1, first, second, reverse);
                vsstb(restored0, dst0, STORE_CONFIG, all16, POST_UPDATE);
                vsstb(restored1, dst1, STORE_CONFIG, all16, POST_UPDATE);
            }
            dst0 -= KV_NZ_BYTES / 2 - 32;
            dst1 -= KV_NZ_BYTES / 2 - 32;
        }
    }
}

} // namespace detail

// Public instruction memory operands are tiles. The caller owns UB allocation
// and cross-pipe handoffs. Each instruction issues one VF launch.
// Wait events follow PTO conventions.
template <typename... WaitEvents>
PTO_INST RecordEvent TDequantKey(
    DequantKeyTile& dst, DequantScaleTile& scales, RawKeyTile& raw, PairControlTile& controls, WaitEvents&... events)
{
    WaitAllEvents(events...);
    detail::dequantKeyImpl(
        (__ubuf__ uint8_t*)raw.data(), (__ubuf__ uint8_t*)controls.data(), (__ubuf__ bfloat16_t*)scales.data(),
        (__ubuf__ bfloat16_t*)dst.data());
    return {};
}

} // namespace custom
} // namespace flash_mla_bf16
#endif // __CCE_AICORE__
#endif // SPARSE_FLASH_MLA_CUSTOM_TDEQUANTKEY_HPP_
