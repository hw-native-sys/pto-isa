/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SPARSE_FLASH_MLA_CUSTOM_TGLOBALUPDATE_HPP_
#define SPARSE_FLASH_MLA_CUSTOM_TGLOBALUPDATE_HPP_

#include "common.hpp"

#if defined(__CCE_AICORE__)
namespace flash_mla_bf16 {
namespace custom {
using PartialOutputTile = Tile<TileType::Vec, float, AIV_M, PV_N>;
// Fragment tiles retain the physical row stride of the complete running O.
using OutputFragmentTile = Tile<TileType::Vec, float, AIV_M, D, BLayout::RowMajor, AIV_M, PV_N>;
using PackedOutputFragmentTile = Tile<TileType::Vec, bfloat16_t, AIV_M, D * 2, BLayout::RowMajor, AIV_M, PV_N>;
enum class GlobalUpdateMode { STEADY, FINAL };

namespace detail {
// Combine the preceding resident segment with running O before rescaling.
// Keep addition and multiplication separate to retain their rounding boundaries.
template <bool Initialize>
PTO_INTERNAL void globalUpdateSteadyRows(__ubuf__ float* partial, __ubuf__ float* running, __ubuf__ float* alphas)
{
    const auto all = pset_b32(PAT_ALL);
    vector_f32 zero;
    if constexpr (Initialize) {
        // Preserve the +0 + partial rounding and signed-zero behavior.
        vbr(zero, 0.0f);
    }
#pragma unroll 1
    for (uint16_t row = 0; row < AIV_M; ++row) {
        vector_f32 alpha;
        vlds(alpha, alphas, 1, BRC_B32, POST_UPDATE);
        // Four fixed pointer steps traverse each output fragment.
        // Stores advance running O after loading its prior value.
        {
            vector_f32 old, value;
            if constexpr (!Initialize) {
                vlds(old, running, 0, NORM);
            }
            vlds(value, partial, 64, NORM, POST_UPDATE);
            if constexpr (Initialize) {
                vadd(value, zero, value, all, MODE_ZEROING);
            } else {
                vadd(value, old, value, all, MODE_ZEROING);
            }
            vmul(value, value, alpha, all, MODE_ZEROING);
            vsts(value, running, 64, NORM_B32, all, POST_UPDATE);
        }
        {
            vector_f32 old, value;
            if constexpr (!Initialize) {
                vlds(old, running, 0, NORM);
            }
            vlds(value, partial, 64, NORM, POST_UPDATE);
            if constexpr (Initialize) {
                vadd(value, zero, value, all, MODE_ZEROING);
            } else {
                vadd(value, old, value, all, MODE_ZEROING);
            }
            vmul(value, value, alpha, all, MODE_ZEROING);
            vsts(value, running, 64, NORM_B32, all, POST_UPDATE);
        }
        {
            vector_f32 old, value;
            if constexpr (!Initialize) {
                vlds(old, running, 0, NORM);
            }
            vlds(value, partial, 64, NORM, POST_UPDATE);
            if constexpr (Initialize) {
                vadd(value, zero, value, all, MODE_ZEROING);
            } else {
                vadd(value, old, value, all, MODE_ZEROING);
            }
            vmul(value, value, alpha, all, MODE_ZEROING);
            vsts(value, running, 64, NORM_B32, all, POST_UPDATE);
        }
        {
            vector_f32 old, value;
            if constexpr (!Initialize) {
                vlds(old, running, 0, NORM);
            }
            vlds(value, partial, 64, NORM, POST_UPDATE);
            if constexpr (Initialize) {
                vadd(value, zero, value, all, MODE_ZEROING);
            } else {
                vadd(value, old, value, all, MODE_ZEROING);
            }
            vmul(value, value, alpha, all, MODE_ZEROING);
            // Skip the other fragment's 256 columns to the next row.
            vsts(value, running, 64 + D - PV_N, NORM_B32, all, POST_UPDATE);
        }
    }
}

PTO_INTERNAL void globalUpdateSteadyImpl(
    __ubuf__ float* partial, __ubuf__ float* running, __ubuf__ float* alphas, bool initialize)
{
    __VEC_SCOPE__
    {
        if (initialize) {
            globalUpdateSteadyRows<true>(partial, running, alphas);
        } else {
            globalUpdateSteadyRows<false>(partial, running, alphas);
        }
    }
}

// The final resident segment uses the held normalization basis.
// Packed output aliases running O with its FP32 byte pitch. Fragment 0 packs
// behind ascending reads; fragment 1 starts after fragment 0 reads its input.
template <bool Initialize>
PTO_INTERNAL void globalUpdateFinalRows(
    __ubuf__ float* partial, __ubuf__ float* running, __ubuf__ bfloat16_t* output, __ubuf__ float* multipliers)
{
    const auto all = pset_b32(PAT_ALL);
    vector_f32 zero;
    if constexpr (Initialize) {
        // Preserve the +0 + partial rounding and signed-zero behavior.
        vbr(zero, 0.0f);
    }
#pragma unroll 1
    for (uint16_t row = 0; row < AIV_M; ++row) {
        vector_f32 multiplier;
        vlds(multiplier, multipliers, 1, BRC_B32, POST_UPDATE);
        {
            vector_f32 old, value;
            vector_bf16 converted;
            if constexpr (!Initialize) {
                vlds(old, running, 64, NORM, POST_UPDATE);
            }
            vlds(value, partial, 64, NORM, POST_UPDATE);
            if constexpr (Initialize) {
                vadd(value, zero, value, all, MODE_ZEROING);
            } else {
                vadd(value, old, value, all, MODE_ZEROING);
            }
            vmul(value, value, multiplier, all, MODE_ZEROING);
            vcvt(converted, value, all, ROUND_R, RS_DISABLE, PART_EVEN);
            vsts(converted, output, 64, PK_B32, all, POST_UPDATE);
        }
        {
            vector_f32 old, value;
            vector_bf16 converted;
            if constexpr (!Initialize) {
                vlds(old, running, 64, NORM, POST_UPDATE);
            }
            vlds(value, partial, 64, NORM, POST_UPDATE);
            if constexpr (Initialize) {
                vadd(value, zero, value, all, MODE_ZEROING);
            } else {
                vadd(value, old, value, all, MODE_ZEROING);
            }
            vmul(value, value, multiplier, all, MODE_ZEROING);
            vcvt(converted, value, all, ROUND_R, RS_DISABLE, PART_EVEN);
            vsts(converted, output, 64, PK_B32, all, POST_UPDATE);
        }
        {
            vector_f32 old, value;
            vector_bf16 converted;
            if constexpr (!Initialize) {
                vlds(old, running, 64, NORM, POST_UPDATE);
            }
            vlds(value, partial, 64, NORM, POST_UPDATE);
            if constexpr (Initialize) {
                vadd(value, zero, value, all, MODE_ZEROING);
            } else {
                vadd(value, old, value, all, MODE_ZEROING);
            }
            vmul(value, value, multiplier, all, MODE_ZEROING);
            vcvt(converted, value, all, ROUND_R, RS_DISABLE, PART_EVEN);
            vsts(converted, output, 64, PK_B32, all, POST_UPDATE);
        }
        {
            vector_f32 old, value;
            vector_bf16 converted;
            if constexpr (!Initialize) {
                vlds(old, running, 320, NORM, POST_UPDATE);
            }
            vlds(value, partial, 64, NORM, POST_UPDATE);
            if constexpr (Initialize) {
                vadd(value, zero, value, all, MODE_ZEROING);
            } else {
                vadd(value, old, value, all, MODE_ZEROING);
            }
            vmul(value, value, multiplier, all, MODE_ZEROING);
            vcvt(converted, value, all, ROUND_R, RS_DISABLE, PART_EVEN);
            vsts(converted, output, 832, PK_B32, all, POST_UPDATE);
        }
    }
}

PTO_INTERNAL void globalUpdateFinalImpl(
    __ubuf__ float* partial, __ubuf__ float* running, __ubuf__ bfloat16_t* output, __ubuf__ float* multipliers,
    bool initialize)
{
    __VEC_SCOPE__
    {
        if (initialize) {
            globalUpdateFinalRows<true>(partial, running, output, multipliers);
        } else {
            globalUpdateFinalRows<false>(partial, running, output, multipliers);
        }
    }
}

} // namespace detail

// Public instruction memory operands are tiles. The caller owns UB allocation
// and cross-pipe handoffs. Each instruction issues one VF launch.
// Wait events follow PTO conventions.
template <GlobalUpdateMode Mode, typename... WaitEvents>
PTO_INST RecordEvent TGlobalUpdate(
    OutputFragmentTile& running, PartialOutputTile& partial, RowStateTile& alphas, bool initialize,
    WaitEvents&... events)
{
    WaitAllEvents(events...);
    static_assert(Mode == GlobalUpdateMode::STEADY);
    detail::globalUpdateSteadyImpl(
        (__ubuf__ float*)partial.data(), (__ubuf__ float*)running.data(), (__ubuf__ float*)alphas.data(), initialize);
    return {};
}

template <GlobalUpdateMode Mode, typename... WaitEvents>
PTO_INST RecordEvent TGlobalUpdate(
    PackedOutputFragmentTile& output, OutputFragmentTile& running, PartialOutputTile& partial,
    RowStateTile& multipliers, bool initialize, WaitEvents&... events)
{
    WaitAllEvents(events...);
    static_assert(Mode == GlobalUpdateMode::FINAL);
    detail::globalUpdateFinalImpl(
        (__ubuf__ float*)partial.data(), (__ubuf__ float*)running.data(), (__ubuf__ bfloat16_t*)output.data(),
        (__ubuf__ float*)multipliers.data(), initialize);
    return {};
}

} // namespace custom
} // namespace flash_mla_bf16
#endif // __CCE_AICORE__
#endif // SPARSE_FLASH_MLA_CUSTOM_TGLOBALUPDATE_HPP_
