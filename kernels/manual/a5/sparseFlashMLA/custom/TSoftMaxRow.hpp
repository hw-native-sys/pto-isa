/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SPARSE_FLASH_MLA_CUSTOM_TSOFTMAXROW_HPP_
#define SPARSE_FLASH_MLA_CUSTOM_TSOFTMAXROW_HPP_

#include "common.hpp"

#if defined(__CCE_AICORE__)
namespace flash_mla_bf16 {
namespace custom {
using DecisionTile = Tile<TileType::Vec, uint32_t, 1, 8>;

namespace detail {
template <bool AllValid, bool FirstTile>
PTO_INTERNAL void softMaxRowImpl(
    float scale, __ubuf__ float* score, __ubuf__ float* maxima, __ubuf__ float* anchors, __ubuf__ uint32_t* decision,
    __ubuf__ float* sums, __ubuf__ float* alphas, __ubuf__ float* mask, __ubuf__ float* rowScratch)
{
    __VEC_SCOPE__
    {
        const auto all = pset_b32(PAT_ALL);
        const auto rows = pset_b32(PAT_VL32);
        vector_f32 zero, lowest;
        vbr(zero, 0.0f);
        vbr(lowest, -__FLT_MAX__);
        MaskReg valid0, valid1;
        if constexpr (!AllValid) {
            vector_f32 flags;
            vlds(flags, mask, 0, NORM);
            vcmp_ne(valid0, flags, zero, all);
            if constexpr (N == 128) {
                vlds(flags, mask, 64, NORM);
                vcmp_ne(valid1, flags, zero, all);
            }
        }
        {
            UnalignReg stream;
            auto* out = rowScratch;
#pragma unroll 1
            for (uint16_t row = 0; row < AIV_M; ++row) {
                vector_f32 x0, x1, localMax;
                vlds(x0, score, row * N, NORM);
                vmuls(x0, x0, scale, all, MODE_ZEROING);
                if constexpr (!AllValid)
                    vsel(x0, x0, lowest, valid0);
                if constexpr (N == 128) {
                    vlds(x1, score, row * N + 64, NORM);
                    vmuls(x1, x1, scale, all, MODE_ZEROING);
                    if constexpr (!AllValid)
                        vsel(x1, x1, lowest, valid1);
                    vmax(x0, x0, x1, all, MODE_ZEROING);
                }
                vcmax(localMax, x0, all, MODE_ZEROING);
                vstus(stream, 1, localMax, out, POST_UPDATE);
            }
            vstas(stream, out, 0, POST_UPDATE);
        }
        mem_bar(VST_VLD);
        {
            vector_f32 localMax, newMax, newAnchor, alpha;
            vector_u32 triggerBits, zeroBits;
            vbr(zeroBits, 0u);
            vlds(localMax, rowScratch, 0, NORM);
            if constexpr (FirstTile) {
                // The initial maximum is clamped to -FLT_MAX, including
                // valid finite inputs whose QK product overflows to -inf.
                vmax(newMax, lowest, localMax, rows, MODE_ZEROING);
                newAnchor = newMax;
                alpha = zero;
                triggerBits = zeroBits;
            } else {
                vector_f32 oldMax, oldAnchor, oldSum, delta, maximumDelta, threshold;
                vector_u32 oneBits;
                MaskReg trigger, seen;
                vbr(oneBits, 1u);
                vlds(oldMax, maxima, 0, NORM);
                vlds(oldAnchor, anchors, 0, NORM);
                vlds(oldSum, sums, 0, NORM);
                vmax(newMax, oldMax, localMax, rows, MODE_ZEROING);
                vsub(delta, newMax, oldAnchor, rows, MODE_ZEROING);
                vcmax(maximumDelta, delta, rows, MODE_ZEROING);
                vdup(maximumDelta, maximumDelta, rows, POS_LOWEST, MODE_ZEROING);
                vbr(threshold, RESCALE_THRESHOLD);
                vcmp_gt(trigger, maximumDelta, threshold, rows);
                vsel(newAnchor, newMax, oldAnchor, trigger);
                vsel(triggerBits, oneBits, zeroBits, trigger);
                vcmp_gt(seen, oldSum, zero, rows);
                vsub(alpha, oldAnchor, newAnchor, rows, MODE_ZEROING);
                vexp(alpha, alpha, rows, MODE_ZEROING);
                vsel(alpha, alpha, zero, seen);
            }
            vsts(newMax, maxima, 0, NORM_B32, rows);
            vsts(newAnchor, anchors, 0, NORM_B32, rows);
            vsts(alpha, alphas, 0, NORM_B32, rows);
            const auto firstLane = pset_b32(PAT_VL1);
            vsts(triggerBits, decision, 0, NORM_B32, firstLane);
        }
    }
}

} // namespace detail

// Public instruction memory operands are tiles. The caller owns UB allocation
// and cross-pipe handoffs. Each instruction issues one VF launch.
// Wait events follow PTO conventions.
template <bool AllValid, bool FirstTile, typename... WaitEvents>
PTO_INST RecordEvent TSoftMaxRow(
    RowStateTile& maxima, RowStateTile& anchors, DecisionTile& decision, RowStateTile& alphas,
    SoftMaxScratchTile& scratch, VecScore& score, RowStateTile& sums, IndexMaskTile& mask, float scale,
    WaitEvents&... events)
{
    WaitAllEvents(events...);
    detail::softMaxRowImpl<AllValid, FirstTile>(
        scale, (__ubuf__ float*)score.data(), (__ubuf__ float*)maxima.data(), (__ubuf__ float*)anchors.data(),
        (__ubuf__ uint32_t*)decision.data(), (__ubuf__ float*)sums.data(), (__ubuf__ float*)alphas.data(),
        (__ubuf__ float*)mask.data(), (__ubuf__ float*)scratch.data());
    return {};
}

} // namespace custom
} // namespace flash_mla_bf16
#endif // __CCE_AICORE__
#endif // SPARSE_FLASH_MLA_CUSTOM_TSOFTMAXROW_HPP_
