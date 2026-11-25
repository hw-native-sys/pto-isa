/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SPARSE_FLASH_MLA_CUSTOM_TGATHERMETADATA_HPP_
#define SPARSE_FLASH_MLA_CUSTOM_TGATHERMETADATA_HPP_

#include "common.hpp"

#if defined(__CCE_AICORE__)
namespace flash_mla_bf16 {
namespace custom {
using GatherMetadataTile = Tile<TileType::Vec, uint32_t, 1, INDEX_SLOT_STRIDE / sizeof(uint32_t)>;

namespace detail {
PTO_INTERNAL void gatherMetaDataImpl(__ubuf__ uint32_t* mask, uint32_t count, uint32_t capacity)
{
    __VEC_SCOPE__
    {
        const auto all = pset_b32(PAT_ALL);
        vector_u32 lane, stride, invalid, one, zero;
        vci((vector_s32&)lane, 0, INC_ORDER);
        vbr(stride, CACHE_RECORD_BYTES);
        vbr(invalid, 0xffffffffu);
        vbr(one, 0x3f800000u); // Float 1.0, consumed directly by softmax.
        vbr(zero, 0u);
#pragma unroll 1
        for (uint16_t part = 0; part < INDEX_TOKENS / 128; ++part) {
            const uint32_t token = static_cast<uint32_t>(part) * 128;
            vector_u32 index0, index1, position, flags0, flags1;
            MaskReg active, valid;
            vlds(index0, mask, token, NORM);
            vlds(index1, mask, token + 64, NORM);
            vadds(position, lane, token, all, MODE_ZEROING);
            vcmps_lt(active, position, count, all);
            vsel(index0, index0, invalid, active);
            vadds(position, lane, token + 64, all, MODE_ZEROING);
            vcmps_lt(active, position, count, all);
            vsel(index1, index1, invalid, active);
            {
                // Two index vectors supply 64 unique pairs. Packed halves
                // encode pair descriptors; validity masks follow token order.
                // A single valid partner is duplicated; two invalid partners
                // disable the gather burst.
                vector_u32 even, odd, lo, hi, distance, low, high, strideLow, strideHigh;
                vector_u32 pairFlags, reverseFlags, packed0, packed1;
                MaskReg anyValid, paired, reverse;
                vdintlv(even, odd, index0, index1);
                vmin(lo, even, odd, all, MODE_ZEROING);
                vmax(hi, even, odd, all, MODE_ZEROING);
                vcmps_lt(anyValid, lo, capacity, all);
                vcmps_lt(paired, hi, capacity, all);
                vcmp_gt(reverse, even, odd, paired);
                vsub(distance, hi, lo, paired, MODE_ZEROING);
                vmull(strideLow, strideHigh, distance, stride, all);
                vbr(pairFlags, 0x80000000u);
                vsel(pairFlags, pairFlags, zero, anyValid);
                vadds(pairFlags, pairFlags, RAW_STRIDE << 8, all, MODE_ZEROING);
                vbr(reverseFlags, 0x40000000u);
                vsel(reverseFlags, reverseFlags, zero, reverse);
                vor(strideHigh, strideHigh, pairFlags, all, MODE_ZEROING);
                vor(strideHigh, strideHigh, reverseFlags, all, MODE_ZEROING);
                vintlv(packed0, packed1, strideLow, strideHigh);
                vsts(packed0, mask, INDEX_PAIR_OFFSET / sizeof(uint32_t) + token, NORM_B32, all);
                vsts(packed1, mask, INDEX_PAIR_OFFSET / sizeof(uint32_t) + token + 64, NORM_B32, all);
                vsel(lo, lo, zero, anyValid);
                vmull(low, high, lo, stride, all);
                vintlv(packed0, packed1, low, high);
                vsts(packed0, mask, INDEX_OFFSET / sizeof(uint32_t) + token, NORM_B32, all);
                vsts(packed1, mask, INDEX_OFFSET / sizeof(uint32_t) + token + 64, NORM_B32, all);
            }
            vcmps_lt(valid, index0, capacity, all);
            vsel(flags0, one, zero, valid);
            vsts(flags0, mask, token, NORM_B32, all);
            vcmps_lt(valid, index1, capacity, all);
            vsel(flags1, one, zero, valid);
            vsts(flags1, mask, token + 64, NORM_B32, all);
            vector_f32 validCount;
            vbr(validCount, 0.0f);
            vadd(validCount, validCount, (vector_f32&)flags0, all, MODE_ZEROING);
            if constexpr (N == 128) {
                vadd(validCount, validCount, (vector_f32&)flags1, all, MODE_ZEROING);
            }
            vcadd(validCount, validCount, all, MODE_ZEROING);
            vsts(
                validCount, reinterpret_cast<__ubuf__ float*>(mask), INDEX_ANY_OFFSET / sizeof(float) + token / N,
                ONEPT_B32, pset_b32(PAT_VL1));
            if constexpr (N == 64) {
                vbr(validCount, 0.0f);
                vadd(validCount, validCount, (vector_f32&)flags1, all, MODE_ZEROING);
                vcadd(validCount, validCount, all, MODE_ZEROING);
                vsts(
                    validCount, reinterpret_cast<__ubuf__ float*>(mask),
                    INDEX_ANY_OFFSET / sizeof(float) + token / N + 1, ONEPT_B32, pset_b32(PAT_VL1));
            }
        }
    }
}

} // namespace detail

// Public instruction memory operands are tiles. The caller owns UB allocation
// and cross-pipe handoffs. Each instruction issues one VF launch.
// Wait events follow PTO conventions.
template <typename... WaitEvents>
PTO_INST RecordEvent
TGatherMetaData(GatherMetadataTile& metadata, uint32_t count, uint32_t capacity, WaitEvents&... events)
{
    WaitAllEvents(events...);
    detail::gatherMetaDataImpl((__ubuf__ uint32_t*)metadata.data(), count, capacity);
    return {};
}

} // namespace custom
} // namespace flash_mla_bf16
#endif // __CCE_AICORE__
#endif // SPARSE_FLASH_MLA_CUSTOM_TGATHERMETADATA_HPP_
