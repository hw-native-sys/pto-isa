/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INT64_REARRANGE_HPP
#define INT64_REARRANGE_HPP

#include <pto/npu/a5/Int64Binary.hpp>

namespace pto {

template <typename T, unsigned DstCols>
PTO_INTERNAL void Int64Fill(__ubuf__ T* dst, T scalar, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 lowReg, highReg;
        uint64_t bits = static_cast<uint64_t>(scalar);
        vbr(lowReg, static_cast<int32_t>(bits));
        vbr(highReg, static_cast<int32_t>(bits >> 32));
        uint32_t count = validCols;
        MaskReg mask = plt_b32(count, POST_UPDATE);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            vsts(lowReg, highReg, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}

template <typename T, unsigned DstCols>
PTO_INTERNAL void Int64Tri(__ubuf__ T* dst, unsigned validRows, unsigned validCols, int diagonal, bool upper)
{
    __VEC_SCOPE__
    {
        vector_s32 zero, one;
        vbr(zero, 0);
        vbr(one, 1);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            int boundary = static_cast<int>(row) + diagonal;
            uint32_t ones;
            uint32_t zeros;
            if (upper) {
                zeros = boundary <= 0 ? 0 : boundary >= static_cast<int>(validCols) ? validCols : boundary;
                ones = validCols - zeros;
            } else {
                ones = boundary < 0 ? 0 : boundary + 1 >= static_cast<int>(validCols) ? validCols : boundary + 1;
                zeros = validCols - ones;
            }
            uint32_t cols = validCols;
            MaskReg storeMask = plt_b32(cols, POST_UPDATE);
            uint32_t prefix = upper ? zeros : ones;
            MaskReg prefixMask = plt_b32(prefix, POST_UPDATE);
            vector_s32 low;
            if (upper)
                vsel(low, zero, one, prefixMask);
            else
                vsel(low, one, zero, prefixMask);
            vsts(low, zero, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, storeMask);
        }
    }
}

template <typename T, typename I, unsigned DstCols, unsigned IdxCols>
PTO_INTERNAL void Int64Gather(
    __ubuf__ T* dst, __ubuf__ T* src, __ubuf__ I* index, unsigned validRows, unsigned validCols)
{
    static_assert(sizeof(I) == sizeof(uint32_t), "Int64Gather requires b32 indices");
    __VEC_SCOPE__
    {
        vector_u32 idx, wordIdx, highIdx, low, high;
        uint32_t count = validCols;
        MaskReg mask = plt_b32(count, POST_UPDATE);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            vlds(idx, (__ubuf__ uint32_t*)index + row * IdxCols, 0, NORM);
            vadd(wordIdx, idx, idx, mask, MODE_ZEROING);
            vadds(highIdx, wordIdx, 1u, mask, MODE_ZEROING);
            vgather2(low, (__ubuf__ uint32_t*)src, wordIdx, mask);
            vgather2(high, (__ubuf__ uint32_t*)src, highIdx, mask);
            vsts((vector_s32&)low, (vector_s32&)high, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}

template <typename T, typename I, unsigned DstNumel, unsigned SrcCols, unsigned IdxCols>
PTO_INTERNAL void Int64Scatter(
    __ubuf__ T* dst, __ubuf__ T* src, __ubuf__ I* index, unsigned validRows, unsigned validCols)
{
    static_assert(sizeof(I) == sizeof(uint32_t), "Int64Scatter requires b32 indices");
    __VEC_SCOPE__
    {
        vector_u32 zero, idx, wordIdx, highIdx, low, high;
        vbr(zero, 0u);
        uint32_t remaining = DstNumel * 2;
        constexpr uint16_t wordsPerRepeat = CCE_VL / sizeof(uint32_t);
        constexpr uint16_t initRepeats = (DstNumel * 2 + wordsPerRepeat - 1) / wordsPerRepeat;
        for (uint16_t repeat = 0; repeat < initRepeats; ++repeat) {
            MaskReg initMask = plt_b32(remaining, POST_UPDATE);
            vsts(zero, (__ubuf__ uint32_t*)dst + repeat * wordsPerRepeat, 0, NORM_B32, initMask);
        }
        uint32_t count = validCols;
        MaskReg mask = plt_b32(count, POST_UPDATE);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            vlds(low, high, (__ubuf__ uint32_t*)src + row * SrcCols * 2, 0, DINTLV_B32);
            vlds(idx, (__ubuf__ uint32_t*)index + row * IdxCols, 0, NORM);
            vadd(wordIdx, idx, idx, mask, MODE_ZEROING);
            vadds(highIdx, wordIdx, 1u, mask, MODE_ZEROING);
            vscatter(low, (__ubuf__ uint32_t*)dst, wordIdx, mask);
            vscatter(high, (__ubuf__ uint32_t*)dst, highIdx, mask);
        }
    }
}

template <MaskPattern Pattern, ScatterAxis Axis, typename T, unsigned DstNumel, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64ScatterPattern(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    constexpr unsigned times = GetTimesByMask<Pattern>();
    constexpr unsigned offset = Int64MaskPatternOffset<Pattern>();
    __VEC_SCOPE__
    {
        vector_s32 z, l0, h0;
        vector_u32 lane, elemIndex, lowIndex, highIndex;
        vbr(z, 0);
        uint32_t total = DstNumel * 2;
        uint32_t rem = total;
        constexpr uint16_t vl = CCE_VL / sizeof(uint32_t);
        constexpr uint16_t repeats = (DstNumel * 2 + vl - 1) / vl;
        for (uint16_t r = 0; r < repeats; ++r) {
            MaskReg m = plt_b32(rem, POST_UPDATE);
            vsts(z, (__ubuf__ int32_t*)dst + r * vl, 0, NORM_B32, m);
        }
        uint32_t count = validCols;
        MaskReg m = plt_b32(count, POST_UPDATE);
        vci((vector_s32&)lane, 0, INC_ORDER);
        vmuls(elemIndex, lane, static_cast<uint32_t>(times), m, MODE_ZEROING);
        vadds(elemIndex, elemIndex, static_cast<uint32_t>(offset), m, MODE_ZEROING);
        vadd(lowIndex, elemIndex, elemIndex, m, MODE_ZEROING);
        vadds(highIndex, lowIndex, 1u, m, MODE_ZEROING);
        uint16_t rows = validRows;
        for (uint16_t i = 0; i < rows; ++i) {
            vlds(l0, h0, (__ubuf__ int32_t*)src + i * SrcCols * 2, 0, DINTLV_B32);
            if constexpr (Axis == ScatterAxis::SCATTER_COL) {
                vsts(l0, h0, (__ubuf__ int32_t*)dst + (i * times + offset) * DstCols * 2, 0, INTLV_B32, m);
            } else {
                __ubuf__ uint32_t* rowDst = (__ubuf__ uint32_t*)dst + i * DstCols * 2;
                vscatter((vector_u32&)l0, rowDst, lowIndex, m);
                vscatter((vector_u32&)h0, rowDst, highIndex, m);
            }
        }
    }
}

template <typename T, unsigned DstCols, unsigned SrcRowStride>
PTO_INTERNAL void Int64RowExpand(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 lowReg, highReg, srcLow, srcHigh;
        uint32_t count = validCols;
        MaskReg mask = plt_b32(count, POST_UPDATE);
        MaskReg allMask = pset_b32(PAT_ALL);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            vlds(srcLow, srcHigh, (__ubuf__ int32_t*)src + row * SrcRowStride * 2, 0, DINTLV_B32);
            vdup(lowReg, srcLow, allMask, POS_LOWEST, MODE_ZEROING);
            vdup(highReg, srcHigh, allMask, POS_LOWEST, MODE_ZEROING);
            vsts(lowReg, highReg, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}

template <typename T, unsigned DstCols>
PTO_INTERNAL void Int64ColExpand(__ubuf__ T* dst, __ubuf__ T* src, unsigned validRows, unsigned validCols)
{
    __VEC_SCOPE__
    {
        vector_s32 lowReg, highReg;
        vlds(lowReg, highReg, (__ubuf__ int32_t*)src, 0, DINTLV_B32);
        uint32_t count = validCols;
        MaskReg mask = plt_b32(count, POST_UPDATE);
        uint16_t rows = validRows;
        for (uint16_t row = 0; row < rows; ++row) {
            vsts(lowReg, highReg, (__ubuf__ int32_t*)dst + row * DstCols * 2, 0, INTLV_B32, mask);
        }
    }
}

} // namespace pto

#endif
