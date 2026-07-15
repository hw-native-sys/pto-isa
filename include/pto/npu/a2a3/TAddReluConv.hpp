/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * @file TAddReluConv.hpp
 * @brief Fused Add + ReLU + Type-Conversion (TADDRELUCONV) for NPU A2/A3.
 *
 * Computes, per element: dst[i] = convert(max(0, src0[i] + src1[i]))
 *
 * The hardware fuses the add, the ReLU clamp and the down-cast into a single
 * `vaddreluconv_*` instruction, removing the intermediate register traffic that
 * a TADD -> TRELU -> TCVT sequence would otherwise generate.
 *
 * Supported source -> destination type pairs (dav-c220 / a2a3):
 *   - float   -> half     (vaddreluconv_f322f16)
 *   - half    -> int8_t   (vaddreluconv_f162s8)
 *   - int16_t -> int8_t   (vaddreluconv_s162s8)
 */

#ifndef TADDRELUCONV_HPP
#define TADDRELUCONV_HPP

#include "common.hpp"

namespace pto {

// ============================================================================
// Intrinsic dispatcher: select the fused add/relu/convert by (src, dst) type.
// ============================================================================
template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void AddReluConvCall(
    __ubuf__ typename TileDataDst::DType* dst, __ubuf__ typename TileDataSrc::DType* src0,
    __ubuf__ typename TileDataSrc::DType* src1, uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
    uint8_t src1BlockStride, uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride)
{
    using DT = typename TileDataDst::DType;
    using ST = typename TileDataSrc::DType;
    // The trailing `false` selects the low-half destination packing (h = 0).
    if constexpr (std::is_same<ST, float>::value && std::is_same<DT, half>::value) {
        vaddreluconv_f322f16(
            dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride, false);
    } else if constexpr (std::is_same<ST, half>::value && std::is_same<DT, int8_t>::value) {
        vaddreluconv_f162s8(
            dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride, false);
    } else if constexpr (std::is_same<ST, int16_t>::value && std::is_same<DT, int8_t>::value) {
        vaddreluconv_s162s8(
            dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride, false);
    }
}

// ============================================================================
// Repeat configuration: number of elements processed per repeat plus the dst
// and src repeat strides (in 32-byte blocks). The element count is bounded by
// the wider of the two element types so neither side overruns the 256-byte
// repeat window, matching the convention used by TCVT.
// ============================================================================
template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void ComputeAddReluConvConfig(
    unsigned& elementsPerRepeat, unsigned& dstRepeatStride, unsigned& srcRepeatStride)
{
    constexpr unsigned dstSize = sizeof(typename TileDataDst::DType);
    constexpr unsigned srcSize = sizeof(typename TileDataSrc::DType);
    constexpr unsigned repeatWidth = dstSize > srcSize ? dstSize : srcSize;
    dstRepeatStride = (repeatWidth == dstSize) ? BLOCK_MAX_PER_REPEAT : (BLOCK_MAX_PER_REPEAT / srcSize * dstSize);
    srcRepeatStride = (repeatWidth == srcSize) ? BLOCK_MAX_PER_REPEAT : (BLOCK_MAX_PER_REPEAT / dstSize * srcSize);
    elementsPerRepeat = REPEAT_BYTE / repeatWidth;
}

// ============================================================================
// Core tile kernel: iterate over rows, emitting full-repeat blocks (head) and
// a masked remainder block (tail) per row.
//   SS0/SS1 : src0/src1 row strides (in elements)
//   DS      : dst row stride (in elements)
// ============================================================================
template <typename TileDataDst, typename TileDataSrc, unsigned SS0, unsigned SS1, unsigned DS>
__tf__ PTO_INTERNAL void TAddReluConv(
    typename TileDataDst::TileDType __out__ dstData, typename TileDataSrc::TileDType __in__ src0Data,
    typename TileDataSrc::TileDType __in__ src1Data, unsigned numRepeatPerLine, unsigned numRemainPerLine,
    unsigned validRow, unsigned elementsPerRepeat, unsigned dstRepeatStride, unsigned srcRepeatStride)
{
    using DT = typename TileDataDst::DType;
    using ST = typename TileDataSrc::DType;
    __ubuf__ DT* dstPtr = (__ubuf__ DT*)__cce_get_tile_ptr(dstData);
    __ubuf__ ST* src0Ptr = (__ubuf__ ST*)__cce_get_tile_ptr(src0Data);
    __ubuf__ ST* src1Ptr = (__ubuf__ ST*)__cce_get_tile_ptr(src1Data);

    // Head region: complete repeat blocks.
    if (numRepeatPerLine > 0) {
        unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (unsigned i = 0; i < validRow; i++) {
            for (unsigned j = 0; j < numLoop; j++) {
                unsigned span = j * elementsPerRepeat * REPEAT_MAX;
                AddReluConvCall<TileDataDst, TileDataSrc>(
                    dstPtr + i * DS + span, src0Ptr + i * SS0 + span, src1Ptr + i * SS1 + span, (uint8_t)REPEAT_MAX, 1,
                    1, 1, (uint8_t)dstRepeatStride, (uint8_t)srcRepeatStride, (uint8_t)srcRepeatStride);
            }
            if (remainAfterLoop > 0) {
                unsigned span = numLoop * elementsPerRepeat * REPEAT_MAX;
                AddReluConvCall<TileDataDst, TileDataSrc>(
                    dstPtr + i * DS + span, src0Ptr + i * SS0 + span, src1Ptr + i * SS1 + span,
                    (uint8_t)remainAfterLoop, 1, 1, 1, (uint8_t)dstRepeatStride, (uint8_t)srcRepeatStride,
                    (uint8_t)srcRepeatStride);
            }
        }
    }

    // Tail region: remainder columns handled with a continuous mask.
    if (numRemainPerLine > 0) {
        unsigned base = numRepeatPerLine * elementsPerRepeat;
        SetContinuousMask(numRemainPerLine);
        for (unsigned i = 0; i < validRow; i++) {
            AddReluConvCall<TileDataDst, TileDataSrc>(
                dstPtr + i * DS + base, src0Ptr + i * SS0 + base, src1Ptr + i * SS1 + base, (uint8_t)1, 1, 1, 1,
                (uint8_t)dstRepeatStride, (uint8_t)srcRepeatStride, (uint8_t)srcRepeatStride);
        }
        set_vector_mask(-1, -1);
    }
}

// ============================================================================
// Static / dynamic validation shared by the public entry point.
// ============================================================================
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TAddReluConvCheck(const TileDataDst& dst, const TileDataSrc0& src0, const TileDataSrc1& src1)
{
    using DT = typename TileDataDst::DType;
    using ST = typename TileDataSrc0::DType;
    static_assert(
        std::is_same<ST, typename TileDataSrc1::DType>::value,
        "Fix: TADDRELUCONV src0 and src1 must have the same element type.");
    static_assert(
        (std::is_same<ST, float>::value && std::is_same<DT, half>::value) ||
            (std::is_same<ST, half>::value && std::is_same<DT, int8_t>::value) ||
            (std::is_same<ST, int16_t>::value && std::is_same<DT, int8_t>::value),
        "Fix: TADDRELUCONV supports float->half, half->int8 and int16->int8 only.");
    static_assert(
        TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
        "Fix: TADDRELUCONV only supports row major layout.");
    static_assert(
        TileDataDst::Loc == TileType::Vec && TileDataSrc0::Loc == TileType::Vec && TileDataSrc1::Loc == TileType::Vec,
        "Fix: TADDRELUCONV tiles must live in TileType::Vec.");
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    PTO_ASSERT(validRow > 0 && validCol > 0, "Fix: TADDRELUCONV valid rows and columns must be greater than 0.");
    PTO_ASSERT(
        src0.GetValidRow() == validRow && src0.GetValidCol() == validCol,
        "Fix: TADDRELUCONV input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(
        src1.GetValidRow() == validRow && src1.GetValidCol() == validCol,
        "Fix: TADDRELUCONV input tile src1 valid shape mismatch with output tile dst shape.");
}

// ============================================================================
// Public implementation entry point.
// ============================================================================
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TADDRELUCONV_IMPL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1)
{
    TAddReluConvCheck<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);

    unsigned elementsPerRepeat, dstRepeatStride, srcRepeatStride;
    ComputeAddReluConvConfig<TileDataDst, TileDataSrc0>(elementsPerRepeat, dstRepeatStride, srcRepeatStride);

    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    unsigned numRepeatPerLine = validCol / elementsPerRepeat;
    unsigned numRemainPerLine = validCol % elementsPerRepeat;

    constexpr unsigned SS0 = TileDataSrc0::RowStride;
    constexpr unsigned SS1 = TileDataSrc1::RowStride;
    constexpr unsigned DS = TileDataDst::RowStride;

    TAddReluConv<TileDataDst, TileDataSrc0, SS0, SS1, DS>(
        dst.data(), src0.data(), src1.data(), numRepeatPerLine, numRemainPerLine, validRow, elementsPerRepeat,
        dstRepeatStride, srcRepeatStride);
}
} // namespace pto

#endif
