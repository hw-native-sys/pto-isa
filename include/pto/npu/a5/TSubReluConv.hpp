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
 * @file TSubReluConv.hpp
 * @brief Fused Sub + ReLU + Type-Conversion (TSUBRELUCONV) for NPU A5.
 *
 * Computes, per element: dst[i] = convert(max(0, src0[i] - src1[i]))
 *
 * A5 has no single `vsubreluconv_*` intrinsic (unlike a2a3); instead the fusion
 * is expressed in the VF register-compute model, mirroring the SubDeqRelu
 * reference (load -> sub -> relu -> cast -> store) but without the dequant step:
 *   vlds(src0); vlds(src1);  -> sub(vdiff)  -> maxs(vdiff, 0)  -> cvt(vout) -> sts
 *
 * Supported source -> destination type pairs:
 *   - float   -> half     (sub in fp32,  vcvt fp32->fp16  with PK_B32)
 *   - half    -> int8_t   (sub in fp16,  vcvt fp16->int8  with PK_B16)
 *   - int16_t -> int8_t   (sub in int16, clamp [0,127] then narrow with PK_B16)
 *
 * Because the ReLU clamps every result to be non-negative, the narrowing
 * conversions only ever produce values in [0, 127], matching the saturating
 * clamp used by the golden reference. The fp32->fp16 and fp16->int8 vcvt
 * intrinsics saturate intrinsically. A5 has no s16->s8 vcvt (only s16->u8),
 * so the int16 path clamps the (already non-negative) diff to 127 and narrows
 * to uint8; for values in [0,127] the uint8 and int8 byte patterns are equal.
 */

#ifndef TSUBRELUCONV_HPP
#define TSUBRELUCONV_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include <pto/common/debug.h>

namespace pto {

// Rounding type used by the float/half down-casts (round-to-nearest-even),
// matching the TCVT default (RoundMode::CAST_RINT) for these pairs.
using __cce_simd::RoundRType;

// ============================================================================
// Core tile kernel: per-row iteration, processing one repeat (a full vector) at
// a time. The compute/predicate width follows the source element type; the
// store packs the narrower destination via PK_B32 (32->16) or PK_B16 (16->8).
//   DS / SS0 / SS1 : dst / src0 / src1 row strides (in elements)
// ============================================================================
template <typename TileDataDst, typename TileDataSrc, unsigned DS, unsigned SS0, unsigned SS1>
__tf__ PTO_INTERNAL OP_NAME(TSUBRELUCONV) OP_TYPE(element_wise) void TSubReluConv(
    typename TileDataDst::TileDType __out__ dstData, typename TileDataSrc::TileDType __in__ src0Data,
    typename TileDataSrc::TileDType __in__ src1Data, unsigned validRows, unsigned validCols)
{
    using DT = typename TileDataDst::DType;
    using ST = typename TileDataSrc::DType;
    __ubuf__ DT* dstPtr = (__ubuf__ DT*)__cce_get_tile_ptr(dstData);
    __ubuf__ ST* src0Ptr = (__ubuf__ ST*)__cce_get_tile_ptr(src0Data);
    __ubuf__ ST* src1Ptr = (__ubuf__ ST*)__cce_get_tile_ptr(src1Data);

    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(ST);
    uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);

    __VEC_SCOPE__
    {
        RegTensor<ST> vsrc0, vsrc1, vdiff;
        MaskReg preg;
        for (uint16_t i = 0; i < (uint16_t)validRows; ++i) {
            uint32_t sreg = (uint32_t)validCols;
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                preg = CreatePredicate<ST>(sreg);
                vlds(vsrc0, src0Ptr, i * SS0 + j * elementsPerRepeat, NORM);
                vlds(vsrc1, src1Ptr, i * SS1 + j * elementsPerRepeat, NORM);
                vsub(vdiff, vsrc0, vsrc1, preg, MODE_ZEROING);
                vmaxs(vdiff, vdiff, (ST)0, preg, MODE_ZEROING);
                if constexpr (std::is_same<ST, float>::value) {
                    // fp32 -> fp16 (vcvt saturates intrinsically)
                    RegTensor<DT> vout;
                    vcvt(vout, vdiff, preg, RoundRType(), RS_DISABLE, PART_EVEN);
                    vsts(vout, dstPtr, i * DS + j * elementsPerRepeat, PK_B32, preg);
                } else if constexpr (std::is_same<ST, half>::value) {
                    // fp16 -> int8 (vcvt saturates to [-128,127]; ReLU -> [0,127])
                    RegTensor<DT> vout;
                    vcvt(vout, vdiff, preg, RoundRType(), RS_DISABLE, PART_EVEN);
                    vsts(vout, dstPtr, i * DS + j * elementsPerRepeat, PK_B16, preg);
                } else {
                    // int16 -> int8: A5 has no s16->s8 vcvt. Clamp the already
                    // non-negative diff to [0,127] and narrow via s16->u8; the
                    // uint8 byte pattern equals int8 for values in [0,127].
                    vmins(vdiff, vdiff, (ST)127, preg, MODE_ZEROING);
                    RegTensor<uint8_t> vout;
                    vcvt(vout, vdiff, preg, RS_DISABLE, PART_EVEN);
                    vsts(vout, (__ubuf__ uint8_t*)dstPtr, i * DS + j * elementsPerRepeat, PK_B16, preg);
                }
            }
        }
    }
}

// ============================================================================
// Static / dynamic validation shared by the public entry point.
// ============================================================================
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TSubReluConvCheck(const TileDataDst& dst, const TileDataSrc0& src0, const TileDataSrc1& src1)
{
    using DT = typename TileDataDst::DType;
    using ST = typename TileDataSrc0::DType;
    static_assert(
        std::is_same<ST, typename TileDataSrc1::DType>::value,
        "Fix: TSUBRELUCONV src0 and src1 must have the same element type.");
    static_assert(
        TileDataDst::Loc == TileType::Vec && TileDataSrc0::Loc == TileType::Vec && TileDataSrc1::Loc == TileType::Vec,
        "Fix: TSUBRELUCONV tiles must live in TileType::Vec.");
    static_assert(
        (std::is_same<ST, float>::value && std::is_same<DT, half>::value) ||
            (std::is_same<ST, int16_t>::value && std::is_same<DT, int8_t>::value) ||
            (std::is_same<ST, half>::value && std::is_same<DT, int8_t>::value),
        "Fix: TSUBRELUCONV supports float->half, half->int8 and int16->int8 only.");
    static_assert(
        TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
        "Fix: TSUBRELUCONV only supports row major layout.");
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    PTO_ASSERT(
        src0.GetValidRow() == validRow && src0.GetValidCol() == validCol,
        "Fix: TSUBRELUCONV input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(
        src1.GetValidRow() == validRow && src1.GetValidCol() == validCol,
        "Fix: TSUBRELUCONV input tile src1 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(validRow > 0 && validCol > 0, "Fix: TSUBRELUCONV valid rows and columns must be greater than 0.");
}

// ============================================================================
// Public implementation entry point.
// ============================================================================
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TSUBRELUCONV_IMPL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1)
{
    TSubReluConvCheck<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);

    constexpr unsigned DS = TileDataDst::RowStride;
    constexpr unsigned SS0 = TileDataSrc0::RowStride;
    constexpr unsigned SS1 = TileDataSrc1::RowStride;

    TSubReluConv<TileDataDst, TileDataSrc0, DS, SS0, SS1>(
        dst.data(), src0.data(), src1.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto
#endif
