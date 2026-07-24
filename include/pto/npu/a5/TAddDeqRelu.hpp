/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * @file TAddDeqRelu.hpp
 * @brief Fused Add + Dequantize + ReLU (TADDDEQRELU) for NPU A5.
 *
 * Computes, per element:
 *   dst[i] = max(0, (src0[i] + src1[i]) * deqScale)  converted to half
 *
 * A5 has no single fused intrinsic for this operation; instead the fusion is
 * expressed in the register-compute model following the same precision-
 * compensated scaling path as the a2a3 implementation:
 *   vlds(src0); vlds(src1) -> vadd -> vcvt(s32->f32) -> vmuls(1/131072)
 *   -> vmuls(deqScale) -> vmuls(131072) -> vmaxs(0) -> vcvt(f32->f16) -> vsts
 *
 * Because A5 uses the register-compute model, no separate UB tmp buffer is
 * needed — all intermediate values stay in vector registers.
 *
 * Supported types:
 *   src0, src1 : int32_t
 *   dst        : half
 *   deqScale   : float scalar
 */

#ifndef TADDDEQRELU_HPP
#define TADDDEQRELU_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include <pto/common/debug.h>

namespace pto {

constexpr float DEQ_SHIFT_RIGHT_17_BIT = 1.0f / 131072.0f;
constexpr float DEQ_SHIFT_LEFT_17_BIT = 131072.0f;

using __cce_simd::RoundRType;

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned DS, unsigned SS0, unsigned SS1>
__tf__ PTO_INTERNAL OP_NAME(TADDDEQRELU) OP_TYPE(element_wise) void TAddDeqRelu(
    typename TileDataDst::TileDType __out__ dstData, typename TileDataSrc0::TileDType __in__ src0Data,
    typename TileDataSrc1::TileDType __in__ src1Data, float deqScale, unsigned validRows, unsigned validCols)
{
    using DT = typename TileDataDst::DType;
    using ST = typename TileDataSrc0::DType;
    __ubuf__ DT* dstPtr = (__ubuf__ DT*)__cce_get_tile_ptr(dstData);
    __ubuf__ ST* src0Ptr = (__ubuf__ ST*)__cce_get_tile_ptr(src0Data);
    __ubuf__ ST* src1Ptr = (__ubuf__ ST*)__cce_get_tile_ptr(src1Data);

    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(ST);
    uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat);

    __VEC_SCOPE__
    {
        RegTensor<ST> vsrc0, vsrc1, vsum;
        RegTensor<float> vfloat;
        MaskReg preg;
        for (uint16_t i = 0; i < (uint16_t)validRows; ++i) {
            uint32_t sreg = (uint32_t)validCols;
            for (uint16_t j = 0; j < (uint16_t)repeatTimes; ++j) {
                preg = CreatePredicate<ST>(sreg);
                vlds(vsrc0, src0Ptr, i * SS0 + j * elementsPerRepeat, NORM);
                vlds(vsrc1, src1Ptr, i * SS1 + j * elementsPerRepeat, NORM);
                vadd(vsum, vsrc0, vsrc1, preg, MODE_ZEROING);
                vcvt(vfloat, vsum, preg, RoundRType());
                vmuls(vfloat, vfloat, static_cast<float>(DEQ_SHIFT_RIGHT_17_BIT), preg, MODE_ZEROING);
                vmuls(vfloat, vfloat, static_cast<float>(deqScale), preg, MODE_ZEROING);
                vmuls(vfloat, vfloat, static_cast<float>(DEQ_SHIFT_LEFT_17_BIT), preg, MODE_ZEROING);
                vmaxs(vfloat, vfloat, (float)0, preg, MODE_ZEROING);
                RegTensor<DT> vout;
                vcvt(vout, vfloat, preg, RoundRType(), RS_DISABLE, PART_EVEN);
                vsts(vout, dstPtr, i * DS + j * elementsPerRepeat, PK_B32, preg);
            }
        }
    }
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TAddDeqReluCheck(const TileDataDst& dst, const TileDataSrc0& src0, const TileDataSrc1& src1)
{
    static_assert(std::is_same<typename TileDataSrc0::DType, int32_t>::value, "Fix: TADDDEQRELU src0 must be int32_t.");
    static_assert(std::is_same<typename TileDataSrc1::DType, int32_t>::value, "Fix: TADDDEQRELU src1 must be int32_t.");
    static_assert(std::is_same<typename TileDataDst::DType, half>::value, "Fix: TADDDEQRELU dst must be half.");
    static_assert(
        TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
        "Fix: TADDDEQRELU only supports row major layout.");
    static_assert(
        TileDataDst::Loc == TileType::Vec && TileDataSrc0::Loc == TileType::Vec && TileDataSrc1::Loc == TileType::Vec,
        "Fix: TADDDEQRELU tiles must live in TileType::Vec.");
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    PTO_ASSERT(validRow > 0 && validCol > 0, "Fix: TADDDEQRELU valid rows and columns must be greater than 0.");
    PTO_ASSERT(
        src0.GetValidRow() == validRow && src0.GetValidCol() == validCol,
        "Fix: TADDDEQRELU src0 valid shape mismatch with dst.");
    PTO_ASSERT(
        src1.GetValidRow() == validRow && src1.GetValidCol() == validCol,
        "Fix: TADDDEQRELU src1 valid shape mismatch with dst.");
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TADDDEQRELU_IMPL(
    TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, float deqScale, TileDataTmp& tmp)
{
    TAddDeqReluCheck<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
    constexpr unsigned DS = TileDataDst::RowStride;
    constexpr unsigned SS0 = TileDataSrc0::RowStride;
    constexpr unsigned SS1 = TileDataSrc1::RowStride;
    TAddDeqRelu<TileDataDst, TileDataSrc0, TileDataSrc1, DS, SS0, SS1>(
        dst.data(), src0.data(), src1.data(), deqScale, dst.GetValidRow(), dst.GetValidCol());
}

} // namespace pto
#endif
