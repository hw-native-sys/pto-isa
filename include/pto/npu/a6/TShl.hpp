/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A6_TSHL_HPP
#define PTO_NPU_A6_TSHL_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a6/common.hpp>
#include <pto/npu/a6/utils.hpp>
#include <pto/npu/a5/TBinOp.hpp>
#include <pto/npu/a6/TShiftOp.hpp>
#include <pto/common/debug.h>

namespace pto {

// A6 TSHL: vector left shift with a SIGNED shift amount (a5-compatible).
//
// A6 vshl/vshr only accept an UNSIGNED shift-amount operand (vector_u8/u16/u32)
// and never reverse direction for negative values.  To reproduce a5 semantics,
// where a negative shift amount reverses the direction of the shift, the amount
// is interpreted as signed: lanes with src1 < 0 must perform a RIGHT shift by
// |src1| instead of a left shift.  Both results are computed (vshl and vshr on
// the magnitude) and the right-shift result is selected on the negative lanes.
//
//   negMask  = (src1 < 0)                 // vcmp_lt on the signed reinterpretaion
//   absSrc1  = |src1|                      // vsub(0,src1) under negMask + vsel
//   tmpL     = src0 << |src1|              // vshl (unsigned amount)
//   tmpR     = src0 >> |src1|              // vshr (arith if src0 signed, logic if unsigned)
//   dst      = negMask ? tmpR : tmpL       // vsel
template <typename T>
struct ShlOp {
    PTO_INTERNAL static void BinInstr(
        RegTensor<T>& reg_dst, RegTensor<T>& reg_src0, RegTensor<T>& reg_src1, MaskReg& preg)
    {
        using SS = std::conditional_t<sizeof(T) == 1, int8_t, std::conditional_t<sizeof(T) == 2, int16_t, int32_t>>;
        using US = std::conditional_t<sizeof(T) == 1, uint8_t, std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>>;
        RegTensor<SS>& sSrc1 = (RegTensor<SS>&)reg_src1;
        RegTensor<SS> zeroVec, negSrc1, absSrc1S;
        RegTensor<US>& absSrc1 = (RegTensor<US>&)absSrc1S;
        RegTensor<T> tmpL, tmpR;
        MaskReg negMask;
        vbr(zeroVec, (SS)0);
        vcmp_lt(negMask, sSrc1, zeroVec, preg);
        vsub(negSrc1, zeroVec, sSrc1, negMask, MODE_ZEROING);
        vsel(absSrc1S, negSrc1, sSrc1, negMask);
        vshl(tmpL, reg_src0, absSrc1, preg, MODE_ZEROING);
        vshr(tmpR, reg_src0, absSrc1, preg, MODE_ZEROING);
        vsel(reg_dst, tmpR, tmpL, negMask);
    }
};

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned ElementsPerRepeat,
    unsigned BlockSizeElem>
__tf__ PTO_INTERNAL OP_NAME(TSHL) OP_TYPE(element_wise) void TShl(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc0::TileDType __in__ src0,
    typename TileDataSrc1::TileDType __in__ src1, unsigned validRows, unsigned validCols,
    VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    __ubuf__ T* src1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src1);
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        Int64ShiftBinary<false, T, TileDataDst::Cols, TileDataSrc0::Cols, TileDataSrc1::Cols>(
            dstPtr, src0Ptr, src1Ptr, validRows, validCols);
    } else {
        BinaryInstr<ShlOp<T>, TileDataDst, TileDataSrc0, TileDataSrc1, ElementsPerRepeat, BlockSizeElem>(
            dstPtr, src0Ptr, src1Ptr, validRows, validCols, version);
    }
    return;
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TShlCheck(const TileDataDst& dst, const TileDataSrc0& src0, const TileDataSrc1& src1)
{
    using T = typename TileDataDst::DType;
    static_assert(
        std::is_same<T, typename TileDataSrc0::DType>::value && std::is_same<T, typename TileDataSrc1::DType>::value,
        "Fix: TSHL has invalid data type.");
    static_assert(
        std::is_same<T, uint64_t>::value || std::is_same<T, int64_t>::value || std::is_same<T, uint8_t>::value ||
            std::is_same<T, int8_t>::value || std::is_same<T, uint16_t>::value || std::is_same<T, int16_t>::value ||
            std::is_same<T, uint32_t>::value || std::is_same<T, int32_t>::value,
        "Fix: TSHL has invalid data type.");
    static_assert(
        TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
        "Fix: TSHL only support row major layout.");
    static_assert(
        std::is_same_v<T, typename TileDataSrc0::DType> && std::is_same_v<T, typename TileDataSrc1::DType>,
        "Fix: TSHL input tile src0, src1 and dst tile data type mismatch.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(
        src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
        "Fix: TSHL input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(
        src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
        "Fix: TSHL input tile src1 valid shape mismatch with output tile dst shape.");
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TSHL_IMPL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1)
{
    using T = typename TileDataDst::DType;
    TShlCheck<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    TShl<TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem>(
        dst.data(), src0.data(), src1.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto
#endif
