/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A6_TSHRS_HPP
#define PTO_NPU_A6_TSHRS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a6/common.hpp>
#include <pto/npu/a6/utils.hpp>
#include <pto/npu/a5/TBinSOp.hpp>
#include <pto/npu/a6/TShiftOp.hpp>
#include <pto/common/debug.h>

namespace pto {

// A6 TSHRS: scalar right shift with a SIGNED shift amount (a5-compatible).
//
// Symmetric to TSHLS: a negative amount reverses the direction, so a negative
// scalar emits a vshls by |src1| instead of a vshrs.
//
// Strategy: the sign is resolved OUTSIDE __VEC_SCOPE, in the __tf__ TShrS
// function, where runtime branching is fine.  There we take |src1| and pick one
// of two compile-time variants of the Op via a template bool `kIsNeg`.  Each
// variant's BinSInstr then emits exactly ONE shift intrinsic (selected by
// `if constexpr`, so no runtime branch inlines into __VEC_SCOPE).
template <typename T, bool kIsNeg>
struct ShrSOp {
    static constexpr bool isDynFunc = false;
    PTO_INTERNAL static void BinSInstr(RegTensor<T>& reg_dst, RegTensor<T>& reg_src0, T src1, MaskReg& preg)
    {
        // `src1` arrives already as the non-negative magnitude (resolved in __tf__).
        using ScT = std::conditional_t<sizeof(T) <= 2, uint16_t, uint32_t>;
        if constexpr (kIsNeg) {
            // a negative amount reverses a right shift into a left shift by |src1|
            vshls(reg_dst, reg_src0, (ScT)src1, preg, MODE_ZEROING);
        } else {
            vshrs(reg_dst, reg_src0, (ScT)src1, preg, MODE_ZEROING);
        }
    }
};

template <
    typename TileDataDst, typename TileDataSrc, unsigned elementsPerRepeat, unsigned blockSizeElem,
    unsigned dstRowStride, unsigned src0RowStride>
__tf__ PTO_INTERNAL OP_NAME(TSHRS) OP_TYPE(element_wise) void TShrS(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src0,
    typename TileDataSrc::DType src1, unsigned kValidRows, unsigned kValidCols,
    VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        Int64ShiftScalar<true, T, TileDataDst::Cols, TileDataSrc::Cols>(dstPtr, src0Ptr, src1, kValidRows, kValidCols);
    } else {
        // Resolve the sign of the scalar amount here, in __tf__ (outside __VEC_SCOPE),
        // where a runtime branch is idiomatic.  Feed the result into the template bool
        // kIsNeg by instantiating both variants and selecting one at runtime.
        using SS = std::conditional_t<sizeof(T) == 1, int8_t, std::conditional_t<sizeof(T) == 2, int16_t, int32_t>>;
        using US = std::conditional_t<sizeof(T) == 1, uint8_t, std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>>;
        using ScT = std::conditional_t<sizeof(T) <= 2, uint16_t, uint32_t>;
        SS signedSrc1 = (SS)src1;
        bool isNeg = signedSrc1 < 0;
        ScT absScalar = isNeg ? (ScT)(-signedSrc1) : (ScT)(US)signedSrc1;
        if (isNeg) {
            BinaryInstr<
                ShrSOp<T, true>, TileDataDst, TileDataSrc, T, elementsPerRepeat, blockSizeElem, dstRowStride,
                src0RowStride>(dstPtr, src0Ptr, (T)absScalar, kValidRows, kValidCols, version);
        } else {
            BinaryInstr<
                ShrSOp<T, false>, TileDataDst, TileDataSrc, T, elementsPerRepeat, blockSizeElem, dstRowStride,
                src0RowStride>(dstPtr, src0Ptr, (T)absScalar, kValidRows, kValidCols, version);
        }
    }
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TSHRS_IMPL(TileDataDst& dst, TileDataSrc& src0, typename TileDataDst::DType src1)
{
    using T = typename TileDataDst::DType;
    static_assert(
        std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>::value || std::is_same<T, int32_t>::value ||
            std::is_same<T, int16_t>::value || std::is_same<T, int8_t>::value || std::is_same<T, uint32_t>::value ||
            std::is_same<T, uint16_t>::value || std::is_same<T, uint8_t>::value,
        "TSHRS: Invalid data type");
    static_assert(
        (TileDataDst::Loc == TileType::Vec) && (TileDataSrc::Loc == TileType::Vec),
        "TileType of dst and src tiles must be TileType::Vec.");
    static_assert(
        (TileDataDst::ValidCol <= TileDataDst::Cols) && (TileDataDst::ValidRow <= TileDataDst::Rows) &&
            (TileDataSrc::ValidCol <= TileDataSrc::Cols) && (TileDataSrc::ValidRow <= TileDataSrc::Rows),
        "Number of valid columns and rows must not be greater than number of tile columns and rows.");

    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned src0RowStride = TileDataSrc::RowStride;

    PTO_ASSERT(src0.GetValidCol() == dst.GetValidCol(), "Number of columns of src and dst must be the same.");
    PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    TShrS<TileDataDst, TileDataSrc, elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride>(
        dst.data(), src0.data(), src1, validRow, validCol);
}
} // namespace pto
#endif
