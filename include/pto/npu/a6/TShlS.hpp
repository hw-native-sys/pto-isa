/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A6_TSHLS_HPP
#define PTO_NPU_A6_TSHLS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a6/common.hpp>
#include <pto/npu/a6/utils.hpp>
#include <pto/npu/a6/TShiftOp.hpp>
#include <pto/common/debug.h>

namespace pto {

// A6 TSHLS: scalar left shift with a SIGNED shift amount (a5-compatible).
//
// A6 vshls/vshrs only accept an UNSIGNED scalar amount (uint16_t for 8/16-bit
// data, uint32_t for 32-bit data) and never reverse direction.  A negative
// amount must reverse the shift, which a6 cannot do natively.
//
// Strategy: the sign is resolved OUTSIDE __VEC_SCOPE, in the __tf__ TShlS
// function, where runtime branching is fine.  There we take |src1| and pick one
// of two compile-time variants of the Op via a template bool `kIsNeg`.  Each
// variant's BinSInstr then emits exactly ONE shift intrinsic (selected by
// `if constexpr`, so no runtime branch inlines into __VEC_SCOPE).
template <typename T, bool kIsNeg>
struct ShlSOp {
    static constexpr bool isDynFunc = false;
    PTO_INTERNAL static void BinSInstr(RegTensor<T>& reg_dst, RegTensor<T>& reg_src0, T src1, MaskReg& preg)
    {
        // `src1` arrives already as the non-negative magnitude (resolved in __tf__).
        using ScT = std::conditional_t<sizeof(T) <= 2, uint16_t, uint32_t>;
        if constexpr (kIsNeg) {
            // a negative amount reverses a left shift into a right shift by |src1|
            vshrs(reg_dst, reg_src0, (ScT)src1, preg, MODE_ZEROING);
        } else {
            vshls(reg_dst, reg_src0, (ScT)src1, preg, MODE_ZEROING);
        }
    }
};

template <
    typename TileDataDst, typename TileDataSrc, unsigned elementsPerRepeat, unsigned blockSizeElem,
    unsigned dstRowStride, unsigned src0RowStride>
__tf__ PTO_INTERNAL OP_NAME(TSHLS) OP_TYPE(element_wise) void TShlS(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src0,
    typename TileDataSrc::DType src1, unsigned kValidRows, unsigned kValidCols,
    VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        Int64ShiftScalar<false, T, TileDataDst::Cols, TileDataSrc::Cols>(dstPtr, src0Ptr, src1, kValidRows, kValidCols);
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
                ShlSOp<T, true>, TileDataDst, TileDataSrc, T, elementsPerRepeat, blockSizeElem, dstRowStride,
                src0RowStride>(dstPtr, src0Ptr, (T)absScalar, kValidRows, kValidCols, version);
        } else {
            BinaryInstr<
                ShlSOp<T, false>, TileDataDst, TileDataSrc, T, elementsPerRepeat, blockSizeElem, dstRowStride,
                src0RowStride>(dstPtr, src0Ptr, (T)absScalar, kValidRows, kValidCols, version);
        }
    }
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TSHLS_IMPL(TileDataDst& dst, TileDataSrc& src0, typename TileDataDst::DType src1)
{
    using T = typename TileDataDst::DType;
    static_assert(
        std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>::value || std::is_same<T, int32_t>::value ||
            std::is_same<T, int16_t>::value || std::is_same<T, int8_t>::value || std::is_same<T, uint32_t>::value ||
            std::is_same<T, uint16_t>::value || std::is_same<T, uint8_t>::value,
        "TSHLS: Invalid data type");
    static_assert(TileDataDst::Loc == TileType::Vec, "TileType of dst tiles must be TileType::Vec.");
    static_assert(
        TileDataDst::ValidCol <= TileDataDst::Cols,
        "Number of valid columns must not be greater than number of tile columns.");
    static_assert(
        TileDataDst::ValidRow <= TileDataDst::Rows,
        "Number of valid rows must not be greater than number of tile rows.");
    static_assert(TileDataSrc::Loc == TileType::Vec, "TileType of src tiles must be TileType::Vec.");
    static_assert(
        TileDataSrc::ValidCol <= TileDataSrc::Cols,
        "Number of valid columns must not be greater than number of tile columns.");
    static_assert(
        TileDataSrc::ValidRow <= TileDataSrc::Rows,
        "Number of valid rows must not be greater than number of tile rows.");

    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);
    constexpr unsigned dstRowStride = TileDataDst::RowStride;
    constexpr unsigned src0RowStride = TileDataSrc::RowStride;

    PTO_ASSERT(src0.GetValidCol() == dst.GetValidCol(), "Number of columns of src and dst must be the same.");
    PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    TShlS<TileDataDst, TileDataSrc, elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride>(
        dst.data(), src0.data(), src1, validRow, validCol);
}
} // namespace pto
#endif
