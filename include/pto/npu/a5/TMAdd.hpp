/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMADD_HPP
#define TMADD_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include <pto/npu/a5/TTernOp.hpp>

namespace pto {

template <typename T>
struct MAddOp {
    PTO_INTERNAL static void TernInstr(
        RegTensor<T>& reg_dst, RegTensor<T>& reg_src0, RegTensor<T>& reg_src1, MaskReg& preg)
    {
        if constexpr (std::is_same_v<T, half>) {
            vmul(reg_dst, reg_dst, reg_src0, preg, MODE_ZEROING);
            vadd(reg_dst, reg_dst, reg_src1, preg, MODE_ZEROING);
        } else {
            vmadd(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
        }
    }
};

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned ElementsPerRepeat,
    unsigned BlockSizeElem>
__tf__ PTO_INTERNAL OP_NAME(TMADD) OP_TYPE(element_wise) void TMAdd(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc0::TileDType __in__ src0,
    typename TileDataSrc1::TileDType __in__ src1, unsigned validRows, unsigned validCols,
    VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    __ubuf__ T* src1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src1);

    TernaryInstr<MAddOp<T>, TileDataDst, TileDataSrc0, TileDataSrc1, ElementsPerRepeat, BlockSizeElem>(
        dstPtr, src0Ptr, src1Ptr, validRows, validCols, version);
    return;
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TMAddCheck(const TileDataDst& dst, const TileDataSrc0& src0, const TileDataSrc1& src1)
{
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(
        src0.GetValidRow() == validRows && src0.GetValidCol() == validCols && src1.GetValidRow() == validRows &&
            src1.GetValidCol() == validCols,
        "Fix: TMADD input tile src0 valid shape mismatch with output tile dst shape.");
    using T = typename TileDataDst::DType;
    static_assert(
        std::is_same_v<T, typename TileDataSrc0::DType> && std::is_same_v<T, typename TileDataSrc1::DType>,
        "Fix: TMADD the data type of dst must be consistent with of src0 and src1.");
    static_assert(
        TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
        "Fix: TMADD only support row major layout.");
    static_assert(
        std::is_same_v<T, half> || std::is_same_v<T, float16_t> || std::is_same_v<T, float> ||
            std::is_same_v<T, float32_t>,
        "Fix: TMADD has invalid data type.");
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TMADD_IMPL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1)
{
    using T = typename TileDataDst::DType;
    TMAddCheck<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);

    TMAdd<TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem>(
        dst.data(), src0.data(), src1.data(), dst.GetValidRow(), dst.GetValidCol());
}

#if defined(PTO_NPU_ARCH_A6)
// A6 sign variants: fused multiply with per-term negation (swap form).
//   NegProduct=false, NegAddend=false → dst = src0*dst + src1 (same as TMADD)
//   NegProduct=true,  NegAddend=false → dst = -(src0*dst) + src1 (vmsub)
//   NegProduct=true,  NegAddend=true  → dst = -(src0*dst) - src1 (vnmadd)
//   NegProduct=false, NegAddend=true  → dst = src0*dst - src1   (vnmsub)
template <typename T, bool NegProduct, bool NegAddend>
struct MAddSignOp {
    PTO_INTERNAL static void TernInstr(
        RegTensor<T>& reg_dst, RegTensor<T>& reg_src0, RegTensor<T>& reg_src1, MaskReg& preg)
    {
        if constexpr (NegProduct && !NegAddend) {
            vmsub(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
        } else if constexpr (NegProduct && NegAddend) {
            vnmadd(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
        } else if constexpr (!NegProduct && NegAddend) {
            vnmsub(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
        } else {
            vmadd(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
        }
    }
};

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, unsigned ElementsPerRepeat,
    unsigned BlockSizeElem, bool NegProduct, bool NegAddend>
__tf__ PTO_INTERNAL OP_NAME(TMADD) OP_TYPE(element_wise) void TMAddSign(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc0::TileDType __in__ src0,
    typename TileDataSrc1::TileDType __in__ src1, unsigned validRows, unsigned validCols,
    VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    __ubuf__ T* src1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src1);

    TernaryInstr<
        MAddSignOp<T, NegProduct, NegAddend>, TileDataDst, TileDataSrc0, TileDataSrc1, ElementsPerRepeat,
        BlockSizeElem>(dstPtr, src0Ptr, src1Ptr, validRows, validCols, version);
    return;
}

// Sign-variant overload: TMADD_IMPL<NegProduct, NegAddend>(dst, src0, src1).
// Bool-first template head keeps it unambiguous with the deduced 3-type form above.
template <bool NegProduct, bool NegAddend, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TMADD_IMPL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1)
{
    static_assert(
        std::is_same_v<typename TileDataDst::DType, half> || std::is_same_v<typename TileDataDst::DType, bfloat16_t> ||
            std::is_same_v<typename TileDataDst::DType, float>,
        "TMADD sign variants require float types");
    using T = typename TileDataDst::DType;
    TMAddCheck<TileDataDst, TileDataSrc0, TileDataSrc1>(dst, src0, src1);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);

    TMAddSign<TileDataDst, TileDataSrc0, TileDataSrc1, elementsPerRepeat, blockSizeElem, NegProduct, NegAddend>(
        dst.data(), src0.data(), src1.data(), dst.GetValidRow(), dst.GetValidCol());
}
#endif

} // namespace pto

#endif
