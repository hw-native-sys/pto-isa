/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TREM_HPP
#define TREM_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include <pto/npu/a5/TBinOp.hpp>
#include <pto/common/debug.h>
#include "custom/TFmodRemHp.hpp"
#include "TDiv.hpp"

namespace pto {

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
template <typename T>
PTO_INTERNAL void Int64RemRegs(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& mask)
{
    vector_s32 qLow, qHigh, productLow, productHigh;
    if constexpr (std::is_same_v<T, int64_t>)
        Int64DivSignedRegs(qLow, qHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
    else
        Int64DivUnsignedRegs(qLow, qHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
    Int64MulRegs(productLow, productHigh, qLow, qHigh, rhsLow, rhsHigh, mask);
    Int64SubRegs(dstLow, dstHigh, lhsLow, lhsHigh, productLow, productHigh, mask);
    vector_s32 zeroLow, zeroHigh;
    Int64DuplicateRegs(zeroLow, zeroHigh, 0, 0);
    MaskReg zeroMask;
    Int64CompareEqRegs(zeroMask, rhsLow, rhsHigh, zeroLow, zeroHigh, mask);
    Int64SelectRegs(dstLow, dstHigh, zeroLow, zeroHigh, dstLow, dstHigh, zeroMask);
}

template <typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols, bool FullLoad>
PTO_INTERNAL void Int64RemRepeat(
    __ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint16_t row, uint32_t colOffset, MaskReg& mask)
{
    vector_s32 dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh;
    Int64LoadRegs<T, Src0Cols, FullLoad>(lhsLow, lhsHigh, src0, row, colOffset);
    Int64LoadRegs<T, Src1Cols, FullLoad>(rhsLow, rhsHigh, src1, row, colOffset);
    Int64RemRegs<T>(dstLow, dstHigh, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
    Int64StoreRegs<T, DstCols>(dstLow, dstHigh, dst, row, colOffset, mask);
}

template <typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Rem(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL * 2 / sizeof(T);
    uint16_t colRepeats = CeilDivision(validCols, elementsPerRepeat);
    uint16_t fullRepeats = Int64FullLoadRepeats<Int64MinCols<Src0Cols, Src1Cols>, elementsPerRepeat>(colRepeats);
    __VEC_SCOPE__
    {
        vector_s32 dl, dh, al, ah, bl, bh, half0, half1;
        MaskReg lowMask, highMask;
        uint16_t rows = validRows;

        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t sreg = validCols;

            for (uint16_t colRepeat = 0; colRepeat < fullRepeats; ++colRepeat) {
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                Int64LoadBounded<Src0Cols, true>(
                    al, ah, (__ubuf__ int32_t*)src0 + (row * Src0Cols + colOffset) * 2, colOffset);
                Int64LoadBounded<Src1Cols, true>(
                    bl, bh, (__ubuf__ int32_t*)src1 + (row * Src1Cols + colOffset) * 2, colOffset);
                MaskReg preg = CreatePredicate<uint32_t>(sreg);
                Int64RemRegs<T>(dl, dh, al, ah, bl, bh, preg);
                pintlv_b32(lowMask, highMask, preg, preg);
                vintlv(half0, half1, dl, dh);
                vsts(half0, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, NORM_B32, lowMask);
                vsts(
                    half1, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2 + CCE_VL / sizeof(int32_t), 0,
                    NORM_B32, highMask);
            }
            for (uint16_t colRepeat = fullRepeats; colRepeat < colRepeats; ++colRepeat) {
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                Int64LoadBounded<Src0Cols, false>(
                    al, ah, (__ubuf__ int32_t*)src0 + (row * Src0Cols + colOffset) * 2, colOffset);
                Int64LoadBounded<Src1Cols, false>(
                    bl, bh, (__ubuf__ int32_t*)src1 + (row * Src1Cols + colOffset) * 2, colOffset);
                MaskReg preg = CreatePredicate<uint32_t>(sreg);
                Int64RemRegs<T>(dl, dh, al, ah, bl, bh, preg);
                pintlv_b32(lowMask, highMask, preg, preg);
                vintlv(half0, half1, dl, dh);
                vsts(half0, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, NORM_B32, lowMask);
                vsts(
                    half1, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2 + CCE_VL / sizeof(int32_t), 0,
                    NORM_B32, highMask);
            }
        }
    }
}

template <typename T, unsigned DstCols>
PTO_INTERNAL void Int64Zero(__ubuf__ T* dst, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = CCE_VL * 2 / sizeof(T);
    __VEC_SCOPE__
    {
        vector_s32 zero, half0, half1;
        MaskReg lowMask, highMask;
        vbr(zero, 0);
        uint16_t rows = validRows;
        uint16_t colRepeats = CeilDivision(validCols, elementsPerRepeat);
        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t sreg = validCols;
            for (uint16_t colRepeat = 0; colRepeat < colRepeats; ++colRepeat) {
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                MaskReg preg = CreatePredicate<uint32_t>(sreg);
                pintlv_b32(lowMask, highMask, preg, preg);
                vintlv(half0, half1, zero, zero);
                vsts(half0, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, NORM_B32, lowMask);
                vsts(
                    half1, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2 + CCE_VL / sizeof(int32_t), 0,
                    NORM_B32, highMask);
            }
        }
    }
}

template <typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RemScalar(__ubuf__ T* dst, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols)
{
    if constexpr (std::is_same_v<T, uint64_t>) {
        if (scalar == 0) {
            Int64Zero<T, DstCols>(dst, validRows, validCols);
            return;
        }
    }
    constexpr unsigned elementsPerRepeat = CCE_VL * 2 / sizeof(T);
    uint16_t colRepeats = CeilDivision(validCols, elementsPerRepeat);
    uint16_t fullRepeats = Int64FullLoadRepeats<SrcCols, elementsPerRepeat>(colRepeats);
    __VEC_SCOPE__
    {
        vector_s32 dl, dh, al, ah, bl, bh, half0, half1;
        MaskReg lowMask, highMask;
        uint64_t bits = static_cast<uint64_t>(scalar);
        Int64DuplicateRegs(bl, bh, static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32));
        uint16_t rows = validRows;

        for (uint16_t row = 0; row < rows; ++row) {
            uint32_t sreg = validCols;

            for (uint16_t colRepeat = 0; colRepeat < fullRepeats; ++colRepeat) {
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                Int64LoadBounded<SrcCols, true>(
                    al, ah, (__ubuf__ int32_t*)src + (row * SrcCols + colOffset) * 2, colOffset);
                MaskReg preg = CreatePredicate<uint32_t>(sreg);
                Int64RemRegs<T>(dl, dh, al, ah, bl, bh, preg);
                pintlv_b32(lowMask, highMask, preg, preg);
                vintlv(half0, half1, dl, dh);
                vsts(half0, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, NORM_B32, lowMask);
                vsts(
                    half1, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2 + CCE_VL / sizeof(int32_t), 0,
                    NORM_B32, highMask);
            }
            for (uint16_t colRepeat = fullRepeats; colRepeat < colRepeats; ++colRepeat) {
                uint32_t colOffset = colRepeat * elementsPerRepeat;
                Int64LoadBounded<SrcCols, false>(
                    al, ah, (__ubuf__ int32_t*)src + (row * SrcCols + colOffset) * 2, colOffset);
                MaskReg preg = CreatePredicate<uint32_t>(sreg);
                Int64RemRegs<T>(dl, dh, al, ah, bl, bh, preg);
                pintlv_b32(lowMask, highMask, preg, preg);
                vintlv(half0, half1, dl, dh);
                vsts(half0, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2, 0, NORM_B32, lowMask);
                vsts(
                    half1, (__ubuf__ int32_t*)dst + (row * DstCols + colOffset) * 2 + CCE_VL / sizeof(int32_t), 0,
                    NORM_B32, highMask);
            }
        }
    }
}
#else
// Declaration-only stubs for kirin9030/kirinX90 (no 64-bit intrinsics).
// See TBinOp.hpp for details.
template <typename T, unsigned DstCols, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Rem(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows, unsigned validCols);

template <typename T, unsigned DstCols, unsigned SrcCols>
PTO_INTERNAL void Int64RemScalar(__ubuf__ T* dst, __ubuf__ T* src, T scalar, unsigned validRows, unsigned validCols);
#endif

template <RemAlgorithm PrecisionType, typename T>
struct RemOp {
    PTO_INTERNAL static void BinInstr(
        RegTensor<T>& reg_dst, RegTensor<T>& reg_src0, RegTensor<T>& reg_src1, MaskReg& preg)
    {
        if constexpr (PrecisionType == RemAlgorithm::HIGH_PRECISION && std::is_same<T, float>::value) {
            TFmodRemHP<false>(reg_dst, reg_src0, reg_src1, preg);
        } else if constexpr (std::is_same<T, float>::value) {
            MaskReg sign_diff_mask;
            RegTensor<T> reg_tmp;
            vdiv(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
            vtrc(reg_dst, reg_dst, ROUND_F, preg);
            vmul(reg_dst, reg_dst, reg_src1, preg, MODE_ZEROING);
            vsub(reg_dst, reg_src0, reg_dst, preg, MODE_ZEROING);

            vmul(reg_tmp, reg_src1, reg_dst, preg, MODE_ZEROING);
            vcmps_lt(sign_diff_mask, reg_tmp, 0.0f, preg);
            vadd(reg_tmp, reg_dst, reg_src1, sign_diff_mask, MODE_ZEROING);
            vsel(reg_dst, reg_tmp, reg_dst, sign_diff_mask);
        } else if constexpr (std::is_same<T, half>::value) {
            RegTensor<float> reg_tmp_even0, reg_tmp_even1, reg_tmp_even2, reg_tmp_odd0, reg_tmp_odd1, reg_tmp_odd2;
            RegTensor<T> reg_dst_even, reg_dst_odd, reg_tmp;
            MaskReg sign_diff_mask;
            vcvt(reg_tmp_even0, reg_src0, preg, PART_EVEN);
            vcvt(reg_tmp_even1, reg_src1, preg, PART_EVEN);
            vcvt(reg_tmp_odd0, reg_src0, preg, PART_ODD);
            vcvt(reg_tmp_odd1, reg_src1, preg, PART_ODD);

            vdiv(reg_tmp_even2, reg_tmp_even0, reg_tmp_even1, preg, MODE_ZEROING);
            vdiv(reg_tmp_odd2, reg_tmp_odd0, reg_tmp_odd1, preg, MODE_ZEROING);

            vtrc(reg_tmp_even2, reg_tmp_even2, ROUND_F, preg);
            vtrc(reg_tmp_odd2, reg_tmp_odd2, ROUND_F, preg);

            vmul(reg_tmp_even2, reg_tmp_even2, reg_tmp_even1, preg, MODE_ZEROING);
            vmul(reg_tmp_odd2, reg_tmp_odd2, reg_tmp_odd1, preg, MODE_ZEROING);

            vsub(reg_tmp_even2, reg_tmp_even0, reg_tmp_even2, preg, MODE_ZEROING);
            vsub(reg_tmp_odd2, reg_tmp_odd0, reg_tmp_odd2, preg, MODE_ZEROING);

            vcvt(reg_dst_even, reg_tmp_even2, preg, ROUND_Z, RS_ENABLE, PART_EVEN);
            vcvt(reg_dst_odd, reg_tmp_odd2, preg, ROUND_Z, RS_ENABLE, PART_ODD);

            vor(reg_dst, reg_dst_even, reg_dst_odd, preg);

            vmul(reg_tmp, reg_src1, reg_dst, preg, MODE_ZEROING);
            vcmps_lt(sign_diff_mask, reg_tmp, 0.0f, preg);
            vadd(reg_tmp, reg_dst, reg_src1, sign_diff_mask, MODE_ZEROING);
            vsel(reg_dst, reg_tmp, reg_dst, sign_diff_mask);
        } else {
            vmod(reg_dst, reg_src0, reg_src1, preg, MODE_ZEROING);
        }
    }
};

template <
    typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp,
    unsigned ElementsPerRepeat, unsigned BlockSizeElem, auto PrecisionType = RemAlgorithm::DEFAULT>
__tf__ PTO_INTERNAL OP_NAME(TREM) OP_TYPE(element_wise) void TRem(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc0::TileDType __in__ src0,
    typename TileDataSrc1::TileDType __in__ src1, typename TileDataTmp::TileDType __in__ tmp, unsigned validRows,
    unsigned validCols, VFImplKind version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataDst::DType;
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    __ubuf__ T* src1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src1);
    // Note: tmp parameter is not used in a5 implementation (no sign correction needed)
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        Int64Rem<T, TileDataDst::Cols, TileDataSrc0::Cols, TileDataSrc1::Cols>(
            dstPtr, src0Ptr, src1Ptr, validRows, validCols);
    } else {
        BinaryInstr<RemOp<PrecisionType, T>, TileDataDst, TileDataSrc0, TileDataSrc1, ElementsPerRepeat, BlockSizeElem>(
            dstPtr, src0Ptr, src1Ptr, validRows, validCols, version);
    }
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
PTO_INTERNAL void TRemCheck(
    const TileDataDst& dst, const TileDataSrc0& src0, const TileDataSrc1& src1, const TileDataTmp& tmp)
{
    using T = typename TileDataDst::DType;
    static_assert(
        std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>::value || std::is_same<T, half>::value ||
            std::is_same<T, float>::value || std::is_same<T, uint16_t>::value || std::is_same<T, int16_t>::value ||
            std::is_same<T, uint32_t>::value || std::is_same<T, int32_t>::value,
        "Fix: TREM has invalid data type.");
    static_assert(
        TileDataDst::isRowMajor && TileDataSrc0::isRowMajor && TileDataSrc1::isRowMajor,
        "Fix: TREM only support row major layout.");
    static_assert(
        std::is_same<T, typename TileDataSrc0::DType>::value && std::is_same<T, typename TileDataSrc1::DType>::value,
        "Fix: TREM input tile src0, src1 and dst tile data type mismatch.");
    unsigned validRows = dst.GetValidRow();
    unsigned validCols = dst.GetValidCol();
    PTO_ASSERT(
        src0.GetValidRow() == validRows && src0.GetValidCol() == validCols,
        "Fix: TREM input tile src0 valid shape mismatch with output tile dst shape.");
    PTO_ASSERT(
        src1.GetValidRow() == validRows && src1.GetValidCol() == validCols,
        "Fix: TREM input tile src1 valid shape mismatch with output tile dst shape.");
}

template <
    auto PrecisionType = RemAlgorithm::DEFAULT, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
    typename TileDataTmp>
PTO_INTERNAL void TREM_IMPL(TileDataDst& dst, TileDataSrc0& src0, TileDataSrc1& src1, TileDataTmp& tmp)
{
    using T = typename TileDataDst::DType;
    TRemCheck<TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp>(dst, src0, src1, tmp);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = CCE_VL / sizeof(T);

    TRem<TileDataDst, TileDataSrc0, TileDataSrc1, TileDataTmp, elementsPerRepeat, blockSizeElem, PrecisionType>(
        dst.data(), src0.data(), src1.data(), tmp.data(), dst.GetValidRow(), dst.GetValidCol());
}
} // namespace pto
#endif
