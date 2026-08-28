/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCMP_HPP
#define TCMP_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "TBinOp.hpp"
#include "utils.hpp"

namespace pto {

#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
template <CmpMode Mode, typename T>
PTO_INTERNAL void Int64CompareRelationalRegs(
    MaskReg& dst, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow, vector_s32& rhsHigh, MaskReg& mask)
{
    MaskReg lowEq, highCmp, lowCmp;
    vcmp_eq(lowEq, lhsHigh, rhsHigh, mask);
    if constexpr (Mode == CmpMode::LT || Mode == CmpMode::LE) {
        if constexpr (Mode == CmpMode::LT)
            vcmp_lt(lowCmp, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
        else
            vcmp_le(lowCmp, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
        if constexpr (std::is_same_v<T, int64_t>) {
            vcmp_lt(highCmp, lhsHigh, rhsHigh, mask);
        } else {
            vcmp_lt(highCmp, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);
        }
    } else {
        if constexpr (Mode == CmpMode::GT)
            vcmp_gt(lowCmp, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
        else
            vcmp_ge(lowCmp, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
        if constexpr (std::is_same_v<T, int64_t>) {
            vcmp_gt(highCmp, lhsHigh, rhsHigh, mask);
        } else {
            vcmp_gt(highCmp, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);
        }
    }
    psel(dst, lowCmp, highCmp, lowEq);
}

PTO_INTERNAL void Int64CompareEqualRegs(
    MaskReg& dst, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow, vector_s32& rhsHigh, MaskReg& mask)
{
    MaskReg lowEq;
    vcmp_eq(lowEq, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
    vcmp_eq(dst, lhsHigh, rhsHigh, lowEq);
}

PTO_INTERNAL void Int64CompareNotEqualRegs(
    MaskReg& dst, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow, vector_s32& rhsHigh, MaskReg& mask)
{
    MaskReg lowNe, highNe;
    vcmp_ne(lowNe, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
    vcmp_ne(highNe, lhsHigh, rhsHigh, mask);
    por(dst, lowNe, highNe, mask);
}

template <CmpMode Mode, typename T>
PTO_INTERNAL void Int64CompareRegs(
    MaskReg& dst, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow, vector_s32& rhsHigh, MaskReg& mask)
{
    if constexpr (Mode == CmpMode::EQ) {
        Int64CompareEqualRegs(dst, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
    } else if constexpr (Mode == CmpMode::NE) {
        Int64CompareNotEqualRegs(dst, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
    } else {
        Int64CompareRelationalRegs<Mode, T>(dst, lhsLow, lhsHigh, rhsLow, rhsHigh, mask);
    }
}

template <CmpMode Mode, typename T, unsigned DstRowBytes, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64CompareMode(
    __ubuf__ uint8_t* dst, __ubuf__ T* src0, __ubuf__ T* src1, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elementsPerRepeat = 64; // vlds+DINTLV_B32 loads 64 int64 elements (512B)
    uint16_t repeatTimes = CeilDivision(validCols, elementsPerRepeat) + 1; // +1 to ensure even pair
    __VEC_SCOPE__
    {
        vector_s32 lhsLow0, lhsHigh0, rhsLow0, rhsHigh0;
        vector_s32 lhsLow1, lhsHigh1, rhsLow1, rhsHigh1;
        uint16_t rows = validRows;
        constexpr int32_t dstRepeatStride = 2 * elementsPerRepeat / 32;
        for (uint16_t row = 0; row < rows; ++row) {
            __ubuf__ uint32_t* rowDst = (__ubuf__ uint32_t*)(dst + row * DstRowBytes);
            uint32_t sreg = validCols;
            for (uint16_t j = 0; j < (uint16_t)(repeatTimes / 2); ++j) {
                MaskReg preg;
                MaskReg result0, result1, dstReg, tmpMask;
                // batch 0
                uint32_t colOffset0 = j * 2 * elementsPerRepeat;
                vlds(lhsLow0, lhsHigh0, (__ubuf__ int32_t*)src0, (row * Src0Cols + colOffset0) * 2, DINTLV_B32);
                vlds(rhsLow0, rhsHigh0, (__ubuf__ int32_t*)src1, (row * Src1Cols + colOffset0) * 2, DINTLV_B32);
                preg = plt_b32(sreg, POST_UPDATE);

                Int64CompareRegs<Mode, T>(result0, lhsLow0, lhsHigh0, rhsLow0, rhsHigh0, preg);
                // batch 1
                uint32_t colOffset1 = (j * 2 + 1) * elementsPerRepeat;
                vlds(lhsLow1, lhsHigh1, (__ubuf__ int32_t*)src0, (row * Src0Cols + colOffset1) * 2, DINTLV_B32);
                vlds(rhsLow1, rhsHigh1, (__ubuf__ int32_t*)src1, (row * Src1Cols + colOffset1) * 2, DINTLV_B32);
                preg = plt_b32(sreg, POST_UPDATE);
                Int64CompareRegs<Mode, T>(result1, lhsLow1, lhsHigh1, rhsLow1, rhsHigh1, preg);
                // Same pattern as TCmp_32B: pdintlv_b8 + PK
                pdintlv_b8(dstReg, tmpMask, result0, result1);
                psts(dstReg, rowDst + j * dstRepeatStride, 0, PK);
            }
        }
    }
}

template <typename T, unsigned DstRowBytes, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Compare(
    __ubuf__ uint8_t* dst, __ubuf__ T* src0, __ubuf__ T* src1, CmpMode mode, unsigned validRows, unsigned validCols)
{
    switch (mode) {
        case CmpMode::NE:
            Int64CompareMode<CmpMode::NE, T, DstRowBytes, Src0Cols, Src1Cols>(dst, src0, src1, validRows, validCols);
            break;
        case CmpMode::LT:
            Int64CompareMode<CmpMode::LT, T, DstRowBytes, Src0Cols, Src1Cols>(dst, src0, src1, validRows, validCols);
            break;
        case CmpMode::GT:
            Int64CompareMode<CmpMode::GT, T, DstRowBytes, Src0Cols, Src1Cols>(dst, src0, src1, validRows, validCols);
            break;
        case CmpMode::GE:
            Int64CompareMode<CmpMode::GE, T, DstRowBytes, Src0Cols, Src1Cols>(dst, src0, src1, validRows, validCols);
            break;
        case CmpMode::LE:
            Int64CompareMode<CmpMode::LE, T, DstRowBytes, Src0Cols, Src1Cols>(dst, src0, src1, validRows, validCols);
            break;
        case CmpMode::EQ:
        default:
            Int64CompareMode<CmpMode::EQ, T, DstRowBytes, Src0Cols, Src1Cols>(dst, src0, src1, validRows, validCols);
            break;
    }
}
#else
// Declaration-only stubs for kirin9030/kirinX90 (no 64-bit intrinsics).
// See TBinOp.hpp for details.
template <typename T, unsigned DstRowBytes, unsigned Src0Cols, unsigned Src1Cols>
PTO_INTERNAL void Int64Compare(
    __ubuf__ uint8_t* dst, __ubuf__ T* src0, __ubuf__ T* src1, CmpMode mode, unsigned validRows, unsigned validCols);
#endif

const int32_t CMP_BITS_PER_INDEX = 32;

template <typename T>
AICORE void CmpCall(MaskReg& dst, T& src0, T& src1, CmpMode cmpMode, MaskReg& preg)
{
    switch (static_cast<CmpMode>(cmpMode)) {
        case CmpMode::EQ:
            vcmp_eq(dst, src0, src1, preg);
            break;
        case CmpMode::NE:
            vcmp_ne(dst, src0, src1, preg);
            break;
        case CmpMode::LT:
            vcmp_lt(dst, src0, src1, preg);
            break;
        case CmpMode::GT:
            vcmp_gt(dst, src0, src1, preg);
            break;
        case CmpMode::GE:
            vcmp_ge(dst, src0, src1, preg);
            break;
        case CmpMode::LE:
            vcmp_le(dst, src0, src1, preg);
            break;
        default:
            vcmp_eq(dst, src0, src1, preg);
            break;
    }
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
__tf__ PTO_INTERNAL OP_NAME(TCMP) OP_TYPE(element_wise) void TCmp_8B_16B(
    typename DstTile::TileDType __out__ dstData, typename SrcTile0::TileDType __in__ src0Data,
    typename SrcTile1::TileDType __in__ src1Data, CmpMode mode, unsigned validRow, unsigned validCol,
    unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using TIN = typename SrcTile0::DType;
    using TOUT = typename DstTile::DType;
    __ubuf__ TIN* src0 = (__ubuf__ TIN*)__cce_get_tile_ptr(src0Data);
    __ubuf__ TIN* src1 = (__ubuf__ TIN*)__cce_get_tile_ptr(src1Data);
    __ubuf__ uint32_t* dst = (__ubuf__ uint32_t*)__cce_get_tile_ptr(dstData);
    constexpr uint32_t repeatElm = CCE_VL / sizeof(TIN);
    uint16_t repeatTimes = CeilDivision(validCol, repeatElm);
    __VEC_SCOPE__
    {
        RegTensor<TIN> src0Reg;
        RegTensor<TIN> src1Reg;
        uint32_t sReg;
        MaskReg pReg;
        MaskReg dstReg;
        using DistType = std::conditional_t<sizeof(TIN) == 2, decltype(PK), decltype(NORM)>;
        constexpr DistType distValue{};
        constexpr int32_t dstRepeatStride = repeatElm / CMP_BITS_PER_INDEX;
        constexpr uint32_t dstStride = DstTile::RowStride * sizeof(TOUT) / sizeof(uint32_t);
        for (uint16_t i = 0; i < (uint16_t)(validRow); i++) {
            sReg = validCol;
            for (uint16_t j = 0; j < (uint16_t)(repeatTimes); j++) {
                pReg = CreatePredicate<TIN>(sReg);
                vlds(src0Reg, src0, i * SrcTile0::RowStride + j * repeatElm, NORM);
                vlds(src1Reg, src1, i * SrcTile1::RowStride + j * repeatElm, NORM);
                CmpCall(dstReg, src0Reg, src1Reg, mode, pReg);
                psts(dstReg, dst + i * dstStride + j * dstRepeatStride, 0, distValue);
            }
        }
    }
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
__tf__ PTO_INTERNAL OP_NAME(TCMP) OP_TYPE(element_wise) void TCmp_32B(
    typename DstTile::TileDType __out__ dstData, typename SrcTile0::TileDType __in__ src0Data,
    typename SrcTile1::TileDType __in__ src1Data, CmpMode mode, unsigned validRow, unsigned validCol,
    unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using TIN = typename SrcTile0::DType;
    using TOUT = typename DstTile::DType;
    __ubuf__ TIN* src0 = (__ubuf__ TIN*)__cce_get_tile_ptr(src0Data);
    __ubuf__ TIN* src1 = (__ubuf__ TIN*)__cce_get_tile_ptr(src1Data);
    __ubuf__ uint32_t* dst = (__ubuf__ uint32_t*)__cce_get_tile_ptr(dstData);
    constexpr uint32_t repeatElm = CCE_VL / sizeof(uint32_t);
    uint16_t repeatTimes = CeilDivision(validCol, repeatElm) + 1;
    __VEC_SCOPE__
    {
        uint32_t sReg;
        RegTensor<TIN> src0Reg0;
        RegTensor<TIN> src0Reg1;
        RegTensor<TIN> src1Reg0;
        RegTensor<TIN> src1Reg1;
        MaskReg pReg;
        MaskReg tmpMask0;
        MaskReg tmpMask1;
        MaskReg dstReg;
        MaskReg tmpMask2;
        constexpr uint32_t dstStride = DstTile::RowStride * sizeof(TOUT) / sizeof(uint32_t);
        constexpr int32_t dstRepeatStride = 2 * repeatElm / CMP_BITS_PER_INDEX;
        for (uint16_t i = 0; i < (uint16_t)(validRow); i++) {
            sReg = validCol;
            uint16_t halfRepeatTimes = static_cast<uint16_t>(repeatTimes / 2);
            for (uint16_t j = 0; j < halfRepeatTimes; j++) {
                vlds(src0Reg0, src0, i * SrcTile0::RowStride + j * 2 * repeatElm, NORM);
                vlds(src1Reg0, src1, i * SrcTile0::RowStride + j * 2 * repeatElm, NORM);
                vlds(src0Reg1, src0, i * SrcTile1::RowStride + (j * 2 + 1) * repeatElm, NORM);
                vlds(src1Reg1, src1, i * SrcTile1::RowStride + (j * 2 + 1) * repeatElm, NORM);
                pReg = plt_b32(sReg, POST_UPDATE);
                CmpCall(tmpMask0, src0Reg0, src1Reg0, mode, pReg);
                pReg = plt_b32(sReg, POST_UPDATE);
                CmpCall(tmpMask1, src0Reg1, src1Reg1, mode, pReg);
                pdintlv_b8(dstReg, tmpMask2, tmpMask0, tmpMask1);
                psts(dstReg, dst + i * dstStride + j * dstRepeatStride, 0, PK);
            }
        }
    }
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
PTO_INTERNAL void TcmpCheck()
{
    using T = typename SrcTile0::DType;
    static_assert(
        std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t> || std::is_same_v<T, int32_t> ||
            std::is_same_v<T, uint32_t> || std::is_same_v<T, float> || std::is_same_v<T, int16_t> ||
            std::is_same_v<T, uint16_t> || std::is_same_v<T, half> || std::is_same_v<T, uint8_t> ||
            std::is_same_v<T, int8_t> || std::is_same_v<T, bfloat16_t>,
        "TCMP: Invalid data type.");
    static_assert(std::is_same_v<T, typename SrcTile1::DType>, "TCMP: src0 and src1 must have same type");
    static_assert(
        DstTile::isRowMajor && SrcTile0::isRowMajor && SrcTile1::isRowMajor, "TCMP: not supported Layout type");
    static_assert(
        DstTile::Loc == TileType::Vec && SrcTile0::Loc == TileType::Vec && SrcTile1::Loc == TileType::Vec,
        "TCMP: TileType of tile must be TileType::Vec.");
    static_assert(
        DstTile::ValidCol <= DstTile::Cols && SrcTile0::ValidCol <= SrcTile0::Cols &&
            SrcTile1::ValidCol <= SrcTile1::Cols,
        "TCMP: Number of valid columns must not be greater than number of tile columns.");
    static_assert(
        DstTile::ValidRow <= DstTile::Rows && SrcTile0::ValidRow <= SrcTile0::Rows &&
            SrcTile1::ValidRow <= SrcTile1::Rows,
        "TCMP: Number of valid rows must not be greater than number of tile rows.");
}

template <typename DstTile, typename SrcTile0, typename SrcTile1>
PTO_INTERNAL void TCMP_IMPL(DstTile& dst, SrcTile0& src0, SrcTile1& src1, CmpMode cmpMode)
{
    TcmpCheck<DstTile, SrcTile0, SrcTile1>();
    using T = typename SrcTile0::DType;
    unsigned validRow = src0.GetValidRow();
    unsigned validCol = src0.GetValidCol();
    if constexpr (sizeof(T) == 8) {
        constexpr unsigned dstRowBytes = DstTile::RowStride * sizeof(typename DstTile::DType);
        Int64Compare<T, dstRowBytes, SrcTile0::Cols, SrcTile1::Cols>(
            (__ubuf__ uint8_t*)dst.data(), (__ubuf__ T*)src0.data(), (__ubuf__ T*)src1.data(), cmpMode, validRow,
            validCol);
    } else if constexpr (sizeof(T) == 4) {
        TCmp_32B<DstTile, SrcTile0, SrcTile1>(dst.data(), src0.data(), src1.data(), cmpMode, validRow, validCol);
    } else if constexpr ((sizeof(T) == 2) || (sizeof(T) == 1)) {
        TCmp_8B_16B<DstTile, SrcTile0, SrcTile1>(dst.data(), src0.data(), src1.data(), cmpMode, validRow, validCol);
    }
}

} // namespace pto
#endif
