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
#include "utils.hpp"

namespace pto {

const int32_t CMP_BITS_PER_INDEX = 32;

template <typename T>
AICORE void CmpCall(MaskReg &dst, T &src0, T &src1, CmpMode cmpMode, MaskReg &preg)
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

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL OP_NAME(TCMP)
    OP_TYPE(element_wise) void TCmp_8B_16B(typename DstTile::TileDType __out__ dstData,
                                           typename SrcTile::TileDType __in__ src0Data,
                                           typename SrcTile::TileDType __in__ src1Data, CmpMode mode, unsigned validRow,
                                           unsigned validCol, unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using TIN = typename SrcTile::DType;
    using TOUT = typename DstTile::DType;
    __ubuf__ TIN *src0 = (__ubuf__ TIN *)__cce_get_tile_ptr(src0Data);
    __ubuf__ TIN *src1 = (__ubuf__ TIN *)__cce_get_tile_ptr(src1Data);
    __ubuf__ uint32_t *dst = (__ubuf__ uint32_t *)__cce_get_tile_ptr(dstData);

    __VEC_SCOPE__
    {
        RegTensor<TIN> src0Reg;
        RegTensor<TIN> src1Reg;
        uint32_t sReg;
        MaskReg pReg;
        MaskReg dstReg;
        using DistType = std::conditional_t<sizeof(TIN) == 2, decltype(PK), decltype(NORM)>;
        constexpr DistType distValue{};
        constexpr uint32_t repeatElm = CCE_VL / sizeof(TIN);
        constexpr int32_t dstRepeatStride = repeatElm / CMP_BITS_PER_INDEX;
        constexpr uint32_t dstStride = DstTile::RowStride * sizeof(TOUT) / sizeof(uint32_t);
        uint16_t repeatTimes = CeilDivision(validCol, repeatElm);
        for (uint16_t i = 0; i < (uint16_t)(validRow); ++i) {
            sReg = validCol;
            for (uint16_t j = 0; j < (uint16_t)(repeatTimes); ++j) {
                pReg = CreatePredicate<TIN>(sReg);

                vlds(src0Reg, src0, i * SrcTile::RowStride + j * repeatElm, NORM);
                vlds(src1Reg, src1, i * SrcTile::RowStride + j * repeatElm, NORM);
                CmpCall(dstReg, src0Reg, src1Reg, mode, pReg);
                psts(dstReg, dst + i * dstStride + j * dstRepeatStride, 0, distValue);
            }
        }
    }
}

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL OP_NAME(TCMP)
    OP_TYPE(element_wise) void TCmp_32B(typename DstTile::TileDType __out__ dstData,
                                        typename SrcTile::TileDType __in__ src0Data,
                                        typename SrcTile::TileDType __in__ src1Data, CmpMode mode, unsigned validRow,
                                        unsigned validCol, unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using TIN = typename SrcTile::DType;
    using TOUT = typename DstTile::DType;
    __ubuf__ TIN *src0 = (__ubuf__ TIN *)__cce_get_tile_ptr(src0Data);
    __ubuf__ TIN *src1 = (__ubuf__ TIN *)__cce_get_tile_ptr(src1Data);
    __ubuf__ uint32_t *dst = (__ubuf__ uint32_t *)__cce_get_tile_ptr(dstData);

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
        constexpr uint32_t repeatElm = CCE_VL / sizeof(uint32_t);
        constexpr uint32_t dstStride = DstTile::RowStride * sizeof(TOUT) / sizeof(uint32_t);
        uint16_t repeatTimes = CeilDivision(validCol, repeatElm) + 1;

        for (uint16_t i = 0; i < (uint16_t)(validRow); ++i) {
            sReg = validCol;
            for (uint16_t j = 0; j < (uint16_t)(repeatTimes / 2); ++j) {
                vlds(src0Reg0, src0, i * SrcTile::RowStride + j * 2 * repeatElm, NORM);
                vlds(src0Reg1, src1, i * SrcTile::RowStride + j * 2 * repeatElm, NORM);
                vlds(src1Reg0, src0, i * SrcTile::RowStride + (j * 2 + 1) * repeatElm, NORM);
                vlds(src1Reg1, src1, i * SrcTile::RowStride + (j * 2 + 1) * repeatElm, NORM);
                pReg = plt_b32(sReg, POST_UPDATE);
                CmpCall(tmpMask0, src0Reg0, src0Reg1, mode, pReg);
                pReg = plt_b32(sReg, POST_UPDATE);
                CmpCall(tmpMask1, src1Reg0, src1Reg1, mode, pReg);
                pdintlv_b8(dstReg, tmpMask2, tmpMask0, tmpMask1);
                psts(dstReg, dst + i * dstStride + j * 4, 0, PK);
            }
        }
    }
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TcmpCheck()
{
    using T = typename SrcTile::DType;
    static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, float> ||
                      std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, half> ||
                      std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                  "TCMP: Invalid data type.");
    static_assert(DstTile::isRowMajor && SrcTile::isRowMajor, "TCMP: not supported Layout type");
    static_assert(DstTile::Loc == TileType::Vec && SrcTile::Loc == TileType::Vec,
                  "TCMP: TileType of tile must be TileType::Vec.");
    static_assert(DstTile::ValidCol <= DstTile::Cols && SrcTile::ValidCol <= SrcTile::Cols,
                  "TCMP: Number of valid columns must not be greater than number of tile columns.");
    static_assert(DstTile::ValidRow <= DstTile::Rows && SrcTile::ValidRow <= SrcTile::Rows,
                  "TCMP: Number of valid rows must not be greater than number of tile rows.");
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TCMP_IMPL(DstTile &dst, SrcTile &src0, SrcTile &src1, CmpMode cmpMode)
{
    TcmpCheck<DstTile, SrcTile>();
    using T = typename SrcTile::DType;
    unsigned validRow = src0.GetValidRow();
    unsigned validCol = src0.GetValidCol();
    if constexpr (sizeof(T) == 4) {
        TCmp_32B<DstTile, SrcTile>(dst.data(), src0.data(), src1.data(), cmpMode, validRow, validCol);
    } else if constexpr ((sizeof(T) == 2) || (sizeof(T) == 1)) {
        TCmp_8B_16B<DstTile, SrcTile>(dst.data(), src0.data(), src1.data(), cmpMode, validRow, validCol);
    }
}

} // namespace pto
#endif
