/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCMPS_HPP
#define TCMPS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "common.hpp"
#include "utils.hpp"

namespace pto {
constexpr const uint16_t RESULT_NUM_PER_INT32 = 32;
template <typename T>
AICORE void GenCmpCall(MaskReg &dst, RegTensor<T> &src0, T src1, CmpMode cmpMode, MaskReg &preg)
{
    switch (static_cast<CmpMode>(cmpMode)) {
        case CmpMode::EQ:
            vcmps_eq(dst, src0, src1, preg);
            break;
        case CmpMode::NE:
            vcmps_ne(dst, src0, src1, preg);
            break;
        case CmpMode::LT:
            vcmps_lt(dst, src0, src1, preg);
            break;
        case CmpMode::GT:
            vcmps_gt(dst, src0, src1, preg);
            break;
        case CmpMode::GE:
            vcmps_ge(dst, src0, src1, preg);
            break;
        case CmpMode::LE:
            vcmps_le(dst, src0, src1, preg);
            break;
        default:
            vcmps_eq(dst, src0, src1, preg);
            break;
    }
}

template <typename T, uint32_t SrcStride, uint32_t DstStride>
PTO_INTERNAL void TCmps_8B_16B(__ubuf__ uint32_t *dst, __ubuf__ T *src0, T src1, CmpMode mode, unsigned validRow,
                               unsigned validCol)
{
    constexpr uint32_t repeatElm = CCE_VL / sizeof(T);
    constexpr uint16_t dstOffset = repeatElm / RESULT_NUM_PER_INT32;
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg;
        MaskReg pReg;
        MaskReg dstReg;

        using DistType = std::conditional_t<sizeof(T) == 2, decltype(PK), decltype(NORM)>;
        constexpr DistType distValue{};
        uint32_t sReg;
        uint16_t repeatTimes = CeilDivision(validCol, repeatElm);

        for (uint16_t i = 0; i < (uint16_t)(validRow); ++i) {
            sReg = validCol;
            for (uint16_t j = 0; j < (uint16_t)(repeatTimes); ++j) {
                pReg = CreatePredicate<T>(sReg);
                vlds(srcReg, src0, i * SrcStride + j * repeatElm, NORM);
                GenCmpCall<T>(dstReg, srcReg, src1, mode, pReg);
                psts(dstReg, dst + i * DstStride + j * dstOffset, 0, distValue);
            }
        }
    }
}

template <typename T, uint32_t SrcStride, uint32_t DstStride>
PTO_INTERNAL void TCmps_32B(__ubuf__ uint32_t *dst, __ubuf__ T *src0, T src1, CmpMode mode, unsigned validRow,
                            unsigned validCol)
{
    constexpr uint32_t repeatElm = CCE_VL / sizeof(T);
    constexpr uint16_t dstOffset = 2 * repeatElm / RESULT_NUM_PER_INT32;
    __VEC_SCOPE__
    {
        RegTensor<T> srcReg0;
        RegTensor<T> srcReg1;
        uint32_t sReg;
        MaskReg pReg;
        MaskReg tmpReg0;
        MaskReg tmpReg1;
        MaskReg tmpReg2;
        MaskReg dstReg;

        for (uint16_t i = 0; i < (uint16_t)(validRow); ++i) {
            sReg = validCol;
            // for odd repeat number, add 1 to include remainder
            uint16_t repeatTimes = CeilDivision(validCol, repeatElm) + 1;
            for (uint16_t j = 0; j < (uint16_t)(repeatTimes / 2); ++j) {
                vlds(srcReg0, src0, i * SrcStride + j * 2 * repeatElm, NORM);
                vlds(srcReg1, src0, i * SrcStride + (j * 2 + 1) * repeatElm, NORM);
                pReg = CreatePredicate<T>(sReg);
                GenCmpCall<T>(tmpReg0, srcReg0, src1, mode, pReg);
                pReg = CreatePredicate<T>(sReg);
                GenCmpCall<T>(tmpReg1, srcReg1, src1, mode, pReg);
                pdintlv_b8(dstReg, tmpReg2, tmpReg0, tmpReg1);
                psts(dstReg, dst + i * DstStride + j * dstOffset, 0, PK);
            }
        }
    }
}

template <typename TileDataDst, typename TileDataSrc, typename T>
__tf__ PTO_INTERNAL OP_NAME(TCMPS)
    OP_TYPE(element_wise) void TCmps_Scalar(typename TileDataDst::TileDType __out__ dstData,
                                            typename TileDataSrc::TileDType __in__ src0Data, T src1, CmpMode mode,
                                            unsigned validRow, unsigned validCol,
                                            unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    __ubuf__ T *src0 = (__ubuf__ T *)__cce_get_tile_ptr(src0Data);
    __ubuf__ uint32_t *dst = (__ubuf__ uint32_t *)__cce_get_tile_ptr(dstData);

    constexpr uint32_t srcStride = TileDataSrc::RowStride;
    constexpr uint32_t dstStride = TileDataDst::RowStride * sizeof(typename TileDataDst::DType) / sizeof(uint32_t);

    if constexpr (sizeof(T) == 4) {
        TCmps_32B<T, srcStride, dstStride>(dst, src0, src1, mode, validRow, validCol);
    } else {
        TCmps_8B_16B<T, srcStride, dstStride>(dst, src0, src1, mode, validRow, validCol);
    }
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
__tf__ PTO_INTERNAL OP_NAME(TCMPS)
    OP_TYPE(element_wise) void TCmps_Tile(typename TileDataDst::TileDType __out__ dstData,
                                          typename TileDataSrc0::TileDType __in__ src0Data,
                                          typename TileDataSrc1::TileDType __in__ src1Data, CmpMode mode,
                                          unsigned validRow, unsigned validCol,
                                          unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename TileDataSrc0::DType;
    __ubuf__ T *src0 = (__ubuf__ T *)__cce_get_tile_ptr(src0Data);
    __ubuf__ T *src1 = (__ubuf__ T *)__cce_get_tile_ptr(src1Data);
    __ubuf__ uint32_t *dst = (__ubuf__ uint32_t *)__cce_get_tile_ptr(dstData);

    constexpr uint32_t srcStride = TileDataSrc0::RowStride;
    constexpr uint32_t dstStride = TileDataDst::RowStride * sizeof(typename TileDataDst::DType) / sizeof(uint32_t);

    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    T src1Value = *src1;
    if constexpr (sizeof(T) == 4) {
        TCmps_32B<T, srcStride, dstStride>(dst, src0, src1Value, mode, validRow, validCol);
    } else {
        TCmps_8B_16B<T, srcStride, dstStride>(dst, src0, src1Value, mode, validRow, validCol);
    }
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TcmpsCheck()
{
    using T = typename TileDataSrc::DType;
    static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> || std::is_same_v<T, float> ||
                      std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, half> ||
                      std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                  "TCMPS: Invalid data type.");
    static_assert(TileDataDst::isRowMajor, "TCMPS: not supported Layout type");
    static_assert(TileDataDst::Loc == TileType::Vec, "TileType of dst tile must be TileType::Vec.");
    static_assert(TileDataDst::ValidCol <= TileDataDst::Cols,
                  "Number of valid columns for dst must not be greater than number of tile columns.");
    static_assert(TileDataDst::ValidRow <= TileDataDst::Rows,
                  "Number of valid rows for dst must not be greater than number of tile rows.");
    static_assert(TileDataSrc::Loc == TileType::Vec, "TileType of src tile must be TileType::Vec.");
    static_assert(TileDataSrc::ValidCol <= TileDataSrc::Cols,
                  "Number of valid columns for scr must not be greater than number of tile columns.");
    static_assert(TileDataSrc::ValidRow <= TileDataSrc::Rows,
                  "Number of valid rows for src must not be greater than number of tile rows.");
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TCMPS_IMPL(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType src1, CmpMode mode)
{
    TcmpsCheck<TileDataDst, TileDataSrc>();
    PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");
    unsigned validRow = src0.GetValidRow();
    unsigned validCol = src0.GetValidCol();
    TCmps_Scalar<TileDataDst, TileDataSrc, typename TileDataSrc::DType>(dst.data(), src0.data(), src1, mode, validRow,
                                                                        validCol);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1,
          typename = std::void_t<typename TileDataSrc1::DType>>
PTO_INTERNAL void TCMPS_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, CmpMode mode)
{
    TcmpsCheck<TileDataDst, TileDataSrc0>();
    static_assert(std::is_same_v<typename TileDataSrc0::DType, typename TileDataSrc1::DType>,
                  "TCMPS: The input data type must be consistent with the scalar data type.");
    PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");
    unsigned validRow = src0.GetValidRow();
    unsigned validCol = src0.GetValidCol();
    TCmps_Tile<TileDataDst, TileDataSrc0, TileDataSrc1>(dst.data(), src0.data(), src1.data(), mode, validRow, validCol);
}
} // namespace pto
#endif
