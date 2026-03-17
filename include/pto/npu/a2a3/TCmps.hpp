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

namespace pto {

constexpr const uint64_t NUM_BITS_IN_BYTE = 8;
constexpr const uint8_t TCMPS_REPEAT_MAX = 240;

template <typename T_src, typename T_scalar>
AICORE void vcmp_dispatch(__ubuf__ uint8_t *dst, __ubuf__ T_src *src0, T_scalar scalar, CmpMode mode, uint8_t rep,
                          uint16_t b_dst, uint16_t b_src, uint16_t r_dst, uint16_t r_src)
{
    switch (mode) {
        case CmpMode::NE:
            vcmpvs_ne(dst, src0, scalar, rep, b_dst, b_src, r_dst, r_src);
            break;
        case CmpMode::LT:
            vcmpvs_lt(dst, src0, scalar, rep, b_dst, b_src, r_dst, r_src);
            break;
        case CmpMode::GT:
            vcmpvs_gt(dst, src0, scalar, rep, b_dst, b_src, r_dst, r_src);
            break;
        case CmpMode::GE:
            vcmpvs_ge(dst, src0, scalar, rep, b_dst, b_src, r_dst, r_src);
            break;
        case CmpMode::LE:
            vcmpvs_le(dst, src0, scalar, rep, b_dst, b_src, r_dst, r_src);
            break;
        case CmpMode::EQ:
        default:
            vcmpvs_eq(dst, src0, scalar, rep, b_dst, b_src, r_dst, r_src);
            break;
    }
}

template <typename TIN, typename TOUT>
AICORE void GenCmpCall(__ubuf__ TOUT *dst, __ubuf__ TIN *src0, TIN src1, CmpMode cmpMode, uint8_t repeat,
                       uint16_t dstblockstride, uint16_t srcblockstride, uint16_t dstrepeatstride,
                       uint16_t srcrepeatstride)
{
    if constexpr (std::is_same<TIN, int32_t>::value) {
        vcmpvs_eq(dst, src0, src1, repeat, dstblockstride, srcblockstride, dstrepeatstride, srcrepeatstride);
    } else {
        if (sizeof(TIN) == 4) {
            vcmp_dispatch(dst, src0, src1, cmpMode, repeat, dstblockstride, srcblockstride, dstrepeatstride,
                          srcrepeatstride);
        } else {
            half scalar;
            if constexpr (std::is_same<TIN, uint16_t>::value || std::is_same<TIN, int16_t>::value) {
                scalar = *reinterpret_cast<half *>(&src1);
            } else {
                scalar = src1;
            }
            auto *src_ptr = reinterpret_cast<__ubuf__ half *>(src0);
            vcmp_dispatch(dst, src_ptr, scalar, cmpMode, repeat, dstblockstride, srcblockstride, dstrepeatstride,
                          srcrepeatstride);
        }
    }
}

template <typename TIN, typename TOUT>
AICORE void TCmps(__ubuf__ TOUT *dst, __ubuf__ TIN *src0, TIN src1Value, CmpMode mode, unsigned validRow,
                  unsigned repeatPerLine, int srcAlignCols, int dstAlignCols)
{
    constexpr int srcOffset = TCMPS_REPEAT_MAX * REPEAT_BYTE / sizeof(TIN);
    constexpr int dstOffset = TCMPS_REPEAT_MAX * REPEAT_BYTE / sizeof(TIN) / NUM_BITS_IN_BYTE;
    size_t nLoop = repeatPerLine / TCMPS_REPEAT_MAX;
    int remainPerLine = repeatPerLine % TCMPS_REPEAT_MAX;

    set_mask_norm();
    set_vector_mask(-1, -1);
    for (size_t i = 0; i < validRow; i++) {
        for (size_t j = 0; j < nLoop; j++) {
            GenCmpCall<TIN, TOUT>(dst + i * dstAlignCols + j * dstOffset, src0 + i * srcAlignCols + j * srcOffset,
                                  src1Value, mode, TCMPS_REPEAT_MAX, 1, 1, 8, 8);
        }
    }
    if (remainPerLine) {
        for (size_t i = 0; i < validRow; i++) {
            GenCmpCall<TIN, TOUT>(dst + i * dstAlignCols + nLoop * dstOffset,
                                  src0 + i * srcAlignCols + nLoop * srcOffset, src1Value, mode, remainPerLine, 1, 1, 8,
                                  8);
        }
    }
}

template <typename TileDataDst, typename TileDataSrc, typename T>
__tf__ AICORE void TCmps_Scalar(typename TileDataDst::TileDType __out__ dst,
                                typename TileDataSrc::TileDType __in__ src0, T src1, CmpMode mode,
                                unsigned numRepeatPerLine, unsigned validRow)
{
    using TIN = typename TileDataSrc::DType;
    using TOUT = typename TileDataDst::DType;
    __ubuf__ TOUT *dstPtr = (__ubuf__ TOUT *)__cce_get_tile_ptr(dst);
    __ubuf__ TIN *srcPtr = (__ubuf__ TIN *)__cce_get_tile_ptr(src0);

    constexpr int srcAlignCols = TileDataSrc::Cols;
    constexpr int dstAlignCols = TileDataDst::Cols;

    TCmps<TIN, TOUT>(dstPtr, srcPtr, src1, mode, validRow, numRepeatPerLine, srcAlignCols, dstAlignCols);
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
__tf__ AICORE void TCmps_Tile(typename TileDataDst::TileDType __out__ dstData,
                              typename TileDataSrc0::TileDType __in__ src0Data,
                              typename TileDataSrc1::TileDType __in__ src1Data, CmpMode mode, unsigned validRow,
                              unsigned validCol)
{
    using TIN = typename TileDataSrc0::DType;
    using TOUT = typename TileDataDst::DType;
    __ubuf__ TOUT *dst = (__ubuf__ TOUT *)__cce_get_tile_ptr(dstData);
    __ubuf__ TIN *src0 = (__ubuf__ TIN *)__cce_get_tile_ptr(src0Data);
    __ubuf__ TIN *src1 = (__ubuf__ TIN *)__cce_get_tile_ptr(src1Data);

    PtoSetWaitFlag<PIPE_V, PIPE_S>();
    PtoSetWaitFlag<PIPE_MTE1, PIPE_S>();
    PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
    TIN src1Value = *src1;
    constexpr int srcAlignCols = TileDataSrc0::Cols;
    constexpr int dstAlignCols = TileDataDst::Cols;
    constexpr int elemPerRepeat = REPEAT_BYTE / sizeof(TIN);

    unsigned repeatPerLine = CeilDivision(validCol, elemPerRepeat);

    TCmps<TIN, TOUT>(dst, src0, src1Value, mode, validRow, repeatPerLine, srcAlignCols, dstAlignCols);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TcmpsCheck()
{
    static_assert(std::is_same<typename TileDataSrc::DType, int32_t>::value ||
                      std::is_same<typename TileDataSrc::DType, float>::value ||
                      std::is_same<typename TileDataSrc::DType, half>::value ||
                      std::is_same<typename TileDataSrc::DType, uint16_t>::value ||
                      std::is_same<typename TileDataSrc::DType, int16_t>::value,
                  "TCMPS: Invalid data type.");
    static_assert(TileDataDst::isRowMajor, "TCMPS: not supported Layout type");
    static_assert(TileDataDst::Loc == TileType::Vec, "TileType of dst tile must be TileType::Vec.");
    static_assert(TileDataDst::ValidRow <= TileDataDst::Rows,
                  "Number of valid rows for dst must not be greater than number of tile rows.");
    static_assert(TileDataDst::ValidCol <= TileDataDst::Cols,
                  "Number of valid columns for dst must not be greater than number of tile columns.");
    static_assert(TileDataSrc::Loc == TileType::Vec, "TileType of src tile must be TileType::Vec.");
    static_assert(TileDataSrc::ValidCol <= TileDataSrc::Cols,
                  "Number of valid columns for scr must not be greater than number of tile columns.");
    static_assert(TileDataSrc::ValidRow <= TileDataSrc::Rows,
                  "Number of valid rows for src must not be greater than number of tile rows.");
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

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TCMPS_IMPL(TileDataDst &dst, TileDataSrc &src0, typename TileDataSrc::DType src1, CmpMode mode)
{
    TcmpsCheck<TileDataDst, TileDataSrc>();
    PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");

    using T = typename TileDataSrc::DType;
    unsigned validRow = src0.GetValidRow();
    unsigned numRepeatPerLine = CeilDivision(src0.GetValidCol(), (REPEAT_BYTE / sizeof(T)));

    TCmps_Scalar<TileDataDst, TileDataSrc, T>(dst.data(), src0.data(), src1, mode, numRepeatPerLine, validRow);
}

} // namespace pto
#endif
