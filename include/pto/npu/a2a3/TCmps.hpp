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

template <typename TileDataDst, typename TileDataSrc, typename T>
AICORE void GenCmpCall(__ubuf__ typename TileDataDst::DType *dst, __ubuf__ typename TileDataSrc::DType *src0, T src1,
                       CmpMode cmpMode, uint8_t repeat, uint16_t dstblockstride, uint16_t srcblockstride,
                       uint16_t dstrepeatstride, uint16_t srcrepeatstride)
{
    using SrcT = typename TileDataSrc::DType;

    if constexpr (std::is_same<SrcT, int32_t>::value) {
        vcmpvs_eq(dst, src0, src1, repeat, dstblockstride, srcblockstride, dstrepeatstride, srcrepeatstride);
    } else {
        if (sizeof(SrcT) == 4) {
            vcmp_dispatch(dst, src0, src1, cmpMode, repeat, dstblockstride, srcblockstride, dstrepeatstride,
                          srcrepeatstride);
        } else {
            half scalar;
            if constexpr (std::is_same<T, uint16_t>::value || std::is_same<T, int16_t>::value) {
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

template <typename TileDataDst, typename TileDataSrc, typename T>
__tf__ AICORE void TCmps(typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src0,
                         T src1, CmpMode mode, unsigned numRepeatPerLine, unsigned validRow, unsigned elementsPerRepeat)
{
    __ubuf__ typename TileDataDst::DType *dstPtr = (__ubuf__ typename TileDataDst::DType *)__cce_get_tile_ptr(dst);
    __ubuf__ typename TileDataSrc::DType *srcPtr = (__ubuf__ typename TileDataSrc::DType *)__cce_get_tile_ptr(src0);

    size_t numLoop = numRepeatPerLine / TCMPS_REPEAT_MAX;
    int numRemainPerLine = numRepeatPerLine % TCMPS_REPEAT_MAX;
    constexpr int srcAlignCols = TileDataSrc::Cols;
    constexpr int dstAlignCols = TileDataDst::Cols;
    constexpr int srcOffset = TCMPS_REPEAT_MAX * REPEAT_BYTE / sizeof(T);
    constexpr int dstOffset = TCMPS_REPEAT_MAX * REPEAT_BYTE / sizeof(T) / NUM_BITS_IN_BYTE;

    set_mask_norm();
    set_vector_mask(-1, -1);
    for (size_t i = 0; i < validRow; i++) {
        for (size_t j = 0; j < numLoop; j++) {
            GenCmpCall<TileDataDst, TileDataSrc, T>(dstPtr + i * dstAlignCols + j * dstOffset,
                                                    srcPtr + i * srcAlignCols + j * srcOffset, src1, mode,
                                                    TCMPS_REPEAT_MAX, 1, 1, 8, 8);
        }
        if (numRemainPerLine) {
            GenCmpCall<TileDataDst, TileDataSrc, T>(dstPtr + i * dstAlignCols + numLoop * dstOffset,
                                                    srcPtr + i * srcAlignCols + numLoop * srcOffset, src1, mode,
                                                    numRemainPerLine, 1, 1, 8, 8);
        }
    }
}

template <typename TileDataDst, typename TileDataSrc0, typename T>
PTO_INTERNAL void TCMPS_IMPL(TileDataDst &dst, TileDataSrc0 &src0, T src1, CmpMode cmpMode)
{
    static_assert(std::is_same<typename TileDataSrc0::DType, int32_t>::value ||
                      std::is_same<typename TileDataSrc0::DType, float>::value ||
                      std::is_same<typename TileDataSrc0::DType, half>::value ||
                      std::is_same<typename TileDataSrc0::DType, uint16_t>::value ||
                      std::is_same<typename TileDataSrc0::DType, int16_t>::value,
                  "TCMPS: Invalid data type.");
    static_assert(TileDataDst::isRowMajor, "TCMPS: not supported Layout type");

    static_assert(TileDataDst::Loc == TileType::Vec, "TileType of dst tile must be TileType::Vec.");
    static_assert(TileDataDst::ValidCol <= TileDataDst::Cols,
                  "Number of valid columns for dst must not be greater than number of tile columns.");
    static_assert(TileDataDst::ValidRow <= TileDataDst::Rows,
                  "Number of valid rows for dst must not be greater than number of tile rows.");

    static_assert(TileDataSrc0::Loc == TileType::Vec, "TileType of src tile must be TileType::Vec.");
    static_assert(TileDataSrc0::ValidCol <= TileDataSrc0::Cols,
                  "Number of valid columns for scr must not be greater than number of tile columns.");
    static_assert(TileDataSrc0::ValidRow <= TileDataSrc0::Rows,
                  "Number of valid rows for src must not be greater than number of tile rows.");
    PTO_ASSERT(src0.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");

    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(typename TileDataSrc0::DType);
    unsigned numRepeatPerLine = CeilDivision(src0.GetValidCol(), elementsPerRepeat);
    unsigned validRow = src0.GetValidRow();

    TCmps<TileDataDst, TileDataSrc0, T>(dst.data(), src0.data(), src1, cmpMode, numRepeatPerLine, validRow,
                                        elementsPerRepeat);
}
} // namespace pto
#endif
