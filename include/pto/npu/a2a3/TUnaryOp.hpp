/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TUNARYOP_HPP
#define TUNARYOP_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {
#define SMALL_RPT (4)
template <typename Op, typename T>
PTO_INTERNAL void Unary1LCountMode(__ubuf__ T *dst, __ubuf__ T *src, unsigned validRow, unsigned validCol)
{
    set_mask_count();
    SetVectorCount(validRow * validCol);
    Op::UnaryInstr(dst, src, 0);
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T>
PTO_INTERNAL void Unary1LNormMode(__ubuf__ T *dst, __ubuf__ T *src, unsigned validRow, unsigned validCol)
{
    constexpr unsigned nRepeatElem = REPEAT_BYTE / sizeof(T);
    unsigned nElem = validRow * validCol;
    unsigned headRepeats = nElem / nRepeatElem;
    unsigned tailElements = nElem % nRepeatElem;

    Op::UnaryInstr(dst, src, headRepeats);
    if (tailElements) {
        unsigned offset = headRepeats * nRepeatElem;
        SetContMaskByDType<T>(tailElements);
        Op::UnaryInstr(dst + offset, src + offset, 1);
        SetFullVecMaskByDType<T>();
    }
}

template <typename Op, typename T, unsigned dstRowStride, unsigned srcRowStride>
PTO_INTERNAL void Unary2LCountMode(__ubuf__ T *dst, __ubuf__ T *src, unsigned validRow, unsigned validCol)
{
    set_mask_count();
    SetVectorCount(validCol);
    for (uint32_t i = 0; i < validRow; i++) {
        Op::UnaryInstr(dst + i * dstRowStride, src + i * srcRowStride, 0);
    }
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, unsigned dstRowStride, unsigned srcRowStride, unsigned nRepeatElem>
PTO_INTERNAL void Unary2LNormModeColVLAlign(__ubuf__ T *dst, __ubuf__ T *src, unsigned validRow, unsigned validCol)
{
    unsigned headRepeats = validCol / nRepeatElem;
    for (uint32_t i = 0; i < validRow; i++) {
        Op::UnaryInstr(dst + i * dstRowStride, src + i * srcRowStride, headRepeats);
    }
}

template <typename Op, typename T, unsigned dstRowStride, unsigned srcRowStride, unsigned nRepeatElem>
PTO_INTERNAL void Unary2LNormModeHead(__ubuf__ T *dst, __ubuf__ T *src, unsigned validRow, unsigned nRepeatPerLine)
{
    if (nRepeatPerLine) {
        unsigned loop = nRepeatPerLine / REPEAT_MAX;
        unsigned remain = nRepeatPerLine % REPEAT_MAX;
        for (unsigned i = 0; i < validRow; i++) {
            if (loop) {
                for (unsigned j = 0; j < loop; j++) {
                    Op::UnaryInstr(dst + i * dstRowStride + j * nRepeatElem * REPEAT_MAX,
                                   src + i * srcRowStride + j * nRepeatElem * REPEAT_MAX, REPEAT_MAX);
                }
            }
            if (remain) {
                Op::UnaryInstr(dst + i * dstRowStride + loop * nRepeatElem * REPEAT_MAX,
                               src + i * srcRowStride + loop * nRepeatElem * REPEAT_MAX, remain);
            }
        }
    }
}

template <typename Op, typename T, unsigned dstRowStride, unsigned srcRowStride, unsigned dstRow, unsigned srcRow,
          unsigned nRepeatElem, unsigned blockSizeElem>
PTO_INTERNAL void Unary2LNormModeTail(__ubuf__ T *dst, __ubuf__ T *src, unsigned validRow, unsigned nRemainPerLine)
{
    constexpr unsigned dstStride = dstRowStride / blockSizeElem;
    constexpr unsigned srcStride = srcRowStride / blockSizeElem;
    unsigned loop = 0;
    unsigned remain = validRow;
    constexpr bool strideOverFlag = (dstStride > REPEAT_STRIDE_MAX || srcStride > REPEAT_STRIDE_MAX);
    SetContMaskByDType<T>(nRemainPerLine);
    if constexpr (dstRow > pto::REPEAT_MAX || srcRow > pto::REPEAT_MAX) {
        loop = validRow / REPEAT_MAX;
        for (uint32_t i = 0; i < loop; i++) {
            if constexpr (strideOverFlag) {
                for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                    Op::UnaryInstr(dst + (i * REPEAT_MAX + j) * dstRowStride, src + (i * REPEAT_MAX + j) * srcRowStride,
                                   1, 1, 1);
                }
            } else {
                Op::UnaryInstr(dst + i * REPEAT_MAX * dstRowStride, src + i * REPEAT_MAX * srcRowStride, REPEAT_MAX,
                               dstStride, srcStride);
            }
        }
        remain = validRow % REPEAT_MAX;
    }
    if (remain) {
        if constexpr (strideOverFlag) {
            for (uint32_t j = 0; j < remain; j++) {
                Op::UnaryInstr(dst + (loop * REPEAT_MAX + j) * dstRowStride,
                               src + (loop * REPEAT_MAX + j) * srcRowStride, 1, 1, 1);
            }
        } else {
            Op::UnaryInstr(dst + loop * REPEAT_MAX * dstRowStride, src + loop * REPEAT_MAX * srcRowStride, remain,
                           dstStride, srcStride);
        }
    }
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename T, unsigned dstRowStride, unsigned srcRowStride, unsigned dstRow, unsigned srcRow,
          unsigned nRepeatElem>
PTO_INTERNAL void Unary2LNormModeRowRpt(__ubuf__ T *dst, __ubuf__ T *src, unsigned validRow, unsigned validCol)
{
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned dstStride = dstRowStride / blockSizeElem;
    constexpr unsigned srcStride = srcRowStride / blockSizeElem;
    constexpr bool condRowRpt = ((dstRow <= pto::REPEAT_MAX) && (dstStride <= REPEAT_STRIDE_MAX) &&
                                 (srcRow <= pto::REPEAT_MAX) && (srcStride <= REPEAT_STRIDE_MAX));
    if constexpr (condRowRpt) {
        unsigned loop = validCol / nRepeatElem;
        unsigned tailElements = validCol % nRepeatElem;
        for (uint32_t i = 0; i < loop; i++) {
            Op::UnaryInstr(dst + i * nRepeatElem, src + i * nRepeatElem, validRow, dstStride, srcStride);
        }

        if (tailElements) {
            SetContMaskByDType<T>(tailElements);
            Op::UnaryInstr(dst + loop * nRepeatElem, src + loop * nRepeatElem, validRow, dstStride, srcStride);
            SetFullVecMaskByDType<T>();
        }
    } else {
        unsigned nRepeatPerLine = validCol / nRepeatElem;
        unsigned remain = validCol % nRepeatElem;
        if constexpr (dstRow > nRepeatElem) {
            Unary2LNormModeHead<Op, T, dstRowStride, srcRowStride, nRepeatElem>(dst, src, validRow, nRepeatPerLine);
            dst += nRepeatPerLine * nRepeatElem;
            src += nRepeatPerLine * nRepeatElem;
        }
        if (remain) {
            Unary2LNormModeTail<Op, T, dstRowStride, srcRowStride, dstRow, srcRow, nRepeatElem, blockSizeElem>(
                dst, src, validRow, remain);
        }
    }
}

template <typename Op, typename T, unsigned dstRowStride, unsigned srcRowStride, unsigned dstRow, unsigned srcRow,
          unsigned dstCol, unsigned nRepeatElem>
PTO_INTERNAL void Unary2LProcess(__ubuf__ T *dst, __ubuf__ T *src, unsigned validRow, unsigned validCol)
{
    constexpr unsigned normColRepeat = dstCol / nRepeatElem;
    if constexpr ((normColRepeat > 1) && ((dstRow * normColRepeat) < SMALL_RPT)) {
        Unary2LCountMode<Op, T, dstRowStride, srcRowStride>(dst, src, validRow, validCol);
    } else if constexpr (dstRow < (normColRepeat + 1)) {
        unsigned tailElements = validCol % nRepeatElem;
        if (tailElements) {
            Unary2LCountMode<Op, T, dstRowStride, srcRowStride>(dst, src, validRow, validCol);
        } else {
            Unary2LNormModeColVLAlign<Op, T, dstRowStride, srcRowStride, nRepeatElem>(dst, src, validRow, validCol);
        }
    } else {
        Unary2LNormModeRowRpt<Op, T, dstRowStride, srcRowStride, dstRow, srcRow, nRepeatElem>(dst, src, validRow,
                                                                                              validCol);
    }
}

template <typename Op, typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TUnaryOp(typename DstTile::TileDType __out__ dstData,
                                  typename SrcTile::TileDType __in__ srcData, unsigned validRow, unsigned validCol)
{
    using TRANS = B82B16Trait<typename DstTile::DType>;
    using T = typename TRANS::TransType;
    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *src = (__ubuf__ T *)__cce_get_tile_ptr(srcData);
    constexpr int nRepeatElem = REPEAT_BYTE / sizeof(T);
    constexpr unsigned dstRow = DstTile::Rows;
    constexpr unsigned srcRow = SrcTile::Rows;
    constexpr unsigned dstRowStride = TRANS::template TransStride<DstTile::RowStride>();
    constexpr unsigned srcRowStride = TRANS::template TransStride<SrcTile::RowStride>();
    constexpr unsigned dstCol = TRANS::template TransStride<DstTile::Cols>();
    int transValidCol = TRANS::TransSize(validCol);

    constexpr bool isCombined = ((DstTile::ValidCol == DstTile::Cols) && (SrcTile::ValidCol == SrcTile::Cols)) ||
                                ((dstRow == 1) && (srcRow == 1));

    if constexpr (isCombined) {
        constexpr unsigned totalRepeats = (dstRow * dstCol + nRepeatElem - 1) / nRepeatElem;
        if constexpr (totalRepeats > pto::REPEAT_MAX) {
            Unary1LCountMode<Op, T>(dst, src, validRow, transValidCol);
        } else {
            Unary1LNormMode<Op, T>(dst, src, validRow, transValidCol);
        }
    } else {
        constexpr bool isSameShape = (DstTile::Cols == SrcTile::Cols) && (dstRow == srcRow);
        if constexpr (isSameShape) {
            if ((transValidCol == dstCol) || (validRow == 1)) {
                unsigned totalRepeats = (validRow * transValidCol + nRepeatElem - 1) / nRepeatElem;
                if (totalRepeats > pto::REPEAT_MAX) {
                    Unary1LCountMode<Op, T>(dst, src, validRow, transValidCol);
                } else {
                    Unary1LNormMode<Op, T>(dst, src, validRow, transValidCol);
                }
            } else {
                Unary2LProcess<Op, T, dstRowStride, srcRowStride, dstRow, srcRow, dstCol, nRepeatElem>(
                    dst, src, validRow, transValidCol);
            }
        } else {
            Unary2LProcess<Op, T, dstRowStride, srcRowStride, dstRow, srcRow, dstCol, nRepeatElem>(dst, src, validRow,
                                                                                                   transValidCol);
        }
    }
}

template <typename DstTile, typename SrcTile, bool floatOnly = true>
PTO_INTERNAL void TunaryCheck()
{
    static_assert(DstTile::isRowMajor && SrcTile::isRowMajor, "TUnaryOp: Not supported Layout type");
    static_assert(DstTile::Loc == TileType::Vec && SrcTile::Loc == TileType::Vec,
                  "TUnaryOp: TileType of src and dst tiles must be TileType::Vec.");
    static_assert(SrcTile::ValidCol <= SrcTile::Cols,
                  "TUnaryOp: Number of src's valid columns must not be greater than number of tile columns.");
    static_assert(DstTile::ValidCol <= DstTile::Cols,
                  "TUnaryOp: Number of dst's valid columns must not be greater than number of tile columns.");
    static_assert(SrcTile::ValidRow <= SrcTile::Rows,
                  "TUnaryOp: Number of src's valid rows must not be greater than number of tile rows.");
    static_assert(DstTile::ValidRow <= DstTile::Rows,
                  "TUnaryOp: Number of dst's valid rows must not be greater than number of tile rows.");
    static_assert(std::is_same<typename DstTile::DType, typename SrcTile::DType>::value,
                  "TUnaryOp: The data type of dst must be consistent with of src");
    static_assert(!floatOnly || std::is_same<typename DstTile::DType, float32_t>::value ||
                      std::is_same<typename DstTile::DType, float>::value ||
                      std::is_same<typename DstTile::DType, half>::value ||
                      std::is_same<typename DstTile::DType, float16_t>::value,
                  "TUNARY: Invalid data type");
}

template <typename Op, typename DstTile, typename SrcTile, bool floatOnly = true>
PTO_INTERNAL void TUNARY_IMPL(DstTile &dst, SrcTile &src)
{
    TunaryCheck<DstTile, SrcTile, floatOnly>();
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    PTO_ASSERT(dstValidRow == src.GetValidRow(), "TUNARY: Number of rows of src and dst must be the same.");
    PTO_ASSERT(dstValidCol == src.GetValidCol(), "TUNARY: Number of columns of src and dst must be the same.");
    TUnaryOp<Op, DstTile, SrcTile>(dst.data(), src.data(), dstValidRow, dstValidCol);
}

/* RSQRT */
template <typename T>
struct RsqrtOp {
    PTO_INTERNAL static void UnaryInstr(__ubuf__ T *dst, __ubuf__ T *src, uint8_t repeat,
                                        uint8_t dstStride = BLOCK_MAX_PER_REPEAT,
                                        uint8_t srcStride = BLOCK_MAX_PER_REPEAT)
    {
        vrsqrt(dst, src, repeat, 1, 1, dstStride, srcStride);
    }
};
template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TRSQRT_IMPL(DstTile &dst, SrcTile &src)
{
    TUNARY_IMPL<RsqrtOp<typename DstTile::DType>>(dst, src);
}

template <typename DstTile, typename SrcTile, typename TmpTile>
__tf__ PTO_INTERNAL void TRsqrtHighPrecision(typename DstTile::TileDType __out__ dstData,
                                             typename SrcTile::TileDType __in__ srcData,
                                             typename TmpTile::TileDType __in__ tmpData, unsigned validRow,
                                             unsigned validCol)
{
    using T = typename DstTile::DType;
    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *src = (__ubuf__ T *)__cce_get_tile_ptr(srcData);
    __ubuf__ T *tmp = (__ubuf__ T *)__cce_get_tile_ptr(tmpData);

    constexpr unsigned dstStride = DstTile::RowStride;
    constexpr unsigned srcStride = SrcTile::RowStride;

    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);

    set_mask_count();
    set_vector_mask(0, blockSizeElem);

    vector_dup(tmp, (T)(1.0), 0, 0, 0, 0, 0);

    set_vector_mask(0, validCol);
    for (uint32_t i = 0; i < validRow; ++i) {
        vsqrt(dst + i * dstStride, src + i * srcStride, 0, 1, 1, 8, 8);
    }

    pipe_barrier(PIPE_V);

    for (uint32_t i = 0; i < validRow; ++i) {
        vdiv(dst + i * dstStride, tmp, dst + i * dstStride, 0, 1, 0, 1, 8, 0, 8);
    }

    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <typename DstTile, typename SrcTile, typename TmpTile>
PTO_INTERNAL void TRSQRT_IMPL(DstTile &dst, SrcTile &src, TmpTile &tmp)
{
    TunaryCheck<DstTile, SrcTile>();
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    PTO_ASSERT(dstValidRow == src.GetValidRow(), "TRSQRT: Number of rows of src and dst must be the same.");
    PTO_ASSERT(dstValidCol == src.GetValidCol(), "TRSQRT: Number of columns of src and dst must be the same.");
    TRsqrtHighPrecision<DstTile, SrcTile, TmpTile>(dst.data(), src.data(), tmp.data(), dstValidRow, dstValidCol);
}

/* SQRT */
template <typename T>
struct SqrtOp {
    PTO_INTERNAL static void UnaryInstr(__ubuf__ T *dst, __ubuf__ T *src, uint8_t repeat,
                                        uint8_t dstStride = BLOCK_MAX_PER_REPEAT,
                                        uint8_t srcStride = BLOCK_MAX_PER_REPEAT)
    {
        vsqrt(dst, src, repeat, 1, 1, dstStride, srcStride);
    }
};
template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TSQRT_IMPL(DstTile &dst, SrcTile &src)
{
    TUNARY_IMPL<SqrtOp<typename DstTile::DType>>(dst, src);
}

/* EXP */
template <typename T>
struct ExpOp {
    PTO_INTERNAL static void UnaryInstr(__ubuf__ T *dst, __ubuf__ T *src, uint8_t repeat,
                                        uint8_t dstStride = BLOCK_MAX_PER_REPEAT,
                                        uint8_t srcStride = BLOCK_MAX_PER_REPEAT)
    {
        vexp(dst, src, repeat, 1, 1, dstStride, srcStride);
    }
};
template <auto PrecisionType = ExpAlgorithm::DEFAULT, typename DstTile, typename SrcTile>
PTO_INTERNAL void TEXP_IMPL(DstTile &dst, SrcTile &src)
{
    TUNARY_IMPL<ExpOp<typename DstTile::DType>>(dst, src);
}

/* NOT */
template <typename T>
struct NotOp {
    PTO_INTERNAL static void UnaryInstr(__ubuf__ T *dst, __ubuf__ T *src, uint8_t repeat,
                                        uint8_t dstStride = BLOCK_MAX_PER_REPEAT,
                                        uint8_t srcStride = BLOCK_MAX_PER_REPEAT)
    {
        vnot(dst, src, repeat, 1, 1, dstStride, srcStride);
    }
};
template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TNOT_IMPL(DstTile &dst, SrcTile &src)
{
    using TransType = typename B82B16Trait<typename DstTile::DType>::TransType;
    TUNARY_IMPL<NotOp<TransType>, DstTile, SrcTile, false>(dst, src);
}

/* RELU */
template <typename T>
struct ReluOp {
    PTO_INTERNAL static void UnaryInstr(__ubuf__ T *dst, __ubuf__ T *src, uint8_t repeat,
                                        uint8_t dstStride = BLOCK_MAX_PER_REPEAT,
                                        uint8_t srcStride = BLOCK_MAX_PER_REPEAT)
    {
        vrelu(dst, src, repeat, 1, 1, dstStride, srcStride);
    }
};
template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TRELU_IMPL(DstTile &dst, SrcTile &src)
{
    TUNARY_IMPL<ReluOp<typename DstTile::DType>, DstTile, SrcTile, false>(dst, src);
}

/* ABS */
template <typename T>
struct AbsOp {
    PTO_INTERNAL static void UnaryInstr(__ubuf__ T *dst, __ubuf__ T *src, uint8_t repeat,
                                        uint8_t dstStride = BLOCK_MAX_PER_REPEAT,
                                        uint8_t srcStride = BLOCK_MAX_PER_REPEAT)
    {
        vabs(dst, src, repeat, 1, 1, dstStride, srcStride);
    }
};
template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TABS_IMPL(DstTile &dst, SrcTile &src)
{
    TUNARY_IMPL<AbsOp<typename DstTile::DType>>(dst, src);
}

/* LOG */
template <typename T>
struct LogOp {
    PTO_INTERNAL static void UnaryInstr(__ubuf__ T *dst, __ubuf__ T *src, uint8_t repeat,
                                        uint8_t dstStride = BLOCK_MAX_PER_REPEAT,
                                        uint8_t srcStride = BLOCK_MAX_PER_REPEAT)
    {
        vln(dst, src, repeat, 1, 1, dstStride, srcStride);
    }
};
template <auto PrecisionType = LogAlgorithm::DEFAULT, typename DstTile, typename SrcTile>
PTO_INTERNAL void TLOG_IMPL(DstTile &dst, SrcTile &src)
{
    TUNARY_IMPL<LogOp<typename DstTile::DType>>(dst, src);
}

/* TNEG */
template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TNEG_IMPL(DstTile &dst, SrcTile &src)
{
    TMULS_IMPL(dst, src, -1);
}
} // namespace pto

#endif
