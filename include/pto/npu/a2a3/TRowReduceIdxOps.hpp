/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef T_ROW_REDUCE_IDX_OPS_HPP
#define T_ROW_REDUCE_IDX_OPS_HPP

#include <pto/common/utils.hpp>
#include <pto/common/type.hpp>

#ifndef B16_REPEAT_MAX
#define B16_REPEAT_MAX 65535
#endif

namespace pto {
template <typename TDst, typename TSrc, typename InstrOp>
struct TRowReduceIdxOp {
    PTO_INTERNAL static void ReduceIdxInstr(__ubuf__ TDst *dst, __ubuf__ TSrc *src, uint8_t rptTimes,
                                            uint16_t dstRptStride, uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        InstrOp::ReduceIdxInstrImpl((__ubuf__ TSrc *)dst, src, rptTimes, dstRptStride, srcBlkStride, srcRptStride);
    }
    PTO_INTERNAL static void ReduceValIdxInstr(__ubuf__ TSrc *dst, __ubuf__ TSrc *src, uint8_t rptTimes,
                                               uint16_t dstRptStride, uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        InstrOp::ReduceValIdxInstrImpl(dst, src, rptTimes, dstRptStride, srcBlkStride, srcRptStride);
    }

    template <bool CntModeEn, int Cols, uint32_t DstStride, uint32_t SrcStride>
    PTO_INTERNAL static void ReduceInstrByModeIdx(__ubuf__ TDst *dst, __ubuf__ TSrc *src, unsigned rptTimes)
    {
        constexpr uint8_t elemPerRpt = REPEAT_BYTE / sizeof(TSrc);
        if constexpr (DstStride > B16_REPEAT_MAX) {
            for (int i = 0; i < rptTimes; i++) {
                ReduceIdxInstr(dst + i * DstStride, src + i * Cols, 1, 0, 1, 0);
            }
        } else if constexpr (CntModeEn) {
            set_mask_count();
            set_vector_mask(0, (uint32_t)rptTimes * elemPerRpt);
            ReduceIdxInstr(dst, src, 0, DstStride, 1, SrcStride);
            set_mask_norm();
            set_vector_mask(-1, -1);
        } else {
            ReduceIdxInstr(dst, src, rptTimes, DstStride, 1, SrcStride);
        }
    }
};

template <typename InstrOp, typename TVal, typename TIdx>
PTO_INTERNAL void ReduceThenGroupValIdx(__ubuf__ TVal *dstVal, __ubuf__ TIdx *dstIdx, __ubuf__ TVal *src,
                                        uint32_t count)
{
    constexpr uint8_t elemPerRpt = REPEAT_BYTE / sizeof(TVal);
    constexpr uint8_t elemPerBlock = BLOCK_BYTE_SIZE / sizeof(TVal);
    set_vector_mask(0, count);
    InstrOp::ReduceValIdxInstr(reinterpret_cast<__ubuf__ TVal *>(dstIdx), src, 0, 1, 1, REPEAT_BYTE / BLOCK_BYTE_SIZE);
    pipe_barrier(PIPE_V);
    set_vector_mask(0, CeilDivision(count, elemPerRpt) * 2);
    vreducev2(reinterpret_cast<__ubuf__ TIdx *>(dstVal), dstIdx, dstIdx, 1, 1, 1, elemPerBlock, elemPerBlock);
    pipe_barrier(PIPE_V);
    vreducev2(dstIdx, dstIdx, dstIdx, 1, 1, 2, elemPerBlock, elemPerBlock);
    pipe_barrier(PIPE_V);
}

template <typename InstrOp, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void ProcReduceIdxStage2(__ubuf__ typename TileDataOut::DType *dst,
                                      __ubuf__ typename TileDataIn::DType *src,
                                      __ubuf__ typename TileDataTmp::DType *tmp, int validRow, int validCol)
{
    using T = typename TileDataIn::DType;
    using U = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint16_t>;
    using TDest = typename TileDataOut::DType;
    constexpr uint8_t elemPerRpt = REPEAT_BYTE / sizeof(T);
    constexpr uint8_t elemPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr uint32_t srcRptStride = TileDataIn::Cols / elemPerBlock;
    size_t tempElementsStage1 = CeilDivision(validCol, elemPerRpt);
    size_t tempElementsStage2 = CeilDivision(tempElementsStage1, elemPerRpt);
    constexpr size_t tempIdxOffsetStage1 = 0;
    size_t tempValOffsetStage1 =
        tempIdxOffsetStage1 + CeilDivision(tempElementsStage1 * 2, elemPerBlock) * elemPerBlock;
    size_t tempOffsetStage2 = tempIdxOffsetStage1 + CeilDivision(tempElementsStage1, elemPerBlock) * elemPerBlock;
    size_t tempIdxOffsetStage2 = tempOffsetStage2;
    size_t tempValOffsetStage2 =
        tempIdxOffsetStage2 + CeilDivision(tempElementsStage2 * 2, elemPerBlock) * elemPerBlock;
    size_t tempIdxOffsetFinal = tempValOffsetStage2;
    set_mask_count();
    for (int i = 0; i < validRow; i++) {
        ReduceThenGroupValIdx<InstrOp, T, U>(
            reinterpret_cast<__ubuf__ T *>(tmp + i * TileDataTmp::Cols + tempValOffsetStage1),
            reinterpret_cast<__ubuf__ U *>(tmp + i * TileDataTmp::Cols + tempIdxOffsetStage1),
            src + i * TileDataIn::Cols, (uint32_t)validCol);
        ReduceThenGroupValIdx<InstrOp, T, U>(
            reinterpret_cast<__ubuf__ T *>(tmp + i * TileDataTmp::Cols + tempValOffsetStage2),
            reinterpret_cast<__ubuf__ U *>(tmp + i * TileDataTmp::Cols + tempIdxOffsetStage2),
            tmp + i * TileDataTmp::Cols + tempValOffsetStage1, (uint32_t)tempElementsStage1);
        set_vector_mask(0, (uint32_t)(tempElementsStage2));
        InstrOp::ReduceIdxInstr(
            reinterpret_cast<__ubuf__ typename TileDataOut::DType *>(tmp + i * TileDataTmp::Cols + tempIdxOffsetFinal),
            reinterpret_cast<__ubuf__ T *>(tmp + i * TileDataTmp::Cols + tempValOffsetStage2), 1, 1, 1, 0);
        pipe_barrier(PIPE_V);
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
    for (int i = 0; i < validRow; i++) {
        __ubuf__ U *idxArrStage1 = (reinterpret_cast<__ubuf__ U *>(tmp + i * TileDataTmp::Cols + tempIdxOffsetStage1));
        __ubuf__ U *idxArrStage2 = (reinterpret_cast<__ubuf__ U *>(tmp + i * TileDataTmp::Cols + tempIdxOffsetStage2));
        U idxStage2 = *(reinterpret_cast<__ubuf__ U *>(tmp + i * TileDataTmp::Cols + tempIdxOffsetFinal));
        TDest idxStage1 = (TDest)idxStage2 * elemPerRpt + idxArrStage2[idxStage2];
        *(dst + i * TileDataOut::Cols) = idxStage1 * elemPerRpt + idxArrStage1[idxStage1];
    }
}

template <typename InstrOp, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void ProcReduceIdxStage1(__ubuf__ typename TileDataOut::DType *dst,
                                      __ubuf__ typename TileDataIn::DType *src,
                                      __ubuf__ typename TileDataTmp::DType *tmp, int validRow, int validCol)
{
    using T = typename TileDataIn::DType;
    using U = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint16_t>;
    using TDest = typename TileDataOut::DType;
    constexpr uint8_t elemPerRpt = REPEAT_BYTE / sizeof(T);
    constexpr uint8_t elemPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr uint32_t srcRptStride = TileDataIn::Cols / elemPerBlock;
    size_t tempElements = CeilDivision(validCol, elemPerRpt);
    size_t tempOffset = CeilDivision(tempElements * 2, elemPerBlock) * elemPerBlock;
    U idxStage1;
    __ubuf__ U *idxArr;

    set_mask_count();
    for (int i = 0; i < validRow; i++) {
        ReduceThenGroupValIdx<InstrOp, T, U>(reinterpret_cast<__ubuf__ T *>(tmp + i * TileDataTmp::Cols + tempOffset),
                                             reinterpret_cast<__ubuf__ U *>(tmp + i * TileDataTmp::Cols),
                                             src + i * TileDataIn::Cols, (uint32_t)validCol);
        set_vector_mask(0, (uint32_t)(tempElements));
        InstrOp::ReduceIdxInstr(
            reinterpret_cast<__ubuf__ typename TileDataOut::DType *>(tmp + i * TileDataTmp::Cols + tempOffset),
            reinterpret_cast<__ubuf__ T *>(tmp + i * TileDataTmp::Cols + tempOffset), 1, 1, 1, 0);
        pipe_barrier(PIPE_V);
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
    for (int i = 0; i < validRow; i++) {
        idxArr = reinterpret_cast<__ubuf__ U *>(tmp + i * TileDataTmp::Cols);
        idxStage1 = *(reinterpret_cast<__ubuf__ U *>(tmp + i * TileDataTmp::Cols + tempOffset));
        TDest idx_final = (TDest)idxStage1 * elemPerRpt + idxArr[idxStage1];
        *(dst + i * TileDataOut::Cols) = idx_final;
    }
}

template <typename InstrOp, typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void OneRepeatProcIdx(__ubuf__ typename TileDataOut::DType *dst, __ubuf__ typename TileDataIn::DType *src,
                                   int validRow, int validCol)
{
    constexpr uint8_t elemPerRpt = REPEAT_BYTE / sizeof(typename TileDataIn::DType);
    constexpr uint8_t elemPerBlock = BLOCK_BYTE_SIZE / sizeof(typename TileDataIn::DType);
    constexpr uint32_t srcRptStride = TileDataIn::Cols / elemPerBlock;

    if (validCol == elemPerRpt) {
        InstrOp::template ReduceInstrByModeIdx<true, TileDataIn::Cols, TileDataOut::Cols, srcRptStride>(dst, src,
                                                                                                        validRow);
        pipe_barrier(PIPE_V);
        return;
    }

    int remain = validCol % elemPerRpt;
    int rowRptTimes = validRow / REPEAT_MAX;
    unsigned rptTimes;
    SetContinuousMask(remain);
    do {
        rptTimes = (rowRptTimes == 0 ? (validRow % REPEAT_MAX) : REPEAT_MAX);
        InstrOp::template ReduceInstrByModeIdx<false, TileDataIn::Cols, TileDataOut::Cols, srcRptStride>(dst, src,
                                                                                                         rptTimes);
        pipe_barrier(PIPE_V);
        rowRptTimes -= 1;
        dst += rptTimes * TileDataOut::Cols;
        src += rptTimes * TileDataIn::Cols;
    } while (rowRptTimes >= 0);

    set_vector_mask(-1, -1);
}

template <typename InstrOp, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TRowReduceIdxInstr(__ubuf__ typename TileDataOut::DType *dst,
                                     __ubuf__ typename TileDataIn::DType *src,
                                     __ubuf__ typename TileDataTmp::DType *tmp, int validRow, int validCol,
                                     int dstValidRow)
{
    TRowReduceCheck<TileDataOut, TileDataIn, true>(validRow, validCol, dstValidRow);
    constexpr uint8_t elemPerRpt = REPEAT_BYTE / sizeof(typename TileDataIn::DType);
    if (validCol <= elemPerRpt) {
        OneRepeatProcIdx<InstrOp, TileDataOut, TileDataIn>(dst, src, validRow, validCol);
        return;
    } else if (validCol <= elemPerRpt * elemPerRpt) {
        ProcReduceIdxStage1<InstrOp, TileDataOut, TileDataIn, TileDataTmp>(dst, src, tmp, validRow, validCol);
        return;
    } else {
        ProcReduceIdxStage2<InstrOp, TileDataOut, TileDataIn, TileDataTmp>(dst, src, tmp, validRow, validCol);
        return;
    }
}

template <typename TDst, typename TSrc>
struct TRowArgMaxOp : TRowReduceIdxOp<TDst, TSrc, TRowArgMaxOp<TDst, TSrc>> {
    PTO_INTERNAL static void ReduceIdxInstrImpl(__ubuf__ TSrc *dst, __ubuf__ TSrc *src, uint8_t rptTimes,
                                                uint16_t dstRptStride, uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        vcmax(dst, src, rptTimes, dstRptStride, srcBlkStride, srcRptStride, ONLY_INDEX);
    }

    PTO_INTERNAL static void ReduceValIdxInstrImpl(__ubuf__ TSrc *dst, __ubuf__ TSrc *src, uint8_t rptTimes,
                                                   uint16_t dstRptStride, uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        vcmax(dst, src, rptTimes, dstRptStride, srcBlkStride, srcRptStride, VALUE_INDEX);
    }
};

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
__tf__ PTO_INTERNAL void TRowIdxMax(typename TileDataOut::TileDType __out__ dstData,
                                    typename TileDataIn::TileDType __in__ srcData,
                                    typename TileDataTmp::TileDType __in__ tmpData, int validRow, int validCol,
                                    int dstValidRow, unsigned version)
{
    using TDst = typename TileDataOut::DType;
    using TSrc = typename TileDataIn::DType;
    __ubuf__ TDst *dst = (__ubuf__ TDst *)__cce_get_tile_ptr(dstData);
    __ubuf__ TSrc *src = (__ubuf__ TSrc *)__cce_get_tile_ptr(srcData);
    __ubuf__ TSrc *tmp = (__ubuf__ TSrc *)__cce_get_tile_ptr(tmpData);
    TRowReduceIdxInstr<TRowArgMaxOp<TDst, TSrc>, TileDataOut, TileDataIn, TileDataTmp>(dst, src, tmp, validRow,
                                                                                       validCol, dstValidRow);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWARGMAX_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    int validCol = src.GetValidCol();
    int validRow = src.GetValidRow();
    int dstValidRow = dst.GetValidRow();
    TRowIdxMax<TileDataOut, TileDataIn, TileDataTmp>(dst.data(), src.data(), tmp.data(), validRow, validCol,
                                                     dstValidRow, VFImplKind::VFIMPL_DEFAULT);
}

template <typename TDst, typename TSrc>
struct TRowArgMinOp : TRowReduceIdxOp<TDst, TSrc, TRowArgMinOp<TDst, TSrc>> {
    PTO_INTERNAL static void ReduceIdxInstrImpl(__ubuf__ TSrc *dst, __ubuf__ TSrc *src, uint8_t rptTimes,
                                                uint16_t dstRptStride, uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        vcmin(dst, src, rptTimes, dstRptStride, srcBlkStride, srcRptStride, ONLY_INDEX);
    }

    PTO_INTERNAL static void ReduceValIdxInstrImpl(__ubuf__ TSrc *dst, __ubuf__ TSrc *src, uint8_t rptTimes,
                                                   uint16_t dstRptStride, uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        vcmin(dst, src, rptTimes, dstRptStride, srcBlkStride, srcRptStride, VALUE_INDEX);
    }
};

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
__tf__ PTO_INTERNAL void TRowIdxMin(typename TileDataOut::TileDType __out__ dstData,
                                    typename TileDataIn::TileDType __in__ srcData,
                                    typename TileDataTmp::TileDType __in__ tmpData, int validRow, int validCol,
                                    int dstValidRow, unsigned version)
{
    using TDst = typename TileDataOut::DType;
    using TSrc = typename TileDataIn::DType;
    __ubuf__ TDst *dst = (__ubuf__ TDst *)__cce_get_tile_ptr(dstData);
    __ubuf__ TSrc *src = (__ubuf__ TSrc *)__cce_get_tile_ptr(srcData);
    __ubuf__ TSrc *tmp = (__ubuf__ TSrc *)__cce_get_tile_ptr(tmpData);
    TRowReduceIdxInstr<TRowArgMinOp<TDst, TSrc>, TileDataOut, TileDataIn, TileDataTmp>(dst, src, tmp, validRow,
                                                                                       validCol, dstValidRow);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWARGMIN_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    int validCol = src.GetValidCol();
    int validRow = src.GetValidRow();
    int dstValidRow = dst.GetValidRow();
    TRowIdxMin<TileDataOut, TileDataIn, TileDataTmp>(dst.data(), src.data(), tmp.data(), validRow, validCol,
                                                     dstValidRow, VFImplKind::VFIMPL_DEFAULT);
}

} // namespace pto
#endif
