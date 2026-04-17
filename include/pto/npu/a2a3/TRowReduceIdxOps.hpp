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
template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, bool outputVal>
PTO_INTERNAL void TRowReduceIdxCheck(uint32_t srcValidRows, uint32_t srcValidCols, uint32_t dstValValidRow,
                                     uint32_t dstIdxValidRow)
{
    using TIdx = typename TileDataOutIdx::DType;
    using TVal = typename TileDataIn::DType;
    if constexpr (outputVal) {
        static_assert(
            (sizeof(TVal) == sizeof(float) && (std::is_same_v<int32_t, TIdx> || std::is_same_v<uint32_t, TIdx>)) ||
                (sizeof(TVal) == sizeof(half) && (std::is_same_v<int16_t, TIdx> || std::is_same_v<uint16_t, TIdx>)),
            "Input and output tile data types must match. "
            "Fix: Ensure TileDataOutIdx uses the same DType as TileDataIn.");
        TRowReduceCheck<TileDataOutVal, TileDataIn, false>(srcValidRows, srcValidCols, dstValValidRow);
    } else {
        static_assert(std::is_same_v<uint32_t, TIdx> || std::is_same_v<int32_t, TIdx>,
                      "Input and output tile data types must match. "
                      "Fix: Ensure TileDataOutIdx uses the same DType as TileDataIn.");
    }
    TRowReduceCheck<TileDataOutIdx, TileDataIn, true>(srcValidRows, srcValidCols, dstIdxValidRow);
}

template <typename InstrOp, typename TVal, typename TIdx>
PTO_INTERNAL void ReduceThenGroupValIdx(__ubuf__ TVal *dstVal, __ubuf__ TIdx *dstIdx, __ubuf__ TVal *src,
                                        uint32_t count)
{
    constexpr uint8_t elemPerRpt = REPEAT_BYTE / sizeof(TVal);
    constexpr uint8_t elemPerBlock = BLOCK_BYTE_SIZE / sizeof(TVal);
    set_vector_mask(0, count);
    InstrOp::ReduceValIdxInstrImpl(reinterpret_cast<__ubuf__ TVal *>(dstIdx), src, 0, 1, 1,
                                   REPEAT_BYTE / BLOCK_BYTE_SIZE);
    pipe_barrier(PIPE_V);
    set_vector_mask(0, CeilDivision(count, elemPerRpt) * 2);
    vreducev2(reinterpret_cast<__ubuf__ TIdx *>(dstVal), dstIdx, dstIdx, 1, 1, 1, elemPerBlock, elemPerBlock);
    pipe_barrier(PIPE_V);
    vreducev2(dstIdx, dstIdx, dstIdx, 1, 1, 2, elemPerBlock, elemPerBlock);
    pipe_barrier(PIPE_V);
}

template <bool outputVal, typename InstrOp, typename TileDataOutVal, typename TileDataOut, typename TileDataIn,
          typename TileDataTmp>
PTO_INTERNAL void ProcReduceIdxStage2(__ubuf__ typename TileDataOutVal::DType *dstVal,
                                      __ubuf__ typename TileDataOut::DType *dst,
                                      __ubuf__ typename TileDataIn::DType *src,
                                      __ubuf__ typename TileDataTmp::DType *tmp, int validRow, int validCol)
{
    using TIdxFinal = typename TileDataOut::DType;
    using T = typename TileDataIn::DType;
    using U = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint16_t>;
    constexpr uint8_t elemPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr uint8_t elemPerRpt = REPEAT_BYTE / sizeof(T);
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
        InstrOp::ReduceValIdxInstrImpl(
            reinterpret_cast<__ubuf__ T *>(tmp + i * TileDataTmp::Cols + tempIdxOffsetFinal),
            reinterpret_cast<__ubuf__ T *>(tmp + i * TileDataTmp::Cols + tempValOffsetStage2), 1, 1, 1, 0);
        pipe_barrier(PIPE_V);
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
    for (int i = 0; i < validRow; i++) {
        __ubuf__ U *idxArrStage1 = (reinterpret_cast<__ubuf__ U *>(tmp + i * TileDataTmp::Cols + tempIdxOffsetStage1));
        __ubuf__ U *idxArrStage2 = (reinterpret_cast<__ubuf__ U *>(tmp + i * TileDataTmp::Cols + tempIdxOffsetStage2));
        U idxStage2 = *(reinterpret_cast<__ubuf__ U *>(tmp + i * TileDataTmp::Cols + tempIdxOffsetFinal + 1));
        TIdxFinal idxStage1 = (TIdxFinal)idxStage2 * elemPerRpt + idxArrStage2[idxStage2];
        *(dst + i * TileDataOut::Cols) = idxStage1 * elemPerRpt + idxArrStage1[idxStage1];
        if constexpr (outputVal) {
            *(dstVal + i * TileDataOutVal::Cols) =
                *(reinterpret_cast<__ubuf__ T *>(tmp + i * TileDataTmp::Cols + tempIdxOffsetFinal));
        }
    }
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
}

template <bool outputVal, typename InstrOp, typename TileDataOutVal, typename TileDataOut, typename TileDataIn,
          typename TileDataTmp>
PTO_INTERNAL void ProcReduceIdxStage1(__ubuf__ typename TileDataOutVal::DType *dstVal,
                                      __ubuf__ typename TileDataOut::DType *dst,
                                      __ubuf__ typename TileDataIn::DType *src,
                                      __ubuf__ typename TileDataTmp::DType *tmp, int validRow, int validCol)
{
    using T = typename TileDataIn::DType;
    using U = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint16_t>;
    using TIdxFinal = typename TileDataOut::DType;
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
        InstrOp::ReduceValIdxInstrImpl(reinterpret_cast<__ubuf__ T *>(tmp + i * TileDataTmp::Cols + tempOffset),
                                       reinterpret_cast<__ubuf__ T *>(tmp + i * TileDataTmp::Cols + tempOffset), 1, 1,
                                       1, 0);
        pipe_barrier(PIPE_V);
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
    for (int i = 0; i < validRow; i++) {
        idxArr = reinterpret_cast<__ubuf__ U *>(tmp + i * TileDataTmp::Cols);
        idxStage1 = *(reinterpret_cast<__ubuf__ U *>(tmp + i * TileDataTmp::Cols + tempOffset + 1));
        TIdxFinal idx_final = (TIdxFinal)idxStage1 * elemPerRpt + idxArr[idxStage1];
        *(dst + i * TileDataOut::Cols) = idx_final;
        if constexpr (outputVal) {
            *(dstVal + i * TileDataOutVal::Cols) =
                *(reinterpret_cast<__ubuf__ T *>(tmp + i * TileDataTmp::Cols + tempOffset));
        }
    }
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
}

template <typename InstrOp, typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void OneRepeatProcIdx(__ubuf__ typename TileDataOut::DType *dst, __ubuf__ typename TileDataIn::DType *src,
                                   int validRow, int validCol)
{
    using T = typename TileDataIn::DType;
    constexpr uint8_t elemPerRpt = REPEAT_BYTE / sizeof(typename TileDataIn::DType);
    constexpr uint8_t elemPerBlock = BLOCK_BYTE_SIZE / sizeof(typename TileDataIn::DType);
    constexpr uint32_t srcRptStride = TileDataIn::Cols / elemPerBlock;

    if constexpr (TileDataOut::Cols > B16_REPEAT_MAX) {
        set_mask_count();
        set_vector_mask(0, validCol);
        for (int i = 0; i < validRow; i++) {
            InstrOp::ReduceIdxInstrImpl(reinterpret_cast<__ubuf__ T *>(dst) + i * TileDataOut::Cols,
                                        src + i * TileDataIn::Cols, 1, 0, 1, 0);
        }
        pipe_barrier(PIPE_V);
    } else {
        if (validCol == elemPerRpt) {
            set_mask_count();
            set_vector_mask(0, (uint32_t)validRow * elemPerRpt);
            InstrOp::ReduceIdxInstrImpl(reinterpret_cast<__ubuf__ T *>(dst), src, 0, TileDataOut::Cols, 1,
                                        srcRptStride);
            pipe_barrier(PIPE_V);
        } else {
            int remain = validCol % elemPerRpt;
            int rowRptTimes = validRow / REPEAT_MAX;
            unsigned rptTimes;
            set_mask_norm();
            SetContinuousMask(remain);
            do {
                rptTimes = (rowRptTimes == 0 ? (validRow % REPEAT_MAX) : REPEAT_MAX);
                InstrOp::ReduceIdxInstrImpl(reinterpret_cast<__ubuf__ T *>(dst), src, rptTimes, TileDataOut::Cols, 1,
                                            srcRptStride);
                pipe_barrier(PIPE_V);
                rowRptTimes -= 1;
                dst += rptTimes * TileDataOut::Cols;
                src += rptTimes * TileDataIn::Cols;
            } while (rowRptTimes >= 0);
        }
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <typename TileDataOutVal, typename TileDataOutIdx, typename TileDataTmp>
PTO_INTERNAL void ExtractValIdxFromTmp(__ubuf__ typename TileDataOutVal::DType *dstVal,
                                       __ubuf__ typename TileDataOutIdx::DType *dstIdx,
                                       __ubuf__ typename TileDataTmp::DType *tmp, int validRow)
{
    using T = typename TileDataOutVal::DType;
    using U = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint16_t>;
    constexpr uint8_t elemPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
    if constexpr (TileDataOutVal::Cols == 1 || TileDataOutIdx::Cols == 1) {
        set_mask_count();
        set_vector_mask(0, validRow * 2);
        if constexpr (TileDataOutIdx::Cols == 1) {
            vreducev2(reinterpret_cast<__ubuf__ U *>(dstIdx), reinterpret_cast<__ubuf__ U *>(tmp),
                      reinterpret_cast<__ubuf__ U *>(tmp), 1, 1, 2, elemPerBlock, elemPerBlock);
            pipe_barrier(PIPE_V);
        }
        if constexpr (TileDataOutVal::Cols == 1) {
            vreducev2(reinterpret_cast<__ubuf__ U *>(dstVal), reinterpret_cast<__ubuf__ U *>(tmp),
                      reinterpret_cast<__ubuf__ U *>(tmp), 1, 1, 1, elemPerBlock, elemPerBlock);
            pipe_barrier(PIPE_V);
        }
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
    if constexpr (TileDataOutVal::Cols != 1 || TileDataOutIdx::Cols != 1) {
        PtoSetWaitFlag<PIPE_V, PIPE_S>();
        if constexpr (TileDataOutIdx::Cols != 1) {
            for (int i = 0; i < validRow; i++) {
                *(reinterpret_cast<__ubuf__ U *>(dstIdx) + i * TileDataOutIdx::Cols) =
                    *(reinterpret_cast<__ubuf__ U *>(tmp) + i * 2 + 1);
            }
        }
        if constexpr (TileDataOutVal::Cols != 1) {
            for (int i = 0; i < validRow; i++) {
                *(reinterpret_cast<__ubuf__ U *>(dstVal) + i * TileDataOutVal::Cols) =
                    *(reinterpret_cast<__ubuf__ U *>(tmp) + i * 2);
            }
        }
        PtoSetWaitFlag<PIPE_S, PIPE_V>();
    }
}

template <typename InstrOp, typename TileDataOutVal, typename TileDataOutIdx, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void OneRepeatProcValIdx(__ubuf__ typename TileDataOutVal::DType *dstVal,
                                      __ubuf__ typename TileDataOutIdx::DType *dstIdx,
                                      __ubuf__ typename TileDataIn::DType *src,
                                      __ubuf__ typename TileDataTmp::DType *tmp, int validRow, int validCol)
{
    using T = typename TileDataIn::DType;
    constexpr uint8_t elemPerRpt = REPEAT_BYTE / sizeof(typename TileDataIn::DType);
    constexpr uint8_t elemPerBlock = BLOCK_BYTE_SIZE / sizeof(typename TileDataIn::DType);
    constexpr uint32_t srcRptStride = TileDataIn::Cols / elemPerBlock;

    if (validCol == elemPerRpt) {
        set_mask_count();
        set_vector_mask(0, (uint32_t)validRow * elemPerRpt);
        InstrOp::ReduceValIdxInstrImpl(reinterpret_cast<__ubuf__ T *>(tmp), src, 0, 1, 1, srcRptStride);
        pipe_barrier(PIPE_V);
    } else {
        int remain = validCol % elemPerRpt;
        int rowRptTimes = validRow / REPEAT_MAX;
        __ubuf__ T *tmpPtr = reinterpret_cast<__ubuf__ T *>(tmp);
        unsigned rptTimes;
        SetContinuousMask(remain);
        do {
            rptTimes = (rowRptTimes == 0 ? (validRow % REPEAT_MAX) : REPEAT_MAX);
            InstrOp::ReduceValIdxInstrImpl(tmpPtr, src, rptTimes, 1, 1, srcRptStride);
            pipe_barrier(PIPE_V);
            rowRptTimes -= 1;
            tmpPtr += rptTimes * 2;
            src += rptTimes * TileDataIn::Cols;
        } while (rowRptTimes >= 0);
    }
    ExtractValIdxFromTmp<TileDataOutVal, TileDataOutIdx, TileDataTmp>(dstVal, dstIdx, tmp, validRow);
}

template <bool outputVal, typename InstrOp, typename TileDataOutVal, typename TileDataOut, typename TileDataIn,
          typename TileDataTmp>
PTO_INTERNAL void TRowReduceIdxInstr(__ubuf__ typename TileDataOutVal::DType *dstVal,
                                     __ubuf__ typename TileDataOut::DType *dst,
                                     __ubuf__ typename TileDataIn::DType *src,
                                     __ubuf__ typename TileDataTmp::DType *tmp, int validRow, int validCol,
                                     int dstValValidRow, int dstIdxValidRow)
{
    TRowReduceIdxCheck<TileDataOutVal, TileDataOut, TileDataIn, outputVal>(validRow, validCol, dstValValidRow,
                                                                           dstIdxValidRow);
    constexpr uint8_t elemPerRpt = REPEAT_BYTE / sizeof(typename TileDataIn::DType);
    if (validCol <= elemPerRpt) {
        if constexpr (outputVal) {
            OneRepeatProcValIdx<InstrOp, TileDataOutVal, TileDataOut, TileDataIn, TileDataTmp>(dstVal, dst, src, tmp,
                                                                                               validRow, validCol);
        } else {
            OneRepeatProcIdx<InstrOp, TileDataOut, TileDataIn>(dst, src, validRow, validCol);
        }
    } else if (validCol <= elemPerRpt * elemPerRpt) {
        ProcReduceIdxStage1<outputVal, InstrOp, TileDataOutVal, TileDataOut, TileDataIn, TileDataTmp>(
            dstVal, dst, src, tmp, validRow, validCol);
    } else {
        ProcReduceIdxStage2<outputVal, InstrOp, TileDataOutVal, TileDataOut, TileDataIn, TileDataTmp>(
            dstVal, dst, src, tmp, validRow, validCol);
    }
}

template <typename TIdx, typename TVal>
struct TRowArgMaxOp {
    PTO_INTERNAL static void ReduceIdxInstrImpl(__ubuf__ TVal *dst, __ubuf__ TVal *src, uint8_t rptTimes,
                                                uint16_t dstRptStride, uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        vcmax(dst, src, rptTimes, dstRptStride, srcBlkStride, srcRptStride, ONLY_INDEX);
    }

    PTO_INTERNAL static void ReduceValIdxInstrImpl(__ubuf__ TVal *dst, __ubuf__ TVal *src, uint8_t rptTimes,
                                                   uint16_t dstRptStride, uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        vcmax(dst, src, rptTimes, dstRptStride, srcBlkStride, srcRptStride, VALUE_INDEX);
    }
};

template <bool outputVal, typename TVal, typename TIdx, typename TIn, typename TTmp>
__tf__ PTO_INTERNAL void TRowIdxMax(typename TVal::TileDType __out__ dstValData,
                                    typename TIdx::TileDType __out__ dstData, typename TIn::TileDType __in__ srcData,
                                    typename TTmp::TileDType __in__ tmpData, int validRow, int validCol,
                                    int dstValValidRow, int dstIdxValidRow, unsigned version)
{
    __ubuf__ typename TVal::DType *dstVal = (__ubuf__ typename TVal::DType *)__cce_get_tile_ptr(dstValData);
    __ubuf__ typename TIdx::DType *dst = (__ubuf__ typename TIdx::DType *)__cce_get_tile_ptr(dstData);
    __ubuf__ typename TIn::DType *src = (__ubuf__ typename TIn::DType *)__cce_get_tile_ptr(srcData);
    __ubuf__ typename TTmp::DType *tmp = (__ubuf__ typename TTmp::DType *)__cce_get_tile_ptr(tmpData);
    TRowReduceIdxInstr<outputVal, TRowArgMaxOp<typename TIdx::DType, typename TVal::DType>, TVal, TIdx, TIn, TTmp>(
        dstVal, dst, src, tmp, validRow, validCol, dstValValidRow, dstIdxValidRow);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWARGMAX_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    TRowIdxMax<false, TileDataIn, TileDataOut, TileDataIn, TileDataTmp>(
        src.data(), dst.data(), src.data(), tmp.data(), src.GetValidRow(), src.GetValidCol(), dst.GetValidRow(),
        dst.GetValidRow(), VFImplKind::VFIMPL_DEFAULT);
}

template <typename TileDataOutVal, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWARGMAX_IMPL(TileDataOutVal &dstVal, TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    TRowIdxMax<true, TileDataOutVal, TileDataOut, TileDataIn, TileDataTmp>(
        dstVal.data(), dst.data(), src.data(), tmp.data(), src.GetValidRow(), src.GetValidCol(), dstVal.GetValidRow(),
        dst.GetValidRow(), VFImplKind::VFIMPL_DEFAULT);
}

template <typename TIdx, typename TVal>
struct TRowArgMinOp {
    PTO_INTERNAL static void ReduceIdxInstrImpl(__ubuf__ TVal *dst, __ubuf__ TVal *src, uint8_t rptTimes,
                                                uint16_t dstRptStride, uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        vcmin(dst, src, rptTimes, dstRptStride, srcBlkStride, srcRptStride, ONLY_INDEX);
    }

    PTO_INTERNAL static void ReduceValIdxInstrImpl(__ubuf__ TVal *dst, __ubuf__ TVal *src, uint8_t rptTimes,
                                                   uint16_t dstRptStride, uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        vcmin(dst, src, rptTimes, dstRptStride, srcBlkStride, srcRptStride, VALUE_INDEX);
    }
};

template <bool outputVal, typename TileDataOutVal, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
__tf__ PTO_INTERNAL void TRowIdxMin(typename TileDataOutVal::TileDType __out__ dstValData,
                                    typename TileDataOut::TileDType __out__ dstData,
                                    typename TileDataIn::TileDType __in__ srcData,
                                    typename TileDataTmp::TileDType __in__ tmpData, int validRow, int validCol,
                                    int dstValValidRow, int dstIdxValidRow, unsigned version)
{
    using TIdx = typename TileDataOut::DType;
    using TVal = typename TileDataIn::DType;
    __ubuf__ TVal *dstVal = (__ubuf__ TVal *)__cce_get_tile_ptr(dstValData);
    __ubuf__ TIdx *dst = (__ubuf__ TIdx *)__cce_get_tile_ptr(dstData);
    __ubuf__ TVal *src = (__ubuf__ TVal *)__cce_get_tile_ptr(srcData);
    __ubuf__ typename TileDataTmp::DType *tmp = (__ubuf__ typename TileDataTmp::DType *)__cce_get_tile_ptr(tmpData);
    TRowReduceIdxInstr<outputVal, TRowArgMinOp<TIdx, TVal>, TileDataOutVal, TileDataOut, TileDataIn, TileDataTmp>(
        dstVal, dst, src, tmp, validRow, validCol, dstValValidRow, dstIdxValidRow);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWARGMIN_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    TRowIdxMin<false, TileDataIn, TileDataOut, TileDataIn, TileDataTmp>(
        src.data(), dst.data(), src.data(), tmp.data(), src.GetValidRow(), src.GetValidCol(), dst.GetValidRow(),
        dst.GetValidRow(), VFImplKind::VFIMPL_DEFAULT);
}

template <typename TileDataOutVal, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWARGMIN_IMPL(TileDataOutVal &dstVal, TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    TRowIdxMin<true, TileDataOutVal, TileDataOut, TileDataIn, TileDataTmp>(
        dstVal.data(), dst.data(), src.data(), tmp.data(), src.GetValidRow(), src.GetValidCol(), dstVal.GetValidRow(),
        dst.GetValidRow(), VFImplKind::VFIMPL_DEFAULT);
}

} // namespace pto
#endif
