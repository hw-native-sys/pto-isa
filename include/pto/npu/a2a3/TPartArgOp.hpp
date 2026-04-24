/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPARTARGOP_HPP
#define TPARTARGOP_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {
template <typename T, unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void TPartArgCopyInstr(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, uint64_t validRow, uint64_t validCol,
                                    uint64_t startRow)
{
    constexpr uint64_t elemPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
    validRow -= startRow;
    srcPtr += startRow * srcStride;
    dstPtr += startRow * dstStride;
    if constexpr (dstStride == srcStride) {
        set_mask_count();
        SetVectorCount(dstStride * validRow);
        uint64_t blockCount = CeilDivision(dstStride * validRow, elemPerBlock);
        pto_copy_ubuf_to_ubuf(dstPtr, srcPtr, 1, blockCount, 1, 1);
    } else {
        set_mask_count();
        SetVectorCount(validCol);
        uint64_t blockCount = CeilDivision(validCol, elemPerBlock);
        for (uint64_t i = 0; i < validRow; i++) {
            pto_copy_ubuf_to_ubuf(dstPtr + i * dstStride, srcPtr + i * srcStride, 1, blockCount, 1, 1);
        }
    }
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <typename Op, typename TVal, typename TIdx, typename TileDstVal, typename TileDstIdx, typename TileSrcVal0,
          typename TileSrcIdx0, typename TileSrcVal1, typename TileSrcIdx1>
PTO_INTERNAL void TPartArgOps(__ubuf__ TVal *dstValPtr, __ubuf__ TIdx *dstIdxPtr, __ubuf__ TVal *srcVal0Ptr,
                              __ubuf__ TIdx *srcIdx0Ptr, __ubuf__ TVal *srcVal1Ptr, __ubuf__ TIdx *srcIdx1Ptr,
                              unsigned validRow, unsigned validCol)
{
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(TVal);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(TVal);
    if (validRow == 0 || validCol == 0) {
        return;
    }
    unsigned completeRepeats = validCol / elementsPerRepeat;
    set_mask_norm();
    set_vector_mask(-1, -1);
    for (unsigned i = 0; i < completeRepeats; i++) {
        for (unsigned row = 0; row < validRow; row++) {
            Op::CmpInstr(srcVal0Ptr + row * TileSrcVal0::RowStride + i * elementsPerRepeat,
                         srcVal1Ptr + row * TileSrcVal1::RowStride + i * elementsPerRepeat);
            pipe_barrier(PIPE_V);
            vsel(dstValPtr + row * TileDstVal::RowStride + i * elementsPerRepeat,
                 srcVal0Ptr + row * TileSrcVal0::RowStride + i * elementsPerRepeat,
                 srcVal1Ptr + row * TileSrcVal0::RowStride + i * elementsPerRepeat, 1, 1, 1, 1, 0, 0, 0);
            vsel((__ubuf__ TVal *)dstIdxPtr + row * TileDstIdx::RowStride + i * elementsPerRepeat,
                 (__ubuf__ TVal *)srcIdx0Ptr + row * TileSrcIdx0::RowStride + i * elementsPerRepeat,
                 (__ubuf__ TVal *)srcIdx1Ptr + row * TileSrcIdx1::RowStride + i * elementsPerRepeat, 1, 1, 1, 1, 0, 0,
                 0);
            pipe_barrier(PIPE_V);
        }
    }
    unsigned remainElem = validCol % elementsPerRepeat;
    if (remainElem) {
        SetContinuousMask(remainElem);
        for (unsigned row = 0; row < validRow; row++) {
            Op::CmpInstr(srcVal0Ptr + row * TileSrcVal0::RowStride + completeRepeats * elementsPerRepeat,
                         srcVal1Ptr + row * TileSrcVal1::RowStride + completeRepeats * elementsPerRepeat);
            pipe_barrier(PIPE_V);
            vsel(dstValPtr + row * TileDstVal::RowStride + completeRepeats * elementsPerRepeat,
                 srcVal0Ptr + row * TileSrcVal0::RowStride + completeRepeats * elementsPerRepeat,
                 srcVal1Ptr + row * TileSrcVal0::RowStride + completeRepeats * elementsPerRepeat, 1, 1, 1, 1, 0, 0, 0);
            vsel((__ubuf__ TVal *)dstIdxPtr + row * TileDstIdx::RowStride + completeRepeats * elementsPerRepeat,
                 (__ubuf__ TVal *)srcIdx0Ptr + row * TileSrcIdx0::RowStride + completeRepeats * elementsPerRepeat,
                 (__ubuf__ TVal *)srcIdx1Ptr + row * TileSrcIdx1::RowStride + completeRepeats * elementsPerRepeat, 1, 1,
                 1, 1, 0, 0, 0);
            pipe_barrier(PIPE_V);
        }
    }
    set_vector_mask(-1, -1);
}

template <typename Op, typename TVal, typename TIdx, typename TileDstVal, typename TileDstIdx, typename TileSrcVal0,
          typename TileSrcIdx0, typename TileSrcVal1, typename TileSrcIdx1>
PTO_INTERNAL void TPartArgInstr(__ubuf__ TVal *dstValPtr, __ubuf__ TIdx *dstIdxPtr, __ubuf__ TVal *srcVal0Ptr,
                                __ubuf__ TIdx *srcIdx0Ptr, __ubuf__ TVal *srcVal1Ptr, __ubuf__ TIdx *srcIdx1Ptr,
                                unsigned src0ValidRow, unsigned src0ValidCol, unsigned src1ValidRow,
                                unsigned src1ValidCol)
{
    if (src0ValidRow == src1ValidRow && src0ValidCol == src1ValidCol) {
        TPartArgOps<Op, TVal, TIdx, TileDstVal, TileDstIdx, TileSrcVal0, TileSrcIdx0, TileSrcVal1, TileSrcIdx1>(
            dstValPtr, dstIdxPtr, srcVal0Ptr, srcIdx0Ptr, srcVal1Ptr, srcIdx1Ptr, src0ValidRow, src0ValidCol);
    } else if (src0ValidCol == src1ValidCol) {
        TPartArgOps<Op, TVal, TIdx, TileDstVal, TileDstIdx, TileSrcVal0, TileSrcIdx0, TileSrcVal1, TileSrcIdx1>(
            dstValPtr, dstIdxPtr, srcVal0Ptr, srcIdx0Ptr, srcVal1Ptr, srcIdx1Ptr, min(src0ValidRow, src1ValidRow),
            src0ValidCol);
        if (src0ValidRow < src1ValidRow) {
            TPartArgCopyInstr<TVal, TileDstVal::RowStride, TileSrcVal1::RowStride>(dstValPtr, srcVal1Ptr, src1ValidRow,
                                                                                   src1ValidCol, src0ValidRow);
            TPartArgCopyInstr<TIdx, TileDstIdx::RowStride, TileSrcIdx1::RowStride>(dstIdxPtr, srcIdx1Ptr, src1ValidRow,
                                                                                   src1ValidCol, src0ValidRow);
        } else {
            TPartArgCopyInstr<TVal, TileDstVal::RowStride, TileSrcVal0::RowStride>(dstValPtr, srcVal0Ptr, src0ValidRow,
                                                                                   src0ValidCol, src1ValidRow);
            TPartArgCopyInstr<TIdx, TileDstIdx::RowStride, TileSrcIdx0::RowStride>(dstIdxPtr, srcIdx0Ptr, src0ValidRow,
                                                                                   src0ValidCol, src1ValidRow);
        }
    } else if ((src0ValidRow <= src1ValidRow && src0ValidCol < src1ValidCol) ||
               (src1ValidRow <= src0ValidRow && src1ValidCol < src0ValidCol)) {
        if (src0ValidCol < src1ValidCol) {
            TPartArgCopyInstr<TVal, TileDstVal::RowStride, TileSrcVal1::RowStride>(dstValPtr, srcVal1Ptr, src1ValidRow,
                                                                                   src1ValidCol, 0);
            TPartArgCopyInstr<TIdx, TileDstIdx::RowStride, TileSrcIdx1::RowStride>(dstIdxPtr, srcIdx1Ptr, src1ValidRow,
                                                                                   src1ValidCol, 0);
        } else {
            TPartArgCopyInstr<TVal, TileDstVal::RowStride, TileSrcVal0::RowStride>(dstValPtr, srcVal0Ptr, src0ValidRow,
                                                                                   src0ValidCol, 0);
            TPartArgCopyInstr<TIdx, TileDstIdx::RowStride, TileSrcIdx0::RowStride>(dstIdxPtr, srcIdx0Ptr, src0ValidRow,
                                                                                   src0ValidCol, 0);
        }
        TPartArgOps<Op, TVal, TIdx, TileDstVal, TileDstIdx, TileSrcVal0, TileSrcIdx0, TileSrcVal1, TileSrcIdx1>(
            dstValPtr, dstIdxPtr, srcVal0Ptr, srcIdx0Ptr, srcVal1Ptr, srcIdx1Ptr, min(src0ValidRow, src1ValidRow),
            min(src0ValidCol, src1ValidCol));
    } else {
        PTO_ASSERT(false, "TPARTARGOPS: Only one of src0 and src1 is smaller than dst.");
    }
}

template <typename Op, typename TileDstVal, typename TileDstIdx, typename TileSrcVal0, typename TileSrcIdx0,
          typename TileSrcVal1, typename TileSrcIdx1>
__tf__ PTO_INTERNAL void TPartArgOp(typename TileDstVal::TileDType __out__ dstVal,
                                    typename TileDstIdx::TileDType __out__ dstIdx,
                                    typename TileSrcVal0::TileDType __in__ srcVal0,
                                    typename TileSrcIdx0::TileDType __in__ srcIdx0,
                                    typename TileSrcVal1::TileDType __in__ srcVal1,
                                    typename TileSrcIdx1::TileDType __in__ srcIdx1, unsigned src0ValidRow,
                                    unsigned src0ValidCol, unsigned src1ValidRow, unsigned src1ValidCol)
{
    using TVal = typename TileDstVal::DType;
    using TIdx = typename TileDstIdx::DType;
    __ubuf__ TVal *dstValPtr = (__ubuf__ TVal *)__cce_get_tile_ptr(dstVal);
    __ubuf__ TVal *srcVal0Ptr = (__ubuf__ TVal *)__cce_get_tile_ptr(srcVal0);
    __ubuf__ TVal *srcVal1Ptr = (__ubuf__ TVal *)__cce_get_tile_ptr(srcVal1);
    __ubuf__ TIdx *dstIdxPtr = (__ubuf__ TIdx *)__cce_get_tile_ptr(dstIdx);
    __ubuf__ TIdx *srcIdx0Ptr = (__ubuf__ TIdx *)__cce_get_tile_ptr(srcIdx0);
    __ubuf__ TIdx *srcIdx1Ptr = (__ubuf__ TIdx *)__cce_get_tile_ptr(srcIdx1);
    TPartArgInstr<Op, TVal, TIdx, TileDstVal, TileDstIdx, TileSrcVal0, TileSrcIdx0, TileSrcVal1, TileSrcIdx1>(
        dstValPtr, dstIdxPtr, srcVal0Ptr, srcIdx0Ptr, srcVal1Ptr, srcIdx1Ptr, src0ValidRow, src0ValidCol, src1ValidRow,
        src1ValidCol);
}

template <typename TileDstVal, typename TileDstIdx, typename TileSrcVal0, typename TileSrcIdx0, typename TileSrcVal1,
          typename TileSrcIdx1>
PTO_INTERNAL bool checkTiles(TileDstVal &dstVal, TileDstIdx &dstIdx, TileSrcVal0 &srcVal0, TileSrcIdx0 &srcIdx0,
                             TileSrcVal1 &srcVal1, TileSrcIdx1 &srcIdx1)
{
    using TVal = typename TileDstVal::DType;
    using TIdx = typename TileDstIdx::DType;
    static_assert(
        std::is_same_v<TVal, typename TileSrcVal0::DType> && std::is_same_v<TVal, typename TileSrcVal1::DType>,
        "TPARTARGOPS: dstVal srcVal0 srcVal1 type must be consistent");
    static_assert(
        std::is_same_v<TIdx, typename TileSrcIdx0::DType> && std::is_same_v<TIdx, typename TileSrcIdx1::DType>,
        "TPARTARGOPS: dstIdx srcIdx0 srcIdx1 type must be consistent");
    static_assert(std::is_same_v<TVal, float> || std::is_same_v<TVal, half>,
                  "TPARTARGOPS: val type only support float, half.");
    static_assert(std::is_integral_v<TIdx> && sizeof(TIdx) == sizeof(TVal),
                  "TPARTARGOPS: idx must be integral and the same size as val");
    unsigned dstValidRow = dstVal.GetValidRow();
    unsigned dstValidCol = dstVal.GetValidCol();
    unsigned dstIdxValidRow = dstIdx.GetValidRow();
    unsigned dstIdxValidCol = dstIdx.GetValidCol();
    unsigned src0ValidRow = srcVal0.GetValidRow();
    unsigned src0ValidCol = srcVal0.GetValidCol();
    unsigned srcIdx0ValidRow = srcIdx0.GetValidRow();
    unsigned srcIdx0ValidCol = srcIdx0.GetValidCol();
    unsigned src1ValidRow = srcVal1.GetValidRow();
    unsigned src1ValidCol = srcVal1.GetValidCol();
    unsigned srcIdx1ValidRow = srcIdx1.GetValidRow();
    unsigned srcIdx1ValidCol = srcIdx1.GetValidCol();
    if (dstValidRow != dstIdxValidRow || dstValidCol != dstIdxValidCol || src0ValidRow != srcIdx0ValidRow ||
        src0ValidCol != srcIdx0ValidCol || src1ValidRow != srcIdx1ValidRow || src1ValidCol != srcIdx1ValidCol) {
        PTO_ASSERT(false, "TPARTARGOPS: idxTile validRow/validCol must be consistent with of valTile.");
        return false;
    }
    if (dstValidRow == 0 || dstValidCol == 0) {
        PTO_ASSERT(false, "TPARTARGOPS: dst valid size must be non zero.");
        return false;
    }
    if (dstValidRow != max(src0ValidRow, src1ValidRow) || dstValidCol != max(src0ValidCol, src1ValidCol)) {
        PTO_ASSERT(false, "TPARTARGOPS: dst valid size must be consistent with src of bigger size.");
        return false;
    }
    return true;
}

template <typename T>
struct PartArgMaxOp {
    PTO_INTERNAL static void CmpInstr(__ubuf__ T *src0, __ubuf__ T *src1)
    {
        vcmp_gt(src0, src1, 1, 1, 1, 1, 0, 0, 0);
    }
};

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataDstIdx,
          typename TileDataSrc0Idx, typename TileDataSrc1Idx>
PTO_INTERNAL void TPARTARGMAX_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataDstIdx &dstIdx,
                                   TileDataSrc0Idx &src0Idx, TileDataSrc1Idx &src1Idx)
{
    unsigned src0ValidRow = src0.GetValidRow();
    unsigned src0ValidCol = src0.GetValidCol();
    unsigned src1ValidRow = src1.GetValidRow();
    unsigned src1ValidCol = src1.GetValidCol();
    if (!checkTiles(dst, dstIdx, src0, src0Idx, src1, src1Idx)) {
        return;
    }
    TPartArgOp<PartArgMaxOp<typename TileDataDst::DType>, TileDataDst, TileDataDstIdx, TileDataSrc0, TileDataSrc0Idx,
               TileDataSrc1, TileDataSrc1Idx>(dst.data(), dstIdx.data(), src0.data(), src0Idx.data(), src1.data(),
                                              src1Idx.data(), src0ValidRow, src0ValidCol, src1ValidRow, src1ValidCol);
}

template <typename T>
struct PartArgMinOp {
    PTO_INTERNAL static void CmpInstr(__ubuf__ T *src0, __ubuf__ T *src1)
    {
        vcmp_lt(src0, src1, 1, 1, 1, 1, 0, 0, 0);
    }
};

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataDstIdx,
          typename TileDataSrc0Idx, typename TileDataSrc1Idx>
PTO_INTERNAL void TPARTARGMIN_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1, TileDataDstIdx &dstIdx,
                                   TileDataSrc0Idx &src0Idx, TileDataSrc1Idx &src1Idx)
{
    unsigned src0ValidRow = src0.GetValidRow();
    unsigned src0ValidCol = src0.GetValidCol();
    unsigned src1ValidRow = src1.GetValidRow();
    unsigned src1ValidCol = src1.GetValidCol();
    if (!checkTiles(dst, dstIdx, src0, src0Idx, src1, src1Idx)) {
        return;
    }
    TPartArgOp<PartArgMinOp<typename TileDataDst::DType>, TileDataDst, TileDataDstIdx, TileDataSrc0, TileDataSrc0Idx,
               TileDataSrc1, TileDataSrc1Idx>(dst.data(), dstIdx.data(), src0.data(), src0Idx.data(), src1.data(),
                                              src1Idx.data(), src0ValidRow, src0ValidCol, src1ValidRow, src1ValidCol);
}

} // namespace pto
#endif