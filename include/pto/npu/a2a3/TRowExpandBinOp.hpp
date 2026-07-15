/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TROWEXPANDBIN_HPP
#define TROWEXPANDBIN_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {
template <typename Op, typename T, unsigned blockSizeElem, unsigned DstRowStride, unsigned Src0RowStride>
PTO_INTERNAL void TRowExpandBinaryCountMode(
    __ubuf__ T* dstPtr, __ubuf__ T* src0Ptr, __ubuf__ T* src1Ptr, unsigned validRow, unsigned validCol)
{
    set_mask_count();
    SetVectorCount(validCol);
    for (unsigned i = 0; i < validRow; i++) {
        Op::RowExpandBinInstr(dstPtr + i * DstRowStride, src0Ptr + i * Src0RowStride, src1Ptr + i * blockSizeElem, 0);
    }
    set_mask_norm();
    SetFullVecMaskByDType<T>();
}

template <
    typename Op, typename T, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned DstRowStride,
    unsigned Src0RowStride>
PTO_INTERNAL void TRowExpandBinaryNormModeTail(
    __ubuf__ T* dstPtr, __ubuf__ T* src0Ptr, __ubuf__ T* src1Ptr, unsigned validRow, unsigned validCol)
{
    // rowStride / blockSizeElem > 255不会进到norm mode
    constexpr uint8_t DstRepeatStride = (uint8_t)(DstRowStride / blockSizeElem);
    constexpr uint8_t SrcRepeatStride = (uint8_t)(Src0RowStride / blockSizeElem);
    if constexpr (DstRowStride < elementsPerRepeat || Src0RowStride < elementsPerRepeat) {
        SetContMaskByDType<T>(validCol);
        Op::RowExpandBinInstr(dstPtr, src0Ptr, src1Ptr, validRow, DstRepeatStride, SrcRepeatStride);
        SetFullVecMaskByDType<T>();
    } else {
        unsigned numLoop = validCol / elementsPerRepeat;
        unsigned numRemainAfterLoop = validCol % elementsPerRepeat;
        for (unsigned i = 0; i < numLoop; i++) {
            Op::RowExpandBinInstr(dstPtr, src0Ptr, src1Ptr, validRow, DstRepeatStride, SrcRepeatStride);
            dstPtr += elementsPerRepeat;
            src0Ptr += elementsPerRepeat;
        }
        if (numRemainAfterLoop) {
            SetContMaskByDType<T>(numRemainAfterLoop);
            Op::RowExpandBinInstr(dstPtr, src0Ptr, src1Ptr, validRow, DstRepeatStride, SrcRepeatStride);
            SetFullVecMaskByDType<T>();
        }
    }
}

template <
    typename Op, typename T, typename U, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned DstRowStride,
    unsigned Src0RowStride>
PTO_INTERNAL void TRowExpandBinaryNormMode(
    __ubuf__ T* dstPtr, __ubuf__ T* src0Ptr, __ubuf__ U* src1Ptr, __ubuf__ T* tmpPtr, __ubuf__ U* tmpPtr_,
    unsigned validRow, unsigned validCol)
{
    // tmpbuf8KB只能存放vbrcb 32个repeat的数据,共256行大于REPEAT_MAX
    // NormMode计算255个repeat后需单独处理第256行，所以vbrcb取30个repeat，240行
    unsigned repeatTimes = CeilDivision(validRow, 8);
    constexpr unsigned BRCB_REPEAT_MAX = 30;
    constexpr unsigned MAX_ROW = 240;
    unsigned numLoop = repeatTimes / BRCB_REPEAT_MAX;
    unsigned numRemainAfterLoop = repeatTimes % BRCB_REPEAT_MAX;
    constexpr unsigned DstOffset = MAX_ROW * DstRowStride;
    constexpr unsigned SrcOffset = MAX_ROW * Src0RowStride;
    for (unsigned i = 0; i < numLoop; i++) {
        vbrcb(tmpPtr_, src1Ptr, 1, 8, BRCB_REPEAT_MAX);
        pipe_barrier(PIPE_V);
        TRowExpandBinaryNormModeTail<Op, T, elementsPerRepeat, blockSizeElem, DstRowStride, Src0RowStride>(
            dstPtr, src0Ptr, tmpPtr, MAX_ROW, validCol);
        pipe_barrier(PIPE_V);
        dstPtr += DstOffset;
        src0Ptr += SrcOffset;
        src1Ptr += MAX_ROW;
    }
    if (numRemainAfterLoop) {
        vbrcb(tmpPtr_, src1Ptr, 1, 8, numRemainAfterLoop);
        pipe_barrier(PIPE_V);
        TRowExpandBinaryNormModeTail<Op, T, elementsPerRepeat, blockSizeElem, DstRowStride, Src0RowStride>(
            dstPtr, src0Ptr, tmpPtr, validRow % MAX_ROW, validCol);
    }
}

template <typename Op, typename T, typename U, int row, unsigned DstRowStride, unsigned Src0RowStride>
PTO_INTERNAL void TRowExpandBinaryInstr(
    __ubuf__ T* dstPtr, __ubuf__ T* src0Ptr, __ubuf__ U* src1Ptr, __ubuf__ T* tmpPtr, __ubuf__ U* tmpPtr_,
    unsigned validRow, unsigned validCol)
{
    constexpr unsigned elementsPerRepeat = pto::REPEAT_BYTE / sizeof(T);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    unsigned repeatTimes = CeilDivision(validRow, 8);
    // repeatStride为uint8_t类型，超过255越界
    constexpr bool repeatStrideOverflow = DstRowStride / blockSizeElem > 255 || Src0RowStride / blockSizeElem > 255;
    // repeatStride越界，或NormMode计算次数大于CountMode计算次数时用CountMode
    bool useCountMode = repeatStrideOverflow || validCol / elementsPerRepeat > validRow;
    if constexpr (row < 256) {
        vbrcb(tmpPtr_, src1Ptr, 1, 8, repeatTimes);
        pipe_barrier(PIPE_V);
        if (useCountMode) {
            TRowExpandBinaryCountMode<Op, T, blockSizeElem, DstRowStride, Src0RowStride>(
                dstPtr, src0Ptr, tmpPtr, validRow, validCol);
        } else {
            TRowExpandBinaryNormModeTail<Op, T, elementsPerRepeat, blockSizeElem, DstRowStride, Src0RowStride>(
                dstPtr, src0Ptr, tmpPtr, validRow, validCol);
        }
    } else {
        if (validRow < 256) {
            vbrcb(tmpPtr_, src1Ptr, 1, 8, repeatTimes);
            pipe_barrier(PIPE_V);
            if (useCountMode) {
                TRowExpandBinaryCountMode<Op, T, blockSizeElem, DstRowStride, Src0RowStride>(
                    dstPtr, src0Ptr, tmpPtr, validRow, validCol);
            } else {
                TRowExpandBinaryNormModeTail<Op, T, elementsPerRepeat, blockSizeElem, DstRowStride, Src0RowStride>(
                    dstPtr, src0Ptr, tmpPtr, validRow, validCol);
            }
        } else { // 大于256行时repeatstride不会越界，且norm mode计算次数较少，使用norm mode
            TRowExpandBinaryNormMode<Op, T, U, elementsPerRepeat, blockSizeElem, DstRowStride, Src0RowStride>(
                dstPtr, src0Ptr, src1Ptr, tmpPtr, tmpPtr_, validRow, validCol);
        }
    }
}

template <
    typename Op, typename T, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned DstRowStride,
    unsigned Src0RowStride, unsigned Src1RowStride>
PTO_INTERNAL void TRowExpandBinaryNormMode32B(
    __ubuf__ T* dstPtr, __ubuf__ T* src0Ptr, __ubuf__ T* src1Ptr, unsigned validRow, unsigned validCol)
{
    unsigned numLoop = validRow / REPEAT_MAX;
    unsigned numRemainAfterLoop = validRow % REPEAT_MAX;
    constexpr unsigned dstOffset = REPEAT_MAX * DstRowStride;
    constexpr unsigned src0Offset = REPEAT_MAX * Src0RowStride;
    constexpr unsigned src1Offset = REPEAT_MAX * Src1RowStride;
    for (unsigned i = 0; i < numLoop; i++) {
        TRowExpandBinaryNormModeTail<Op, T, elementsPerRepeat, blockSizeElem, DstRowStride, Src0RowStride>(
            dstPtr, src0Ptr, src1Ptr, REPEAT_MAX, validCol);
        dstPtr += dstOffset;
        src0Ptr += src0Offset;
        src1Ptr += src1Offset;
    }
    if (numRemainAfterLoop) {
        TRowExpandBinaryNormModeTail<Op, T, elementsPerRepeat, blockSizeElem, DstRowStride, Src0RowStride>(
            dstPtr, src0Ptr, src1Ptr, numRemainAfterLoop, validCol);
    }
}

template <typename Op, typename T, int row, unsigned DstRowStride, unsigned Src0RowStride, unsigned Src1RowStride>
PTO_INTERNAL void TRowExpandBinaryInstr32B(
    __ubuf__ T* dstPtr, __ubuf__ T* src0Ptr, __ubuf__ T* src1Ptr, unsigned validRow, unsigned validCol)
{
    constexpr unsigned elementsPerRepeat = pto::REPEAT_BYTE / sizeof(T);
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    unsigned repeatTimes = CeilDivision(validRow, 8);
    constexpr bool repeatStrideOverflow = DstRowStride / blockSizeElem > 255 || Src0RowStride / blockSizeElem > 255;
    bool useCountMode = repeatStrideOverflow || validCol / elementsPerRepeat > validRow;
    if constexpr (row < 256) {
        if (useCountMode) {
            TRowExpandBinaryCountMode<Op, T, blockSizeElem, DstRowStride, Src0RowStride>(
                dstPtr, src0Ptr, src1Ptr, validRow, validCol);
        } else {
            TRowExpandBinaryNormModeTail<Op, T, elementsPerRepeat, blockSizeElem, DstRowStride, Src0RowStride>(
                dstPtr, src0Ptr, src1Ptr, validRow, validCol);
        }
    } else {
        if (validRow < 256) {
            if (useCountMode) {
                TRowExpandBinaryCountMode<Op, T, blockSizeElem, DstRowStride, Src0RowStride>(
                    dstPtr, src0Ptr, src1Ptr, validRow, validCol);
            } else {
                TRowExpandBinaryNormModeTail<Op, T, elementsPerRepeat, blockSizeElem, DstRowStride, Src0RowStride>(
                    dstPtr, src0Ptr, src1Ptr, validRow, validCol);
            }
        } else { // 大于256行时repeatstride不会越界，可以用norm mode
            TRowExpandBinaryNormMode32B<
                Op, T, elementsPerRepeat, blockSizeElem, DstRowStride, Src0RowStride, Src1RowStride>(
                dstPtr, src0Ptr, src1Ptr, validRow, validCol);
        }
    }
}

template <typename Op, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
__tf__ PTO_INTERNAL void TRowExpandBin(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc0::TileDType __in__ src0,
    typename TileDataSrc1::TileDType __in__ src1, unsigned validRow, unsigned validCol)
{
    using T = typename TileDataDst::DType;
    using U = typename std::conditional<sizeof(typename TileDataDst::DType) == 4, uint32_t, uint16_t>::type;
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    if constexpr (TileDataSrc1::isRowMajor) {
        __ubuf__ T* src1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src1);
        TRowExpandBinaryInstr32B<
            Op, T, TileDataDst::Rows, TileDataDst::RowStride, TileDataSrc0::RowStride, TileDataSrc1::RowStride>(
            dstPtr, src0Ptr, src1Ptr, validRow, validCol);
    } else {
        __ubuf__ U* src1Ptr = (__ubuf__ U*)__cce_get_tile_ptr(src1);
        __ubuf__ T* tmpPtr = (__ubuf__ T*)(TMP_UB_OFFSET);  // 8KB tmpbuf address
        __ubuf__ U* tmpPtr_ = (__ubuf__ U*)(TMP_UB_OFFSET); // 8KB tmpbuf address
        TRowExpandBinaryInstr<Op, T, U, TileDataDst::Rows, TileDataDst::RowStride, TileDataSrc0::RowStride>(
            dstPtr, src0Ptr, src1Ptr, tmpPtr, tmpPtr_, validRow, validCol);
    }
}

template <typename Op, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1, typename TileDataTmp>
__tf__ PTO_INTERNAL void TRowExpandBin(
    typename TileDataDst::TileDType __out__ dst, typename TileDataSrc0::TileDType __in__ src0,
    typename TileDataSrc1::TileDType __in__ src1, typename TileDataTmp::TileDType __out__ tmp, unsigned validRow,
    unsigned validCol)
{
    using T = typename TileDataDst::DType;
    using U = typename std::conditional<sizeof(typename TileDataDst::DType) == 4, uint32_t, uint16_t>::type;
    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    __ubuf__ U* src1Ptr = (__ubuf__ U*)__cce_get_tile_ptr(src1);
    __ubuf__ T* tmpPtr = (__ubuf__ T*)__cce_get_tile_ptr(tmp);
    __ubuf__ U* tmpPtr_ = (__ubuf__ U*)__cce_get_tile_ptr(tmp);
    TRowExpandBinaryInstr<Op, T, U, TileDataDst::Rows, TileDataDst::RowStride, TileDataSrc0::RowStride>(
        dstPtr, src0Ptr, src1Ptr, tmpPtr, tmpPtr_, validRow, validCol);
}
} // namespace pto
#endif