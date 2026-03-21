/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TTRANS_HPP
#define TTRANS_HPP

#include "pto/common/constants.hpp"
#include "pto/common/utils.hpp"

namespace pto {

constexpr int ADDR_NUM = 16;
constexpr int HALF_ADDR_NUM = 8;
// for b8,  [32, 32] -> [32, 32]
constexpr int X_ELEM_B8 = 32;
constexpr int Y_ELEM_B8 = 32;
// for b16,  [16, 16] -> [16, 16]
// for b32,  [16, 8] -> [8,16]
constexpr int X_ELEM_B32 = 8;
constexpr int Y_ELEM_OTHER = 16;

template <typename T>
struct TransOp {
    PTO_INTERNAL static void TransB8Instr(uint8_t repeat, uint16_t dstStride, uint16_t srcStride)
    {
        // [32, 8] -> [8, 32]
        scatter_vnchwconv_b8(VA0, VA2, repeat, dstStride, srcStride, false, false);
        scatter_vnchwconv_b8(VA6, VA2, repeat, dstStride, srcStride, false, true);
        scatter_vnchwconv_b8(VA0, VA4, repeat, dstStride, srcStride, true, false);
        scatter_vnchwconv_b8(VA6, VA4, repeat, dstStride, srcStride, true, true);
    }

    PTO_INTERNAL static void TransB16Instr(uint8_t repeat, uint16_t dstStride, uint16_t srcStride)
    {
        // [16, 16] -> [16, 16]
        scatter_vnchwconv_b16(VA0, VA2, repeat, dstStride, srcStride);
    }

    PTO_INTERNAL static void TransB32Instr(uint8_t repeat, uint16_t dstStride, uint16_t srcStride)
    {
        // [16,8] -> [8,16]
        scatter_vnchwconv_b32(VA0, VA2, repeat, dstStride, srcStride);
    }

    PTO_INTERNAL static void CopyInstr(__ubuf__ uint32_t *dstPtr, __ubuf__ uint32_t *srcPtr, uint8_t repeat,
                                       uint16_t dstRepeatStride, uint16_t srcRepeatStride)
    {
        vcopy(dstPtr, srcPtr, repeat, 1, 1, dstRepeatStride, srcRepeatStride);
    }

    PTO_INTERNAL static void CopyInstr(__ubuf__ uint16_t *dstPtr, __ubuf__ uint16_t *srcPtr, uint8_t repeat,
                                       uint16_t dstRepeatStride, uint16_t srcRepeatStride)
    {
        vcopy(dstPtr, srcPtr, repeat, 1, 1, dstRepeatStride, srcRepeatStride);
    }
};

template <typename Op, typename T, unsigned blockSizeElem>
PTO_INTERNAL void TransFullSubTiles(__ubuf__ T *tmpPtr, __ubuf__ T *srcPtr, unsigned tmpStride, unsigned numSubTileX,
                                    unsigned numSubTileY, unsigned srcStride)
{
    for (int i = 0; i < numSubTileX; i++) {
        uint64_t srcUb[ADDR_NUM] = {0}, tmpUb[ADDR_NUM] = {0};
        uint64_t offset = i * blockSizeElem;
        uint16_t vconvSrcStride = Y_ELEM_OTHER * srcStride * sizeof(T) / BLOCK_BYTE_SIZE;
        for (int j = 0; j < ADDR_NUM; j++) {
            srcUb[j] = (uint64_t)(srcPtr + offset + j * srcStride);
            tmpUb[j] = (sizeof(T) == 2) ?
                           (uint64_t)(tmpPtr + (j + offset) * tmpStride) :
                           (uint64_t)(tmpPtr + ((j >> 1) + offset) * tmpStride + (j & 1) * blockSizeElem);
        }
        set_va_reg_sb(VA2, srcUb);
        set_va_reg_sb(VA3, &srcUb[HALF_ADDR_NUM]);
        set_va_reg_sb(VA0, tmpUb);
        set_va_reg_sb(VA1, &tmpUb[HALF_ADDR_NUM]);
        if constexpr (sizeof(T) == 2) {
            if (numSubTileY == 1) {
                Op::TransB16Instr(1, 0, 0);
            } else {
                Op::TransB16Instr(numSubTileY, 1, vconvSrcStride);
            }
        } else {
            if (numSubTileY == 1) {
                Op::TransB32Instr(1, 0, 0);
            } else {
                Op::TransB32Instr(numSubTileY, 2, vconvSrcStride);
            }
        }
    }
}

template <typename Op, typename T, unsigned blockSizeElem>
PTO_INTERNAL void TransB8FullSubTiles(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned tmpStride, unsigned numSubTileX,
                                      unsigned numSubTileY, unsigned srcStride)
{
    if (numSubTileX > 0) { // [32, 32] aligned
        uint64_t srcUb[ADDR_NUM] = {0}, srcUb1[ADDR_NUM] = {0}, tmpUb[ADDR_NUM] = {0}, tmpUb1[ADDR_NUM] = {0};
        uint16_t vconvSrcStride = Y_ELEM_B8 * srcStride * sizeof(T) / BLOCK_BYTE_SIZE;
        for (int i = 0; i < numSubTileX; i++) {
            uint64_t offset = i * blockSizeElem;
            for (int j = 0; j < ADDR_NUM; j++) {
                srcUb[j] = (uint64_t)(srcPtr + offset + j * srcStride);
                srcUb1[j] = srcUb[j] + ADDR_NUM * srcStride;
                tmpUb[j] = (uint64_t)(dstPtr + (j + offset) * tmpStride);
                tmpUb1[j] = tmpUb[j] + ADDR_NUM * tmpStride;
            }
            set_va_reg_sb(VA2, srcUb);
            set_va_reg_sb(VA3, &srcUb[HALF_ADDR_NUM]);
            set_va_reg_sb(VA0, tmpUb);
            set_va_reg_sb(VA1, &tmpUb[HALF_ADDR_NUM]);

            set_va_reg_sb(VA4, srcUb1);
            set_va_reg_sb(VA5, &srcUb1[HALF_ADDR_NUM]);
            set_va_reg_sb(VA6, tmpUb1);
            set_va_reg_sb(VA7, &tmpUb1[HALF_ADDR_NUM]);
            if (numSubTileY == 1) { // [32, 32]
                Op::TransB8Instr(1, 0, 0);
            } else { // larger then [32, 32], e.g, [32, 64]
                Op::TransB8Instr(numSubTileY, 1, vconvSrcStride);
            }
        } // end of numSubTileX
    }
}

template <typename Op, typename T, unsigned blockSizeElem>
PTO_INTERNAL void TransYTailTiles(__ubuf__ T *tmpPtr, __ubuf__ T *srcPtr, unsigned tmpStride, unsigned numSubTileX,
                                  unsigned numSubTileY, unsigned remainY, unsigned srcStride)
{
    uint64_t srcUb[ADDR_NUM] = {0}, tmpUb[ADDR_NUM] = {0};
    uint64_t offset = numSubTileY * Y_ELEM_OTHER;
    for (int i = 0; i < remainY; i++) {
        srcUb[i] = (uint64_t)(srcPtr + (offset + i) * srcStride);
    }
    for (int i = 0; i < ADDR_NUM; i++) {
        tmpUb[i] = (sizeof(T) == 2) ? (uint64_t)(tmpPtr + offset + i * tmpStride) :
                                      (uint64_t)(tmpPtr + offset + (i & 1) * blockSizeElem + (i >> 1) * tmpStride);
    }
    set_va_reg_sb(VA2, srcUb);
    set_va_reg_sb(VA3, &srcUb[HALF_ADDR_NUM]);
    set_va_reg_sb(VA0, tmpUb);
    set_va_reg_sb(VA1, &tmpUb[HALF_ADDR_NUM]);
    if constexpr (sizeof(T) == 2) {
        if (numSubTileX == 1) {
            Op::TransB16Instr(1, 0, 0);
        } else {
            Op::TransB16Instr(numSubTileX, blockSizeElem * tmpStride * sizeof(T) / BLOCK_BYTE_SIZE, 1);
        }
    } else {
        if (numSubTileX == 1) {
            Op::TransB32Instr(1, 0, 0);
        } else {
            Op::TransB32Instr(numSubTileX, blockSizeElem * tmpStride * sizeof(T) / BLOCK_BYTE_SIZE, 1);
        }
    }
}

template <typename Op, typename T, unsigned blockSizeEleme>
PTO_INTERNAL void TransB8YTailTiles(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned tmpStride, unsigned numSubTileX,
                                    unsigned numSubTileY, unsigned remainY, unsigned srcStride)
{
    uint64_t srcUb[ADDR_NUM] = {0}, tmpUb[ADDR_NUM] = {0};
    uint64_t srcUb1[ADDR_NUM] = {0}, tmpUb1[ADDR_NUM] = {0};
    uint64_t offset = numSubTileY * Y_ELEM_B8;
    for (int i = 0; i < remainY; i++) {
        if (i < ADDR_NUM) {
            srcUb[i] = (uint64_t)(srcPtr + (offset + i) * srcStride);
        } else {
            srcUb1[i - ADDR_NUM] = (uint64_t)(srcPtr + (offset + i) * srcStride);
        }
    }
    for (int i = 0; i < ADDR_NUM; i++) {
        tmpUb[i] = (uint64_t)(dstPtr + offset + i * tmpStride);
        tmpUb1[i] = tmpUb[i] + ADDR_NUM * tmpStride;
    }
    set_va_reg_sb(VA2, srcUb);
    set_va_reg_sb(VA3, &srcUb[HALF_ADDR_NUM]);
    set_va_reg_sb(VA0, tmpUb);
    set_va_reg_sb(VA1, &tmpUb[HALF_ADDR_NUM]);

    set_va_reg_sb(VA4, srcUb1);
    set_va_reg_sb(VA5, &srcUb1[HALF_ADDR_NUM]);
    set_va_reg_sb(VA6, tmpUb1);
    set_va_reg_sb(VA7, &tmpUb1[HALF_ADDR_NUM]);
    if (numSubTileX == 1) {
        Op::TransB8Instr(1, 0, 0);
    } else {
        Op::TransB8Instr(numSubTileX, Y_ELEM_B8 * tmpStride * sizeof(T) / BLOCK_BYTE_SIZE, 1);
    }
}

template <typename T, unsigned blockSizeElem, unsigned yTileSizeElem>
PTO_INTERNAL void TransTailTiles(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned tmpStride, unsigned validRow,
                                 unsigned validCol, unsigned dstStride, unsigned srcStride)
{
    // we can use constexpr if tmpStride is known in static way

    PtoSetWaitFlag<PIPE_V, PIPE_S>();
    for (int i = 0; i < validRow; i++) {
        for (int j = 0; j < validCol; j++) {
            dstPtr[j * dstStride + i] = srcPtr[i * srcStride + j];
        }
    }
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
    return;
}

template <typename T, unsigned blockSizeElem>
PTO_INTERNAL void TTransOperation(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, __ubuf__ T *tmpPtr, unsigned validRow,
                                  unsigned validCol, unsigned dstStride, unsigned srcStride)
{
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? Y_ELEM_B8 : Y_ELEM_OTHER;
    // tmpStride should computed in static way
    unsigned tmpStride = (validRow + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem;
    if (((dstStride % yTileSizeElem) != 0) || ((srcStride % blockSizeElem) != 0) ||
        ((tmpStride % yTileSizeElem) != 0)) {
        TransTailTiles<T, blockSizeElem, yTileSizeElem>(dstPtr, srcPtr, tmpStride, validRow, validCol, dstStride,
                                                        srcStride);
        return;
    }
    // go by subtile column, a.k.a. iter in row direction
    int numSubTileX = (validCol + blockSizeElem - 1) / blockSizeElem;
    int numSubTileY = validRow / yTileSizeElem;
    if (numSubTileY > 0) {
        if constexpr (sizeof(T) == 1) {
            TransB8FullSubTiles<TransOp<T>, T, blockSizeElem>(tmpPtr, srcPtr, tmpStride, numSubTileX, numSubTileY,
                                                              srcStride);
        } else {
            TransFullSubTiles<TransOp<T>, T, blockSizeElem>(tmpPtr, srcPtr, tmpStride, numSubTileX, numSubTileY,
                                                            srcStride);
        }
    }
    // tail
    int remainY = validRow % yTileSizeElem;
    if (remainY > 0) {
        if constexpr (sizeof(T) == 1) {
            TransB8YTailTiles<TransOp<T>, T, blockSizeElem>(tmpPtr, srcPtr, tmpStride, numSubTileX, numSubTileY,
                                                            remainY, srcStride);
        } else {
            TransYTailTiles<TransOp<T>, T, blockSizeElem>(tmpPtr, srcPtr, tmpStride, numSubTileX, numSubTileY, remainY,
                                                          srcStride);
        }
    }
    // copy to dst
    uint16_t lenBurst = (validRow * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
    uint16_t srcGap = tmpStride * sizeof(T) / BLOCK_BYTE_SIZE - lenBurst;
    uint16_t dstGap = dstStride * sizeof(T) / BLOCK_BYTE_SIZE - lenBurst;
    pipe_barrier(PIPE_V);
    copy_ubuf_to_ubuf(dstPtr, tmpPtr, 0, validCol, lenBurst, srcGap, dstGap);
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp, unsigned blockSizeElem>
__tf__ PTO_INTERNAL void TTrans(typename TileDataDst::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src,
                                typename TileDataTmp::TileDType __in__ tmp, unsigned validRow, unsigned validCol,
                                unsigned dstStride, unsigned srcStride)
{
    using T = typename TileDataSrc::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ T *tmpPtr = (__ubuf__ T *)__cce_get_tile_ptr(tmp);
    TTransOperation<T, blockSizeElem>(dstPtr, srcPtr, tmpPtr, validRow, validCol, dstStride, srcStride);
}

////////////// TTrans Repeat X/////

template <typename Op, typename T, unsigned blockSizeElem>
PTO_INTERNAL void TransRepeatXFullSubTiles(__ubuf__ T *tmpPtr, __ubuf__ T *srcPtr, unsigned tmpStride,
                                           unsigned numSubTileX, unsigned numSubTileY, unsigned srcStride)
{
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? Y_ELEM_B8 : Y_ELEM_OTHER;
    for (int i = 0; i < numSubTileY; i++) {
        uint64_t srcUb[ADDR_NUM] = {0}, tmpUb[ADDR_NUM] = {0};
        uint64_t offset = i * srcStride * yTileSizeElem;
        uint64_t vconvDstStride = X_ELEM_B32 * tmpStride * sizeof(T) / BLOCK_BYTE_SIZE; // for float
        uint64_t dstOffset = Y_ELEM_OTHER * i;
        // for float, src: 16 * 8, dst: 8 * 16;
        // for half, src: 16 * 16, dst: 16 * 16;
        for (int j = 0; j < ADDR_NUM; j++) {
            srcUb[j] = (uint64_t)(srcPtr + offset + j * srcStride);
            tmpUb[j] = (sizeof(T) == 2) ?
                           (uint64_t)(tmpPtr + j * tmpStride + dstOffset) :
                           (uint64_t)(tmpPtr + (j >> 1) * tmpStride + dstOffset + (j & 1) * blockSizeElem);
        }
        set_va_reg_sb(VA2, srcUb);
        set_va_reg_sb(VA3, &srcUb[HALF_ADDR_NUM]);
        set_va_reg_sb(VA0, tmpUb);
        set_va_reg_sb(VA1, &tmpUb[HALF_ADDR_NUM]);
        if constexpr (sizeof(T) == 2) {
            if (numSubTileX == 1) {
                Op::TransB16Instr(1, 0, 0);
            } else {
                Op::TransB16Instr(numSubTileX, vconvDstStride * 2, 1);
            }
        } else {
            if (numSubTileX == 1) {
                Op::TransB32Instr(1, 0, 0);
            } else {
                Op::TransB32Instr(numSubTileX, vconvDstStride, 1);
            }
        }
    }
}

template <typename Op, typename T, unsigned blockSizeElem>
PTO_INTERNAL void TransRepeatXB8FullSubTiles(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, unsigned tmpStride,
                                             unsigned numSubTileX, unsigned numSubTileY, unsigned srcStride)
{
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? Y_ELEM_B8 : Y_ELEM_OTHER;
    if (numSubTileY > 0) { // [32, 32] aligned
        uint64_t srcUb[ADDR_NUM] = {0}, srcUb1[ADDR_NUM] = {0}, tmpUb[ADDR_NUM] = {0}, tmpUb1[ADDR_NUM] = {0};
        uint64_t vconvDstStride = X_ELEM_B8 * tmpStride * sizeof(T) / BLOCK_BYTE_SIZE;
        for (int i = 0; i < numSubTileY; i++) {
            uint64_t offset = i * yTileSizeElem * srcStride;
            for (int j = 0; j < ADDR_NUM; j++) {
                srcUb[j] = (uint64_t)(srcPtr + offset + j * srcStride);
                srcUb1[j] = srcUb[j] + ADDR_NUM * srcStride;
                tmpUb[j] = (uint64_t)(dstPtr + j * tmpStride + 32 * i);
                tmpUb1[j] = tmpUb[j] + ADDR_NUM * tmpStride;
            }
            set_va_reg_sb(VA4, srcUb1);
            set_va_reg_sb(VA5, &srcUb1[HALF_ADDR_NUM]);
            set_va_reg_sb(VA6, tmpUb1);
            set_va_reg_sb(VA7, &tmpUb1[HALF_ADDR_NUM]);

            set_va_reg_sb(VA2, srcUb);
            set_va_reg_sb(VA3, &srcUb[HALF_ADDR_NUM]);
            set_va_reg_sb(VA0, tmpUb);
            set_va_reg_sb(VA1, &tmpUb[HALF_ADDR_NUM]);
            if (numSubTileX == 1) { // [32, 32]
                Op::TransB8Instr(1, 0, 0);
            } else { // larger than [32, 32], e.g, [32, 64]
                Op::TransB8Instr(numSubTileX, vconvDstStride, 1);
            }
        } // end of numSubTileY
    }
}

template <typename T, unsigned blockSizeElem>
PTO_INTERNAL void TTransRepeatXOperation(__ubuf__ T *dstPtr, __ubuf__ T *srcPtr, __ubuf__ T *tmpPtr, unsigned validRow,
                                         unsigned validCol, unsigned dstStride, unsigned srcStride)
{
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1) ? Y_ELEM_B8 : Y_ELEM_OTHER;
    // tmpStride should computed in static way
    unsigned tmpStride = (validRow + yTileSizeElem - 1) / yTileSizeElem * yTileSizeElem;
    if (((dstStride % blockSizeElem) != 0) || ((srcStride % blockSizeElem) != 0)) {
        TransTailTiles<T, blockSizeElem, yTileSizeElem>(dstPtr, srcPtr, tmpStride, validRow, validCol, dstStride,
                                                        srcStride);
        return;
    }
    // go by subtile column, a.k.a. iter in row direction
    int numSubTileX = validCol / blockSizeElem;
    int numSubTileY = validRow / yTileSizeElem;
    if (numSubTileX > 0) {
        if constexpr (sizeof(T) == 1) {
            TransRepeatXB8FullSubTiles<TransOp<T>, T, blockSizeElem>(tmpPtr, srcPtr, tmpStride, numSubTileX,
                                                                     numSubTileY, srcStride);
        } else {
            TransRepeatXFullSubTiles<TransOp<T>, T, blockSizeElem>(tmpPtr, srcPtr, tmpStride, numSubTileX, numSubTileY,
                                                                   srcStride);
        }
    }
    // tail
    int remainY = validRow % yTileSizeElem;
    if (remainY > 0) {
        if constexpr (sizeof(T) != 1) {
            TransYTailTiles<TransOp<T>, T, blockSizeElem>(tmpPtr, srcPtr, tmpStride, numSubTileX, numSubTileY, remainY,
                                                          srcStride);
        } else {
            TransB8YTailTiles<TransOp<T>, T, blockSizeElem>(tmpPtr, srcPtr, tmpStride, numSubTileX, numSubTileY,
                                                            remainY, srcStride);
        }
    }
    // copy to dst
    uint16_t lenBurst = (validRow * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
    uint16_t dstGap = dstStride * sizeof(T) / BLOCK_BYTE_SIZE - lenBurst;
    uint16_t srcGap = tmpStride * sizeof(T) / BLOCK_BYTE_SIZE - lenBurst;
    pipe_barrier(PIPE_V);
    copy_ubuf_to_ubuf(dstPtr, tmpPtr, 0, validCol, lenBurst, srcGap, dstGap);
}

///////////////////
template <typename TileData, unsigned blockSizeElem>
__tf__ PTO_INTERNAL void TTransConvNCHW2NC1HWC0(typename TileData::TileDType __out__ dst,
                                                typename TileData::TileDType __in__ src,
                                                typename TileData::TileDType __in__ tmp, unsigned srcN, unsigned srcC,
                                                unsigned srcH, unsigned srcW, unsigned dstC0)
{
    using T = typename TileData::DType;
    __ubuf__ T *dstPtrOrig = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtrOrig = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ T *tmpPtr = (__ubuf__ T *)__cce_get_tile_ptr(tmp);
    unsigned srcStride = srcH * srcW;
    unsigned dstStride = dstC0;
    unsigned validCol = srcH * srcW;
    unsigned validRow = dstC0;
    unsigned dstC1 = (srcC + dstC0 - 1) / dstC0;
    unsigned nStride = dstC1 * dstC0 * srcH * srcW;
    unsigned cStride = dstC0 * srcH * srcW;
    // N C1 C0 HW -> N C1 HW C0
    for (int n = 0; n < srcN; n++) {
        for (int c = 0; c < dstC1; c++) {
            __ubuf__ T *srcPtr = srcPtrOrig + n * nStride + c * cStride;
            __ubuf__ T *dstPtr = dstPtrOrig + n * nStride + c * cStride;
            TTransRepeatXOperation<T, blockSizeElem>(dstPtr, srcPtr, tmpPtr, validRow, validCol, dstStride, srcStride);
        }
    }
}

template <typename TileData, unsigned blockSizeElem>
__tf__ PTO_INTERNAL void TTransConvNC1HWC02C1HWNC0(typename TileData::TileDType __out__ dst,
                                                   typename TileData::TileDType __in__ src,
                                                   typename TileData::TileDType __in__ tmp, unsigned dstN,
                                                   unsigned srcN, unsigned srcC1HW, unsigned srcC0)
{
    using T = typename TileData::DType;
    __ubuf__ T *dstPtrOrig = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtrOrig = (__ubuf__ T *)__cce_get_tile_ptr(src);
    if (srcC0 * sizeof(T) % BLOCK_BYTE_SIZE == 0) {
        uint32_t burstNum = srcC1HW;
        uint32_t lenBurst = (srcC0 * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
        uint32_t srcGap = 0;
        uint32_t dstGap = (dstN * srcC0 * sizeof(T) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE - lenBurst;
        // N C1HW C0 -> C1HW N C0
        uint32_t nStride = srcC1HW * srcC0;
        for (int i = 0; i < srcN; i++) {
            __ubuf__ T *srcPtr = srcPtrOrig + i * nStride;
            __ubuf__ T *dstPtr = dstPtrOrig + i * srcC0;
            copy_ubuf_to_ubuf(dstPtr, srcPtr, 0, burstNum, lenBurst, srcGap, dstGap);
        }
    } else {
        unsigned validCol = srcC0;
        unsigned validRow = srcC1HW;
        unsigned srcStride = srcC0;
        unsigned dstStride = dstN * srcC0;
        unsigned nStride = srcC1HW * srcC0;
        PtoSetWaitFlag<PIPE_V, PIPE_S>();
        for (uint16_t num = 0; num < (uint16_t)srcN; num++) {
            __ubuf__ T *srcPtr = srcPtrOrig + num * nStride;
            __ubuf__ T *dstPtr = dstPtrOrig + num * srcC0;
            for (uint16_t i = 0; i < (uint16_t)validRow; ++i) {
                for (uint16_t j = 0; j < (uint16_t)validCol; ++j) {
                    dstPtr[i * dstStride + j] = srcPtr[i * srcStride + j];
                }
            }
        }
        PtoSetWaitFlag<PIPE_S, PIPE_V>();
    }
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void CheckConvTile()
{
#ifdef _DEBUG
    using T = typename TileDataSrc::DType;
    constexpr const int UB_SIZE = 196608; // 192*1024 B
    if (TileDataSrc::layout == Layout::NCHW && TileDataDst::layout == Layout::NC1HWC0) {
        unsigned dstN = dst.GetShape(GlobalTensorDim::DIM_0);
        unsigned dstC1 = dst.GetShape(GlobalTensorDim::DIM_1);
        unsigned dstH = dst.GetShape(GlobalTensorDim::DIM_2);
        unsigned dstW = dst.GetShape(GlobalTensorDim::DIM_3);
        unsigned dstC0 = dst.GetShape(GlobalTensorDim::DIM_4);
        unsigned srcN = src.GetShape(GlobalTensorDim::DIM_0);
        unsigned srcC = src.GetShape(GlobalTensorDim::DIM_1);
        unsigned srcH = src.GetShape(GlobalTensorDim::DIM_2);
        unsigned srcW = src.GetShape(GlobalTensorDim::DIM_3);
        unsigned srcSize = srcN * srcC * srcH * srcW;
        unsigned dstSize = dstN * dstC1 * dstC0 * dstH * dstW;
        unsigned tmpSize = TileDataTmp::Rows * TileDataTmp::Cols;
        PTO_ASSERT(srcH * srcW * sizeof(T) % BLOCK_BYTE_SIZE == 0, "expect align for H * W");
        PTO_ASSERT(srcN == dstN && srcH == dstH && srcW == dstW && dstC1 == (srcC + dstC0 - 1) / dstC0,
                   "expect same size for src and dst.");
        PTO_ASSERT((srcSize + dstSize + tmpSize) * sizeof(T) < UB_SIZE, "ERROR: memory usage exceeds UB limit!");
    } else if (TileDataSrc::layout == Layout::NC1HWC0 && TileDataDst::layout == Layout::FRACTAL_Z) {
        unsigned dstC1HW = dst.GetShape(GlobalTensorDim::DIM_0);
        unsigned dstN1 = dst.GetShape(GlobalTensorDim::DIM_1);
        unsigned dstN0 = dst.GetShape(GlobalTensorDim::DIM_2);
        unsigned dstC0 = dst.GetShape(GlobalTensorDim::DIM_3);
        unsigned srcN = src.GetShape(GlobalTensorDim::DIM_0);
        unsigned srcC1 = src.GetShape(GlobalTensorDim::DIM_1);
        unsigned srcH = src.GetShape(GlobalTensorDim::DIM_2);
        unsigned srcW = src.GetShape(GlobalTensorDim::DIM_3);
        unsigned srcC0 = src.GetShape(GlobalTensorDim::DIM_4);
        unsigned srcSize = srcN * srcC1 * srcC0 * srcH * srcW;
        unsigned dstSize = dstC1HW * dstN1 * dstN0 * dstC0;
        unsigned tmpSize = TileDataTmp::Rows * TileDataTmp::Cols;
        PTO_ASSERT((srcSize + dstSize + tmpSize) * sizeof(T) < UB_SIZE, "ERROR: memory usage exceeds UB limit!");
        PTO_ASSERT(srcC1 * srcH * srcW == dstC1HW && srcC0 == dstC0 && dstN1 == (srcN + dstN0 - 1) / dstN0,
                   "expect same size for src and dst.");
    }
#endif
}

template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL void TTRANS_IMPL(TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp)
{
    using T = typename TileDataSrc::DType;
    using U = typename TileDataDst::DType;
    static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "Fix: TTRANS has unsupported data type.");
    static_assert(sizeof(T) == sizeof(U), "Fix: TTRANS has inconsistent input and output data type.");
    if constexpr (is_conv_tile_v<TileDataSrc>) {
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
        CheckConvTile<TileDataDst, TileDataSrc, TileDataTmp>();
        if constexpr (TileDataSrc::layout == Layout::NC1HWC0 && TileDataDst::layout == Layout::FRACTAL_Z) {
            unsigned srcN = src.GetShape(GlobalTensorDim::DIM_0);
            unsigned srcC1 = src.GetShape(GlobalTensorDim::DIM_1);
            unsigned srcH = src.GetShape(GlobalTensorDim::DIM_2);
            unsigned srcW = src.GetShape(GlobalTensorDim::DIM_3);
            unsigned srcC0 = src.GetShape(GlobalTensorDim::DIM_4);
            unsigned dstN1 = dst.GetShape(GlobalTensorDim::DIM_1);
            unsigned dstN0 = dst.GetShape(GlobalTensorDim::DIM_2);
            TTransConvNC1HWC02C1HWNC0<TileDataSrc, blockSizeElem>(dst.data(), src.data(), tmp.data(), dstN1 * dstN0,
                                                                  srcN, srcC1 * srcH * srcW, srcC0);
        } else if (TileDataSrc::layout == Layout::NCHW && TileDataDst::layout == Layout::NC1HWC0) {
            unsigned srcN = src.GetShape(GlobalTensorDim::DIM_0);
            unsigned srcC = src.GetShape(GlobalTensorDim::DIM_1);
            unsigned srcH = src.GetShape(GlobalTensorDim::DIM_2);
            unsigned srcW = src.GetShape(GlobalTensorDim::DIM_3);
            unsigned dstC0 = dst.GetShape(GlobalTensorDim::DIM_4);
            TTransConvNCHW2NC1HWC0<TileDataSrc, blockSizeElem>(dst.data(), src.data(), tmp.data(), srcN, srcC, srcH,
                                                               srcW, dstC0);
        }
        return;
    } else {
        static_assert(TileDataSrc::isRowMajor, "Fix: TTRANS has not supported layout type.");
        constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
        constexpr unsigned dstStride = TileDataDst::RowStride;
        constexpr unsigned srcStride = TileDataSrc::RowStride;

        unsigned validRow = src.GetValidRow();
        unsigned validCol = src.GetValidCol();
        TTrans<TileDataDst, TileDataSrc, TileDataTmp, blockSizeElem>(dst.data(), src.data(), tmp.data(), validRow,
                                                                     validCol, dstStride, srcStride);
    }
}
} // namespace pto
#endif