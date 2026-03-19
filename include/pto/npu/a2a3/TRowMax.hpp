/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TROWMAX_HPP
#define TROWMAX_HPP

#include "TRowReduceOps.hpp"
#include <limits>

namespace pto {

// Float/Half operation traits (for vcmax-based implementation)
template <typename T>
struct TRowMaxOp : TRowReduceOp<T, TRowMaxOp<T>> {
    PTO_INTERNAL static void BinInstrImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t rptTimes,
                                          uint16_t dstRptStride, uint16_t src0RptStride, uint16_t src1RptStride,
                                          uint8_t dstBlockStride = 1, uint8_t src0BlockStride = 1,
                                          uint8_t src1BlockStride = 1)
    {
        vmax(dst, src0, src1, rptTimes, dstBlockStride, src0BlockStride, src1BlockStride, dstRptStride, src0RptStride,
             src1RptStride);
    }

    PTO_INTERNAL static void ReduceInstrImpl(__ubuf__ T *dst, __ubuf__ T *src, uint8_t rptTimes, uint16_t dstRptStride,
                                             uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        vcmax(dst, src, rptTimes, dstRptStride, srcBlkStride, srcRptStride, ONLY_VALUE);
    }

    PTO_INTERNAL static void GroupReduceInstrImpl(__ubuf__ T *dst, __ubuf__ T *src, uint8_t rptTimes,
                                                  uint16_t dstRptStride, uint16_t src0Stride, uint16_t src1Stride)
    {
        vcgmax(dst, src, rptTimes, dstRptStride, src0Stride, src1Stride);
    }
};

template <typename T, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
__tf__ PTO_INTERNAL void TRowMax(typename TileDataOut::TileDType __out__ dstData,
                                 typename TileDataIn::TileDType __in__ srcData,
                                 typename TileDataTmp::TileDType __in__ tmpData, int validCol, int validRow)
{
    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *src = (__ubuf__ T *)__cce_get_tile_ptr(srcData);
    __ubuf__ T *tmp = (__ubuf__ T *)__cce_get_tile_ptr(tmpData);

    if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t>) {
        // Integer implementation (follows TROWPROD pattern)
        constexpr unsigned dstRowStride = TileDataOut::RowStride;
        constexpr unsigned srcRowStride = TileDataIn::RowStride;
        constexpr unsigned tmpRowStride = TileDataTmp::RowStride;
        constexpr unsigned elemsPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
        unsigned blocksPerRow = validCol / elemsPerBlock;

        set_mask_count();

        for (unsigned row = 0; row < validRow; ++row, dst += dstRowStride, src += srcRowStride, tmp += tmpRowStride) {
            set_vector_mask(0, elemsPerBlock);

            // Initialize tmp with minimum value
            if constexpr (std::is_same_v<T, int32_t>) {
                vector_dup(tmp, (T)std::numeric_limits<int32_t>::min(), 1, 1, 1, 0, 0);
            } else {
                vector_dup(tmp, (T)std::numeric_limits<int16_t>::min(), 1, 1, 1, 0, 0);
            }
            pipe_barrier(PIPE_V);

            // Accumulate using vmax
            for (unsigned block = 0; block < blocksPerRow; ++block) {
                vmax(tmp, tmp, src + block * elemsPerBlock, 1, 0, 0, 1, 0, 0, 1);
                pipe_barrier(PIPE_V);
            }

            // Handle remaining elements
            unsigned elemsLessThanBlock = validCol % elemsPerBlock;
            if (elemsLessThanBlock > 0) {
                set_vector_mask(0, elemsLessThanBlock);
                vmax(tmp, tmp, src + blocksPerRow * elemsPerBlock, 1, 0, 0, 1, 0, 0, 1);
                pipe_barrier(PIPE_V);
            }

            pipe_barrier(PIPE_ALL);

            // Final scalar reduction
            if constexpr (std::is_same_v<T, int32_t>) {
                T result = tmp[0];
                result = result > tmp[1] ? result : tmp[1];
                result = result > tmp[2] ? result : tmp[2];
                result = result > tmp[3] ? result : tmp[3];
                result = result > tmp[4] ? result : tmp[4];
                result = result > tmp[5] ? result : tmp[5];
                result = result > tmp[6] ? result : tmp[6];
                result = result > tmp[7] ? result : tmp[7];
                dst[0] = result;
            } else {
                T result = tmp[0];
                result = result > tmp[1] ? result : tmp[1];
                result = result > tmp[2] ? result : tmp[2];
                result = result > tmp[3] ? result : tmp[3];
                result = result > tmp[4] ? result : tmp[4];
                result = result > tmp[5] ? result : tmp[5];
                result = result > tmp[6] ? result : tmp[6];
                result = result > tmp[7] ? result : tmp[7];
                result = result > tmp[8] ? result : tmp[8];
                result = result > tmp[9] ? result : tmp[9];
                result = result > tmp[10] ? result : tmp[10];
                result = result > tmp[11] ? result : tmp[11];
                result = result > tmp[12] ? result : tmp[12];
                result = result > tmp[13] ? result : tmp[13];
                result = result > tmp[14] ? result : tmp[14];
                result = result > tmp[15] ? result : tmp[15];
                dst[0] = result;
            }
        }

        set_mask_norm();
        set_vector_mask(-1, -1);
    } else {
        // Float/Half implementation (original vcmax-based)
        TRowReduceInstr<TRowMaxOp<T>, T, TileDataOut, TileDataIn, TileDataTmp>(dst, src, tmp, validCol, validRow);
    }
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWMAX_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    int validCol = src.GetValidCol();
    int validRow = src.GetValidRow();
    TRowReduceCheck<TileDataOut, TileDataIn>(validRow, validCol, dst.GetValidRow());
    if (validCol == 0 || validRow == 0) {
        return;
    }

    TRowMax<typename TileDataIn::DType, TileDataOut, TileDataIn, TileDataTmp>(dst.data(), src.data(), tmp.data(),
                                                                              validCol, validRow);
}
} // namespace pto
#endif
