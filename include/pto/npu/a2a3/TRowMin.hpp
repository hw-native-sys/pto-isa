/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TROWMIN_HPP
#define TROWMIN_HPP

#include "TRowReduceOps.hpp"
#include <limits>

namespace pto {

// Scalar reduction for integer types
template <typename T, unsigned N>
PTO_INTERNAL T ScalarReduceMin(__ubuf__ T *tmp)
{
    T result = tmp[0];
    for (unsigned i = 1; i < N; ++i) {
        result = result < tmp[i] ? result : tmp[i];
    }
    return result;
}

// Float/Half operation traits (for vcmin-based implementation)
template <typename T>
struct TRowMinOp : TRowReduceOp<T, TRowMinOp<T>> {
    PTO_INTERNAL static void BinInstrImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, uint8_t rptTimes,
                                          uint16_t dstRptStride, uint16_t src0RptStride, uint16_t src1RptStride,
                                          uint8_t dstBlockStride = 1, uint8_t src0BlockStride = 1,
                                          uint8_t src1BlockStride = 1)
    {
        vmin(dst, src0, src1, rptTimes, dstBlockStride, src0BlockStride, src1BlockStride, dstRptStride, src0RptStride,
             src1RptStride);
    }

    PTO_INTERNAL static void ReduceInstrImpl(__ubuf__ T *dst, __ubuf__ T *src, uint8_t rptTimes, uint16_t dstRptStride,
                                             uint16_t srcBlkStride, uint16_t srcRptStride)
    {
        vcmin(dst, src, rptTimes, dstRptStride, srcBlkStride, srcRptStride, ONLY_VALUE);
    }

    PTO_INTERNAL static void GroupReduceInstrImpl(__ubuf__ T *dst, __ubuf__ T *src, uint8_t rptTimes,
                                                  uint16_t dstRptStride, uint16_t src0Stride, uint16_t src1Stride)
    {
        vcgmin(dst, src, rptTimes, dstRptStride, src0Stride, src1Stride);
    }
};

template <typename T, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
__tf__ PTO_INTERNAL void TRowMin(typename TileDataOut::TileDType __out__ dstData,
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
        unsigned elemsLessThanBlock = validCol % elemsPerBlock;
        unsigned blocksPerRow = validCol / elemsPerBlock;

        set_mask_count();

        for (unsigned row = 0; row < validRow; ++row, dst += dstRowStride, src += srcRowStride, tmp += tmpRowStride) {
            set_vector_mask(0, elemsPerBlock);

            // Initialize tmp with maximum value
            if constexpr (std::is_same_v<T, int32_t>) {
                vector_dup(tmp, (T)std::numeric_limits<int32_t>::max(), 1, 1, 1, 0, 0);
            } else {
                vector_dup(tmp, (T)std::numeric_limits<int16_t>::max(), 1, 1, 1, 0, 0);
            }
            pipe_barrier(PIPE_V);

            // Accumulate using vmin
            for (unsigned block = 0; block < blocksPerRow; ++block) {
                vmin(tmp, tmp, src + block * elemsPerBlock, 1, 0, 0, 1, 0, 0, 1);
                pipe_barrier(PIPE_V);
            }

            // Handle remaining elements
            if (elemsLessThanBlock > 0) {
                set_vector_mask(0, elemsLessThanBlock);
                vmin(tmp, tmp, src + blocksPerRow * elemsPerBlock, 1, 0, 0, 1, 0, 0, 1);
                pipe_barrier(PIPE_V);
            }

            pipe_barrier(PIPE_ALL);

            // Final scalar reduction
            if constexpr (std::is_same_v<T, int32_t>) {
                dst[0] = ScalarReduceMin<T, 8>(tmp);
            } else {
                dst[0] = ScalarReduceMin<T, 16>(tmp);
            }
        }

        set_mask_norm();
        set_vector_mask(-1, -1);
    } else {
        // Float/Half implementation (original vcmin-based)
        TRowReduceInstr<TRowMinOp<T>, T, TileDataOut, TileDataIn, TileDataTmp>(dst, src, tmp, validCol, validRow);
    }
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWMIN_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    int validCol = src.GetValidCol();
    int validRow = src.GetValidRow();
    TRowReduceCheck<TileDataOut, TileDataIn>(validRow, validCol, dst.GetValidRow());

    TRowMin<typename TileDataIn::DType, TileDataOut, TileDataIn, TileDataTmp>(dst.data(), src.data(), tmp.data(),
                                                                              validCol, validRow);
}
} // namespace pto
#endif
