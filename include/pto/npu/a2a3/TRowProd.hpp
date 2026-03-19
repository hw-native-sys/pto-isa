/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TROWPROD_HPP
#define TROWPROD_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include "TRowReduceOps.hpp"

namespace pto {

template <typename T, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
__tf__ PTO_INTERNAL void TRowProd(typename TileDataOut::TileDType __out__ dst,
                                  typename TileDataIn::TileDType __in__ src, typename TileDataTmp::TileDType __in__ tmp,
                                  int validRow, int validCol)
{
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ T *tmpPtr = (__ubuf__ T *)__cce_get_tile_ptr(tmp);

    constexpr unsigned dstRowStride = TileDataOut::RowStride;
    constexpr unsigned srcRowStride = TileDataIn::RowStride;

    constexpr unsigned elemsPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
    unsigned blocksPerRow = validCol / elemsPerBlock;

    set_mask_count();

    for (unsigned row = 0; row < validRow; ++row, dstPtr += dstRowStride, srcPtr += srcRowStride) {
        set_vector_mask(0, elemsPerBlock);

        if constexpr (std::is_same_v<T, float>) {
            vector_dup(tmpPtr, (T)1.0f, 1, 1, 1, 0, 0);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            vector_dup(tmpPtr, (T)1, 1, 1, 1, 0, 0);
        } else if constexpr (std::is_same_v<T, int16_t>) {
            vector_dup(tmpPtr, (T)1, 1, 1, 1, 0, 0);
        } else if constexpr (std::is_same_v<T, half>) {
            vector_dup(tmpPtr, (T)1.0f, 1, 1, 1, 0, 0);
        }
        pipe_barrier(PIPE_V);

        for (unsigned block = 0; block < blocksPerRow; ++block) {
            vmul(tmpPtr, tmpPtr, srcPtr + block * elemsPerBlock, 1, 0, 0, 1, 0, 0, 1);
            pipe_barrier(PIPE_V);
        }

        unsigned elemsLessThanBlock = validCol % elemsPerBlock;
        if (elemsLessThanBlock > 0) {
            set_vector_mask(0, elemsLessThanBlock);
            vmul(tmpPtr, tmpPtr, srcPtr + blocksPerRow * elemsPerBlock, 1, 0, 0, 1, 0, 0, 1);
            pipe_barrier(PIPE_V);
        }

        pipe_barrier(PIPE_ALL);
        if constexpr (std::is_same_v<T, float>) {
            dstPtr[0] = tmpPtr[0] * tmpPtr[1] * tmpPtr[2] * tmpPtr[3] * tmpPtr[4] * tmpPtr[5] * tmpPtr[6] * tmpPtr[7];
        } else if constexpr (std::is_same_v<T, half>) {
            dstPtr[0] = (half)((float)tmpPtr[0] * (float)tmpPtr[1] * (float)tmpPtr[2] * (float)tmpPtr[3] *
                               (float)tmpPtr[4] * (float)tmpPtr[5] * (float)tmpPtr[6] * (float)tmpPtr[7] *
                               (float)tmpPtr[8] * (float)tmpPtr[9] * (float)tmpPtr[10] * (float)tmpPtr[11] *
                               (float)tmpPtr[12] * (float)tmpPtr[13] * (float)tmpPtr[14] * (float)tmpPtr[15]);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            dstPtr[0] = tmpPtr[0] * tmpPtr[1] * tmpPtr[2] * tmpPtr[3] * tmpPtr[4] * tmpPtr[5] * tmpPtr[6] * tmpPtr[7];
        } else if constexpr (std::is_same_v<T, int16_t>) {
            dstPtr[0] = tmpPtr[0] * tmpPtr[1] * tmpPtr[2] * tmpPtr[3] * tmpPtr[4] * tmpPtr[5] * tmpPtr[6] * tmpPtr[7] *
                        tmpPtr[8] * tmpPtr[9] * tmpPtr[10] * tmpPtr[11] * tmpPtr[12] * tmpPtr[13] * tmpPtr[14] *
                        tmpPtr[15];
        } else {
            static_assert(sizeof(T) == 0, "T must be float, half, int32, or int16");
        }
    }

    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWPROD_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    using T = typename TileDataIn::DType;
    int validCol = src.GetValidCol();
    int validRow = src.GetValidRow();
    TRowReduceCheck<TileDataOut, TileDataIn>(validRow, validCol, dst.GetValidRow());

    TRowProd<T, TileDataOut, TileDataIn, TileDataTmp>(dst.data(), src.data(), tmp.data(), validRow, validCol);
}

} // namespace pto
#endif
