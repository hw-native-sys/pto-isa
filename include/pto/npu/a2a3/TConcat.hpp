/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TCONCAT_HPP
#define TCONCAT_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {
template <typename TileDataD, typename TileDataS0, typename TileDataS1>
__tf__ PTO_INTERNAL void TConcatImpl(typename TileDataD::TileDType __out__ dst,
                                     typename TileDataS0::TileDType __in__ src0,
                                     typename TileDataS1::TileDType __in__ src1, unsigned validRow, unsigned validCol0,
                                     unsigned validCol1)
{
    using TD = typename TileDataD::DType;
    using TS0 = typename TileDataS0::DType;
    using TS1 = typename TileDataS1::DType;

    __ubuf__ TD *dstPtr = (__ubuf__ TD *)__cce_get_tile_ptr(dst);
    __ubuf__ TS0 *src0Ptr = (__ubuf__ TS0 *)__cce_get_tile_ptr(src0);
    __ubuf__ TS1 *src1Ptr = (__ubuf__ TS1 *)__cce_get_tile_ptr(src1);

    constexpr unsigned elementsPerBlock = BLOCK_BYTE_SIZE / sizeof(TD);
    constexpr unsigned dstRowStride = TileDataD::RowStride;
    constexpr unsigned src0RowStride = TileDataS0::RowStride;
    constexpr unsigned src1RowStride = TileDataS1::RowStride;

    unsigned blockLen = (validCol0 * sizeof(TD) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
    unsigned src0Gap = (TileDataS0::Cols * sizeof(TD) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE - blockLen;
    unsigned dstGap = (TileDataD::Cols * sizeof(TD) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE - blockLen;
    for (int i = 0; i < validRow; i++) {
        copy_ubuf_to_ubuf(dstPtr + i * dstRowStride, src0Ptr + i * src0RowStride, 0, 1, blockLen, src0Gap, dstGap);
    }

    bool isAligned = (validCol0 % elementsPerBlock) == 0;
    if (isAligned) {
        unsigned src1Gap = (TileDataS1::Cols * sizeof(TD) + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE - blockLen;
        for (int i = 0; i < validRow; i++) {
            copy_ubuf_to_ubuf(dstPtr + i * dstRowStride + validCol0, src1Ptr + i * src1RowStride, 0, 1, blockLen,
                              src1Gap, dstGap);
        }
    } else {
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        for (unsigned i = 0; i < validRow; i++) {
            for (unsigned j = 0; j < validCol1; j++) {
                dstPtr[i * dstRowStride + validCol0 + j] = src1Ptr[i * src1RowStride + j];
            }
        }
    }
}

template <typename TileDataD, typename TileDataS0, typename TileDataS1>
PTO_INTERNAL void TCONCAT_IMPL(TileDataD &dst, TileDataS0 &src0, TileDataS1 &src1)
{
    using TD = typename TileDataD::DType;
    using TS0 = typename TileDataS0::DType;
    using TS1 = typename TileDataS1::DType;

    static_assert(std::is_same<TD, TS0>::value && std::is_same<TD, TS1>::value,
                  "TCONCAT: Data type of dst, src0 and src1 must be the same.");
    static_assert(std::is_same<TD, int32_t>::value || std::is_same<TD, int16_t>::value ||
                      std::is_same<TD, int8_t>::value || std::is_same<TD, uint32_t>::value ||
                      std::is_same<TD, uint16_t>::value || std::is_same<TD, uint8_t>::value ||
                      std::is_same<TD, half>::value || std::is_same<TD, float16_t>::value ||
                      std::is_same<TD, float32_t>::value || std::is_same<TD, bfloat16_t>::value,
                  "TCONCAT: Invalid data type.");
    static_assert(
        TileDataD::Loc == TileType::Vec && TileDataS0::Loc == TileType::Vec && TileDataS1::Loc == TileType::Vec,
        "TCONCAT: TileType of src and dst tiles must be TileType::Vec.");
    static_assert(TileDataD::ValidRow <= TileDataD::Rows && TileDataS0::ValidRow <= TileDataS0::Rows &&
                      TileDataS1::ValidRow <= TileDataS1::Rows,
                  "TCONCAT: Number of valid rows must not be greater than number of tile rows.");

    unsigned validRow = dst.GetValidRow();
    unsigned validCol0 = src0.GetValidCol();
    unsigned validCol1 = src1.GetValidCol();

    PTO_ASSERT(validRow == src0.GetValidRow(), "TCONCAT: validRow of src0 must match dst.");
    PTO_ASSERT(validRow == src1.GetValidRow(), "TCONCAT: validRow of src1 must match dst.");
    PTO_ASSERT(validCol0 + validCol1 <= TileDataD::Cols, "TCONCAT: Total columns exceed dst capacity.");

    TConcatImpl<TileDataD, TileDataS0, TileDataS1>(dst.data(), src0.data(), src1.data(), validRow, validCol0,
                                                   validCol1);
}
} // namespace pto

#endif
