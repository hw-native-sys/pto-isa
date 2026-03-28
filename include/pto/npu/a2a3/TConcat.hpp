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

template <typename DstTile, typename Src0Tile, typename Src1Tile, typename Src0IdxTile, typename Src1IdxTile>
__tf__ PTO_INTERNAL void TConcatIdx(typename DstTile::TileDType __out__ dst, typename Src0Tile::TileDType __in__ src0,
                                    typename Src1Tile::TileDType __in__ src1,
                                    typename Src0IdxTile::TileDType __in__ idx0,
                                    typename Src1IdxTile::TileDType __in__ idx1, unsigned validRow,
                                    unsigned dstValidCol)
{
    using dataType = typename DstTile::DType;
    using idxType = typename Src0IdxTile::DType;
    using copyType = std::conditional_t<sizeof(dataType) == sizeof(int32_t), __ubuf__ int32_t *, __ubuf__ int16_t *>;

    __ubuf__ dataType *dstPtr = (__ubuf__ dataType *)__cce_get_tile_ptr(dst);
    __ubuf__ dataType *src0Ptr = (__ubuf__ dataType *)__cce_get_tile_ptr(src0);
    __ubuf__ dataType *src1Ptr = (__ubuf__ dataType *)__cce_get_tile_ptr(src1);
    __ubuf__ idxType *idx0Ptr = (__ubuf__ idxType *)__cce_get_tile_ptr(idx0);
    __ubuf__ idxType *idx1Ptr = (__ubuf__ idxType *)__cce_get_tile_ptr(idx1);

    constexpr unsigned elementsPerBlock = BLOCK_BYTE_SIZE / sizeof(dataType);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(dataType);
    constexpr unsigned dstStride = DstTile::RowStride;
    constexpr unsigned src0Stride = Src0Tile::RowStride;
    constexpr unsigned src1Stride = Src1Tile::RowStride;
    constexpr unsigned idx0Stride = Src0IdxTile::RowStride;
    constexpr unsigned idx1Stride = Src1IdxTile::RowStride;

    for (uint16_t i = 0; i < validRow; i++) {
        PtoSetWaitFlag<PIPE_MTE2, PIPE_S>();
        unsigned idx0Num = *(idx0Ptr + i * idx0Stride);
        unsigned idx1Num = *(idx1Ptr + i * idx1Stride);
        unsigned src0Num = idx0Num < dstValidCol ? idx0Num : dstValidCol;
        unsigned src1Col = dstValidCol > src0Num ? dstValidCol - src0Num : 0;
        unsigned src1Num = idx1Num < src1Col ? idx1Num : src1Col;
        unsigned src0RepeatTime = CeilDivision(src0Num, elementsPerRepeat);
        unsigned src1RepeatTime = CeilDivision(src1Num, elementsPerRepeat);
        unsigned src0RepeatOffset = (src0RepeatTime - 1) * elementsPerRepeat;
        unsigned src1RepeatOffset = (src1RepeatTime - 1) * elementsPerRepeat;
        unsigned src0MaskNum = src0Num % elementsPerRepeat;
        unsigned src1MaskNum = src1Num % elementsPerRepeat;
        bool isAligned = (src0Num % elementsPerBlock) == 0;
        PtoSetWaitFlag<PIPE_S, PIPE_V>();

        vcopy((copyType)(dstPtr + i * dstStride), (copyType)(src0Ptr + i * src0Stride), src0RepeatTime - 1, 1, 1, 8, 8);
        SetContinuousMask(src0MaskNum);
        vcopy((copyType)(dstPtr + src0RepeatOffset + i * dstStride),
              (copyType)(src0Ptr + src0RepeatOffset + i * src0Stride), 1, 1, 1, 8, 8);
        set_vector_mask(-1, -1);

        if (isAligned) {
            vcopy((copyType)(dstPtr + i * dstStride + src0Num), (copyType)(src1Ptr + i * src1Stride),
                  src1RepeatTime - 1, 1, 1, 8, 8);
            SetContinuousMask(src1MaskNum);
            vcopy((copyType)(dstPtr + src1RepeatOffset + i * dstStride + src0Num),
                  (copyType)(src1Ptr + src1RepeatOffset + i * src1Stride), 1, 1, 1, 8, 8);
            set_vector_mask(-1, -1);
        } else {
            PtoSetWaitFlag<PIPE_V, PIPE_S>();
            for (unsigned j = 0; j < src1Num; j++) {
                dstPtr[i * dstStride + src0Num + j] = src1Ptr[i * src1Stride + j];
            }
            PtoSetWaitFlag<PIPE_S, PIPE_V>();
            PtoSetWaitFlag<PIPE_S, PIPE_MTE3>();
        }
    }
}

template <typename DstTile, typename Src0Tile, typename Src1Tile, typename Src0IdxTile, typename Src1IdxTile>
PTO_INTERNAL void TCONCAT_IMPL(DstTile &dst, Src0Tile &src0, Src1Tile &src1, Src0IdxTile &src0Idx, Src1IdxTile &src1Idx)
{
    using dataType = typename DstTile::DType;
    using idxType = typename Src0IdxTile::DType;

    static_assert(std::is_same<dataType, typename Src0Tile::DType>::value &&
                      std::is_same<dataType, typename Src1Tile::DType>::value,
                  "TCONCAT: Data type of dst, src0 and src1 must be the same.");
    static_assert(std::is_same<dataType, int32_t>::value || std::is_same<dataType, int16_t>::value ||
                      std::is_same<dataType, int8_t>::value || std::is_same<dataType, uint32_t>::value ||
                      std::is_same<dataType, uint16_t>::value || std::is_same<dataType, uint8_t>::value ||
                      std::is_same<dataType, half>::value || std::is_same<dataType, float32_t>::value ||
                      std::is_same<dataType, bfloat16_t>::value,
                  "TCONCAT: Invalid data type.");
    static_assert(std::is_same<idxType, typename Src1IdxTile::DType>::value,
                  "TCONCAT: Data type of src0Idx and src1Idx must be the same.");
    static_assert(std::is_same<idxType, int32_t>::value || std::is_same<idxType, int16_t>::value ||
                      std::is_same<idxType, int8_t>::value || std::is_same<idxType, uint32_t>::value ||
                      std::is_same<idxType, uint16_t>::value || std::is_same<idxType, uint8_t>::value,
                  "TCONCAT: Invalid data type of src0Idx.");
    static_assert(DstTile::Loc == TileType::Vec && Src0Tile::Loc == TileType::Vec && Src1Tile::Loc == TileType::Vec,
                  "TCONCAT: TileType of src and dst tiles must be TileType::Vec.");
    static_assert(DstTile::ValidRow <= DstTile::Rows && Src0Tile::ValidRow <= Src0Tile::Rows &&
                      Src1Tile::ValidRow <= Src1Tile::Rows,
                  "TCONCAT: Number of valid rows must not be greater than number of tile rows.");

    unsigned validRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();

    PTO_ASSERT(validRow == src0.GetValidRow(), "TCONCAT: validRow of src0 must match dst.");
    PTO_ASSERT(validRow == src1.GetValidRow(), "TCONCAT: validRow of src1 must match dst.");

    TConcatIdx<DstTile, Src0Tile, Src1Tile, Src0IdxTile, Src1IdxTile>(
        dst.data(), src0.data(), src1.data(), src0Idx.data(), src1Idx.data(), validRow, dstValidCol);
}
} // namespace pto

#endif
