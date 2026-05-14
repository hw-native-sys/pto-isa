/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSCATTER_HPP
#define TSCATTER_HPP

#include <pto/common/constants.hpp>

namespace pto {
template <typename DstTile, typename T>
PTO_INTERNAL void InitUBBuffer(__ubuf__ T *dstPtr)
{
    using U = std::conditional_t<sizeof(T) == 4, uint32_t, uint16_t>;
    __ubuf__ U *dst = (__ubuf__ U *)dstPtr;

    // tile must be 32B align, the result is no remainder
    constexpr uint32_t numel = DstTile::Numel * sizeof(T) / sizeof(U);
    set_mask_count();
    set_vector_mask(0, numel);
    vector_dup(dst, (U)0, 0, 1, 1, 8, 8);
    set_mask_norm();
    set_vector_mask(-1, -1);
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_V, PIPE_S>();
#endif
}

template <typename DstTile, typename SrcTile, typename IdxTile>
__tf__ PTO_INTERNAL void TScatterImpl(typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src,
                                      typename IdxTile::TileDType __in__ idx, unsigned validRow, unsigned validCol)
{
    using T = typename DstTile::DType;
    using TI = typename IdxTile::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ TI *indPtr = (__ubuf__ TI *)__cce_get_tile_ptr(idx);

    // Initialize dst UB buffer
    InitUBBuffer<DstTile>(dstPtr);

    for (int i = 0; i < validRow; i++) {
        for (int j = 0; j < validCol; j++) {
            TI ix = *(indPtr + i * IdxTile::Cols + j);
            dstPtr[ix] = srcPtr[i * SrcTile::Cols + j];
        }
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#endif
}

template <typename DstTile, typename SrcTile, typename IdxTile>
PTO_INTERNAL void TSCATTER_IMPL(DstTile &dst, SrcTile &src, IdxTile &idx)
{
    using TD = typename DstTile::DType;
    using TS = typename SrcTile::DType;
    using TI = typename IdxTile::DType;
    static_assert(std::is_same<TD, int32_t>::value || std::is_same<TD, int16_t>::value ||
                      std::is_same<TD, int8_t>::value || std::is_same<TD, uint32_t>::value ||
                      std::is_same<TD, uint16_t>::value || std::is_same<TD, uint8_t>::value ||
                      std::is_same<TD, half>::value || std::is_same<TD, float16_t>::value ||
                      std::is_same<TD, float32_t>::value || std::is_same<TD, bfloat16_t>::value,
                  "TSCATTER: Invalid data type.");
    static_assert(std::is_same<TD, TS>::value, "TSCATTER: Data type of dst and src must be the same.");
    static_assert((sizeof(TD) == 4 && sizeof(TI) == 4) || (sizeof(TD) == 2 && sizeof(TI) == 2) ||
                      (sizeof(TD) == 1 && sizeof(TI) == 2),
                  "TSCATTER: Invalid data type of idx.");
    static_assert(std::is_same<TI, uint16_t>::value || std::is_same<TI, uint32_t>::value ||
                      std::is_same<TI, int16_t>::value || std::is_same<TI, int32_t>::value,
                  "TSCATTER: Invalid data type of idx.");
    static_assert(DstTile::Loc == TileType::Vec && SrcTile::Loc == TileType::Vec && IdxTile::Loc == TileType::Vec,
                  "TSCATTER: TileType of src and dst tiles must be TileType::Vec.");
    static_assert(
        DstTile::ValidCol <= DstTile::Cols && SrcTile::ValidCol <= SrcTile::Cols && IdxTile::ValidCol <= IdxTile::Cols,
        "TSCATTER: Number of valid columns must not be greater than number of tile columns.");
    static_assert(
        DstTile::ValidRow <= DstTile::Rows && SrcTile::ValidRow <= SrcTile::Rows && IdxTile::ValidRow <= IdxTile::Rows,
        "TSCATTER: Number of valid rows must not be greater than number of tile rows.");

    unsigned validRow = idx.GetValidRow();
    unsigned validCol = idx.GetValidCol();

    TScatterImpl<DstTile, SrcTile, IdxTile>(dst.data(), src.data(), idx.data(), validRow, validCol);
}

template <MaskPattern mask>
PTO_INTERNAL constexpr int GetTimesByMask()
{
    constexpr uint16_t TIMES_1 = 1;
    constexpr uint16_t TIMES_2 = 2;
    constexpr uint16_t TIMES_4 = 4;

    switch (mask) {
        case MaskPattern::P1111:
            return TIMES_1;
        case MaskPattern::P1010:
            return TIMES_2;
        case MaskPattern::P0101:
            return TIMES_2;
        default:
            return TIMES_4;
    }
}

template <MaskPattern mask, int RowStride>
PTO_INTERNAL int GetIdxByMask(int i, int j)
{
    constexpr uint16_t TIMES_2 = 2;
    constexpr uint16_t TIMES_4 = 4;
    constexpr uint16_t INDEX_0 = 0;
    constexpr uint16_t INDEX_1 = 1;
    constexpr uint16_t INDEX_2 = 2;
    constexpr uint16_t INDEX_3 = 3;

    switch (mask) {
        case MaskPattern::P1010:
            return i * RowStride + TIMES_2 * j + INDEX_0;
        case MaskPattern::P0101:
            return i * RowStride + TIMES_2 * j + INDEX_1;
        case MaskPattern::P1000:
            return i * RowStride + TIMES_4 * j + INDEX_0;
        case MaskPattern::P0100:
            return i * RowStride + TIMES_4 * j + INDEX_1;
        case MaskPattern::P0010:
            return i * RowStride + TIMES_4 * j + INDEX_2;
        case MaskPattern::P0001:
            return i * RowStride + TIMES_4 * j + INDEX_3;
        default:
            return i * RowStride + j;
    }
}

template <MaskPattern mask, typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TScatterMaskImpl(typename DstTile::TileDType __out__ dst,
                                          typename SrcTile::TileDType __in__ src, unsigned validRow, unsigned validCol)
{
    using T = typename DstTile::DType;
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

    // Initialize dst UB buffer
    InitUBBuffer<DstTile>(dstPtr);

    int idx = 0;
    for (int i = 0; i < validRow; i++) {
        for (int j = 0; j < validCol; j++) {
            idx = GetIdxByMask<mask, DstTile::RowStride>(i, j);
            dstPtr[idx] = srcPtr[i * SrcTile::Cols + j];
        }
    }
#ifndef __PTO_AUTO__
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
#endif
}

template <MaskPattern mask, typename DstTile, typename SrcTile>
PTO_INTERNAL void TSCATTER_IMPL(DstTile &dst, SrcTile &src)
{
    if constexpr (mask == MaskPattern::P1111) {
        return TMOV_IMPL(dst, src);
    } else {
        using T = typename DstTile::DType;
        static_assert(std::is_same<T, int32_t>::value || std::is_same<T, int16_t>::value ||
                          std::is_same<T, int8_t>::value || std::is_same<T, uint32_t>::value ||
                          std::is_same<T, uint16_t>::value || std::is_same<T, uint8_t>::value ||
                          std::is_same<T, half>::value || std::is_same<T, float16_t>::value ||
                          std::is_same<T, float32_t>::value || std::is_same<T, bfloat16_t>::value,
                      "TSCATTER: Invalid data type.");
        static_assert(std::is_same_v<T, typename SrcTile::DType>,
                      "TSCATTER: Data type of dst and src must be the same.");

        static_assert(DstTile::Loc == TileType::Vec && SrcTile::Loc == TileType::Vec,
                      "TSCATTER: TileType of src and dst tiles must be TileType::Vec.");
        static_assert(DstTile::ValidCol <= DstTile::Cols && SrcTile::ValidCol <= SrcTile::Cols,
                      "TSCATTER: Number of valid columns must not be greater than number of tile columns.");
        static_assert(DstTile::ValidRow <= DstTile::Rows && SrcTile::ValidRow <= SrcTile::Rows,
                      "TSCATTER: Number of valid rows must not be greater than number of tile rows.");
        static_assert(mask >= MaskPattern::P0101 && mask <= MaskPattern::P1111,
                      "TSCATTER: MaskPattern parameter value out of range: must be P0101...P1111 inclusive.");
        unsigned validRow = src.GetValidRow();
        unsigned validCol = src.GetValidCol();

        PTO_ASSERT(validRow == dst.GetValidRow(), "TSCATTER: validRow of src must match dst.");
        PTO_ASSERT(validCol == dst.GetValidCol() * GetTimesByMask<mask>, "TSCATTER: validRow of src must match dst.");

        TScatterMaskImpl<mask, DstTile, SrcTile>(dst.data(), src.data(), validRow, validCol);
    }
}
} // namespace pto

#endif