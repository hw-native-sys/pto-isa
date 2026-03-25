/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TROWEXPAND_HPP
#define TROWEXPAND_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {
constexpr const int vbrcbElem = 8;

template <typename TileDataDst, typename TileDataSrc>
__tf__ PTO_INTERNAL void TRowExpand(typename TileDataDst::TileDType __out__ dst,
                                    typename TileDataSrc::TileDType __in__ src, int validRow, int validCol)
{
    using T = typename TileDataSrc::DType;
    using TRANS = B82B16Trait<T>;
    using TRANSTYPE = typename TRANS::TransType;
    int transValidCol = TRANS::TransSize(validCol);
    __ubuf__ T *dstPtr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);

    constexpr int dstStride = TRANS::template TransStride<TileDataDst::RowStride>();
    constexpr int srcStride = TileDataSrc::RowStride;
    __ubuf__ TRANSTYPE *transDst = (__ubuf__ TRANSTYPE *)dstPtr;

    set_mask_count();
    set_vector_mask(0, transValidCol);
    for (int i = 0; i < validRow; i++) {
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        T tempValue = (T)(*(srcPtr + i * srcStride));
        typename TRANS::TransType transValue = TRANS::TransValue(tempValue);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        vector_dup(transDst + i * dstStride, transValue, 0, 1, 1, BLOCK_MAX_PER_REPEAT, 0);
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <typename T, typename TileDataDst, typename TileDataSrc>
__tf__ PTO_INTERNAL void TRowExpandBrcb(typename TileDataDst::TileDType __out__ dstData,
                                        typename TileDataSrc::TileDType __in__ srcData)
{
    using BrcbType = std::conditional_t<sizeof(T) == sizeof(uint16_t), uint16_t,
                                        std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, T>>;
    __ubuf__ BrcbType *dst = (__ubuf__ BrcbType *)__cce_get_tile_ptr(dstData);
    __ubuf__ BrcbType *src = (__ubuf__ BrcbType *)__cce_get_tile_ptr(srcData);
    constexpr int repeat = TileDataSrc::Numel / vbrcbElem;
    constexpr int elemPerRepeat = REPEAT_BYTE / sizeof(T);

    // vbrcb requires src to be 32B aligned, and offset REPEAT_MAX * vbrcbElem is non-32B aligned
    constexpr int loop = repeat / (REPEAT_MAX - 1);
    constexpr int remain = repeat % (REPEAT_MAX - 1);
    if constexpr (loop > 0) {
        for (int i = 0; i < loop; ++i) {
            vbrcb(dst + i * (REPEAT_MAX - 1) * elemPerRepeat, src + i * (REPEAT_MAX - 1) * vbrcbElem, 1,
                  BLOCK_MAX_PER_REPEAT, (REPEAT_MAX - 1));
        }
    }
    if constexpr (remain > 0) {
        vbrcb(dst + loop * (REPEAT_MAX - 1) * elemPerRepeat, src + loop * (REPEAT_MAX - 1) * vbrcbElem, 1,
              BLOCK_MAX_PER_REPEAT, remain);
    }
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TROWEXPAND_IMPL(TileDataDst &dst, TileDataSrc &src)
{
    using T = typename TileDataSrc::DType;
    static_assert((sizeof(typename TileDataSrc::DType) == 1) || (sizeof(typename TileDataSrc::DType) == 2) ||
                      (sizeof(typename TileDataSrc::DType) == 4),
                  "Fix: TROWEXPAND Data type must be b8/b16/b32");
    static_assert(std::is_same_v<typename TileDataDst::DType, T>,
                  "Fix: TROWEXPAND input data type must be consistent with the output data type");
    static_assert(TileDataSrc::Loc == pto::TileType::Vec, "Fix: TROWEXPAND Src TileType must be Vec!");
    static_assert(TileDataDst::Loc == pto::TileType::Vec, "Fix: TROWEXPAND Dst TileType must be Vec!");
    static_assert(TileDataSrc::SFractal == SLayout::NoneBox, "Fix: TROWEXPAND Src layout must be ND or DN!");
    static_assert((TileDataDst::isRowMajor && (TileDataDst::SFractal == SLayout::NoneBox)),
                  "Fix: TROWEXPAND Src and dst layout must be ND!");
    int srcValidRow = src.GetValidRow();
    int srcValidCol = src.GetValidCol();
    int dstValidRow = dst.GetValidRow();
    int dstValidCol = dst.GetValidCol();
    PTO_ASSERT(srcValidRow == dstValidRow,
               "Fix: TROWEXPAND src tile's valid row must be consistent with dst tile's valid row!");
    PTO_ASSERT(srcValidRow != 0 && srcValidCol != 0 && dstValidRow != 0 && dstValidCol != 0,
               "Fix: TROWEXPAND input/output shape is invalid, validCol or validRow is 0.");

    constexpr bool isBroadcastSupportType = (sizeof(T) == 2 || sizeof(T) == 4);

    constexpr bool isStaticShape =
        (TileDataSrc::Rows == TileDataSrc::ValidRow) && (TileDataSrc::Cols == TileDataSrc::ValidCol) &&
        (TileDataDst::Rows == TileDataDst::ValidRow) && (TileDataDst::Cols == TileDataDst::ValidCol);

    constexpr unsigned elemPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr bool isBroadcast = TileDataSrc::isRowMajor ?
                                     ((TileDataSrc::Rows == 1) && (TileDataSrc::Cols == TileDataDst::Rows) &&
                                      (TileDataDst::Cols == elemPerBlock)) :
                                     ((TileDataSrc::Cols == 1) && (TileDataSrc::Rows == TileDataDst::Rows) &&
                                      (TileDataDst::Cols == elemPerBlock));

    if constexpr (isBroadcastSupportType && isStaticShape && isBroadcast) {
        /*
          isBroadcastSupportType:
            Only b16 and b32 are supported.
          isStaticShape:
            Broadcast is a special case where the src tile is a single row or column,
            src and dst tile are static shapes to ensure that the tile data is saved continuously.
          isBroadcast:
            [1, M] -> [M, elemPerBlock], src is row major.
            [M, 1] -> [M, elemPerBlock], src is column major.
            The value of sizeof(T) x M is a multiple of 32Byte, it also means that M must be a multiple of 8,
            this constraint is implemented by the Tile basic definition.
        */
        TRowExpandBrcb<T, TileDataDst, TileDataSrc>(dst.data(), src.data());
    } else {
        TRowExpand<TileDataDst, TileDataSrc>(dst.data(), src.data(), dstValidRow, dstValidCol);
    }
}
} // namespace pto
#endif
