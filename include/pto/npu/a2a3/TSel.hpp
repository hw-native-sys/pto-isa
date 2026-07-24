/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSEL_HPP
#define TSEL_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>

namespace pto {
enum class SELMODE : uint8_t {
    VSEL_CMPMASK_SPR = 0,
    VSEL_TENSOR_SCALAR_MODE = 1,
    VSEL_TENSOR_TENSOR_MODE = 2,
};

template <typename DstTile, typename MaskTile, typename Src0Tile, typename Src1Tile, typename TmpTile>
__tf__ PTO_INTERNAL void TSel(
    typename DstTile::TileDType __out__ dst, typename MaskTile::TileDType __in__ selMask,
    typename Src0Tile::TileDType __in__ src0, typename Src1Tile::TileDType __in__ src1,
    typename TmpTile::TileDType __in__ tmp, unsigned validRow, unsigned validCol)
{
    using T = std::conditional_t<sizeof(typename DstTile::DType) == 4, float, half>;
    using MaskT = typename MaskTile::DType;

    constexpr unsigned dstRowStride = DstTile::RowStride;
    constexpr unsigned src0RowStride = Src0Tile::RowStride;
    constexpr unsigned src1RowStride = Src1Tile::RowStride;
    constexpr unsigned maskRowStride = MaskTile::RowStride;
    constexpr unsigned cmpmaskLen = sizeof(T) == 2 ? 4 : 2; // 128bit for B16 and 64bit for B32

    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* src0Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src0);
    __ubuf__ T* src1Ptr = (__ubuf__ T*)__cce_get_tile_ptr(src1);
    __ubuf__ MaskT* maskPtr = (__ubuf__ MaskT*)__cce_get_tile_ptr(selMask);

    __ubuf__ uint32_t* cmpMaskPtr = (__ubuf__ uint32_t*)__cce_get_tile_ptr(tmp);

    uint32_t maskAddr;
    set_mask_count();
    for (unsigned i = 0; i < validRow; i++) {
        set_vector_mask(0, cmpmaskLen);
        maskAddr = static_cast<uint32_t>(reinterpret_cast<int64_t>(maskPtr + i * maskRowStride));
        vector_dup(cmpMaskPtr, maskAddr, 1, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        set_cmpmask(cmpMaskPtr);
        pipe_barrier(PIPE_V);
        set_vector_mask(0, validCol);
        vsel(
            (__ubuf__ T*)(dstPtr + i * dstRowStride), (__ubuf__ T*)(src0Ptr + i * src0RowStride),
            (__ubuf__ T*)(src1Ptr + i * src1RowStride), 1, 1, 1, 1, 8, 8, 8, SELMODE::VSEL_TENSOR_TENSOR_MODE);
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <typename DstTile, typename MaskTile, typename Src0Tile, typename Src1Tile, typename TmpTile>
PTO_INTERNAL void TSEL_IMPL(DstTile& dst, MaskTile& selMask, Src0Tile& src0, Src1Tile& src1, TmpTile& tmp)
{
    static_assert(
        sizeof(typename DstTile::DType) == 4 || sizeof(typename DstTile::DType) == 2,
        "Fix: TSEL only support 16B and 32B data type.");
    static_assert(
        std::is_same_v<typename DstTile::DType, typename Src0Tile::DType> ||
            std::is_same_v<typename DstTile::DType, typename Src1Tile::DType>,
        "Fix: TSEL only support same data type between dst, src0, and src1.");
    static_assert(
        DstTile::isRowMajor && Src0Tile::isRowMajor && Src1Tile::isRowMajor,
        "Fix: TSEL only support RowMajor layout type.");
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    TSel<DstTile, MaskTile, Src0Tile, Src1Tile, TmpTile>(
        dst.data(), selMask.data(), src0.data(), src1.data(), tmp.data(), validRow, validCol);
}
} // namespace pto
#endif
