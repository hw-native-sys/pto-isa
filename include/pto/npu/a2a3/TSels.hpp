/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSELS_HPP
#define TSELS_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a2a3/TSel.hpp>

namespace pto {
template <typename DstTile, typename MaskTile, typename SrcTile, typename TmpTile>
__tf__ PTO_INTERNAL void TSels(
    typename DstTile::TileDType __out__ dst, typename MaskTile::TileDType __in__ mask,
    typename SrcTile::TileDType __in__ src, typename TmpTile::TileDType __in__ tmp,
    typename SrcTile::DType __in__ scalar, unsigned validRow, unsigned validCol)
{
    using T = std::conditional_t<sizeof(typename DstTile::DType) == 4, float, half>;
    using MaskT = typename MaskTile::DType;
    constexpr unsigned maskRowStride = MaskTile::RowStride;
    constexpr unsigned cmpmaskLen = sizeof(T) == 2 ? 4 : 2; // 128bit for B16 and 64bit for B32

    __ubuf__ T* dstPtr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    __ubuf__ MaskT* maskPtr = (__ubuf__ MaskT*)__cce_get_tile_ptr(mask);
    __ubuf__ typename SrcTile::DType* scalarPtr = (__ubuf__ typename SrcTile::DType*)__cce_get_tile_ptr(tmp);

    *scalarPtr = scalar;
    set_mask_count();
    set_vector_mask(0, validCol);
    PtoSetWaitFlag<PIPE_S, PIPE_V>();
    set_cmpmask(scalarPtr);
    pipe_barrier(PIPE_V);
    for (unsigned i = 0; i < validRow; i++) {
        vsel(
            dstPtr + i * DstTile::RowStride, srcPtr + i * SrcTile::RowStride, maskPtr + i * MaskTile::RowStride, 1, 1,
            1, 1, 8, 8, 8, SELMODE::VSEL_TENSOR_SCALAR_MODE);
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <typename TileDataDst, typename TileDataMask, typename TileDataSrc, typename TmpTile>
PTO_INTERNAL void TSELS_IMPL(
    TileDataDst& dst, TileDataMask& mask, TileDataSrc& src, TmpTile& tmp, typename TileDataSrc::DType scalar)
{
    static_assert(
        sizeof(typename TileDataDst::DType) == 4 || sizeof(typename TileDataDst::DType) == 2,
        "Fix: TSEL only support 16B and 32B data type.");
    static_assert(
        std::is_same_v<typename TileDataDst::DType, typename TileDataSrc::DType>,
        "Fix: TSEL only support same data type between dst, src.");
    static_assert(TileDataDst::isRowMajor && TileDataSrc::isRowMajor, "Fix: TSEL only support RowMajor layout type.");
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();

    PTO_ASSERT(src.GetValidCol() == dst.GetValidCol(), "Number of columns of src and dst must be the same.");
    PTO_ASSERT(src.GetValidRow() == dst.GetValidRow(), "Number of rows of src and dst must be the same.");

    TSels<TileDataDst, TileDataMask, TileDataSrc, TmpTile>(
        dst.data(), mask.data(), src.data(), tmp.data(), scalar, validRow, validCol);
}

} // namespace pto
#endif
