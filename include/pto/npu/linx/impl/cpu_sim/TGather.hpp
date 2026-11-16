/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Integrated from LinxISA/Linx-TileOP-API; see ../../README.md for provenance.

#ifndef PTO_LINX_VENDOR_TGATHER_HPP
#define PTO_LINX_VENDOR_TGATHER_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape_dst, typename tile_shape_src, typename tile_shape_indices>
void TGather_RowMajor_Imp(
    typename tile_shape_dst::TileDType dst, const typename tile_shape_src::TileDType src,
    const typename tile_shape_indices::TileDType indices)
{
    for (size_t i = 0; i < tile_shape_dst::ValidRow; ++i)
        for (size_t j = 0; j < tile_shape_dst::ValidCol; ++j) {
            size_t index = i * tile_shape_indices::RowStride + j;
            size_t idx = indices[index] * tile_shape_src::RowStride + j;
            dst[index] = src[idx];
        }
}

template <typename tile_shape_dst, typename tile_shape_src, typename tile_shape_indices>
void TGather_ColMajor_Imp(
    typename tile_shape_dst::TileDType dst, const typename tile_shape_src::TileDType src,
    const typename tile_shape_indices::TileDType indices)
{
    for (size_t i = 0; i < tile_shape_dst::ValidCol; ++i)
        for (size_t j = 0; j < tile_shape_dst::ValidRow; ++j) {
            size_t index = i * tile_shape_indices::ColStride + j;
            size_t idx = indices[index] + i * tile_shape_src::ColStride;
            dst[index] = src[idx];
        }
}

template <is_tile_data_v tile_shape_dst, is_tile_data_v tile_shape_src, is_tile_data_v tile_shape_indices>
void TGATHER_Impl(tile_shape_dst& dst, tile_shape_src& src, tile_shape_indices& indices)
{
    static_assert(
        tile_shape_dst::Rows == tile_shape_indices::Rows && tile_shape_dst::Cols == tile_shape_indices::Cols,
        "Error! Output shape != Index(indices) shape");
    static_assert(
        tile_shape_dst::ValidRow != DYNAMIC && tile_shape_dst::ValidCol != DYNAMIC &&
            tile_shape_src::ValidRow != DYNAMIC && tile_shape_src::ValidCol != DYNAMIC &&
            tile_shape_indices::ValidRow != DYNAMIC && tile_shape_indices::ValidCol != DYNAMIC,
        "TODO: Support tile dynamic shape!");
    if constexpr (tile_shape_src::isRowMajor && tile_shape_indices::isRowMajor && tile_shape_dst::isRowMajor) {
        TGather_RowMajor_Imp<tile_shape_dst, tile_shape_src, tile_shape_indices>(
            dst.data(), src.data(), indices.data());
    } else if constexpr (
        !tile_shape_src::isRowMajor && !tile_shape_indices::isRowMajor && !tile_shape_dst::isRowMajor) {
        TGather_ColMajor_Imp<tile_shape_dst, tile_shape_src, tile_shape_indices>(
            dst.data(), src.data(), indices.data());
    } else {
        static_assert(
            tile_shape_src::isRowMajor && tile_shape_indices::isRowMajor && tile_shape_dst::isRowMajor,
            "Storage type not supported");
    }
}

#endif
