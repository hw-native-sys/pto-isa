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

#ifndef PTO_LINX_VENDOR_TCAST_HPP
#define PTO_LINX_VENDOR_TCAST_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape_out, typename tile_shape_in>
void TCast_NoFractal_Imp(typename tile_shape_out::TileDType dst, const typename tile_shape_in::TileDType src)
{
    using type = typename tile_shape_out::DType;
    for (uint16_t i = 0; i < tile_shape_in::ValidRow; ++i)
        for (uint16_t j = 0; j < tile_shape_in::ValidCol; ++j) {
            size_t idx = i * tile_shape_in::RowStride + j * tile_shape_in::ColStride;
            dst[idx] = static_cast<type>(src[idx]);
        }
}

template <typename tile_shape_out, typename tile_shape_in>
void TCast_NzLayout_Imp(typename tile_shape_out::TileDType dst, const typename tile_shape_in::TileDType src)
{
    using type = typename tile_shape_out::DType;
    for (uint16_t i = 0; i < tile_shape_in::ValidRow; ++i)
        for (uint16_t j = 0; j < tile_shape_in::ValidCol; ++j) {
            size_t sub_tile_i = i / tile_shape_in::InnerRows;
            size_t idx_i = i % tile_shape_in::InnerRows;
            size_t sub_tile_j = j / tile_shape_in::InnerCols;
            size_t idx_j = j % tile_shape_in::InnerCols;

            size_t idx = sub_tile_j * tile_shape_in::Rows * tile_shape_in::InnerCols +
                         sub_tile_i * tile_shape_in::InnerNumel + idx_i * tile_shape_in::InnerCols + idx_j;
            dst[idx] = static_cast<type>(src[idx]);
        }
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TCAST_Impl(tile_shape_out& dst, tile_shape_in& src)
{
    static_assert(
        tile_shape_in::Rows == tile_shape_out::Rows && tile_shape_in::Cols == tile_shape_out::Cols,
        "Error! Input shape != Output shape");
    static_assert(
        tile_shape_in::ValidRow != DYNAMIC && tile_shape_in::ValidCol != DYNAMIC &&
            tile_shape_out::ValidRow != DYNAMIC && tile_shape_out::ValidCol != DYNAMIC,
        "TODO: Support tile dynamic shape!");
    if constexpr (is_Nz_layout<tile_shape_in>::value && is_Nz_layout<tile_shape_out>::value) {
        TCast_NzLayout_Imp<tile_shape_out, tile_shape_in>(dst.data(), src.data());
    } else if constexpr (
        (tile_shape_in::isRowMajor && tile_shape_out::isRowMajor) ||
        (!tile_shape_in::isRowMajor && !tile_shape_out::isRowMajor)) {
        TCast_NoFractal_Imp<tile_shape_out, tile_shape_in>(dst.data(), src.data());
    } else {
        static_assert(tile_shape_in::isRowMajor && tile_shape_out::isRowMajor, "Storage layout type not supported");
    }
}

#endif
