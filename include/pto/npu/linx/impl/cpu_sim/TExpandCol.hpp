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

#ifndef PTO_LINX_VENDOR_TEXPANDCOL_HPP
#define PTO_LINX_VENDOR_TEXPANDCOL_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape_out, typename tile_shape_in>
void TExpandCol_RowMajor_Imp(typename tile_shape_out::TileDType dst, const typename tile_shape_in::TileDType src)
{
    for (size_t i = 0; i < tile_shape_out::ValidRow; ++i)
        for (size_t j = 0; j < tile_shape_out::ValidCol; ++j) {
            size_t index = i * tile_shape_out::Cols + j;
            dst[index] = src[i * tile_shape_in::RowStride];
        }
}
template <typename tile_shape_out, typename tile_shape_in>
void TExpandCol_ColMajor_Imp(typename tile_shape_out::TileDType dst, const typename tile_shape_in::TileDType src)
{
    for (size_t i = 0; i < tile_shape_out::ValidCol; ++i)
        for (size_t j = 0; j < tile_shape_out::ValidRow; ++j) {
            size_t index = i * tile_shape_out::Rows + j;
            dst[index] = src[j * tile_shape_in::RowStride];
        }
}
template <typename tile_shape_out, typename tile_shape_in>
void TExpandCol2NzImp(typename tile_shape_out::TileDType dst, const typename tile_shape_in::TileDType src)
{
    static constexpr int inner_rows = tile_shape_out::InnerRows;
    static constexpr int inner_cols = tile_shape_out::InnerCols;
    for (size_t i = 0; i < tile_shape_out::ValidCol; ++i) {
        size_t sub_tile_i = i / inner_cols;
        size_t idx_i = i % inner_cols;
        for (size_t j = 0; j < tile_shape_out::ValidRow; ++j) {
            size_t sub_tile_j = j / inner_rows;
            size_t idx_j = j % inner_rows;
            size_t index = sub_tile_i * tile_shape_out::Rows * inner_cols + sub_tile_j * tile_shape_out::InnerNumel +
                           idx_j * inner_cols + idx_i;
            dst[index] = src[j * tile_shape_in::RowStride];
        }
    }
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TEXPANDCOL_Impl(tile_shape_out& dst, tile_shape_in& src)
{
    static_assert(
        tile_shape_in::ValidRow != DYNAMIC && tile_shape_in::ValidCol != DYNAMIC &&
            tile_shape_out::ValidRow != DYNAMIC && tile_shape_out::ValidCol != DYNAMIC,
        "TODO: Support tile dynamic shape!");
    static_assert(
        (tile_shape_out::Rows == tile_shape_in::Rows) && (tile_shape_out::ValidRow == tile_shape_in::ValidRow),
        "Error! Cude A:Rows != Cude B:Rows");
    static_assert(tile_shape_out::ValidRow == tile_shape_in::ValidRow, "Error! Cude A:ValidRow != Cude B:ValidRow");
    if constexpr (is_Nz_layout<tile_shape_out>::value) {
        TExpandCol2NzImp<tile_shape_out, tile_shape_in>(dst.data(), src.data());
    } else if constexpr (tile_shape_out::isBoxedLayout == false) {
        if constexpr (tile_shape_out::isRowMajor) {
            TExpandCol_RowMajor_Imp<tile_shape_out, tile_shape_in>(dst.data(), src.data());
        } else {
            TExpandCol_ColMajor_Imp<tile_shape_out, tile_shape_in>(dst.data(), src.data());
        }
    } else {
        static_assert(tile_shape_out::isBoxedLayout == false, "Storage type not supported");
    }
}

#endif
