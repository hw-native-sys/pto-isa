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

#ifndef PTO_LINX_VENDOR_TPAD_HPP
#define PTO_LINX_VENDOR_TPAD_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"
#include <assert.h>
using namespace pto;

template <typename tile_shape_out, typename tile_shape_in, typename T>
void TPad_Vec_RowMajor(
    typename tile_shape_out::TileDType dst, const typename tile_shape_in::TileDType src, const T pad_value,
    const size_t up_pad, const size_t left_pad, const size_t down_pad, const size_t right_pad)
{
    size_t src_valid_row = tile_shape_in::ValidRow;
    size_t src_valid_col = tile_shape_in::ValidCol;
    size_t src_left_col_start = left_pad;
    size_t src_right_col_end = left_pad + src_valid_col;
    size_t src_up_row_start = up_pad;
    size_t src_down_row_end = up_pad + src_valid_row;
    size_t after_pad_row = up_pad + src_valid_row + down_pad;
    size_t after_pad_col = left_pad + src_valid_col + right_pad;

    for (size_t i = 0; i < after_pad_row; ++i)
        for (size_t j = 0; j < after_pad_col; ++j) {
            size_t index = i * tile_shape_out::RowStride + j;
            bool in_src_range = (j >= src_left_col_start) && (j < src_right_col_end) && (i >= src_up_row_start) &&
                                (i < src_down_row_end);
            size_t src_col = j - src_left_col_start;
            size_t src_row = i - src_up_row_start;
            size_t src_index = src_row * tile_shape_in::RowStride + src_col;
            dst[index] = (in_src_range == 1) ? src[src_index] : static_cast<typename tile_shape_out::DType>(pad_value);
        }
}

template <typename tile_shape_out, typename tile_shape_in, typename T>
void TPad_Vec_ColMajor(
    typename tile_shape_out::TileDType dst, const typename tile_shape_in::TileDType src, const T pad_value,
    const size_t up_pad, const size_t left_pad, const size_t down_pad, const size_t right_pad)
{
    size_t src_valid_row = tile_shape_in::ValidRow;
    size_t src_valid_col = tile_shape_in::ValidCol;
    size_t src_left_col_start = left_pad;
    size_t src_right_col_end = left_pad + src_valid_col;
    size_t src_up_row_start = up_pad;
    size_t src_down_row_end = up_pad + src_valid_row;
    size_t after_pad_row = up_pad + src_valid_row + down_pad;
    size_t after_pad_col = left_pad + src_valid_col + right_pad;

    for (size_t i = 0; i < after_pad_col; ++i)
        for (size_t j = 0; j < after_pad_row; ++j) {
            size_t index = i * tile_shape_out::ColStride + j;
            bool in_src_range = (i >= src_left_col_start) && (i < src_right_col_end) && (j >= src_up_row_start) &&
                                (j < src_down_row_end);
            size_t src_col = i - src_left_col_start;
            size_t src_row = j - src_up_row_start;
            size_t src_index = src_col * tile_shape_in::ColStride + src_row;
            dst[index] = (in_src_range == 1) ? src[src_index] : static_cast<typename tile_shape_out::DType>(pad_value);
        }
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in, typename T>
void TPAD_Impl(
    tile_shape_out& dst, const tile_shape_in& src, T pad_value, size_t up_pad, size_t left_pad, size_t down_pad,
    size_t right_pad)
{
    static_assert(!tile_shape_out::isBoxedLayout && !tile_shape_in::isBoxedLayout, "Not support Boxed Layout!");
    static_assert(
        tile_shape_out::Loc == Location::Vec && tile_shape_in::Loc == Location::Vec,
        "Only VEC tile type are supported");
    static_assert(
        tile_shape_out::ValidRow != DYNAMIC && tile_shape_out::ValidCol != DYNAMIC &&
            tile_shape_in::ValidRow != DYNAMIC && tile_shape_in::ValidCol != DYNAMIC,
        "TODO: Support tile dynamic shape!");
    static_assert(
        tile_shape_out::ValidRow >= tile_shape_in::ValidRow && tile_shape_out::ValidCol >= tile_shape_in::ValidCol,
        "Dst and src must be same shape!");

    size_t src_valid_row = src.GetValidRow();
    size_t src_valid_col = src.GetValidCol();
    size_t dst_valid_row = dst.GetValidRow();
    size_t dst_valid_col = dst.GetValidCol();
    size_t after_pad_row = up_pad + src_valid_row + down_pad;
    size_t after_pad_col = left_pad + src_valid_col + right_pad;
    assert(
        (after_pad_row <= dst_valid_row && after_pad_col <= dst_valid_col) &&
        "Padded rows exceed destination tensor rows");

    if constexpr (tile_shape_in::isRowMajor && tile_shape_out::isRowMajor) {
        TPad_Vec_RowMajor<tile_shape_out, tile_shape_in, T>(
            dst.data(), src.data(), pad_value, up_pad, left_pad, down_pad, right_pad);
    } else if constexpr (!tile_shape_in::isRowMajor && !tile_shape_out::isRowMajor) {
        TPad_Vec_ColMajor<tile_shape_out, tile_shape_in, T>(
            dst.data(), src.data(), pad_value, up_pad, left_pad, down_pad, right_pad);
    } else {
        static_assert(tile_shape_in::isRowMajor && tile_shape_out::isRowMajor, "Storage layout type not supported");
    }
}

#endif
