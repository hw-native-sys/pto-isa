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

#ifndef PTO_LINX_VENDOR_TEXP_HPP
#define PTO_LINX_VENDOR_TEXP_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape>
void TExp_RowMajor_Imp(typename tile_shape::TileDType dst, const typename tile_shape::TileDType src)
{
    for (size_t i = 0; i < tile_shape::ValidRow; ++i)
        for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
            size_t index = i * tile_shape::RowStride + j;
            if constexpr (std::is_same<typename tile_shape::DType, __half>::value) {
                const float src_float = static_cast<float>(src[index]);
                const float exp_float = std::exp(src_float);
                dst[index] = static_cast<__half>(exp_float);
            } else {
                dst[index] = std::exp(src[index]);
            }
        }
}
template <typename tile_shape>
void TExp_ColMajor_Imp(typename tile_shape::TileDType dst, const typename tile_shape::TileDType src)
{
    for (size_t i = 0; i < tile_shape::ValidCol; ++i)
        for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
            size_t index = i * tile_shape::ColStride + j;
            if constexpr (std::is_same<typename tile_shape::DType, __half>::value) {
                const float src_float = static_cast<float>(src[index]);
                const float exp_float = std::exp(src_float);
                dst[index] = static_cast<__half>(exp_float);
            } else {
                dst[index] = std::exp(src[index]);
            }
        }
}
template <typename tile_shape>
void TExp2NzImp(typename tile_shape::TileDType dst, const typename tile_shape::TileDType src)
{
    static constexpr int inner_rows = tile_shape::InnerRows;
    static constexpr int inner_cols = tile_shape::InnerCols;
    for (size_t i = 0; i < tile_shape::ValidCol; ++i) {
        size_t sub_tile_i = i / inner_cols;
        size_t idx_i = i % inner_cols;
        for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
            size_t sub_tile_j = j / inner_rows;
            size_t idx_j = j % inner_rows;
            size_t index = sub_tile_i * tile_shape::Rows * inner_cols + sub_tile_j * tile_shape::InnerNumel +
                           idx_j * inner_cols + idx_i;
            if constexpr (std::is_same<typename tile_shape::DType, __half>::value) {
                const float src_float = static_cast<float>(src[index]);
                const float exp_float = std::exp(src_float);
                dst[index] = static_cast<__half>(exp_float);
            } else {
                dst[index] = std::exp(src[index]);
            }
        }
    }
}
template <is_tile_data_v tile_shape>
void TEXP_Impl(tile_shape& dst, tile_shape& src)
{
    static_assert(
        tile_shape::ValidRow != DYNAMIC && tile_shape::ValidCol != DYNAMIC, "TODO: Support tile dynamic shape!");
    if constexpr (is_Nz_layout<tile_shape>::value) {
        TExp2NzImp<tile_shape>(dst.data(), src.data());
    } else if (tile_shape::isBoxedLayout == false) {
        if constexpr (tile_shape::isRowMajor) {
            TExp_RowMajor_Imp<tile_shape>(dst.data(), src.data());
        } else {
            TExp_ColMajor_Imp<tile_shape>(dst.data(), src.data());
        }
    } else {
        static_assert(tile_shape::isBoxedLayout == false, "Storage type not supported");
    }
}

#endif
