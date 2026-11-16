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

#ifndef PTO_LINX_VENDOR_TSUB_HPP
#define PTO_LINX_VENDOR_TSUB_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape>
void Tsub_RowMajor_Imp(
    typename tile_shape::TileDType dst, const typename tile_shape::TileDType src0,
    const typename tile_shape::TileDType src1)
{
    for (size_t i = 0; i < tile_shape::ValidRow; ++i)
        for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
            dst[i * tile_shape::RowStride + j] =
                src0[i * tile_shape::RowStride + j] - src1[i * tile_shape::RowStride + j];
        }
}

template <typename tile_shape>
void Tsub_ColMajor_Imp(
    typename tile_shape::TileDType dst, const typename tile_shape::TileDType src0,
    const typename tile_shape::TileDType src1)
{
    for (size_t i = 0; i < tile_shape::ValidCol; ++i)
        for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
            dst[i * tile_shape::ColStride + j] =
                src0[i * tile_shape::ColStride + j] - src1[i * tile_shape::ColStride + j];
        }
}

template <typename tile_shape>
void Tsub_NzLayout_Imp(
    typename tile_shape::TileDType dst, const typename tile_shape::TileDType src0,
    const typename tile_shape::TileDType src1)
{
    for (size_t i = 0; i < tile_shape::ValidRow; ++i)
        for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
            size_t sub_tile_i = i / tile_shape::InnerRows;
            size_t idx_i = i % tile_shape::InnerRows;
            size_t sub_tile_j = j / tile_shape::InnerCols;
            size_t idx_j = j % tile_shape::InnerCols;

            size_t idx = sub_tile_j * tile_shape::Rows * tile_shape::InnerCols + sub_tile_i * tile_shape::InnerNumel +
                         idx_i * tile_shape::InnerCols + idx_j;
            dst[idx] = src0[idx] - src1[idx];
        }
}

template <is_tile_data_v tile_shape>
void TSUB_Impl(tile_shape& dst, tile_shape& src0, tile_shape& src1)
{
    static_assert(
        tile_shape::ValidRow != DYNAMIC && tile_shape::ValidCol != DYNAMIC, "TODO: Support tile dynamic shape!");
    if constexpr (is_Nz_layout<tile_shape>::value) {
        Tsub_NzLayout_Imp<tile_shape>(dst.data(), src0.data(), src1.data());
    } else if constexpr (tile_shape::isBoxedLayout == false) {
        if constexpr (tile_shape::isRowMajor) {
            Tsub_RowMajor_Imp<tile_shape>(dst.data(), src0.data(), src1.data());
        } else {
            Tsub_ColMajor_Imp<tile_shape>(dst.data(), src0.data(), src1.data());
        }
    } else {
        static_assert(tile_shape::isBoxedLayout == false, "Storage layout type not supported");
    }
}

#endif
