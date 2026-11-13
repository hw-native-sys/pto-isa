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

#ifndef PTO_LINX_VENDOR_TSTORE_BACKEND_HPP
#define PTO_LINX_VENDOR_TSTORE_BACKEND_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename gm_shape, typename tile_shape>
void CopyOut2NzImpl1D(typename gm_shape::DType* dst, const typename tile_shape::TileDType src)
{
    static constexpr int inner_rows = tile_shape::InnerRows;
    static constexpr int inner_cols = tile_shape::InnerCols;

    for (size_t i = 0; i < tile_shape::ValidCol; ++i) {
        size_t sub_tile_i = i / inner_cols;
        size_t idx_i = i % inner_cols;
        for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
            size_t sub_tile_j = j / inner_rows;
            size_t idx_j = j % inner_rows;
            // sub_tile_j and idx_j can be replaced with a constant after loop
            // unrolling
            const typename tile_shape::DType* tile_ptr = src;
            size_t idx_gm = gm_shape::RowStride * j + i;
            size_t idx_tile = sub_tile_i * tile_shape::Rows * inner_cols + sub_tile_j * tile_shape::InnerNumel +
                              idx_j * inner_cols + idx_i;
            dst[idx_gm] = tile_ptr[idx_tile];
        }
    }
}

template <typename gm_shape, typename tile_shape>
void TStore_ColMajor_Impl(typename gm_shape::DType* dst, typename tile_shape::TileDType src)
{
    for (size_t i = 0; i < tile_shape::ValidCol; ++i)
        for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
            size_t idx_gm = i * gm_shape::ColStride + j;
            size_t idx_tile = i * tile_shape::ColStride + j;
            dst[idx_gm] = src[idx_tile];
        }
}

template <typename gm_shape, typename tile_shape>
void TStore_RowMajor_Impl(typename gm_shape::DType* dst, typename tile_shape::TileDType src)
{
    for (size_t i = 0; i < tile_shape::ValidRow; ++i)
        for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
            size_t idx_gm = i * gm_shape::RowStride + j;
            size_t idx_tile = i * tile_shape::RowStride + j;
            dst[idx_gm] = src[idx_tile];
        }
}

template <typename gm_shape, typename tile_shape>
void CopyOut2NzImpl1D_Dynamic(gm_shape& dst, const tile_shape& src)
{
    static constexpr int inner_rows = tile_shape::InnerRows;
    static constexpr int inner_cols = tile_shape::InnerCols;

    for (size_t i = 0; i < src.GetValidCol(); ++i) {
        size_t sub_tile_i = i / inner_cols;
        size_t idx_i = i % inner_cols;
        for (size_t j = 0; j < src.GetValidRow(); ++j) {
            size_t sub_tile_j = j / inner_rows;
            size_t idx_j = j % inner_rows;
            // sub_tile_j and idx_j can be replaced with a constant after loop
            // unrolling
            const typename tile_shape::DType* tile_ptr = src.data();
            size_t idx_gm = dst.GetStride(3) * j + i;
            size_t idx_tile = sub_tile_i * tile_shape::Rows * inner_cols + sub_tile_j * tile_shape::InnerNumel +
                              idx_j * inner_cols + idx_i;
            dst.data()[idx_gm] = tile_ptr[idx_tile];
        }
    }
}

template <typename gm_shape, typename tile_shape>
void TStore_ColMajor_Impl_Dynamic(gm_shape& dst, tile_shape& src)
{
    for (size_t i = 0; i < src.GetValidCol(); ++i) {
        for (size_t j = 0; j < src.GetValidRow(); ++j) {
            size_t idx_gm = i * dst.GetStride(4) + j;
            size_t idx_tile = i * tile_shape::ColStride + j;
            dst.data()[idx_gm] = src.data()[idx_tile];
        }
    }
}

template <typename gm_shape, typename tile_shape>
void TStore_RowMajor_Impl_Dynamic(gm_shape& dst, tile_shape& src)
{
    for (size_t i = 0; i < src.GetValidRow(); ++i) {
        for (size_t j = 0; j < src.GetValidCol(); ++j) {
            size_t idx_gm = i * dst.GetStride(3) + j;
            size_t idx_tile = i * tile_shape::RowStride + j;
            dst.data()[idx_gm] = src.data()[idx_tile];
        }
    }
}

template <is_global_data_v gm_shape, is_tile_data_v tile_shape>
void TSTORE_Impl(gm_shape& dst, tile_shape& src)
{
    if (tile_shape::ValidRow == DYNAMIC || tile_shape::ValidCol == DYNAMIC) { // dynamic
        if constexpr (is_Nz_layout<tile_shape>::value) {
            if constexpr (gm_shape::isRowMajor) {
                CopyOut2NzImpl1D_Dynamic<gm_shape, tile_shape>(dst, src);
            } else {
                static_assert(gm_shape::isRowMajor, "Storage layout type not supported, gm should rowmajor");
            }
        } else if constexpr (tile_shape::isBoxedLayout == false) {
            if constexpr (gm_shape::isRowMajor) {
                TStore_RowMajor_Impl_Dynamic<gm_shape, tile_shape>(dst, src);
            } else {
                TStore_ColMajor_Impl_Dynamic<gm_shape, tile_shape>(dst, src);
            }
        } else {
            static_assert(tile_shape::isBoxedLayout == false, "Data type not supported");
        }
    } else { // static
        if constexpr (is_Nz_layout<tile_shape>::value) {
            if constexpr (gm_shape::isRowMajor) {
                CopyOut2NzImpl1D<gm_shape, tile_shape>(dst.data(), src.data());
            } else {
                static_assert(gm_shape::isRowMajor, "Storage layout type not supported, gm should rowmajor");
            }
        } else if constexpr (tile_shape::isBoxedLayout == false) {
            if constexpr (gm_shape::isRowMajor) {
                TStore_RowMajor_Impl<gm_shape, tile_shape>(dst.data(), src.data());
            } else {
                TStore_ColMajor_Impl<gm_shape, tile_shape>(dst.data(), src.data());
            }
        } else {
            static_assert(tile_shape::isBoxedLayout == false, "Data type not supported");
        }
    }
}

#endif
