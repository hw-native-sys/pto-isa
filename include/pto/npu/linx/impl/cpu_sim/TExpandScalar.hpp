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

#ifndef PTO_LINX_VENDOR_TEXPANDSCALAR_HPP
#define PTO_LINX_VENDOR_TEXPANDSCALAR_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape>
void ExpandScalarImpl_RowMajor(typename tile_shape::TileDType dst, const typename tile_shape::DType val)
{
    for (size_t i = 0; i < tile_shape::ValidRow; ++i)
        for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
            size_t index = i * tile_shape::RowStride + j;
            dst[index] = val;
        }
}
template <typename tile_shape>
void ExpandScalarImpl_ColMajor(typename tile_shape::TileDType dst, const typename tile_shape::DType val)
{
    for (size_t i = 0; i < tile_shape::ValidCol; ++i)
        for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
            size_t index = i * tile_shape::ColStride + j;
            dst[index] = val;
        }
}
template <typename tile_shape>
void ExpandScalar2NzImpl(typename tile_shape::TileDType dst, const typename tile_shape::DType val)
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
            dst[index] = val;
        }
    }
}
template <typename tile_shape>
void ExpandScalar2ZnImpl(typename tile_shape::TileDType dst, const typename tile_shape::DType val)
{
    static constexpr int inner_rows = tile_shape::InnerRows;
    static constexpr int inner_cols = tile_shape::InnerCols;
    for (uint16_t i = 0; i < tile_shape::ValidRow; ++i) {
        unsigned sub_tile_i = i / inner_rows;
        unsigned idx_i = i % inner_rows;
        for (uint16_t j = 0; j < tile_shape::ValidCol; ++j) {
            unsigned sub_tile_j = j / inner_cols;
            unsigned idx_j = j % inner_cols;
            unsigned index = sub_tile_i * tile_shape::Cols * inner_rows + sub_tile_j * tile_shape::InnerNumel +
                             idx_j * inner_rows + idx_i;
            dst[index] = val;
        }
    }
}

template <typename tile_shape>
void ExpandScalarImpl_RowMajor_Dynamic(tile_shape& dst, const typename tile_shape::DType val)
{
    for (size_t i = 0; i < dst.GetValidRow(); ++i)
        for (size_t j = 0; j < dst.GetValidCol(); ++j) {
            size_t index = i * tile_shape::RowStride + j;
            dst.data()[index] = val;
        }
}

template <typename tile_shape>
void ExpandScalarImpl_ColMajor_Dynamic(tile_shape& dst, const typename tile_shape::DType val)
{
    for (size_t i = 0; i < dst.GetValidCol(); ++i)
        for (size_t j = 0; j < dst.GetValidRow(); ++j) {
            size_t index = i * tile_shape::ColStride + j;
            dst.data()[index] = val;
        }
}

template <typename tile_shape>
void ExpandScalar2NzImpl_Dynamic(tile_shape& dst, const typename tile_shape::DType val)
{
    static constexpr int inner_rows = tile_shape::InnerRows;
    static constexpr int inner_cols = tile_shape::InnerCols;
    for (size_t i = 0; i < dst.GetValidCol(); ++i) {
        size_t sub_tile_i = i / inner_cols;
        size_t idx_i = i % inner_cols;
        for (size_t j = 0; j < dst.GetValidRow(); ++j) {
            size_t sub_tile_j = j / inner_rows;
            size_t idx_j = j % inner_rows;
            size_t index = sub_tile_i * tile_shape::Rows * inner_cols + sub_tile_j * tile_shape::InnerNumel +
                           idx_j * inner_cols + idx_i;
            dst.data()[index] = val;
        }
    }
}

template <typename tile_shape>
void ExpandScalar2ZnImpl_Dynamic(tile_shape& dst, const typename tile_shape::DType val)
{
    static constexpr int inner_rows = tile_shape::InnerRows;
    static constexpr int inner_cols = tile_shape::InnerCols;
    for (uint16_t i = 0; i < dst.GetValidRow(); ++i) {
        unsigned sub_tile_i = i / inner_rows;
        unsigned idx_i = i % inner_rows;
        for (uint16_t j = 0; j < dst.GetValidCol(); ++j) {
            unsigned sub_tile_j = j / inner_cols;
            unsigned idx_j = j % inner_cols;
            unsigned index = sub_tile_i * tile_shape::Cols * inner_rows + sub_tile_j * tile_shape::InnerNumel +
                             idx_j * inner_rows + idx_i;
            dst.data()[index] = val;
        }
    }
}

template <is_tile_data_v tile_shape>
void TEXPANDSCALAR_Impl(tile_shape& dst, typename tile_shape::DType s)
{
    if constexpr (tile_shape::ValidRow == DYNAMIC || tile_shape::ValidCol == DYNAMIC) { // dynamic
        if constexpr (is_Nz_layout<tile_shape>::value) {
            ExpandScalar2NzImpl_Dynamic<tile_shape>(dst, s);
        } else if constexpr (is_Zn_layout<tile_shape>::value) {
            ExpandScalar2ZnImpl_Dynamic<tile_shape>(dst, s);
        } else if constexpr (tile_shape::isBoxedLayout == false) {
            if constexpr (tile_shape::isRowMajor) {
                ExpandScalarImpl_RowMajor_Dynamic<tile_shape>(dst, s);
            } else if constexpr (!tile_shape::isRowMajor) {
                ExpandScalarImpl_ColMajor_Dynamic<tile_shape>(dst, s);
            } else {
                static_assert("Storage layout type not supported!");
            }
        } else {
            static_assert(tile_shape::isBoxedLayout == false, "Storage type not supported");
        }
    } else { // static
        if constexpr (is_Nz_layout<tile_shape>::value) {
            ExpandScalar2NzImpl<tile_shape>(dst.data(), s);
        } else if constexpr (is_Zn_layout<tile_shape>::value) {
            ExpandScalar2ZnImpl<tile_shape>(dst.data(), s);
        } else if constexpr (tile_shape::isBoxedLayout == false) {
            if constexpr (tile_shape::isRowMajor) {
                ExpandScalarImpl_RowMajor<tile_shape>(dst.data(), s);
            } else if constexpr (!tile_shape::isRowMajor) {
                ExpandScalarImpl_ColMajor<tile_shape>(dst.data(), s);
            } else {
                static_assert("Storage layout type not supported!");
            }
        } else {
            static_assert(tile_shape::isBoxedLayout == false, "Storage type not supported");
        }
    }
}

#endif
