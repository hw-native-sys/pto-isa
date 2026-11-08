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

#ifndef PTO_LINX_VENDOR_MATMUL_HPP
#define PTO_LINX_VENDOR_MATMUL_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape_A, typename tile_shape_B, typename tile_shape_C>
void MatMul_RowMajor_Imp(
    typename tile_shape_C::TileDType dst, const typename tile_shape_A::TileDType src0,
    const typename tile_shape_B::TileDType src1, const size_t K)
{
    for (size_t i = 0; i < tile_shape_C::ValidRow; ++i) {
        for (size_t j = 0; j < tile_shape_C::ValidCol; ++j) {
            typename tile_shape_C::DType mul_acc = 0;
            for (size_t k = 0; k < K; ++k) {
                size_t idx_0 = i * K + k;
                size_t idx_1 = k * tile_shape_B::RowStride + j;
                mul_acc = mul_acc + src0[idx_0] * src1[idx_1];
            }
            size_t idx = i * tile_shape_C::RowStride + j;
            dst[idx] = mul_acc;
        }
    }
}

template <typename tile_shape_A, typename tile_shape_B, typename tile_shape_C>
void MatMul_ColMajor_Imp(
    typename tile_shape_C::TileDType dst, const typename tile_shape_A::TileDType src0,
    const typename tile_shape_B::TileDType src1, const size_t K)
{
    for (size_t i = 0; i < tile_shape_C::ValidRow; ++i) {
        for (size_t j = 0; j < tile_shape_C::ValidCol; ++j) {
            typename tile_shape_C::DType mul_acc = 0;
            for (size_t k = 0; k < K; ++k) {
                size_t idx_0 = k * tile_shape_A::ColStride + i;
                size_t idx_1 = j * K + k;
                mul_acc = mul_acc + src0[idx_0] * src1[idx_1];
            }
            size_t idx = j * tile_shape_C::ColStride + i;
            dst[idx] = mul_acc;
        }
    }
}

template <typename tile_shape_A, typename tile_shape_B, typename tile_shape_C>
void Matmul_Fractal(
    size_t M, size_t N, size_t K, int anonymous, typename tile_shape_C::TileDType dst,
    const typename tile_shape_A::TileDType src0, const typename tile_shape_B::TileDType src1)
{
    static constexpr int blk_rows_a = tile_shape_A::InnerRows;
    static constexpr int blk_cols_a = tile_shape_A::InnerCols;

    static constexpr int blk_rows_b = tile_shape_B::InnerRows;
    static constexpr int blk_cols_b = tile_shape_B::InnerCols;

    static constexpr int blk_size = tile_shape_A::InnerNumel;

    static constexpr int blk_rows_c = tile_shape_C::InnerRows;
    static constexpr int blk_cols_c = tile_shape_C::InnerCols;
    static constexpr int blk_size_c = tile_shape_C::InnerNumel;

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            typename tile_shape_C::DType mul_acc = 0;
            for (size_t k = 0; k < K; ++k) {
                size_t src0_idx = (k / blk_cols_a) * tile_shape_A::Rows * blk_cols_a + (i / blk_rows_a) * blk_size +
                                  (i % blk_rows_a) * blk_cols_a + k % blk_cols_a;
                size_t src1_idx = (k / blk_rows_b) * tile_shape_B::Cols * blk_rows_b + (j / blk_cols_b) * blk_size +
                                  (j % blk_cols_b) * blk_rows_b + k % blk_rows_b;
                mul_acc = mul_acc + src0[src0_idx] * src1[src1_idx];
            }
            size_t dst_idx = (j / blk_cols_c) * tile_shape_C::Rows * blk_cols_c + (i / blk_rows_c) * blk_size_c +
                             (i % blk_rows_c) * blk_cols_c + j % blk_cols_c;
            dst[dst_idx] = mul_acc;
        }
    }
}

// Matrix Multiply: C[MxN] = A[M×K] x B[KxN]
template <is_tile_data_v tile_shape_A, is_tile_data_v tile_shape_B, is_tile_data_v tile_shape_C>
void MATMUL_Impl(tile_shape_C& dst, tile_shape_A& src0, tile_shape_B& src1)
{
    static_assert(tile_shape_A::ValidCol == tile_shape_B::ValidRow, "Error! Cude A:Columns != Cude B:Rows");

    static constexpr size_t M = tile_shape_C::ValidRow;
    static constexpr size_t N = tile_shape_C::ValidCol;
    static constexpr size_t K = tile_shape_A::ValidCol;
    static_assert(M != DYNAMIC && N != DYNAMIC && K != DYNAMIC, "TODO: Support tile dynamic shape!");
    if constexpr (tile_shape_A::isRowMajor && tile_shape_B::isRowMajor) {
        MatMul_RowMajor_Imp<tile_shape_A, tile_shape_B, tile_shape_C>(dst.data(), src0.data(), src1.data(), K);
    }
    if constexpr (!tile_shape_A::isRowMajor && !tile_shape_B::isRowMajor) {
        MatMul_ColMajor_Imp<tile_shape_A, tile_shape_B, tile_shape_C>(dst.data(), src0.data(), src1.data(), K);
    } else if constexpr (is_Nz_layout<tile_shape_A>::value && is_Zn_layout<tile_shape_B>::value) {
        static_assert(
            tile_shape_C::SFractalSize == 1024 && is_Nz_layout<tile_shape_C>::value,
            "Error! Cude C:FractalSize != 1024 or not Nz_layout");
        Matmul_Fractal<tile_shape_A, tile_shape_B, tile_shape_C>(M, N, K, 1, dst.data(), src0.data(), src1.data());
    } else {
        static_assert(tile_shape_A::isRowMajor && tile_shape_B::isRowMajor, "Storage layout type not supported");
    }
}

#endif
