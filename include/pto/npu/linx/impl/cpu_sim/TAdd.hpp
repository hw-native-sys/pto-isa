#ifndef TADD_HPP
#define TADD_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape>
void TAdd_RowMajor_Imp(typename tile_shape::TileDType dst,
                       const typename tile_shape::TileDType src0,
                       const typename tile_shape::TileDType src1) {
  for (size_t i = 0; i < tile_shape::ValidRow; ++i)
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      size_t idx = i * tile_shape::RowStride + j;
      dst[idx] = src0[idx] + src1[idx];
    }
}

template <typename tile_shape>
void TAdd_ColMajor_Imp(typename tile_shape::TileDType dst,
                       const typename tile_shape::TileDType src0,
                       const typename tile_shape::TileDType src1) {
  for (size_t i = 0; i < tile_shape::ValidCol; ++i)
    for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
      size_t idx = i * tile_shape::ColStride + j;
      dst[idx] = src0[idx] + src1[idx];
    }
}

template <typename tile_shape>
void TAdd_NzLayout_Imp(typename tile_shape::TileDType dst,
                       const typename tile_shape::TileDType src0,
                       const typename tile_shape::TileDType src1) {
  static constexpr int inner_rows = tile_shape::InnerRows;
  static constexpr int inner_cols = tile_shape::InnerCols;

  for (size_t i = 0; i < tile_shape::ValidCol; ++i) {
    size_t sub_tile_i = i / inner_cols;
    size_t idx_i = i % inner_cols;
    for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
      size_t sub_tile_j = j / inner_rows;
      size_t idx_j = j % inner_rows;
      size_t index = sub_tile_i * tile_shape::Rows * inner_cols +
          sub_tile_j * tile_shape::InnerNumel + idx_j * inner_cols + idx_i;
      dst[index] = src0[index] + src1[index];
    }
  }
}

template <is_tile_data_v tile_shape>
void TADD_Impl(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  static_assert(!tile_shape::isBoxedLayout && "Not Support boxed layout!");
  static_assert(tile_shape::ValidRow != DYNAMIC && tile_shape::ValidCol != DYNAMIC,
              "TODO: Support tile dynamic shape!");
  if constexpr (tile_shape::isRowMajor) {
    TAdd_RowMajor_Imp<tile_shape>(dst.data(), src0.data(), src1.data());
  } else {
    TAdd_ColMajor_Imp<tile_shape>(dst.data(), src0.data(), src1.data());
  }
}

#endif
