#ifndef TABS_HPP
#define TABS_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"
#include <cmath>
using namespace pto;

template <typename tile_shape>
void TAbs_RowMajor_Imp(typename tile_shape::TileDType dst,
                       const typename tile_shape::TileDType src) {
  for (size_t i = 0; i < tile_shape::ValidRow; ++i)
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      size_t idx = i * tile_shape::RowStride + j;
      dst[idx] = (src[idx] < 0) ? -src[idx] : src[idx];
    }
}

template <typename tile_shape>
void TAbs_ColMajor_Imp(typename tile_shape::TileDType dst,
                       const typename tile_shape::TileDType src) {
  for (size_t i = 0; i < tile_shape::ValidCol; ++i)
    for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
        size_t idx = i * tile_shape::ColStride + j;
      dst[idx] = (src[idx] < 0) ? -src[idx] : src[idx];
    }
}

template <typename tile_shape>
void TAbs_NzLayout_Imp(typename tile_shape::TileDType dst,
                       const typename tile_shape::TileDType src) {
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
      dst[index] = (src[index] < 0) ? -src[index] : src[index];
    }
  }
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TABS_Impl(tile_shape_out &dst, const tile_shape_in &src) {
  static_assert(std::is_same_v<typename tile_shape_out::DType,
                               typename tile_shape_in::DType>,
                "TABS source and destination dtypes must match");
  static_assert(tile_shape_out::Rows == tile_shape_in::Rows &&
                    tile_shape_out::Cols == tile_shape_in::Cols &&
                    tile_shape_out::BFractal == tile_shape_in::BFractal &&
                    tile_shape_out::SFractal == tile_shape_in::SFractal,
                "TABS source and destination descriptors must match");
  static_assert(tile_shape_in::ValidRow != DYNAMIC && tile_shape_in::ValidCol != DYNAMIC,
              "TODO: Support tile dynamic shape!");
  if constexpr (is_Nz_layout<tile_shape_in>::value) {
      TAbs_NzLayout_Imp<tile_shape_in>(dst.data(), src.data());
  } else if constexpr (is_Zn_layout<tile_shape_in>::value) {
      static_assert(!is_Zn_layout<tile_shape_in>::value,
                    "Zn layout type not supported");
  } else if constexpr (tile_shape_in::isBoxedLayout == false) {
    if constexpr (tile_shape_in::isRowMajor) {
      TAbs_RowMajor_Imp<tile_shape_in>(dst.data(), src.data());
    } else {
      TAbs_ColMajor_Imp<tile_shape_in>(dst.data(), src.data());
    }
  } else {
    static_assert(tile_shape_in::isBoxedLayout == false,
                  "Storage type not supported");
  }
}

#endif
