#ifndef TSELECT_HPP
#define TSELECT_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape, typename tile_shape_index>
void TSelect_RowMajor_Imp(typename tile_shape::TileDType dst,
                          const typename tile_shape_index::TileDType cond,
                          const typename tile_shape::TileDType src0,
                          const typename tile_shape::TileDType src1) {
  for (size_t i = 0; i < tile_shape::ValidRow; ++i)
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      size_t index = i * tile_shape::RowStride + j;
      dst[index] = (cond[index] == 1) ? src0[index] : src1[index];
    }
}

template <typename tile_shape, typename tile_shape_index>
void TSelect_ColMajor_Imp(typename tile_shape::TileDType dst,
                          const typename tile_shape_index::TileDType cond,
                          const typename tile_shape::TileDType src0,
                          const typename tile_shape::TileDType src1) {
  for (size_t i = 0; i < tile_shape::ValidCol; ++i)
    for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
      size_t index = i * tile_shape::ColStride + j;
      dst[index] = (cond[index] == 1) ? src0[index] : src1[index];
    }
}

template <typename tile_shape, typename tile_shape_index>
void TSelect_NzLayout_Imp(typename tile_shape::TileDType dst,
                          const typename tile_shape_index::TileDType cond,
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
                     sub_tile_j * tile_shape::InnerNumel + idx_j * inner_cols +
                     idx_i;
      dst[index] = (cond[index] == 1) ? src0[index] : src1[index];
    }
  }
}

template <is_tile_data_v tile_shape, is_tile_data_v tile_shape_index>
void TSELECT_Impl(tile_shape &dst, tile_shape_index &cond, tile_shape &src0,
                  tile_shape &src1) {
  static_assert(tile_shape_index::Cols == tile_shape::Cols,
                "Error! cond: Columns != src: Columns");
  static_assert(tile_shape_index::Rows == tile_shape::Rows,
                "Error! cond: Rows != src: Rows");
  static_assert(tile_shape::ValidRow != DYNAMIC && tile_shape::ValidCol != DYNAMIC &&
                tile_shape_index::ValidRow != DYNAMIC && tile_shape_index::ValidCol != DYNAMIC,
              "TODO: Support tile dynamic shape!");
  if constexpr (is_Nz_layout<tile_shape>::value &&
                is_Nz_layout<tile_shape_index>::value) {
    TSelect_NzLayout_Imp<tile_shape, tile_shape_index>(
        dst.data(), cond.data(), src0.data(), src1.data());
  } else if constexpr (is_Zn_layout<tile_shape>::value &&
                       is_Zn_layout<tile_shape_index>::value) {
    static_assert(!(is_Zn_layout<tile_shape>::value &&
                    is_Zn_layout<tile_shape_index>::value),
                  "Zn layout type not supported");
  } else if constexpr (tile_shape::isBoxedLayout == false &&
                       tile_shape_index::isBoxedLayout == false) {
    if constexpr (tile_shape::isRowMajor &&
                  tile_shape_index::isRowMajor) {
      TSelect_RowMajor_Imp<tile_shape, tile_shape_index>(
          dst.data(), cond.data(), src0.data(), src1.data());
    } else {
      TSelect_ColMajor_Imp<tile_shape, tile_shape_index>(
          dst.data(), cond.data(), src0.data(), src1.data());
    }
  } else {
    static_assert(tile_shape::isBoxedLayout == false &&
                      tile_shape_index::isBoxedLayout == false,
                  "Storage type not supported");
  }
}

#endif
