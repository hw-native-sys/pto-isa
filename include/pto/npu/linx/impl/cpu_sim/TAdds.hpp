#ifndef TADDS_HPP
#define TADDS_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape>
void TAdds_RowMajor_Imp(typename tile_shape::TileDType dst,
                        const typename tile_shape::TileDType src,
                        const typename tile_shape::DType s) {
  for (size_t i = 0; i < tile_shape::ValidRow; ++i)
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      size_t idx = i * tile_shape::RowStride + j;
      dst[idx] = src[idx] + s;
    }
}

template <typename tile_shape>
void TAdds_ColMajor_Imp(typename tile_shape::TileDType dst,
                        const typename tile_shape::TileDType src,
                        const typename tile_shape::DType s) {
  for (size_t i = 0; i < tile_shape::ValidCol; ++i)
    for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
      size_t idx = i * tile_shape::ColStride + j;
      dst[idx] = src[idx] + s;
    }
}

template <typename tile_shape>
void TAdds_NzLayout_Imp(typename tile_shape::TileDType dst,
                       const typename tile_shape::TileDType src,
                       const typename tile_shape::DType s) {
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
      dst[index] = src[index] + s;
    }
  }
}

template <is_tile_data_v tile_shape>
void TADDS_Impl(tile_shape &dst, tile_shape &src, typename tile_shape::DType s) {
  static_assert(tile_shape::ValidRow != DYNAMIC && tile_shape::ValidCol != DYNAMIC,
              "TODO: Support tile dynamic shape!");
  if constexpr (is_Nz_layout<tile_shape>::value) {
      TAdds_NzLayout_Imp<tile_shape>(dst.data(), src.data(), s);
  } else if constexpr (is_Zn_layout<tile_shape>::value) {
      static_assert(!is_Zn_layout<tile_shape>::value,
                    "Zn layout type not supported");
  } else if constexpr (tile_shape::isBoxedLayout == false) {
    if constexpr (tile_shape::isRowMajor) {
      TAdds_RowMajor_Imp<tile_shape>(dst.data(), src.data(), s);
    } else {
      TAdds_ColMajor_Imp<tile_shape>(dst.data(), src.data(), s);
    }
  } else {
    static_assert(tile_shape::isBoxedLayout == false,
                  "Storage type not supported");
  }
}

#endif
