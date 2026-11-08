#ifndef TSUBS_HPP
#define TSUBS_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape>
void TSubs_RowMajor_Imp(typename tile_shape::TileDType dst,
                        const typename tile_shape::TileDType src,
                        const typename tile_shape::DType s) {
  for (size_t i = 0; i < tile_shape::ValidRow; ++i)
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      dst[i * tile_shape::RowStride + j] =
          src[i * tile_shape::RowStride + j] - s;
    }
}

template <typename tile_shape>
void TSubs_ColMajor_Imp(typename tile_shape::TileDType dst,
                        const typename tile_shape::TileDType src,
                        const typename tile_shape::DType s) {
  for (size_t i = 0; i < tile_shape::ValidRow; ++i)
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      dst[j * tile_shape::ColStride + i] =
          src[j * tile_shape::ColStride + i] - s;
    }
}

template <typename tile_shape>
void TSubs_NzLayout_Imp(typename tile_shape::TileDType dst,
                        const typename tile_shape::TileDType src,
                        const typename tile_shape::DType s) {
  for (size_t i = 0; i < tile_shape::ValidRow; ++i)
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      size_t sub_tile_i = i / tile_shape::InnerRows;
      size_t idx_i = i % tile_shape::InnerRows;
      size_t sub_tile_j = j / tile_shape::InnerCols;
      size_t idx_j = j % tile_shape::InnerCols;

      size_t idx = sub_tile_j * tile_shape::Rows * tile_shape::InnerCols +
                     sub_tile_i * tile_shape::InnerNumel +
                     idx_i * tile_shape::InnerCols + idx_j;
      dst[idx] = src[idx] - s;
    }
}

template <is_tile_data_v tile_shape>
void TSUBS_Impl(tile_shape &dst, tile_shape &src, typename tile_shape::DType s) {
  static_assert(tile_shape::ValidRow != DYNAMIC && tile_shape::ValidCol != DYNAMIC,
              "TODO: Support tile dynamic shape!");
  if constexpr (is_Nz_layout<tile_shape>::value) {
    TSubs_NzLayout_Imp<tile_shape>(dst.data(), src.data(), s);
  } else if constexpr (tile_shape::isBoxedLayout == false) {
    if constexpr (tile_shape::isRowMajor) {
      TSubs_RowMajor_Imp<tile_shape>(dst.data(), src.data(), s);
    } else {
      TSubs_ColMajor_Imp<tile_shape>(dst.data(), src.data(), s);
    }
  } else {
    static_assert(tile_shape::isBoxedLayout == false,
                  "Storage layout type not supported");
  }
}

#endif
