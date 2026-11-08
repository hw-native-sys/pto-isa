#ifndef TROWMAXEXPAND_HPP
#define TROWMAXEXPAND_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape>
void TRowMaxExpand_RowMajor_Imp(typename tile_shape::TileDType dst,
                                const typename tile_shape::TileDType src) {
  for (size_t i = 0; i < tile_shape::ValidRow; ++i) {
    typename tile_shape::DType max = src[i * tile_shape::RowStride + 0];
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      max = std::max(max, src[i * tile_shape::RowStride + j]);
    }
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      dst[i * tile_shape::RowStride + j] = max;
    }
  }
}

template <typename tile_shape>
void TRowMaxExpand_ColMajor_Imp(typename tile_shape::TileDType dst,
                                const typename tile_shape::TileDType src) {
  for (size_t i = 0; i < tile_shape::ValidRow; ++i) {
    typename tile_shape::DType max = src[i];
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      max = std::max(max, src[j * tile_shape::ColStride + i]);
    }
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      dst[j * tile_shape::ColStride + i] = max;
    }
  }
}

template <typename tile_shape>
void TRowMaxExpand_NzLayout_Imp(typename tile_shape::TileDType dst,
                                const typename tile_shape::TileDType src) {
  for (size_t i = 0; i < tile_shape::ValidRow; ++i) {
    size_t sub_tile_i = i / tile_shape::InnerRows;
    size_t idx_i = i % tile_shape::InnerRows;
    typename tile_shape::DType max = src[sub_tile_i * tile_shape::InnerNumel +
                                         idx_i * tile_shape::InnerCols];
    for (size_t j = 1; j < tile_shape::ValidCol; ++j) {
      size_t sub_tile_j = j / tile_shape::InnerCols;
      size_t idx_j = j % tile_shape::InnerCols;
      max = std::max(
          max, src[sub_tile_j * tile_shape::Rows * tile_shape::InnerCols +
                   sub_tile_i * tile_shape::InnerNumel +
                   idx_i * tile_shape::InnerCols + idx_j]);
    }
    for (size_t k = 0; k < tile_shape::ValidCol ; ++k) {
      size_t sub_tile_k = k / tile_shape::InnerCols;
      size_t idx_k = k % tile_shape::InnerCols;
      dst[sub_tile_k * tile_shape::Rows * tile_shape::InnerCols +
          sub_tile_i * tile_shape::InnerNumel +
          idx_i * tile_shape::InnerCols + idx_k] = max;
    }
  }
}
// ROWMAX + EXPAND
template <is_tile_data_v tile_shape>
void TROWMAXEXPAND_Impl(tile_shape &dst, tile_shape &src) {
  static_assert(tile_shape::ValidRow != DYNAMIC && tile_shape::ValidCol != DYNAMIC,
              "TODO: Support tile dynamic shape!");
  if constexpr (is_Nz_layout<tile_shape>::value) {
    TRowMaxExpand_NzLayout_Imp<tile_shape>(dst.data(), src.data());
  } else if constexpr (tile_shape::isBoxedLayout == false) {
    if constexpr (tile_shape::isRowMajor) {
      TRowMaxExpand_RowMajor_Imp<tile_shape>(dst.data(), src.data());
    } else {
      TRowMaxExpand_ColMajor_Imp<tile_shape>(dst.data(), src.data());
    }
  } else {
    static_assert(tile_shape::isBoxedLayout == false,
                  "Storage layout type not supported");
  }
}

#endif
