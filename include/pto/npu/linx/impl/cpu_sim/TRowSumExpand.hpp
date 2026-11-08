#ifndef TROWSUMEXPAND_HPP
#define TROWSUMEXPAND_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape>
void TRowSumExpand_RowMajor_Imp(typename tile_shape::TileDType dst,
                                const typename tile_shape::TileDType src) {
  for (size_t i = 0; i < tile_shape::ValidRow; ++i) {
    typename tile_shape::DType sum = 0;
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      sum = sum + src[i * tile_shape::RowStride + j];
    }
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      dst[i * tile_shape::RowStride + j] = sum;
    }
  }
}

template <typename tile_shape>
void TRowSumExpand_ColMajor_Imp(typename tile_shape::TileDType dst,
                                const typename tile_shape::TileDType src) {
  for (size_t i = 0; i < tile_shape::ValidRow; ++i) {
    typename tile_shape::DType sum = 0;
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      sum = sum + src[j * tile_shape::ColStride + i];
    }
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      dst[j * tile_shape::ColStride + i] = sum;
    }
  }
}

template <typename tile_shape>
void TRowSumExpand_NzLayout_Imp(typename tile_shape::TileDType dst,
                                const typename tile_shape::TileDType src) {
  for (size_t i = 0; i < tile_shape::ValidRow; ++i) {
    size_t sub_tile_i = i / tile_shape::InnerRows;
    size_t idx_i = i % tile_shape::InnerRows;
    typename tile_shape::DType sum = 0;
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      size_t sub_tile_j = j / tile_shape::InnerCols;
      size_t idx_j = j % tile_shape::InnerCols;
      sum = sum + src[sub_tile_j * tile_shape::Rows * tile_shape::InnerCols +
                      sub_tile_i * tile_shape::InnerNumel +
                      idx_i * tile_shape::InnerCols + idx_j];
    }
    for (size_t k = 0; k < tile_shape::ValidCol ; ++k) {
      size_t sub_tile_k = k / tile_shape::InnerCols;
      size_t idx_k = k % tile_shape::InnerCols;
      dst[sub_tile_k * tile_shape::Rows * tile_shape::InnerCols +
          sub_tile_i * tile_shape::InnerNumel +
          idx_i * tile_shape::InnerCols + idx_k] = sum;
    }
  }
}
// ROWSUN + EXPAND
template <is_tile_data_v tile_shape>
void TROWSUMEXPAND_Impl(tile_shape &dst, tile_shape &src) {
  if constexpr (is_Nz_layout<tile_shape>::value) {
    TRowSumExpand_NzLayout_Imp<tile_shape>(dst.data(), src.data());
  } else if constexpr (tile_shape::isBoxedLayout == false) {
    if constexpr (tile_shape::isRowMajor) {
      TRowSumExpand_RowMajor_Imp<tile_shape>(dst.data(), src.data());
    } else {
      TRowSumExpand_ColMajor_Imp<tile_shape>(dst.data(), src.data());
    }
  }
}

#endif
