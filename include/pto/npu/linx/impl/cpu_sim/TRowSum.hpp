#ifndef TROWSUM_HPP
#define TROWSUM_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape_out, typename tile_shape_in>
void TRowSum_RowMajor_Imp(typename tile_shape_out::TileDType dst,
                          const typename tile_shape_in::TileDType src) {
  for (size_t i = 0; i < tile_shape_in::ValidRow; ++i) {
    typename tile_shape_in::DType sum = 0;
    for (size_t j = 0; j < tile_shape_in::ValidCol; ++j) {
      sum = sum + src[i * tile_shape_in::RowStride + j];
    }
    dst[i * tile_shape_out::RowStride] = sum;
  }
}

template <typename tile_shape_out, typename tile_shape_in>
void TRowSum_ColMajor_Imp(typename tile_shape_out::TileDType dst,
                          const typename tile_shape_in::TileDType src) {
  for (size_t i = 0; i < tile_shape_in::ValidRow; ++i) {
    typename tile_shape_in::DType sum = 0;
    for (size_t j = 0; j < tile_shape_in::ValidCol; ++j) {
      sum = sum + src[j * tile_shape_in::ColStride + i];
    }
    dst[i * tile_shape_out::ColStride] = sum;
  }
}

template <typename tile_shape_out, typename tile_shape_in>
void TRowSum_NzLayout_Imp(typename tile_shape_out::TileDType dst,
                          const typename tile_shape_in::TileDType src) {
  for (size_t i = 0; i < tile_shape_in::ValidRow; ++i) {
    size_t sub_tile_i = i / tile_shape_in::InnerRows;
    size_t idx_i = i % tile_shape_in::InnerRows;
    typename tile_shape_in::DType sum = 0;
    for (size_t j = 0; j < tile_shape_in::ValidCol; ++j) {
      size_t sub_tile_j = j / tile_shape_in::InnerCols;
      size_t idx_j = j % tile_shape_in::InnerCols;
      sum = sum +
            src[sub_tile_j * tile_shape_in::Rows * tile_shape_in::InnerCols +
                sub_tile_i * tile_shape_in::InnerNumel +
                idx_i * tile_shape_in::InnerCols + idx_j];
    }

    if constexpr (tile_shape_out::isRowMajor) {
      dst[i * tile_shape_out::RowStride] = sum;
    } else {
      dst[i * tile_shape_out::ColStride] = sum;
    }
  }
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TROWSUM_Impl(tile_shape_out &dst, tile_shape_in &src) {
  if constexpr (is_Nz_layout<tile_shape_in>::value) {
    TRowSum_NzLayout_Imp<tile_shape_out, tile_shape_in>(dst.data(), src.data());
  } else if constexpr (tile_shape_in::isBoxedLayout == false) {
    if constexpr (tile_shape_out::isRowMajor) {
      TRowSum_RowMajor_Imp<tile_shape_out, tile_shape_in>(dst.data(),
                                                          src.data());
    } else {
      TRowSum_ColMajor_Imp<tile_shape_out, tile_shape_in>(dst.data(),
                                                          src.data());
    }
  }
}

#endif
