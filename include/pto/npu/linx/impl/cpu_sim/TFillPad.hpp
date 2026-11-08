#ifndef TFILLPAD_HPP
#define TFILLPAD_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"
#include "pto/npu/linx/impl/jcore/constants.hpp"
using namespace pto;

template <typename tile_shape_out, typename tile_shape_in>
void TFillPad_Vec_RowMajor(typename tile_shape_out::TileDType dst,
                           const typename tile_shape_in::TileDType src) {
  for (size_t i = 0; i < tile_shape_out::Rows; ++i) {
    for (size_t j = 0; j < tile_shape_out::Cols; ++j) {
      size_t index = i * tile_shape_out::RowStride + j;

      if (i < tile_shape_in::ValidRow && j < tile_shape_in::ValidCol) {
        dst[index] = src[index];
      } else {
        dst[index] = 0;
      }
    }
  }
}

template <typename tile_shape_out, typename tile_shape_in>
void TFillPad_Vec_ColMajor(typename tile_shape_out::TileDType dst,
                           const typename tile_shape_in::TileDType src) {
  for (size_t i = 0; i < tile_shape_out::Rows; ++i) {
    for (size_t j = 0; j < tile_shape_out::Cols; ++j) {
      size_t index = j * tile_shape_out::ColStride + i;

      if (i < tile_shape_in::ValidRow && j < tile_shape_in::ValidCol) {
        dst[index] = src[index];
      } else {
        dst[index] = 0;
      }
    }
  }
}

template <typename tile_shape_out, typename tile_shape_in>
void TFillPad_Vec_RowMajor_Dynamic(tile_shape_out &dst,
                                   const tile_shape_in &src) {
  for (size_t i = 0; i < tile_shape_out::Rows; ++i) {
    for (size_t j = 0; j < tile_shape_out::Cols; ++j) {
      size_t index = i * tile_shape_out::RowStride + j;

      if (i < src.GetValidRow() && j < src.GetValidCol()) {
         dst.data()[index] = src.data()[index];
      } else {
         dst.data()[index] = 0;
      }
    }
  }
}

template <typename tile_shape_out, typename tile_shape_in>
void TFillPad_Vec_ColMajor_Dynamic(tile_shape_out &dst,
                                   const tile_shape_in &src) {
  for (size_t i = 0; i < tile_shape_out::Rows; ++i) {
    for (size_t j = 0; j < tile_shape_out::Cols; ++j) {
      size_t index = j * tile_shape_out::ColStride + i;

      if (i < src.GetValidRow() && j < src.GetValidCol()) {
        dst.data()[index] = src.data()[index];
      } else {
        dst.data()[index] = 0;
      }
    }
  }
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TFILLPAD_Impl(tile_shape_out &dst, tile_shape_in &src) {
  static_assert(!tile_shape_out::isBoxedLayout && !tile_shape_in::isBoxedLayout, "Not support Boxed Layout!");
  static_assert(tile_shape_out::Rows == tile_shape_in::Rows && tile_shape_out::Cols == tile_shape_in::Cols,
                "Dst and src must be same shape!");
  static_assert(tile_shape_out::PadVal == PadValue::Zero, "Only support pad zero!");
  static constexpr size_t dst_row = tile_shape_out::Rows;
  static constexpr size_t dst_col = tile_shape_out::Cols;

  if (tile_shape_out::ValidRow == DYNAMIC || tile_shape_out::ValidCol == DYNAMIC) { // dynamic
    if constexpr (tile_shape_out::isRowMajor) {
      static_assert(tile_shape_in::isRowMajor, "Dst Layout should be same as src.");
      TFillPad_Vec_RowMajor_Dynamic<tile_shape_out, tile_shape_in>(dst, src);
    } else {
      static_assert(!tile_shape_in::isRowMajor, "Dst Layout should be same as src.");
      TFillPad_Vec_ColMajor_Dynamic<tile_shape_out, tile_shape_in>(dst, src);
    }
  } else { // static
    if constexpr (tile_shape_out::isRowMajor) {
      static_assert(tile_shape_in::isRowMajor, "Dst Layout should be same as src.");
      TFillPad_Vec_RowMajor<tile_shape_out, tile_shape_in>(dst.data(), src.data());
    } else {
      static_assert(!tile_shape_in::isRowMajor, "Dst Layout should be same as src.");
      TFillPad_Vec_ColMajor<tile_shape_out, tile_shape_in>(dst.data(), src.data());
    }
  }
}

#endif
