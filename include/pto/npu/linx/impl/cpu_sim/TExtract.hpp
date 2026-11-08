#ifndef TEXTRACT_HPP
#define TEXTRACT_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape_out, typename tile_shape_in>
void TExtract_RowMajor_Imp(typename tile_shape_out::TileDType dst,
                           const typename tile_shape_in::TileDType src,
                           const uint16_t offset_row,
                           const uint16_t offset_col) {
  for (uint16_t i = 0; i < tile_shape_out::ValidRow; ++i)
    for (uint16_t j = 0; j < tile_shape_out::ValidCol; ++j) {
      dst[i * tile_shape_out::RowStride + j] =
          src[(offset_row + i) * tile_shape_in::RowStride + (offset_col + j)];
    }
}

template <typename tile_shape_out, typename tile_shape_in>
void TExtract_ColMajor_Imp(typename tile_shape_out::TileDType dst,
                           const typename tile_shape_in::TileDType src,
                           const uint16_t offset_row,
                           const uint16_t offset_col) {
  for (uint16_t i = 0; i < tile_shape_out::ValidRow; ++i)
    for (uint16_t j = 0; j < tile_shape_out::ValidCol; ++j) {
      dst[j * tile_shape_out::ColStride + i] =
          src[(offset_col + j) * tile_shape_in::ColStride + (offset_row + i)];
    }
}

template <typename tile_shape_out, typename tile_shape_in>
void TExtract_NzLayout_Imp(typename tile_shape_out::TileDType dst,
                           const typename tile_shape_in::TileDType src,
                           const uint16_t offest_i, const uint16_t offest_j) {

  for (uint16_t i = 0; i < tile_shape_out::ValidRow; ++i)
    for (uint16_t j = 0; j < tile_shape_out::ValidCol; ++j) {
      uint16_t sub_tile_i = i / tile_shape_out::InnerRows;
      uint16_t idx_i = i % tile_shape_out::InnerRows;
      uint16_t sub_tile_j = j / tile_shape_out::InnerCols;
      uint16_t idx_j = j % tile_shape_out::InnerCols;
      uint16_t idx_dst =
          sub_tile_j * tile_shape_out::Rows * tile_shape_out::InnerCols +
          sub_tile_i * tile_shape_out::InnerNumel +
          idx_i * tile_shape_out::InnerCols + idx_j;

      uint16_t sub_tile_i_src = (i + offest_i) / tile_shape_in::InnerRows;
      uint16_t idx_i_src = (i + offest_i) % tile_shape_in::InnerRows;
      uint16_t sub_tile_j_src = (j + offest_j) / tile_shape_in::InnerCols;
      uint16_t idx_j_src = (j + offest_j) % tile_shape_in::InnerCols;
      uint16_t idx_src =
          sub_tile_j_src * tile_shape_in::Rows * tile_shape_in::InnerCols +
          sub_tile_i_src * tile_shape_in::InnerNumel +
          idx_i_src * tile_shape_in::InnerCols + idx_j_src;

      dst[idx_dst] = src[idx_src];
    }
}

template <typename tile_shape_out, typename tile_shape_in>
void TExtract_RowMajor_Imp_Dynamic(tile_shape_out &dst,
                           const tile_shape_in &src,
                           const uint16_t offset_row,
                           const uint16_t offset_col) {
  for (uint16_t i = 0; i < dst.GetValidRow(); ++i) {
    for (uint16_t j = 0; j < dst.GetValidCol(); ++j) {
      dst.data()[i * tile_shape_out::RowStride + j] =
          src.data()[(offset_row + i) * tile_shape_in::RowStride + (offset_col + j)];
    }
  }
}

template <typename tile_shape_out, typename tile_shape_in>
void TExtract_ColMajor_Imp_Dynamic(tile_shape_out &dst,
                           const tile_shape_in &src,
                           const uint16_t offset_row,
                           const uint16_t offset_col) {
  for (uint16_t i = 0; i < dst.GetValidRow(); ++i) {
    for (uint16_t j = 0; j < dst.GetValidCol(); ++j) {
      dst.data()[j * tile_shape_out::ColStride + i] =
          src.data()[(offset_col + j) * tile_shape_in::ColStride + (offset_row + i)];
    }
  }
}

template <typename tile_shape_out, typename tile_shape_in>
void TExtract_NzLayout_Imp_Dynamic(tile_shape_out &dst,
                           const tile_shape_in &src,
                           const uint16_t offest_i, const uint16_t offest_j) {
  for (uint16_t i = 0; i < dst.GetValidRow(); ++i) {
    for (uint16_t j = 0; j < dst.GetValidCol(); ++j) {
      uint16_t sub_tile_i = i / tile_shape_out::InnerRows;
      uint16_t idx_i = i % tile_shape_out::InnerRows;
      uint16_t sub_tile_j = j / tile_shape_out::InnerCols;
      uint16_t idx_j = j % tile_shape_out::InnerCols;
      uint16_t idx_dst =
          sub_tile_j * tile_shape_out::Rows * tile_shape_out::InnerCols +
          sub_tile_i * tile_shape_out::InnerNumel +
          idx_i * tile_shape_out::InnerCols + idx_j;

      uint16_t sub_tile_i_src = (i + offest_i) / tile_shape_in::InnerRows;
      uint16_t idx_i_src = (i + offest_i) % tile_shape_in::InnerRows;
      uint16_t sub_tile_j_src = (j + offest_j) / tile_shape_in::InnerCols;
      uint16_t idx_j_src = (j + offest_j) % tile_shape_in::InnerCols;
      uint16_t idx_src =
          sub_tile_j_src * tile_shape_in::Rows * tile_shape_in::InnerCols +
          sub_tile_i_src * tile_shape_in::InnerNumel +
          idx_i_src * tile_shape_in::InnerCols + idx_j_src;

      dst.data()[idx_dst] = src.data()[idx_src];
    }
  }
}

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TEXTRACT_Impl(tile_shape_out &dst, tile_shape_in &src, uint16_t offset_i,
              uint16_t offset_j) {
  if (tile_shape_out::ValidRow == DYNAMIC || tile_shape_out::ValidCol == DYNAMIC) { // dynamic
    if constexpr (is_Nz_layout<tile_shape_in>::value &&
                  is_Nz_layout<tile_shape_out>::value) {
      TExtract_NzLayout_Imp_Dynamic<tile_shape_out, tile_shape_in>(dst, src,
                                                          offset_i, offset_j);
    } else if constexpr (tile_shape_in::isRowMajor &&
                        tile_shape_out::isRowMajor) {
      TExtract_RowMajor_Imp_Dynamic<tile_shape_out, tile_shape_in>(dst, src,
                                                          offset_i, offset_j);
    } else if constexpr (!tile_shape_in::isRowMajor &&
                        !tile_shape_out::isRowMajor) {
      TExtract_ColMajor_Imp_Dynamic<tile_shape_out, tile_shape_in>(dst, src,
                                                          offset_i, offset_j);
    } else {
      static_assert(tile_shape_in::isRowMajor && tile_shape_out::isRowMajor,
                    "Storage layout type not supported");
    }
  } else { // static
    if constexpr (is_Nz_layout<tile_shape_in>::value &&
                  is_Nz_layout<tile_shape_out>::value) {
      TExtract_NzLayout_Imp<tile_shape_out, tile_shape_in>(dst.data(), src.data(),
                                                          offset_i, offset_j);
    } else if constexpr (tile_shape_in::isRowMajor &&
                        tile_shape_out::isRowMajor) {
      TExtract_RowMajor_Imp<tile_shape_out, tile_shape_in>(dst.data(), src.data(),
                                                          offset_i, offset_j);
    } else if constexpr (!tile_shape_in::isRowMajor &&
                        !tile_shape_out::isRowMajor) {
      TExtract_ColMajor_Imp<tile_shape_out, tile_shape_in>(dst.data(), src.data(),
                                                          offset_i, offset_j);
    } else {
      static_assert(tile_shape_in::isRowMajor && tile_shape_out::isRowMajor,
                    "Storage layout type not supported");
    }
  }
}

#endif
