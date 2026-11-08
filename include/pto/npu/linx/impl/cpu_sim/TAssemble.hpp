#ifndef TASSEMBLE_HPP
#define TASSEMBLE_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape_in0, typename tile_shape_in1,
          typename tile_shape_in2, typename tile_shape_out>
void TAssemble_RowMajor_Imp(typename tile_shape_out::TileDType dst,
                            const typename tile_shape_in0::TileDType src0,
                            const typename tile_shape_in1::TileDType src1,
                            const typename tile_shape_in2::TileDType src2) {
  for (size_t i = 0; i < tile_shape_out::ValidRow; ++i) {
    for (size_t j = 0; j < tile_shape_in0::ValidCol; ++j) {
      dst[i * tile_shape_out::RowStride + j] =
          src0[i * tile_shape_in0::RowStride + j];
    }
    for (size_t j = 0; j < tile_shape_in1::ValidCol; ++j) {
      dst[tile_shape_in0::ValidCol + i * tile_shape_out::RowStride + j] =
          src1[i * tile_shape_in1::RowStride + j];
    }
    for (size_t j = 0; j < tile_shape_in2::ValidCol; ++j) {
      dst[tile_shape_in0::ValidCol + tile_shape_in1::ValidCol +
          i * tile_shape_out::RowStride + j] =
          src2[i * tile_shape_in2::RowStride + j];
    }
  }
}

template <typename tile_shape_in0, typename tile_shape_in1,
          typename tile_shape_in2, typename tile_shape_out>
void TAssemble_ColMajor_Imp(typename tile_shape_out::TileDType dst,
                            const typename tile_shape_in0::TileDType src0,
                            const typename tile_shape_in1::TileDType src1,
                            const typename tile_shape_in2::TileDType src2) {
  for (size_t i = 0; i < tile_shape_out::ValidRow; ++i) {
    for (size_t j = 0; j < tile_shape_in0::ValidCol; ++j) {
      dst[j * tile_shape_out::ColStride + i] =
          src0[j * tile_shape_in0::ColStride + i];
    }
    for (size_t j = 0; j < tile_shape_in1::ValidCol; ++j) {
      dst[tile_shape_in0::Numel + j * tile_shape_out::ColStride + i] =
          src1[j * tile_shape_in1::ColStride + i];
    }
    for (size_t j = 0; j < tile_shape_in2::ValidCol; ++j) {
      dst[tile_shape_in0::Numel + tile_shape_in1::Numel +
          j * tile_shape_out::ColStride + i] =
          src2[j * tile_shape_in2::ColStride + i];
    }
  }
}

template <typename tile_shape_in0, typename tile_shape_in1,
          typename tile_shape_in2, typename tile_shape_out>
void TAssemble_NzLayout_Imp(typename tile_shape_out::TileDType dst,
                            const typename tile_shape_in0::TileDType src0,
                            const typename tile_shape_in1::TileDType src1,
                            const typename tile_shape_in2::TileDType src2) {
  for (size_t i = 0; i < tile_shape_out::ValidRow; ++i) {
    size_t sub_tile_i = i / tile_shape_out::InnerRows;
    size_t idx_i = i % tile_shape_out::InnerRows;
    for (size_t j = 0; j < tile_shape_in0::ValidCol; ++j) {
      size_t sub_tile_j0 = j / tile_shape_in0::kInnerCols;
      size_t idx_j0 = j % tile_shape_in0::kInnerCols;

      size_t idx0 =
          sub_tile_j0 * tile_shape_in0::Rows * tile_shape_in0::kInnerCols +
          sub_tile_i * tile_shape_in0::kInnerNumel +
          idx_i * tile_shape_in0::kInnerCols + idx_j0;
      dst[idx0] = src0[idx0];
    }
    for (size_t j = 0; j < tile_shape_in1::ValidCol; ++j) {
      size_t sub_tile_j1 = j / tile_shape_in1::kInnerCols;
      size_t idx_j1 = j % tile_shape_in1::kInnerCols;

      size_t idx1 =
          sub_tile_j1 * tile_shape_in1::Rows * tile_shape_in1::kInnerCols +
          sub_tile_i * tile_shape_in1::kInnerNumel +
          idx_i * tile_shape_in1::kInnerCols + idx_j1;
      dst[idx1 + tile_shape_in0::Cols * tile_shape_out::Rows] = src1[idx1];
    }
    for (size_t j = 0; j < tile_shape_in2::ValidCol; ++j) {
      size_t sub_tile_j2 = j / tile_shape_in2::kInnerCols;
      size_t idx_j2 = j % tile_shape_in2::kInnerCols;

      size_t idx2 =
          sub_tile_j2 * tile_shape_in2::Rows * tile_shape_in2::kInnerCols +
          sub_tile_i * tile_shape_in2::kInnerNumel +
          idx_i * tile_shape_in2::kInnerCols + idx_j2;
      dst[idx2 + tile_shape_in0::Cols * tile_shape_out::Rows +
          tile_shape_in1::Cols * tile_shape_out::Rows] = src2[idx2];
    }
  }
}

template <is_tile_data_v tile_shape_in0, is_tile_data_v tile_shape_in1,
          is_tile_data_v tile_shape_in2, is_tile_data_v tile_shape_out>
void TASSEMBLE_Impl(tile_shape_out &dst, tile_shape_in0 &src0, tile_shape_in1 &src1,
               tile_shape_in2 &src2) {
  static_assert(tile_shape_out::ValidRow == tile_shape_in0::ValidRow &&
                    tile_shape_out::ValidRow == tile_shape_in1::ValidRow &&
                    tile_shape_out::ValidRow == tile_shape_in2::ValidRow,
                "Error! All tile rows must be equal");
  static_assert(tile_shape_out::ValidRow != DYNAMIC && tile_shape_out::ValidCol != DYNAMIC &&
                tile_shape_in0::ValidRow != DYNAMIC && tile_shape_in0::ValidCol != DYNAMIC &&
                tile_shape_in1::ValidRow != DYNAMIC && tile_shape_in1::ValidCol != DYNAMIC &&
                tile_shape_in2::ValidRow != DYNAMIC && tile_shape_in2::ValidCol != DYNAMIC,
              "TODO: Support tile dynamic shape!");
  static_assert(std::is_same<typename tile_shape_in0::DType,
                             typename tile_shape_in1::DType>::value &&
                    std::is_same<typename tile_shape_in1::DType,
                                 typename tile_shape_in2::DType>::value &&
                    std::is_same<typename tile_shape_in2::DType,
                                 typename tile_shape_out::DType>::value,
                "Error! All data types must be equal");
  static_assert(
      tile_shape_out::ValidCol ==
          tile_shape_in0::ValidCol + tile_shape_in1::ValidCol + tile_shape_in2::ValidCol,
      "Error! Output columns must equal the sum of the input columns.");
  static_assert(tile_shape_in0::ValidCol == tile_shape_in0::ValidCol &&
                    tile_shape_in1::ValidCol == tile_shape_in1::ValidCol &&
                    tile_shape_in2::ValidCol == tile_shape_in2::ValidCol,
                "Error! Mask for columns are forbidden");

  if constexpr (is_Nz_layout<tile_shape_in0>::value &&
                is_Nz_layout<tile_shape_in1>::value &&
                is_Nz_layout<tile_shape_in2>::value &&
                is_Nz_layout<tile_shape_out>::value) {
    TAssemble_NzLayout_Imp<tile_shape_in0, tile_shape_in1, tile_shape_in2,
                           tile_shape_out>(dst.data(), src0.data(), src1.data(),
                                           src2.data());
  } else if constexpr (tile_shape_in0::isRowMajor &&
                       tile_shape_in1::isRowMajor &&
                       tile_shape_in2::isRowMajor &&
                       tile_shape_out::isRowMajor) {
    TAssemble_RowMajor_Imp<tile_shape_in0, tile_shape_in1, tile_shape_in2,
                           tile_shape_out>(dst.data(), src0.data(), src1.data(),
                                           src2.data());
  } else if constexpr (!tile_shape_in0::isRowMajor &&
                       !tile_shape_in1::isRowMajor &&
                       !tile_shape_in2::isRowMajor &&
                       !tile_shape_out::isRowMajor) {
    TAssemble_ColMajor_Imp<tile_shape_in0, tile_shape_in1, tile_shape_in2,
                           tile_shape_out>(dst.data(), src0.data(), src1.data(),
                                           src2.data());
  } else {
    static_assert(tile_shape_in0::isRowMajor &&
                      tile_shape_in1::isRowMajor &&
                      tile_shape_in2::isRowMajor &&
                      tile_shape_out::isRowMajor,
                  "Storage layout type not supported");
  }
}

#endif
