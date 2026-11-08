#ifndef TXOR_HPP
#define TXOR_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"

using namespace pto;

template <typename tile_shape>
void TXor_RowMajor_Imp(typename tile_shape::TileDType dst,
                       const typename tile_shape::TileDType src0,
                       const typename tile_shape::TileDType src1) {
  for (size_t i = 0; i < tile_shape::ValidRow; ++i)
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      size_t idx = i * tile_shape::RowStride + j;
      dst[idx] = src0[idx] ^ src1[idx];
    }
}

template <typename tile_shape>
void TXor_ColMajor_Imp(typename tile_shape::TileDType dst,
                       const typename tile_shape::TileDType src0,
                       const typename tile_shape::TileDType src1) {
  for (size_t i = 0; i < tile_shape::ValidCol; ++i)
    for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
      size_t idx = i * tile_shape::ColStride + j;
      dst[idx] = src0[idx] ^ src1[idx];
    }
}

template <is_tile_data_v tile_shape>
void TXOR_Impl(tile_shape &dst, tile_shape &src0, tile_shape &src1) {
  static_assert(tile_shape::Loc == Location::Vec, "Only VEC tile type are supported");
  static_assert(!tile_shape::isBoxedLayout, "TXOR not support Boxed Layout!");

  if constexpr (std::is_same<typename tile_shape::DType, int64_t>::value ||
                std::is_same<typename tile_shape::DType, int32_t>::value ||
                std::is_same<typename tile_shape::DType, int16_t>::value ||
                std::is_same<typename tile_shape::DType, int8_t>::value ||
                std::is_same<typename tile_shape::DType, unsigned long>::value ||
                std::is_same<typename tile_shape::DType, unsigned int>::value ||
                std::is_same<typename tile_shape::DType, unsigned short>::value ||
                std::is_same<typename tile_shape::DType, unsigned char>::value) {
    if constexpr (tile_shape::isRowMajor) {
      TXor_RowMajor_Imp<tile_shape>(dst.data(), src0.data(), src1.data());
    } else {
      TXor_ColMajor_Imp<tile_shape>(dst.data(), src0.data(), src1.data());
    }
  } else {
    static_assert(std::is_same<typename tile_shape::DType, int64_t>::value ||
                  std::is_same<typename tile_shape::DType, int32_t>::value ||
                  std::is_same<typename tile_shape::DType, int16_t>::value ||
                  std::is_same<typename tile_shape::DType, int8_t>::value ||
                  std::is_same<typename tile_shape::DType, unsigned long>::value ||
                  std::is_same<typename tile_shape::DType, unsigned int>::value ||
                  std::is_same<typename tile_shape::DType, unsigned short>::value ||
                  std::is_same<typename tile_shape::DType, unsigned char>::value,
                  "Only int data type are supported");
  }
}

#endif
