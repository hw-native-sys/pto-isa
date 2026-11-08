#ifndef TSRL_HPP
#define TSRL_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"
#include <type_traits>

using namespace pto;

// Host reference: uniform logical (unsigned) right shift of the whole tile by
// a scalar amount. src is converted to unsigned before shifting so the sign
// bit does not propagate.
template <typename tile_shape>
void TSrl_RowMajor_Imp(typename tile_shape::TileDType dst,
                       const typename tile_shape::TileDType src,
                       unsigned shamt) {
  using UDType = std::make_unsigned_t<typename tile_shape::DType>;
  for (size_t i = 0; i < tile_shape::ValidRow; ++i)
    for (size_t j = 0; j < tile_shape::ValidCol; ++j) {
      size_t idx = i * tile_shape::RowStride + j;
      dst[idx] = static_cast<typename tile_shape::DType>(
          static_cast<UDType>(src[idx]) >> shamt);
    }
}

template <typename tile_shape>
void TSrl_ColMajor_Imp(typename tile_shape::TileDType dst,
                       const typename tile_shape::TileDType src,
                       unsigned shamt) {
  using UDType = std::make_unsigned_t<typename tile_shape::DType>;
  for (size_t i = 0; i < tile_shape::ValidCol; ++i)
    for (size_t j = 0; j < tile_shape::ValidRow; ++j) {
      size_t idx = i * tile_shape::ColStride + j;
      dst[idx] = static_cast<typename tile_shape::DType>(
          static_cast<UDType>(src[idx]) >> shamt);
    }
}

template <is_tile_data_v tile_shape>
void TSRL_Impl(tile_shape &dst, tile_shape &src, unsigned shamt) {
  static_assert(tile_shape::Loc == Location::Vec, "Only VEC tile type are supported");
  static_assert(!tile_shape::isBoxedLayout, "TSRL not support Boxed Layout!");

  if constexpr (std::is_same<typename tile_shape::DType, int64_t>::value ||
                std::is_same<typename tile_shape::DType, int32_t>::value ||
                std::is_same<typename tile_shape::DType, int16_t>::value ||
                std::is_same<typename tile_shape::DType, int8_t>::value ||
                std::is_same<typename tile_shape::DType, unsigned long>::value ||
                std::is_same<typename tile_shape::DType, unsigned int>::value ||
                std::is_same<typename tile_shape::DType, unsigned short>::value ||
                std::is_same<typename tile_shape::DType, unsigned char>::value) {
    if constexpr (tile_shape::isRowMajor) {
      TSrl_RowMajor_Imp<tile_shape>(dst.data(), src.data(), shamt);
    } else {
      TSrl_ColMajor_Imp<tile_shape>(dst.data(), src.data(), shamt);
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
