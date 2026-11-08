#ifndef TEXPDIF_HPP
#define TEXPDIF_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"
#include <bit>
#include <cmath>
#include <vector>

using namespace pto;

template <typename D, typename A, typename B>
void TEXPDIF_Impl(D &dst, A &src0, B &src1) {
  // Read both sources before publishing any result.  This preserves the
  // operation's read-old/write-new alias contract in the simulator.
  validate_texpdif_operands<D, A, B>();
  const int elements = A::ValidRow * A::ValidCol;
  std::vector<float> left_snapshot(elements);
  std::vector<float> right_snapshot(elements);
  auto index = []<typename T>(int row, int col) {
    if constexpr (T::BFractal == BLayout::CubeM16 ||
                  T::BFractal == BLayout::CubeM32)
      return T::CubeStorageIndex(row, col);
    else if constexpr (T::isRowMajor)
      return row * T::RowStride + col;
    else
      return col * T::ColStride + row;
  };
  for (int row = 0; row < A::ValidRow; ++row) {
    for (int col = 0; col < A::ValidCol; ++col) {
      const int logical = row * A::ValidCol + col;
      const auto left_raw = src0.data()[index.template operator()<A>(row, col)];
      const auto right_raw = src1.data()[index.template operator()<B>(row, col)];
      const auto left = [&] {
        if constexpr (std::is_same_v<texpdif_backing_dtype_t<A>, typename A::DType>)
          return left_raw;
        else
          return std::bit_cast<typename A::DType>(left_raw);
      }();
      const auto right = [&] {
        if constexpr (std::is_same_v<texpdif_backing_dtype_t<B>, typename B::DType>)
          return right_raw;
        else
          return std::bit_cast<typename B::DType>(right_raw);
      }();
      left_snapshot[logical] = static_cast<float>(left);
      right_snapshot[logical] = static_cast<float>(right);
    }
  }
  for (int row = 0; row < A::ValidRow; ++row) {
    for (int col = 0; col < A::ValidCol; ++col) {
      const int logical = row * A::ValidCol + col;
      dst.data()[index.template operator()<D>(row, col)] = static_cast<typename D::DType>(
          std::exp(left_snapshot[logical] - right_snapshot[logical]));
    }
  }
}

#endif
