#ifndef JCORES_UTILS_HPP
#define JCORES_UTILS_HPP

#include <iomanip>
#include "pto/npu/linx/impl/common/pto_tileop.hpp"
#include "pto/npu/linx/impl/common/tileop_api.hpp"

namespace pto {
template <typename tile_shape>
void print_tile_Impl(tile_shape &tile) {
  static constexpr size_t tile_size = tile_shape::Rows * tile_shape::Cols;
  typename tile_shape::DType d[tile_size] = {0};
  using dtype = typename tile_shape::DType;
  using shape = Shape<1, 1, 1, 1, 1>;
  using stride =
       std::conditional_t<tile_shape::isRowMajor || tile_shape::isBoxedLayout,
          Stride<1, 1, tile_shape::Rows * tile_shape::Cols, tile_shape::Cols, 1>,
          Stride<1, 1, tile_shape::Rows * tile_shape::Cols, 1, tile_shape::Rows>>;
  using gm_shape =
      std::conditional_t<tile_shape::isRowMajor || tile_shape::isBoxedLayout,
          GlobalTensor<dtype, shape, stride, Layout::ND>,
          GlobalTensor<dtype, shape, stride, Layout::DN>>;
  gm_shape dst(d);
  TSTORE(dst, tile);

  print_tile_info<tile_shape>();
  std::cout << std::fixed << std::scientific << std::setprecision(4);
  if constexpr (!gm_shape::isRowMajor) {
    for (int i = 0; i < tile_shape::Rows; i++) {
      for (int j = 0; j < tile_shape::Cols; j++) {
        int offset = j * tile_shape::Rows + i;
        std::cout << std::setw(8) << static_cast<float>(*(d + offset)) << "\t";
        if (j == tile.GetValidCol() - 1 && tile.GetValidCol() < tile_shape::Cols)
          std::cout << (i >= tile.GetValidRow() ? " \t" : "|\t");
      }
      if (i == tile.GetValidRow() - 1 && tile.GetValidRow() < tile_shape::Rows) {
        std::cout << std::endl;
        for (int k = 0; k < tile.GetValidCol(); k++)
          std::cout << std::setw(8) << "-" << "\t";
      }
      std::cout << std::endl;
    }
  } else {
      for (int i = 0; i < tile_shape::Rows; i++) {
        for (int j = 0; j < tile_shape::Cols; j++) {
          int offset = i * tile_shape::Cols + j;
          std::cout << std::setw(8) << static_cast<float>(*(d + offset)) << "\t";
          if (j == tile.GetValidCol() - 1 && tile.GetValidCol() < tile_shape::Cols)
				    std::cout << (i >= tile.GetValidRow() ? " \t" : "|\t");
        }
        if (i == tile.GetValidRow() - 1 && tile.GetValidRow() < tile_shape::Rows) {
          std::cout << std::endl;
          for (int k = 0; k < tile.GetValidCol(); k++)
            std::cout << std::setw(8) << "-" << "\t";
        }
        std::cout << std::endl;
      }
  }
}

} // namspace end

#endif // JCORES_UTILS_HPP
