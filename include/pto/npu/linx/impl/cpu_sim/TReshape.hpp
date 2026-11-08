#ifndef PTO_CPU_SIM_T_RESHAPE_HPP_
#define PTO_CPU_SIM_T_RESHAPE_HPP_

#include "pto/npu/linx/impl/common/pto_tile.hpp"
#include <cstring>

using namespace pto;

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TReshapeImpl(tile_shape_out &tile_out, tile_shape_in &tile_in) {
  static_assert(tile_shape_in::ValidRow != DYNAMIC && tile_shape_in::ValidCol != DYNAMIC &&
                tile_shape_out::ValidRow != DYNAMIC && tile_shape_out::ValidCol != DYNAMIC,
              "TODO: Support tile dynamic shape!");
  std::memcpy(tile_out.data(), tile_in.data(),
              tile_shape_in::Rows * tile_shape_in::Cols *
                  sizeof(typename tile_shape_in::DType));
}

#endif  // PTO_CPU_SIM_T_RESHAPE_HPP_
