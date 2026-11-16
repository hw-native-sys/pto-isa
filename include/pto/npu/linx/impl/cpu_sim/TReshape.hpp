/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Integrated from LinxISA/Linx-TileOP-API; see ../../README.md for provenance.

#ifndef PTO_LINX_VENDOR_PTO_CPU_SIM_T_RESHAPE_HPP
#define PTO_LINX_VENDOR_PTO_CPU_SIM_T_RESHAPE_HPP

#include "pto/npu/linx/impl/common/pto_tile.hpp"
#include <cstring>

using namespace pto;

template <is_tile_data_v tile_shape_out, is_tile_data_v tile_shape_in>
void TReshapeImpl(tile_shape_out& tile_out, tile_shape_in& tile_in)
{
    static_assert(
        tile_shape_in::ValidRow != DYNAMIC && tile_shape_in::ValidCol != DYNAMIC &&
            tile_shape_out::ValidRow != DYNAMIC && tile_shape_out::ValidCol != DYNAMIC,
        "TODO: Support tile dynamic shape!");
    std::memcpy(
        tile_out.data(), tile_in.data(),
        tile_shape_in::Rows * tile_shape_in::Cols * sizeof(typename tile_shape_in::DType));
}

#endif // PTO_LINX_VENDOR_PTO_CPU_SIM_T_RESHAPE_HPP
