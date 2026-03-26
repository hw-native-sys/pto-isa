/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TGET_SCALE_ADDR_CPU_HPP
#define TGET_SCALE_ADDR_CPU_HPP

#include <type_traits>
#include "pto/common/pto_tile.hpp"

namespace pto {
template <typename TileDataOut, typename TileDataIn>
PTO_INTERNAL void TGET_SCALE_ADDR_IMPL(TileDataOut &dst, TileDataIn &src)
{
    static_assert(is_tile_data_v<TileDataIn>, "input must be a Tile instance.");
    static_assert(is_tile_data_v<TileDataOut>, "output must be a Tile instance.");
    dst.data() = reinterpret_cast<typename TileDataOut::DType *>(src.data());
}
} // namespace pto

#endif
