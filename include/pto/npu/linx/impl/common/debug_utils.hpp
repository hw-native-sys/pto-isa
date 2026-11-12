/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_LINX_NATIVE_DEBUG_UIILS_HPP
#define PTO_LINX_NATIVE_DEBUG_UIILS_HPP

#if (defined(__linx) || defined(PTO_CPU_SIM_TARGET_LINX)) && defined(PTO_ENABLE_TILE_DEBUG)
#include "pto/npu/linx/impl/jcore/utils.hpp"
#endif

namespace pto {
#if (defined(__linx) || defined(PTO_CPU_SIM_TARGET_LINX)) && !defined(PTO_ENABLE_TILE_DEBUG)
template <typename tile_shape>
void print_tile(tile_shape &) = delete;
#else
template <typename tile_shape>
void print_tile(tile_shape &tile) {
  print_tile_Impl(tile);
}
#endif

} // namespace pto

#endif
