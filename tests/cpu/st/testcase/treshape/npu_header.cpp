/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/common/cpu_stub.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/cpu/TAssign.hpp>

// CPU stubs expose every modeled dtype, masking the missing A2/A3 builtin.
// Keep the real architecture traits, but make direct use of this name fail.
#define float8_e8m0_t PTO_UNAVAILABLE_E8M0_BUILTIN
#include <pto/npu/a2a3/TReshape.hpp>
#undef float8_e8m0_t

namespace {

template <typename Arch>
void CompileScaleViews()
{
    if constexpr (Arch::SupportsFp8) {
        using Element = typename Arch::Float8E8M0Type;
        using Boxed =
            pto::Tile<pto::TileType::Vec, Element, 16, 2, pto::BLayout::RowMajor, 16, 2, pto::SLayout::RowMajor, 32>;
        using Flat = pto::Tile<pto::TileType::Vec, Element, 1, 32>;
        Boxed boxed;
        Flat flat;
        pto::TRESHAPE_IMPL(flat, boxed);
        pto::TRESHAPE_IMPL(boxed, flat);
    }
}

} // namespace

void CompileNpuReshapeHeader()
{
    using Source = pto::Tile<pto::TileType::Vec, float, 2, 16>;
    using Destination = pto::Tile<pto::TileType::Vec, float, 1, 32>;
    Source source;
    Destination destination;
    pto::TRESHAPE_IMPL(destination, source);
    CompileScaleViews<pto::CurrArch>();
}
