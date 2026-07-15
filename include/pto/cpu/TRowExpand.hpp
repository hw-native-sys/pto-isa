/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_TROWEXPAND_HPP
#define PTO_CPU_TROWEXPAND_HPP

#include <type_traits>

#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TROWEXPAND_IMPL(TileDst& dst, TileSrc& src)
{
    const std::size_t rows = static_cast<std::size_t>(dst.GetValidRow());
    const std::size_t cols = static_cast<std::size_t>(dst.GetValidCol());
    if (rows == 0 || cols == 0) {
        return;
    }

    cpu::parallel_for_rows(rows, cols, [&](std::size_t r) {
        const auto v = src.data()[GetTileElementOffset<TileSrc>(r, 0)];
        for (std::size_t c = 0; c < cols; ++c) {
            dst.data()[GetTileElementOffset<TileDst>(r, c)] = static_cast<typename TileDst::DType>(v);
        }
    });
}

} // namespace pto

#endif
