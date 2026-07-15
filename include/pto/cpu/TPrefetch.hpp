/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PREFETCH_HPP
#define PREFETCH_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {
template <typename tile_shape, int stride>
void TPrefetch_Impl(
    typename tile_shape::TileDType dst, typename tile_shape::TileDType src, unsigned validRow, unsigned validCol)
{
    cpu::parallel_for_rows(validRow, validCol, [&](std::size_t r) {
        const std::size_t base = r * tile_shape::Cols;
        PTO_CPU_VECTORIZE_LOOP
        for (std::size_t c = 0; c < validCol; ++c) {
            const std::size_t idx = base + c;
            dst[idx] = src[idx];
        }
    });
}

template <typename TileData, typename GlobalData>
PTO_INTERNAL void TPREFETCH_IMPL(TileData& dst, GlobalData& src)
{
    unsigned row = dst.GetValidRow();
    unsigned col = dst.GetValidCol();
    constexpr unsigned stride = TileData::RowStride;
    TPrefetch_Impl<TileData, stride>(dst.data(), src.data(), row, col);
}
} // namespace pto

#endif
