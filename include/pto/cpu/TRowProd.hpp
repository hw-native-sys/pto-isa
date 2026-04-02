/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TROWPROD_HPP
#define TROWPROD_HPP

#include "pto/cpu/tile_offsets.hpp"
#include "pto/cpu/parallel.hpp"

namespace pto {
template <typename TileDst, typename TileSrc>
void TRowProd(typename TileDst::TileDType dst, typename TileSrc::TileDType src, uint16_t rows, uint16_t cols)
{
    cpu::parallel_for_1d(0, rows, static_cast<std::size_t>(rows) * cols, [&](std::size_t r) {
        typename TileDst::DType prod = static_cast<typename TileDst::DType>(1);
        for (std::size_t c = 0; c < cols; ++c) {
            prod = static_cast<typename TileDst::DType>(
                prod * static_cast<typename TileDst::DType>(src[GetTileElementOffset<TileSrc>(r, c)]));
        }
        dst[GetTileElementOffset<TileDst>(r, 0)] = prod;
    });
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWPROD_IMPL(TileDataOut &dstTile, TileDataIn &srcTile, TileDataTmp &tmp)
{
    (void)tmp;
    TRowProd<TileDataOut, TileDataIn>(dstTile.data(), srcTile.data(), srcTile.GetValidRow(), srcTile.GetValidCol());
}
} // namespace pto

#endif