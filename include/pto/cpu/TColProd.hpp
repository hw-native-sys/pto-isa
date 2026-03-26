/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TCOLPROD_HPP
#define TCOLPROD_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {
template <typename TileDst, typename TileSrc>
void TColProd(typename TileDst::TileDType dst, typename TileSrc::TileDType src, uint16_t rows, uint16_t cols)
{
    for (uint16_t c = 0; c < cols; ++c) {
        typename TileDst::DType prod = static_cast<typename TileDst::DType>(1);
        for (uint16_t r = 0; r < rows; ++r) {
            prod = static_cast<typename TileDst::DType>(
                prod * static_cast<typename TileDst::DType>(src[GetTileElementOffset<TileSrc>(r, c)]));
        }
        dst[GetTileElementOffset<TileDst>(0, c)] = prod;
    }
}

template <typename TileDst, typename TileSrc>
PTO_INTERNAL void TCOLPROD_IMPL(TileDst &dstTile, TileSrc &srcTile)
{
    TColProd<TileDst, TileSrc>(dstTile.data(), srcTile.data(), srcTile.GetValidRow(), srcTile.GetValidCol());
}
} // namespace pto

#endif
