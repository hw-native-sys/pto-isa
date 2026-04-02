/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TDEQUANT_CPU_HPP
#define TDEQUANT_CPU_HPP

#include <algorithm>
#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {
template <typename TileDataDst, typename TileDataSrc, typename TileDataPara>
PTO_INTERNAL void TDEQUANT_IMPL(TileDataDst &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara &offset)
{
    const unsigned rows = dst.GetValidRow();
    const unsigned cols = dst.GetValidCol();
    const unsigned paraCols = std::max(1u, static_cast<unsigned>(scale.GetValidCol()));

    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned c = 0; c < cols; ++c) {
            const unsigned paraCol = std::min(c, paraCols - 1);
            const auto scaleVal = scale.data()[GetTileElementOffset<TileDataPara>(r, paraCol)];
            const auto offsetVal = offset.data()[GetTileElementOffset<TileDataPara>(r, paraCol)];
            const auto srcVal = src.data()[GetTileElementOffset<TileDataSrc>(r, c)];
            dst.data()[GetTileElementOffset<TileDataDst>(r, c)] =
                static_cast<typename TileDataDst::DType>((static_cast<typename TileDataDst::DType>(srcVal) -
                                                          static_cast<typename TileDataDst::DType>(offsetVal)) *
                                                         static_cast<typename TileDataDst::DType>(scaleVal));
        }
    }
}
} // namespace pto

#endif