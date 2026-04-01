/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TCOLREDUCEIDX_CPU_HPP
#define TCOLREDUCEIDX_CPU_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {
template <typename TileDataOut, typename TileDataIn, typename Compare>
PTO_INTERNAL void TColReduceIdxImpl(TileDataOut &dst, TileDataIn &src, Compare compare)
{
    for (unsigned c = 0; c < src.GetValidCol(); ++c) {
        unsigned bestIdx = 0;
        auto bestVal = src.data()[GetTileElementOffset<TileDataIn>(0, c)];
        for (unsigned r = 1; r < src.GetValidRow(); ++r) {
            auto current = src.data()[GetTileElementOffset<TileDataIn>(r, c)];
            if (compare(current, bestVal)) {
                bestVal = current;
                bestIdx = r;
            }
        }
        dst.data()[GetTileElementOffset<TileDataOut>(0, c)] = static_cast<typename TileDataOut::DType>(bestIdx);
    }
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TCOLARGMAX_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    (void)tmp;
    TColReduceIdxImpl(dst, src, [](const auto &lhs, const auto &rhs) { return lhs > rhs; });
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TCOLARGMIN_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    (void)tmp;
    TColReduceIdxImpl(dst, src, [](const auto &lhs, const auto &rhs) { return lhs < rhs; });
}
} // namespace pto

#endif
