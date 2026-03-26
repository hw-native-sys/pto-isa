/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TROWREDUCEIDX_CPU_HPP
#define TROWREDUCEIDX_CPU_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {
template <typename TileDataOut, typename TileDataIn, typename Compare>
PTO_INTERNAL void TRowReduceIdxImpl(TileDataOut &dst, TileDataIn &src, Compare compare)
{
    for (unsigned r = 0; r < src.GetValidRow(); ++r) {
        unsigned bestIdx = 0;
        auto bestVal = src.data()[GetTileElementOffset<TileDataIn>(r, 0)];
        for (unsigned c = 1; c < src.GetValidCol(); ++c) {
            auto current = src.data()[GetTileElementOffset<TileDataIn>(r, c)];
            if (compare(current, bestVal)) {
                bestVal = current;
                bestIdx = c;
            }
        }
        dst.data()[GetTileElementOffset<TileDataOut>(r, 0)] = static_cast<typename TileDataOut::DType>(bestIdx);
    }
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWARGMAX_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    (void)tmp;
    TRowReduceIdxImpl(dst, src, [](const auto &lhs, const auto &rhs) { return lhs > rhs; });
}

template <typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TROWARGMIN_IMPL(TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
{
    (void)tmp;
    TRowReduceIdxImpl(dst, src, [](const auto &lhs, const auto &rhs) { return lhs < rhs; });
}
} // namespace pto

#endif
