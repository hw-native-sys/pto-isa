/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_NZ_UTILS_HPP
#define PTO_CPU_NZ_UTILS_HPP

#include <cstddef>

namespace pto {

template <typename TileData>
PTO_INLINE size_t GetNZTileIndex(size_t r, size_t c)
{
    const size_t subTileR = r / TileData::InnerRows;
    const size_t innerR = r % TileData::InnerRows;
    const size_t subTileC = c / TileData::InnerCols;
    const size_t innerC = c % TileData::InnerCols;
    return GetTileElementOffsetSubfractals<TileData>(subTileR, innerR, subTileC, innerC);
}

PTO_INLINE size_t GetNZGlobalIndex(
    size_t r, size_t c, int gShape1, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2, int gStride3,
    int gStride4)
{
    const size_t outerCol = c / static_cast<size_t>(gShape4);
    const size_t i0 = outerCol / static_cast<size_t>(gShape1);
    const size_t i1 = outerCol % static_cast<size_t>(gShape1);
    const size_t i2 = r / static_cast<size_t>(gShape3);
    const size_t i3 = r % static_cast<size_t>(gShape3);
    const size_t i4 = c % static_cast<size_t>(gShape4);

    return i0 * static_cast<size_t>(gStride0) + i1 * static_cast<size_t>(gStride1) +
           i2 * static_cast<size_t>(gStride2) + i3 * static_cast<size_t>(gStride3) + i4 * static_cast<size_t>(gStride4);
}

template <typename TileData, typename Func>
PTO_INLINE void ForEachNZElement(
    size_t validRow, size_t validCol, int gShape1, int gShape3, int gShape4, int gStride0, int gStride1, int gStride2,
    int gStride3, int gStride4, Func&& func)
{
    for (size_t r = 0; r < validRow; ++r) {
        for (size_t c = 0; c < validCol; ++c) {
            const size_t tile_idx = GetTileElementOffset<TileData>(r, c);
            const size_t gd_idx =
                GetNZGlobalIndex(r, c, gShape1, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3, gStride4);
            func(r, c, tile_idx, gd_idx);
        }
    }
}

} // namespace pto

#endif
