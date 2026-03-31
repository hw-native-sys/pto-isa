/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_GPU_COMMON_TILE_OFFSETS_HPP
#define PTO_GPU_COMMON_TILE_OFFSETS_HPP

#include <cstddef>

namespace pto::gpu {

template <typename TileData>
PTO_INTERNAL std::size_t GetTileElementOffsetSubfractals(std::size_t subTileR, std::size_t innerR,
                                                         std::size_t subTileC, std::size_t innerC)
{
    if constexpr (!TileData::isRowMajor && (TileData::SFractal == SLayout::RowMajor)) {
        return subTileC * TileData::Rows * TileData::InnerCols + subTileR * TileData::InnerNumel +
               innerR * TileData::InnerCols + innerC;
    } else if constexpr (TileData::isRowMajor && (TileData::SFractal == SLayout::ColMajor)) {
        return subTileR * TileData::Cols * TileData::InnerRows + subTileC * TileData::InnerNumel +
               innerC * TileData::InnerRows + innerR;
    } else {
        return subTileR * TileData::Cols * TileData::InnerRows + subTileC * TileData::InnerNumel +
               innerR * TileData::InnerCols + innerC;
    }
}

template <typename TileData>
PTO_INTERNAL std::size_t GetTileElementOffsetPlain(std::size_t r, std::size_t c)
{
    if constexpr (TileData::isRowMajor) {
        return r * TileData::Cols + c;
    }
    return c * TileData::Rows + r;
}

template <typename TileData>
PTO_INTERNAL std::size_t GetTileElementOffset(std::size_t r, std::size_t c)
{
    if constexpr (TileData::SFractal == SLayout::NoneBox) {
        return GetTileElementOffsetPlain<TileData>(r, c);
    }
    return GetTileElementOffsetSubfractals<TileData>(r / TileData::InnerRows, r % TileData::InnerRows,
                                                     c / TileData::InnerCols, c % TileData::InnerCols);
}

} // namespace pto::gpu

#endif
