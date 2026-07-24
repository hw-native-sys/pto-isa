/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSELS_HPP
#define TSELS_HPP

#include "pto/common/pto_tile.hpp"
#include "tile_offsets.hpp"
#include <type_traits>
#include <climits>
#include <cassert>

namespace pto {
template <typename TileDataDst, typename TileDataMask, typename TileDataSrc, typename TileDataTmp>
__aicore__ PTO_INLINE void TSELS_IMPL(
    TileDataDst& dst, TileDataMask& mask, TileDataSrc& src, TileDataTmp& tmp, typename TileDataSrc::DType scalar)
{
    static_assert(std::is_same_v<typename TileDataDst::DType, typename TileDataSrc::DType>);
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    assert(validRow == src.GetValidRow() && validCol == src.GetValidCol());

    using MaskT = typename TileDataMask::DType;
    constexpr size_t bitsPerElement = sizeof(MaskT) * CHAR_BIT;

    for (size_t r = 0; r < validRow; r++) {
        for (size_t c = 0; c < validCol; c++) {
            size_t idxSrc = GetTileElementOffset<TileDataSrc>(r, c);
            size_t idxMask = GetTileElementOffset<TileDataMask>(r, c / bitsPerElement);
            size_t idxDst = GetTileElementOffset<TileDataDst>(r, c);

            MaskT maskBits = mask.data()[idxMask];
            bool isBitSet = (maskBits >> (c % bitsPerElement)) & 1;
            dst.data()[idxDst] = isBitSet ? src.data()[idxSrc] : scalar;
        }
    }
}

} // namespace pto
#endif
