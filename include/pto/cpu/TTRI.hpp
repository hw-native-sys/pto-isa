/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TTRI_HPP
#define TTRI_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {
template <typename TileData, int isUpperOrLower>
PTO_INTERNAL void TTriCheck(const TileData &dst)
{
    using T = typename TileData::DType;
    static_assert(std::is_same<T, int32_t>::value || std::is_same<T, int>::value || std::is_same<T, int16_t>::value ||
                      std::is_same<T, uint32_t>::value || std::is_same<T, uint16_t>::value ||
                      std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value ||
                      std::is_same<T, float>::value,
                  "Fix: TTRI has invalid data type.");
    static_assert(isUpperOrLower == 0 || isUpperOrLower == 1, "Fix: isUpperOrLower must be 0 or 1.");
    static_assert(TileData::isRowMajor, "Fix: TTRI only support row major layout.");
}

template <typename TileData>
PTO_INTERNAL void TTril(__ubuf__ typename TileData::DType *dstPtr, unsigned validRow, unsigned validCol, int diagonal)
{
    for (int r = 0; r < validRow; r++) {
        int base = r * TileData::Cols;
        int ends = std::min(diagonal + r + 1, (int)validCol);
        int starts = std::max(diagonal + r + 1, 0);
        PTO_CPU_VECTORIZE_LOOP
        for (int c = 0; c < ends; c++) {
            dstPtr[base + c] = 1;
        }
        PTO_CPU_VECTORIZE_LOOP
        for (int c = starts; c < validCol; c++) {
            dstPtr[base + c] = 0;
        }
    }
}

template <typename TileData>
PTO_INTERNAL void TTriu(__ubuf__ typename TileData::DType *dstPtr, unsigned validRow, unsigned validCol, int diagonal)
{
    for (int r = 0; r < validRow; r++) {
        int base = r * TileData::Cols;
        int ends = std::min(diagonal + r, (int)validCol);
        int starts = std::max(diagonal + r, 0);
        PTO_CPU_VECTORIZE_LOOP
        for (int c = 0; c < ends; c++) {
            dstPtr[base + c] = 0;
        }
        PTO_CPU_VECTORIZE_LOOP
        for (int c = starts; c < validCol; c++) {
            dstPtr[base + c] = 1;
        }
    }
}

template <typename TileData, int isUpperOrLower>
PTO_INTERNAL void TTRI_IMPL(TileData &dst, int diagonal)
{
    TTriCheck<TileData, isUpperOrLower>(dst);
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    if constexpr (isUpperOrLower == 0)
        TTril<TileData>(dst.data(), validRow, validCol, diagonal);
    else
        TTriu<TileData>(dst.data(), validRow, validCol, diagonal);
}
} // namespace pto
#endif
