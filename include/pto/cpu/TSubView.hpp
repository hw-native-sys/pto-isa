/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TSUBVIEW_CPU_HPP
#define TSUBVIEW_CPU_HPP

#include <cstdint>

namespace pto {
template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TSUBVIEW_IMPL(TileDataDst &dst, TileDataSrc &src, uint16_t rowIdx, uint16_t colIdx)
{
    constexpr int kRowStride = TileDataSrc::RowStride;
    constexpr int kColStride = TileDataSrc::ColStride;
    const uint64_t totalOffset = rowIdx * kRowStride + colIdx * kColStride;

    static_assert(TileDataDst::Loc == TileDataSrc::Loc,
                  "The destination and source tiles must have the same TileType!");

    dst.data() = src.data() + totalOffset;
}
} // namespace pto

#endif