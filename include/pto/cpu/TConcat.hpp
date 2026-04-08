/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TCONCAT_HPP
#define TCONCAT_HPP

#include <pto/common/pto_tile.hpp>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {
template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TCONCAT_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    assert(dst.GetValidRow() == src0.GetValidRow() && dst.GetValidRow() == src1.GetValidRow());
    assert(dst.GetValidCol() >= src0.GetValidCol() + src1.GetValidCol());
    const unsigned rows = dst.GetValidRow();
    const unsigned cols0 = src0.GetValidCol();
    const unsigned cols1 = src1.GetValidCol();

    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned c = 0; c < cols0; ++c) {
            dst.data()[GetTileElementOffset<TileDataDst>(r, c)] = src0.data()[GetTileElementOffset<TileDataSrc0>(r, c)];
        }
        for (unsigned c = 0; c < cols1; ++c) {
            dst.data()[GetTileElementOffset<TileDataDst>(r, cols0 + c)] =
                src1.data()[GetTileElementOffset<TileDataSrc1>(r, c)];
        }
    }
}
} // namespace pto

#endif
