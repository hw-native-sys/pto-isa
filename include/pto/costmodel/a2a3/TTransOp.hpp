/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TTRANS_OP_HPP
#define TTRANS_OP_HPP

#include <vector>
#include "pto/costmodel/costmodel_types.hpp"
#include "pto/costmodel/op_struct.hpp"

namespace pto {
constexpr int Y_ELEM_B8 = 32;
constexpr int Y_ELEM_OTHER = 16;

template <typename DstTile, typename SrcTile, typename TmpTile>
PTO_INTERNAL std::vector<CostModelStats> runTransOp(DstTile & /*dst*/, SrcTile &src, TmpTile & /*tmp*/)
{
    using T = typename SrcTile::DType;
    constexpr unsigned BLOCK_BYTE = 32u;
    constexpr unsigned blockSizeElem = BLOCK_BYTE / sizeof(T);
    constexpr unsigned yTileSizeElem = (sizeof(T) == 1u) ? 32u : 16u;

    unsigned validRow = src.GetValidRow();
    unsigned validCol = src.GetValidCol();

    unsigned numSubTileX = (validCol + blockSizeElem - 1u) / blockSizeElem;
    unsigned numSubTileY = validRow / yTileSizeElem;
    unsigned remainY = validRow % yTileSizeElem;

    std::vector<CostModelStats> stats;

    auto emitTransInstr = [&](unsigned repeats) {
        if constexpr (sizeof(T) == 1u) {
            TransOp::TransB8Instr(stats, repeats);
        } else if constexpr (sizeof(T) == 2u) {
            TransOp::TransB16Instr(stats, repeats);
        } else {
            TransOp::TransB32Instr(stats, repeats);
        }
    };

    // Full Y-tiles
    if (numSubTileY > 0) {
        for (unsigned i = 0; i < numSubTileX; i++) {
            emitTransInstr(numSubTileY);
        }
    }

    // Y-tail (partial last block of rows)
    if (remainY > 0) {
        emitTransInstr(numSubTileX);
    }

    // pipe_barrier(PIPE_V) then copy_ubuf_to_ubuf (MTE1) to copy the transposed result
    stats.emplace_back("pipe_barrier");
    stats.emplace_back("copy_ubuf_to_ubuf", 1);

    return stats;
}

} // namespace pto
#endif // TTRANS_OP_HPP
