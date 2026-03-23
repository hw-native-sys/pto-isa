/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TROWEXPAND_INSTR_HPP
#define TROWEXPAND_INSTR_HPP

#include "pto/costmodel/pto_isa_costmodel.hpp"

namespace pto {
constexpr const int vbrcbElem = 8;
PTO_INTERNAL void TRowExpand(std::vector<CostModelStats> &stats, int validRow, int validCol)
{
    for (int i = 0; i < validRow; i++) {
        stats.emplace_back("vector_dup", BLOCK_MAX_PER_REPEAT);
    }
}

template <typename TileDataSrc>
PTO_INTERNAL void TRowExpandBrcb(std::vector<CostModelStats> &stats)
{
    constexpr int repeat = TileDataSrc::Numel / vbrcbElem;
    constexpr int loop = repeat / (REPEAT_MAX - 1);
    constexpr int remain = repeat % (REPEAT_MAX - 1);
    if constexpr (loop > 0) {
        for (int i = 0; i < loop; ++i) {
            stats.emplace_back("vbrcb", 1);
        }
    }
    if constexpr (remain > 0) {
        stats.emplace_back("vbrcb", remain);
    }
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL std::vector<CostModelStats> runRowExpandOp(TileDataDst &dst, TileDataSrc &src)
{
    std::vector<CostModelStats> stats;
    using T = typename TileDataSrc::DType;
    int dstValidRow = dst.GetValidRow();
    int dstValidCol = dst.GetValidCol();

    constexpr bool isBroadcastSupportType = (sizeof(T) == 2 || sizeof(T) == 4);

    constexpr bool isStaticShape =
        (TileDataSrc::Rows == TileDataSrc::ValidRow) && (TileDataSrc::Cols == TileDataSrc::ValidCol) &&
        (TileDataDst::Rows == TileDataDst::ValidRow) && (TileDataDst::Cols == TileDataDst::ValidCol);

    constexpr unsigned elemPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr bool isBroadcast = TileDataSrc::isRowMajor ?
                                     ((TileDataSrc::Rows == 1) && (TileDataSrc::Cols == TileDataDst::Rows) &&
                                      (TileDataDst::Cols == elemPerBlock)) :
                                     ((TileDataSrc::Cols == 1) && (TileDataSrc::Rows == TileDataDst::Rows) &&
                                      (TileDataDst::Cols == elemPerBlock));

    if constexpr (isBroadcastSupportType && isStaticShape && isBroadcast) {
        TRowExpandBrcb<TileDataSrc>(stats);
    } else {
        TRowExpand(stats, dstValidRow, dstValidCol);
    }
    return stats;
}

} // namespace pto
#endif
