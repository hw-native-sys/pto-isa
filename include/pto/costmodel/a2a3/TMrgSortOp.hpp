/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TMRGSORT_OP_HPP
#define TMRGSORT_OP_HPP

#include <vector>
#include <cstdint>
#include "pto/costmodel/costmodel_types.hpp"

namespace pto {

// Single-src TMRGSORT: vmrgsort4(repeatTimes)
//   repeatTimes = srcCol / (blockLen * 4)
template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL std::vector<CostModelStats> runMrgSortSingleOp(DstTileData & /*dst*/, SrcTileData &src, uint32_t blockLen)
{
    constexpr unsigned BLOCK_NUM = 4;
    uint32_t srcCol = src.GetValidCol();
    int repeatTimes = static_cast<int>(srcCol / (blockLen * BLOCK_NUM));

    std::vector<CostModelStats> stats;
    stats.emplace_back("vmrgsort4", repeatTimes);
    return stats;
}

// Multi-src TMRGSORT (2/3/4 sources):
//   vmrgsort4(1) + pipe_barrier(PIPE_V) + copy_ubuf_to_ubuf (MTE1, nBurst=1, lenBurst=1)
PTO_INTERNAL std::vector<CostModelStats> runMrgSortMultiOp()
{
    std::vector<CostModelStats> stats;
    stats.emplace_back("vmrgsort4", 1);
    stats.emplace_back("pipe_barrier");
    stats.emplace_back("copy_ubuf_to_ubuf", 1); // copy sorted result (MTE1)
    return stats;
}

} // namespace pto
#endif // TMRGSORT_OP_HPP
