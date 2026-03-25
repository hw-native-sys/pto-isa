/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TSORT32_OP_HPP
#define TSORT32_OP_HPP

#include <vector>
#include <pto/common/constants.hpp>
#include "pto/costmodel/costmodel_types.hpp"

namespace pto {

constexpr unsigned SORT_BLOCK = 32;
constexpr unsigned MAX_UB_TMP = 32 * 255;

// TSORT32: vbitsort(repeatNumPerRow) + PIPE_V per row.
// repeatNumPerRow = validCol / 32
template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL std::vector<CostModelStats> runSort32Op(DstTileData &dst, SrcTileData &src)
{
    unsigned validRow = dst.GetValidRow();
    unsigned repeatNumPerRow = src.GetValidCol() / SORT_BLOCK;

    std::vector<CostModelStats> stats;

    if (repeatNumPerRow < REPEAT_MAX) {
        for (unsigned i = 0; i < validRow; i++) {
            stats.emplace_back("vbitsort", repeatNumPerRow);
            stats.emplace_back("pipe_barrier");
        }
    } else {
        unsigned loopNum = ((repeatNumPerRow + REPEAT_MAX) - 1) / REPEAT_MAX;
        unsigned tailRepeatNum = repeatNumPerRow % REPEAT_MAX;
        for (unsigned i = 0; i < validRow; i++) {
            for (unsigned j = 0; j < loopNum; j++) {
                unsigned repeatNum = (j == loopNum - 1) ? tailRepeatNum : REPEAT_MAX;
                stats.emplace_back("vbitsort", repeatNum);
                stats.emplace_back("pipe_barrier");
            }
        }
    }

    return stats;
}

template <typename DstTileData, typename SrcTileData, typename TmpTileData>
PTO_INTERNAL std::vector<CostModelStats> runSort32OpWithTmp(DstTileData &dst, SrcTileData &src, TmpTileData &tmp)
{
    unsigned validRow = dst.GetValidRow();
    unsigned repeatNumPerRow = src.GetValidCol() / SORT_BLOCK;

    std::vector<CostModelStats> stats;

    if (src.GetValidCol() <= MAX_UB_TMP) {
        for (unsigned i = 0; i < validRow; i++) {
            // copy row src cbuf to tmp cbuf
            stats.emplace_back("copy_ubuf_to_ubuf", 1);
            stats.emplace_back("pipe_barrier");

            // dup -NAN of tial value in tmp cbuf
            stats.emplace_back("copy_ubuf_to_ubuf", 1);
            stats.emplace_back("pipe_barrier");

            // sort for tmp and out to dst
            stats.emplace_back("vbitsort", repeatNumPerRow + 1);
            stats.emplace_back("pipe_barrier");
        }
    } else {
        auto loopNum = ((repeatNumPerRow + 1 + REPEAT_MAX) - 1) / REPEAT_MAX;
        unsigned srcTailRepeatNum = ((src.GetValidCol() + SORT_BLOCK - 1) / SORT_BLOCK) % REPEAT_MAX;
        for (int32_t i = 0; i < validRow; i++) {
            for (int32_t j = 0; j < loopNum; j++) {
                if (j < loopNum - 1) {
                    stats.emplace_back("vbitsort", REPEAT_MAX);
                    stats.emplace_back("pipe_barrier");
                } else {
                    // sort for last block
                    stats.emplace_back("vbitsort", srcTailRepeatNum - 1);
                    stats.emplace_back("pipe_barrier");

                    // copy row src cbuf to tmp cbuf
                    stats.emplace_back("copy_ubuf_to_ubuf", 1);
                    stats.emplace_back("pipe_barrier");

                    // dup -inf of tial value in tmp cbuf
                    stats.emplace_back("vector_dup", 1);
                    stats.emplace_back("pipe_barrier");

                    // sort for tmp and out to dst
                    stats.emplace_back("vbitsort", 1);
                    stats.emplace_back("pipe_barrier");
                }
            }
        }
    }

    return stats;
}

} // namespace pto
#endif // TSORT32_OP_HPP
