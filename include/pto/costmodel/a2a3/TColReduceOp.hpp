/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef T_COL_REDUCE_OPS_HPP
#define T_COL_REDUCE_OPS_HPP

#include <pto/common/type.hpp>

namespace pto {

template <typename InstrOp, typename T, typename TileDataOut, typename TileDataIn, unsigned srcstride>
PTO_INTERNAL void ColReduceInstr(std::vector<CostModelStats> &stats, int validRow, int validCol)
{
    using ReduceOp = TColReduceOp<InstrOp>;
    constexpr int DTypeSize = sizeof(T);
    int lenBurst = (validCol * DTypeSize + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;

    stats.emplace_back("copy_ubuf_to_ubuf", 1, lenBurst, 0, 0);

    pipe_barrier(PIPE_V);
    if (validRow == 1) {
        return;
    }

    constexpr int blockSizeElem = BLOCK_BYTE_SIZE / DTypeSize;
    constexpr int numBlockPerLine = (srcstride * DTypeSize + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;
    constexpr int dupSrcStride = numBlockPerLine * blockSizeElem;
    constexpr int elementsPerRepeat = REPEAT_BYTE / DTypeSize;
    int numRepeatPerLine = validCol / elementsPerRepeat;
    int numRemainPerLine = validCol % elementsPerRepeat;
    int elementsPerLine = numRepeatPerLine * elementsPerRepeat;

    ReduceOp::template ColReduceInstrByMode<dupSrcStride>(stats, numRepeatPerLine, numRemainPerLine, elementsPerLine,
                                                          validRow);
    stats.emplace_back("PIPE_V");
}

template <typename T, typename Op, typename TileDataOut, typename TileDataIn>
PTO_INTERNAL std::vector<CostModelStats> runColReduceOps(TileDataOut &dst, TileDataIn &src)
{
    std::vector<CostModelStats> stats;
    int ValidRow = src.GetValidRow();
    int ValidCol = src.GetValidCol();
    if (ValidRow == 0 || ValidCol == 0) {
        return stats;
    }
    constexpr int srcstride = TileDataIn::RowStride;
    ColReduceInstr<Op, T, TileDataOut, TileDataIn, srcstride>(stats, ValidRow, ValidCol);
    return stats;
}

template <typename T, int SrcStride, int DstStride>
PTO_INTERNAL void BinarySum(std::vector<CostModelStats> &stats, int validRow, int validCol)
{
    constexpr int elemPerRpt = REPEAT_BYTE / sizeof(T);
    int repeats = (validCol + elemPerRpt - 1) / elemPerRpt;
    stats.emplace_back("mask", 0, validCol);
    for (uint32_t i = 0; i < validRow / 2; i++) {
        stats.emplace_back("vadd", repeats, 1, 1, 1, 8, 8, 8);
    }
    stats.emplace_back("PIPE_V");

    if (validRow % 2 == 1) {
        stats.emplace_back("vadd", repeats, 1, 1, 1, 8, 8, 8);
        stats.emplace_back("PIPE_V");
    }
    stats.emplace_back("mask", -1, -1);
}

template <typename T, int SrcStride, int DstStride>
PTO_INTERNAL void SequentialSum(std::vector<CostModelStats> &stats, int validRow, int validCol)
{
    stats.emplace_back("mask", 0, validCol);
    for (int i = 1; i < validRow; i++) {
        stats.emplace_back("vadd", 0, 1, 1, 1, 8, 8, 8);
        stats.emplace_back("PIPE_V");
    }
    stats.emplace_back("mask", -1, -1);
}

template <typename T, typename TileDataDst, typename TileDataSrc, typename TileDataTmp, int srcStride, int dstStride,
          int tmpStride, bool IsBinary>
PTO_INTERNAL void TColSum(std::vector<CostModelStats> &stats, int validRow, int validCol)
{
    constexpr int DTypeSize = sizeof(T);
    int lenBurst = (validCol * DTypeSize + BLOCK_BYTE_SIZE - 1) / BLOCK_BYTE_SIZE;

    if (validRow == 1) {
        stats.emplace_back("copy_ubuf_to_ubuf", 1, lenBurst, 0, 0);
        stats.emplace_back("PIPE_V");
        return;
    }

    if (IsBinary) {
        BinarySum<T, srcStride, tmpStride>(stats, validRow, validCol);
        int cnt = validRow / 2;
        while (cnt > 1) {
            BinarySum<T, tmpStride, tmpStride>(stats, cnt, validCol);
            stats.emplace_back("PIPE_V");
            cnt /= 2;
        }

        stats.emplace_back("copy_ubuf_to_ubuf", 1, lenBurst, 0, 0);
        stats.emplace_back("PIPE_V");
    } else {
        stats.emplace_back("copy_ubuf_to_ubuf", 1, lenBurst, 0, 0);
        stats.emplace_back("PIPE_V");
        SequentialSum<T, srcStride, dstStride>(stats, validRow, validCol);
    }
}

template <typename T, typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
PTO_INTERNAL std::vector<CostModelStats> runColSumOp(TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp,
                                                     bool IsBinary)
{
    std::vector<CostModelStats> stats;
    int validRow = src.GetValidRow();
    int validCol = src.GetValidCol();
    constexpr int srcStride = TileDataSrc::RowStride;
    constexpr int dstStride = TileDataDst::RowStride;
    constexpr int tmpStride = TileDataTmp::RowStride * sizeof(typename TileDataTmp::DType) / sizeof(T);

    if (validRow == 0 || validCol == 0) {
        return stats;
    }
    if (IsBinary) {
        TColSum<T, TileDataDst, TileDataSrc, TileDataTmp, srcStride, dstStride, tmpStride, true>(stats, validRow,
                                                                                                 validCol);
    } else {
        TColSum<T, TileDataDst, TileDataSrc, TileDataTmp, srcStride, dstStride, tmpStride, false>(stats, validRow,
                                                                                                  validCol);
    }

    return stats;
}
} // namespace pto
#endif