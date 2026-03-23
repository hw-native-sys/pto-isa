/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef T_ROW_REDUCE_OPS_HPP
#define T_ROW_REDUCE_OPS_HPP

#include <pto/common/type.hpp>

namespace pto {

template <typename InstrOp, typename T, uint32_t DstCols, uint32_t SrcCols, uint8_t elemPerRpt, uint32_t dstRptStride,
          uint32_t srcRptStride>
PTO_INTERNAL void OneRepeatProc(std::vector<CostModelStats> &stats, int validCol, int validRow, int remain,
                                int rowRptTimes)
{
    if (validCol == elemPerRpt) {
        InstrOp::template ReduceInstrByMode<true, SrcCols, dstRptStride, srcRptStride, elemPerRpt>(stats, validRow);
        stats.emplace_back("PIPE_V");
        return;
    }

    unsigned rptTimes;
    stats.emplace_back("mask", GetContinuousMask1(remain), GetContinuousMask0(remain));
    do {
        rptTimes = rowRptTimes == 0 ? (validRow % REPEAT_MAX) : REPEAT_MAX;
        InstrOp::template ReduceInstrByMode<false, SrcCols, dstRptStride, srcRptStride, elemPerRpt>(stats, rptTimes);
        stats.emplace_back("PIPE_V");
        rowRptTimes -= 1;
    } while (rowRptTimes >= 0);

    stats.emplace_back("mask", -1, -1);
}

template <typename InstrOp, typename T, typename TileOut, typename TileIn>
PTO_INTERNAL bool TryOptimizeFP32Reduce(std::vector<CostModelStats> &stats)
{
    if constexpr (!TileOut::isBoxedLayout && !TileOut::isRowMajor && TileOut::ValidCol == 1) {
        if constexpr (std::is_same_v<T, float>) {
            constexpr bool ShapeOf64x128 =
                TileIn::Rows == 64 && TileIn::ValidRow == 64 && TileIn::Cols == 128 && TileIn::ValidCol == 128;
            constexpr bool ShapeOf32x256 =
                TileIn::Rows == 32 && TileIn::ValidRow == 32 && TileIn::Cols == 256 && TileIn::ValidCol == 256;
            constexpr bool ShapeOf16x512 =
                TileIn::Rows == 16 && TileIn::ValidRow == 16 && TileIn::Cols == 512 && TileIn::ValidCol == 512;
            constexpr bool ShapeOf8x1024 =
                TileIn::Rows == 8 && TileIn::ValidRow == 8 && TileIn::Cols == 1024 && TileIn::ValidCol == 1024;
            if constexpr (ShapeOf64x128) {
                InstrOp::template ReduceOptFP32_64x128<TileIn::Rows, TileIn::ValidRow, TileIn::Cols, TileIn::ValidCol>(
                    stats);
                return true;
            } else if constexpr (ShapeOf32x256) {
                InstrOp::template ReduceOptFP32_32x256<TileIn::Rows, TileIn::ValidRow, TileIn::Cols, TileIn::ValidCol>(
                    stats);
                return true;
            } else if constexpr (ShapeOf16x512) {
                InstrOp::template ReduceOptFP32_16x512<TileIn::Rows, TileIn::ValidRow, TileIn::Cols, TileIn::ValidCol>(
                    stats);
                return true;
            } else if constexpr (ShapeOf8x1024) {
                InstrOp::template ReduceOptFP32_8x1024<TileIn::Rows, TileIn::ValidRow, TileIn::Cols, TileIn::ValidCol>(
                    stats);
                return true;
            }
        }
    }
    return false;
}

template <typename InstrOp, typename T, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL void TRowReduceInstr(std::vector<CostModelStats> &stats, int validCol, int validRow)
{
    if (TryOptimizeFP32Reduce<InstrOp, T, TileDataOut, TileDataIn>(stats)) {
        return;
    }
    constexpr uint8_t elemPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr uint8_t elemPerRpt = REPEAT_BYTE / sizeof(T);
    constexpr uint32_t dstRptStride = TileDataOut::Cols;
    constexpr uint32_t srcRptStride = TileDataIn::Cols / elemPerBlock;
    constexpr uint32_t tmpRptStride = TileDataTmp::Cols / elemPerBlock;
    int srcRptPerRow = validCol / elemPerRpt;
    int remain = validCol % elemPerRpt;
    int rowRptTimes = validRow / REPEAT_MAX; // 需要处理的行若超过uint8_max, 则拆分为多次进行循环
    unsigned rptTimes;

    if (validCol <= elemPerRpt) {
        OneRepeatProc<InstrOp, T, TileDataOut::Cols, TileDataIn::Cols, elemPerRpt, dstRptStride, srcRptStride>(
            stats, validCol, validRow, remain, rowRptTimes);
        return;
    }

    if (validCol < 2 * elemPerRpt) {
        // 解决 ccec 编译检查问题； 如果删除会导致copy_ubuf_to_ubuf编译错误，提醒第六、七个参数的范围必须是[0, 65535]
        if constexpr ((srcRptStride < BLOCK_MAX_PER_REPEAT) || (tmpRptStride < BLOCK_MAX_PER_REPEAT)) {
            return;
        }
        // 将满足一次repeat部分copy到dst
        stats.emplace_back("copy_ubuf_to_ubuf", validRow, BLOCK_MAX_PER_REPEAT, srcRptStride - BLOCK_MAX_PER_REPEAT,
                           tmpRptStride - BLOCK_MAX_PER_REPEAT);
        stats.emplace_back("PIPE_V");
    }

    InstrOp::template FillTmp<TileDataTmp::Cols, TileDataIn::Cols, tmpRptStride, srcRptStride, elemPerRpt>(
        stats, srcRptPerRow, validRow, validCol);

    // 不足一次repeat的部分设置mask与tmp计算, 此时tmp必定存在有效数据
    if (remain > 0) {
        stats.emplace_back("mask", GetContinuousMask1(remain), GetContinuousMask0(remain));
        do {
            rptTimes = rowRptTimes == 0 ? (validRow % REPEAT_MAX) : REPEAT_MAX;
            InstrOp::template BinInstrByMode<false, TileDataTmp::Cols, TileDataTmp::Cols, TileDataIn::Cols,
                                             tmpRptStride, tmpRptStride, srcRptStride, elemPerRpt>(stats, rptTimes);
            rowRptTimes -= 1;
        } while (rowRptTimes >= 0);
        stats.emplace_back("mask", -1, -1);
        stats.emplace_back("PIPE_V");
    }

    InstrOp::template TmpProc<TileDataTmp::Cols, TileDataIn::Cols, tmpRptStride, srcRptStride, elemPerRpt>(
        stats, srcRptPerRow, validRow);

    InstrOp::template ReduceInstrByMode<true, TileDataTmp::Cols, dstRptStride, tmpRptStride, elemPerRpt>(stats,
                                                                                                         validRow);
    stats.emplace_back("PIPE_V");
}

template <typename T, typename Op, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL std::vector<CostModelStats> TRowReduce(const std::string &instr_name, int validCol, int validRow)
{
    std::vector<CostModelStats> stats;
    if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, int16_t>) {
        // Integer implementation (follows TROWPROD pattern)
        constexpr unsigned dstRowStride = TileDataOut::RowStride;
        constexpr unsigned srcRowStride = TileDataIn::RowStride;
        constexpr unsigned tmpRowStride = TileDataTmp::RowStride;
        constexpr unsigned elemsPerBlock = BLOCK_BYTE_SIZE / sizeof(T);
        unsigned blocksPerRow = validCol / elemsPerBlock;

        for (unsigned row = 0; row < validRow;) {
            for (unsigned block = 0; block < blocksPerRow; ++block) {
                if (instr_name == "TROWMAX") {
                    stats.emplace_back("vmax", 1, 0, 0, 1, 0, 0, 1);
                } else if (instr_name == "TROWMIN") {
                    stats.emplace_back("vmin", 1, 0, 0, 1, 0, 0, 1);
                } else if (instr_name == "TROWSUM") {
                    stats.emplace_back("vadd", 1, 0, 0, 1, 0, 0, 1);
                }
            }

            unsigned elemsLessThanBlock = validCol % elemsPerBlock;
            if (elemsLessThanBlock > 0) {
                if (instr_name == "TROWMAX") {
                    stats.emplace_back("vmax", 1, 0, 0, 1, 0, 0, 1);
                } else if (instr_name == "TROWMIN") {
                    stats.emplace_back("vmin", 1, 0, 0, 1, 0, 0, 1);
                } else if (instr_name == "TROWSUM") {
                    stats.emplace_back("vadd", 1, 0, 0, 1, 0, 0, 1);
                }
                stats.emplace_back("PIPE_V");
            }
        }
    } else {
        // Float/Half implementation (original vcmax-based)
        TRowReduceInstr<Op, T, TileDataOut, TileDataIn, TileDataTmp>(stats, validCol, validRow);
    }

    return stats;
}

template <typename T, typename Op, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
PTO_INTERNAL std::vector<CostModelStats> runRowReduceOps(const std::string &instr_name, TileDataOut &dst,
                                                         TileDataIn &src, TileDataTmp &tmp)
{
    int validCol = src.GetValidCol();
    int validRow = src.GetValidRow();
    std::vector<CostModelStats> stats;
    if (validCol == 0 || validRow == 0) {
        return stats;
    }

    return TRowReduce<T, Op, TileDataOut, TileDataIn, TileDataTmp>(instr_name, validCol, validRow);
}

} // namespace pto

#endif