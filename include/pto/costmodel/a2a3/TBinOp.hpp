/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TBIN_HPP
#define TBIN_HPP

#include <pto/common/constants.hpp>
#include "pto/costmodel/costmodel_types.hpp"

namespace pto {
constexpr unsigned SMALL_RPT_BINOP = 4;

template <typename Op>
PTO_INTERNAL void Bin1LCountMode(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    Op::BinInstr(stats, 1);
}

template <typename Op>
PTO_INTERNAL void Bin2LCountMode(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    for (unsigned i = 0; i < validRow; i++) {
        Op::BinInstr(stats, 1);
    }
}

template <typename Op>
PTO_INTERNAL void Bin1LNormModeSmall(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    Op::BinInstr(stats, validRow);
}

template <typename Op, unsigned elementsPerRepeat>
PTO_INTERNAL void Bin1LNormMode(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    unsigned numElements = validRow * validCol;
    unsigned headRepeats = numElements / elementsPerRepeat;
    unsigned tailElements = numElements % elementsPerRepeat;
    Op::BinInstr(stats, headRepeats);
    if (tailElements)
        [[unlikely]]
        {
            Op::BinInstr(stats, 1);
        }
}

template <typename Op, unsigned elementsPerRepeat>
PTO_INTERNAL void Bin2LNormModeColVLAlign(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    unsigned headRepeats = validCol / elementsPerRepeat;
    for (unsigned i = 0; i < validRow; i++) {
        Op::BinInstr(stats, headRepeats);
    }
}

template <typename Op>
PTO_INTERNAL void Bin2LNormModeHead(std::vector<CostModelStats> &stats, unsigned validRow, unsigned numRepeatPerLine)
{
    if (numRepeatPerLine > 0) {
        unsigned loopNum = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (int i = 0; i < validRow; i++) {
            for (int j = 0; j < loopNum; j++) {
                Op::BinInstr(stats, REPEAT_MAX);
            }
            if (remainAfterLoop) {
                Op::BinInstr(stats, remainAfterLoop);
            }
        }
    }
}

template <typename Op, bool strideOverFlag, unsigned Rows>
PTO_INTERNAL void RecordTailLoopRepeats(std::vector<CostModelStats> &stats, unsigned validRow)
{
    unsigned loopNum = 0;
    unsigned remainAfterLoop = validRow;
    if constexpr (Rows > pto::REPEAT_MAX) {
        loopNum = validRow / REPEAT_MAX;
        for (int i = 0; i < loopNum; i++) {
            if constexpr (strideOverFlag) {
                for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                    Op::BinInstr(stats, 1);
                }
            } else {
                Op::BinInstr(stats, REPEAT_MAX);
            }
        }
        remainAfterLoop = validRow % REPEAT_MAX;
    }
    if (remainAfterLoop) {
        if constexpr (strideOverFlag) {
            for (unsigned j = 0; j < remainAfterLoop; j++) {
                Op::BinInstr(stats, 1);
            }
        } else {
            Op::BinInstr(stats, remainAfterLoop);
        }
    }
}

template <typename Op, unsigned elementsPerRepeat>
PTO_INTERNAL void RecordRowRptLoopRepeats(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    unsigned loopNum = validCol / elementsPerRepeat;
    unsigned tailElements = validCol % elementsPerRepeat;
    for (unsigned i = 0; i < loopNum; i++) {
        Op::BinInstr(stats, validRow);
    }
    if (tailElements) {
        Op::BinInstr(stats, validRow);
    }
}

template <typename Op, unsigned Rows, unsigned blockSizeElem, unsigned stride>
PTO_INTERNAL void Bin2LNormModeTail(std::vector<CostModelStats> &stats, unsigned validRow, unsigned numRemainPerLine)
{
    constexpr bool strideOverFlag = (stride / blockSizeElem > REPEAT_STRIDE_MAX);
    RecordTailLoopRepeats<Op, strideOverFlag, Rows>(stats, validRow);
}

template <typename Op, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL void Bin2LNormModeRowRpt(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    constexpr unsigned repeatStride = rowStride / blockSizeElem;
    constexpr bool condRowRpt = ((Rows <= pto::REPEAT_MAX) && (repeatStride <= REPEAT_STRIDE_MAX));
    if constexpr (condRowRpt) {
        RecordRowRptLoopRepeats<Op, elementsPerRepeat>(stats, validRow, validCol);
    } else {
        unsigned numRemainPerLine = validCol;
        if constexpr (Rows > elementsPerRepeat) {
            unsigned numRepeatPerLine = validCol / elementsPerRepeat;
            numRemainPerLine = validCol % elementsPerRepeat;
            Bin2LNormModeHead<Op>(stats, validRow, numRepeatPerLine);
        }
        if (numRemainPerLine) {
            Bin2LNormModeTail<Op, Rows, blockSizeElem, rowStride>(stats, validRow, numRemainPerLine);
        }
    }
}

template <typename Op, typename TileData, unsigned elementsPerRepeat>
PTO_INTERNAL void BinaryInstrFastPath(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    constexpr unsigned totalRepeats = (TileData::Rows * TileData::Cols + elementsPerRepeat - 1) / elementsPerRepeat;
    constexpr bool nonVLAligned = (((TileData::Cols % elementsPerRepeat) != 0) && (TileData::Cols > elementsPerRepeat));
    if constexpr (nonVLAligned || (totalRepeats > pto::REPEAT_MAX)) {
        Bin1LCountMode<Op>(stats, validRow, validCol);
    } else {
        Bin1LNormMode<Op, elementsPerRepeat>(stats, validRow, validCol);
    }
}

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL void BinaryInstrGeneralPath(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    // Continuous check in runtime(merge axis)
    if ((TileData::Cols == validCol) || (validRow == 1))
        [[likely]]
        {
            unsigned totalRepeats = (validRow * validCol + elementsPerRepeat - 1) / elementsPerRepeat;
            bool nonVLAligned = ((validCol > elementsPerRepeat) && ((validCol % elementsPerRepeat) != 0));
            if (nonVLAligned || (totalRepeats > pto::REPEAT_MAX))
                [[unlikely]]
                {
                    Bin1LCountMode<Op>(stats, validRow, validCol);
                }
            else {
                Bin1LNormMode<Op, elementsPerRepeat>(stats, validRow, validCol);
            }
        }
    else { // Non continuous
        constexpr unsigned normColRepeat = TileData::Cols / elementsPerRepeat;
        if constexpr ((normColRepeat > 1) && ((TileData::Rows * normColRepeat) < SMALL_RPT_BINOP)) {
            Bin2LCountMode<Op>(stats, validRow, validCol);
        } else if constexpr (TileData::Rows < (normColRepeat + 1)) {
            if ((validCol % elementsPerRepeat) > 0) {
                Bin2LCountMode<Op>(stats, validRow, validCol);
            } else {
                Bin2LNormModeColVLAlign<Op, elementsPerRepeat>(stats, validRow, validCol);
            }
        } else {
            Bin2LNormModeRowRpt<Op, TileData::Rows, elementsPerRepeat, blockSizeElem, rowStride>(stats, validRow,
                                                                                                 validCol);
        }
    }
}

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL void BinaryInstr(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    // Small shape optimization
    if constexpr ((TileData::Rows <= pto::REPEAT_MAX) && (TileData::Cols < elementsPerRepeat)) {
        Bin1LNormModeSmall<Op>(stats, validRow, validCol);
        return;
    }
    // Continuous check in compile time
    if constexpr ((TileData::Cols == TileData::ValidCol) || (TileData::Rows == 1)) {
        BinaryInstrFastPath<Op, TileData, elementsPerRepeat>(stats, validRow, validCol);
    } else {
        BinaryInstrGeneralPath<Op, TileData, elementsPerRepeat, blockSizeElem, rowStride>(stats, validRow, validCol);
    }
}

template <typename Op, unsigned elemPerBlk, unsigned dstStride, unsigned src0Stride, unsigned src1Stride>
PTO_INTERNAL void Bin2LNormModeTail(std::vector<CostModelStats> &stats, unsigned validRow, unsigned remain)
{
    unsigned loopNum = validRow / REPEAT_MAX;
    unsigned remainAfterLP = validRow % REPEAT_MAX;
    constexpr bool src0StrideOverFlag = (src0Stride / elemPerBlk > REPEAT_STRIDE_MAX);
    constexpr bool src1StrideOverFlag = (src1Stride / elemPerBlk > REPEAT_STRIDE_MAX);
    constexpr bool dstStrideOverFlag = (dstStride / elemPerBlk > REPEAT_STRIDE_MAX);
    for (int i = 0; i < loopNum; i++) {
        if constexpr (src0StrideOverFlag || src1StrideOverFlag || dstStrideOverFlag) {
            for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                Op::BinInstr(stats, 1);
            }
        } else {
            Op::BinInstr(stats, REPEAT_MAX);
        }
    }
    remainAfterLP = validRow % REPEAT_MAX;
    if (remainAfterLP) {
        if constexpr (src0StrideOverFlag || src1StrideOverFlag || dstStrideOverFlag) {
            for (unsigned j = 0; j < remainAfterLP; j++) {
                Op::BinInstr(stats, 1);
            }
        } else {
            Op::BinInstr(stats, remainAfterLP);
        }
    }
}

template <typename Op, unsigned elemPerRpt, unsigned elemPerBlk, unsigned dstStride, unsigned src0Stride,
          unsigned src1Stride>
PTO_INTERNAL void Bin2LNormModeRowRpt(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    unsigned rptPerLine = validCol / elemPerRpt;
    unsigned remain = validCol % elemPerRpt;
    Bin2LNormModeHead<Op>(stats, validRow, rptPerLine);
    if (remain) {
        Bin2LNormModeTail<Op, elemPerBlk, dstStride, src0Stride, src1Stride>(stats, validRow, remain);
    }
}

template <typename Op, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned dstRowStride,
          unsigned src0RowStride, unsigned src1RowStride>
PTO_INTERNAL void BinaryInstr(std::vector<CostModelStats> &stats, unsigned validRows, unsigned validCols)
{
    Bin2LNormModeRowRpt<Op, elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride, src1RowStride>(
        stats, validRows, validCols);
}

template <typename Op, typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned dstRowStride,
          unsigned src0RowStride = dstRowStride, unsigned src1RowStride = dstRowStride>
PTO_INTERNAL std::vector<CostModelStats> TBinaryOp(unsigned validRows, unsigned validCols)
{
    std::vector<CostModelStats> stats;
    if constexpr (dstRowStride == src0RowStride && dstRowStride == src1RowStride) {
        BinaryInstr<Op, TileData, elementsPerRepeat, blockSizeElem, dstRowStride>(stats, validRows, validCols);
    } else {
        BinaryInstr<Op, elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride, src1RowStride>(stats, validRows,
                                                                                                      validCols);
    }
    return stats;
}

template <typename Op, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL std::vector<CostModelStats> runBinaryOp(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    using T = typename TileDataDst::DType;
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);

    if constexpr (std::is_same_v<TileDataDst, TileDataSrc0> && std::is_same_v<TileDataDst, TileDataSrc1>) {
        constexpr unsigned dstRowStride = TileDataDst::RowStride;
        return TBinaryOp<Op, TileDataDst, elementsPerRepeat, blockSizeElem, dstRowStride>(dst.GetValidRow(),
                                                                                          dst.GetValidCol());
    } else {
        constexpr unsigned dstRowStride = TileDataDst::RowStride;
        constexpr unsigned src0RowStride = TileDataSrc0::RowStride;
        constexpr unsigned src1RowStride = TileDataSrc1::RowStride;

        return TBinaryOp<Op, TileDataDst, elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride, src1RowStride>(
            dst.GetValidRow(), dst.GetValidCol());
    }
}

} // namespace pto
#endif
