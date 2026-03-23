/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TBINS_HPP
#define TBINS_HPP

#include <pto/common/constants.hpp>
#include "pto/costmodel/costmodel_types.hpp"

namespace pto {
constexpr unsigned PTO_SMALL_RPT = 4;

template <typename Op>
PTO_INTERNAL void BinS1LCountMode(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    Op::BinSInstr(stats, 0);
}

template <typename Op, unsigned elementsPerRepeat>
PTO_INTERNAL void BinS1LNormMode(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    unsigned numElements = validRow * validCol;
    unsigned headRepeats = numElements / elementsPerRepeat;
    unsigned tailElements = numElements % elementsPerRepeat;
    Op::BinSInstr(stats, headRepeats);
    if (tailElements)
        [[unlikely]]
        {
            Op::BinSInstr(stats, 1);
        }
}

template <typename Op>
PTO_INTERNAL void BinS2LCountMode(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    for (unsigned i = 0; i < validRow; i++) {
        Op::BinSInstr(stats, 0);
    }
}

template <typename Op, unsigned elementsPerRepeat>
PTO_INTERNAL void BinS2LNormModeColVLAlign(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    unsigned headRepeats = validCol / elementsPerRepeat;
    for (uint32_t i = 0; i < validRow; i++) {
        Op::BinSInstr(stats, headRepeats);
    }
}

template <typename Op>
PTO_INTERNAL void BinS2LNormModeHead(std::vector<CostModelStats> &stats, unsigned validRow, unsigned numRepeatPerLine)
{
    if (numRepeatPerLine > 0) {
        unsigned numLoop = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (int i = 0; i < validRow; i++) {
            if (numLoop)
                [[unlikely]]
                {
                    for (int j = 0; j < numLoop; j++) {
                        Op::BinSInstr(stats, REPEAT_MAX);
                    }
                }
            if (remainAfterLoop) {
                Op::BinSInstr(stats, remainAfterLoop);
            }
        }
    }
}

template <typename Op, unsigned Rows, unsigned blockSizeElem, unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void BinS2LNormModeTail(std::vector<CostModelStats> &stats, unsigned validRow, unsigned numRemainPerLine)
{
    constexpr bool strideOverFlag =
        (dstStride / blockSizeElem > REPEAT_STRIDE_MAX) || (srcStride / blockSizeElem > REPEAT_STRIDE_MAX);
    unsigned loopNum = 0;
    unsigned remainAfterLoop = validRow;
    if constexpr (Rows > pto::REPEAT_MAX) {
        loopNum = validRow / REPEAT_MAX;
        for (int i = 0; i < loopNum; i++) {
            if constexpr (strideOverFlag) {
                for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                    Op::BinSInstr(stats, 1);
                }
            } else {
                Op::BinSInstr(stats, REPEAT_MAX);
            }
        }
        remainAfterLoop = validRow % REPEAT_MAX;
    }
    if (remainAfterLoop) {
        if constexpr (strideOverFlag) {
            for (unsigned j = 0; j < remainAfterLoop; j++) {
                Op::BinSInstr(stats, 1);
            }
        } else {
            Op::BinSInstr(stats, remainAfterLoop);
        }
    }
}

template <typename Op, unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned dstStride,
          unsigned srcStride>
PTO_INTERNAL void BinS2LNormModeRowRpt(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    constexpr unsigned dstRepeatStride = dstStride / blockSizeElem;
    constexpr unsigned srcRepeatStride = srcStride / blockSizeElem;
    constexpr bool condRowRpt =
        ((Rows <= pto::REPEAT_MAX) && (dstRepeatStride <= REPEAT_STRIDE_MAX) && (srcRepeatStride <= REPEAT_STRIDE_MAX));
    if constexpr (condRowRpt) {
        unsigned loopNum = validCol / elementsPerRepeat;
        unsigned tailElements = validCol % elementsPerRepeat;
        for (unsigned i = 0; i < loopNum; i++) {
            Op::BinSInstr(stats, validRow);
        }
        if (tailElements) {
            Op::BinSInstr(stats, validRow);
        }
    } else {
        unsigned numRemainPerLine = validCol;
        if constexpr (Rows > elementsPerRepeat) {
            unsigned numRepeatPerLine = validCol / elementsPerRepeat;
            numRemainPerLine = validCol % elementsPerRepeat;
            BinS2LNormModeHead<Op>(stats, validRow, numRepeatPerLine);
        }
        if (numRemainPerLine) {
            BinS2LNormModeTail<Op, Rows, blockSizeElem, dstStride, srcStride>(stats, validRow, numRemainPerLine);
        }
    }
}

template <typename Op, typename TileDataDst, typename TileDataSrc, unsigned elementsPerRepeat, unsigned blockSizeElem,
          unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void TBinSInstrNonContinuousPath(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    constexpr unsigned normColRepeat = TileDataDst::Cols / elementsPerRepeat;
    constexpr bool countMode = (normColRepeat > 1) && ((TileDataDst::Rows * normColRepeat) < PTO_SMALL_RPT) &&
                               ((TileDataSrc::Rows * normColRepeat) < PTO_SMALL_RPT);
    constexpr bool isColRpt = (TileDataDst::Rows < (normColRepeat + 1)) && (TileDataSrc::Rows < (normColRepeat + 1));
    if constexpr (countMode) {
        BinS2LCountMode<Op>(stats, validRow, validCol);
    } else if constexpr (isColRpt) {
        unsigned tailElements = validCol % elementsPerRepeat;
        if (tailElements) {
            BinS2LCountMode<Op>(stats, validRow, validCol);
        } else {
            BinS2LNormModeColVLAlign<Op, elementsPerRepeat>(stats, validRow, validCol);
        }
    } else {
        BinS2LNormModeRowRpt<Op, TileDataDst::Rows, elementsPerRepeat, blockSizeElem, dstStride, srcStride>(
            stats, validRow, validCol);
    }
}

template <typename Op, typename TileDataDst, typename TileDataSrc, unsigned elementsPerRepeat, unsigned blockSizeElem,
          unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void TBinSInstr(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    constexpr bool tileDataContinue =
        ((TileDataDst::Cols == TileDataDst::ValidCol) && (TileDataSrc::Cols == TileDataSrc::ValidCol)) ||
        ((TileDataDst::Rows == 1) && (TileDataSrc::Rows == 1));
    if constexpr (tileDataContinue) {
        constexpr unsigned totalRepeats =
            (TileDataDst::Rows * TileDataDst::Cols + elementsPerRepeat - 1) / elementsPerRepeat;
        constexpr bool nonVLAligned =
            (((TileDataDst::Cols % elementsPerRepeat) != 0) && (TileDataDst::Cols > elementsPerRepeat));
        constexpr bool enbleCountMode = nonVLAligned || (totalRepeats > pto::REPEAT_MAX);
        if constexpr (enbleCountMode) {
            BinS1LCountMode<Op>(stats, validRow, validCol);
        } else {
            BinS1LNormMode<Op, elementsPerRepeat>(stats, validRow, validCol);
        }
    } else {
        TBinSInstrNonContinuousPath<Op, TileDataDst, TileDataSrc, elementsPerRepeat, blockSizeElem, dstStride,
                                    srcStride>(stats, validRow, validCol);
    }
}

template <typename Op, typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL std::vector<CostModelStats> TBinaryScalarOp(unsigned validRow, unsigned validCol)
{
    using T = typename TileDataDst::DType;
    std::vector<CostModelStats> stats;
    constexpr unsigned elementsPerRepeat = pto::REPEAT_BYTE / sizeof(T);
    constexpr unsigned blockSizeElem = pto::BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned dstStride = TileDataDst::RowStride;
    constexpr unsigned srcStride = TileDataSrc::RowStride;
    TBinSInstr<Op, TileDataDst, TileDataSrc, elementsPerRepeat, blockSizeElem, dstStride, srcStride>(stats, validRow,
                                                                                                     validCol);
    return stats;
}

template <typename Op, typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL std::vector<CostModelStats> runBinaryScalarOp(TileDataDst &dst, TileDataSrc &src)
{
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    if ((dstValidRow != 0 && dstValidCol != 0) &&
        (dstValidRow == src.GetValidRow() && dstValidCol == src.GetValidCol())) {
        return TBinaryScalarOp<Op, TileDataDst, TileDataSrc>(dstValidRow, dstValidCol);
    } else {
        PTO_ASSERT(false, "TADDS: dstTile validRow/validCol must be consistent with of src");
        return {};
    }
}

} // namespace pto
#endif
