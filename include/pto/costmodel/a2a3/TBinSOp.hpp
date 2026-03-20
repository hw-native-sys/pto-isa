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
#include "pto/costmodel/a2a3/TBinOp.hpp"
namespace pto {
constexpr unsigned PTO_SMALL_RPT = 4;

PTO_INTERNAL void BinS1LCountMode(CostModelStats &stats, unsigned validRow, unsigned validCol)
{
    RecordCountMode(stats);
}

template <unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void BinS2LCountMode(CostModelStats &stats, unsigned validRow, unsigned validCol)
{
    for (unsigned i = 0; i < validRow; i++) {
        RecordCountMode(stats);
    }
}

template <unsigned elementsPerRepeat>
PTO_INTERNAL void BinS2LNormModeColVLAlign(CostModelStats &stats, unsigned validRow, unsigned validCol)
{
    unsigned headRepeats = validCol / elementsPerRepeat;
    for (uint32_t i = 0; i < validRow; i++) {
        RecordRepeat(stats, static_cast<uint8_t>(headRepeats));
    }
}

PTO_INTERNAL void BinS2LNormModeHead(CostModelStats &stats, unsigned validRow, unsigned numRepeatEachLine)
{
    if (numRepeatEachLine > 0) {
        unsigned loopNum = numRepeatEachLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatEachLine % REPEAT_MAX;
        for (int i = 0; i < validRow; i++) {
            if (loopNum)
                [[unlikely]]
                {
                    for (int j = 0; j < loopNum; j++) {
                        RecordRepeat(stats, REPEAT_MAX);
                    }
                }
            if (remainAfterLoop) {
                RecordRepeat(stats, static_cast<uint8_t>(remainAfterLoop));
            }
        }
    }
}

template <unsigned Rows, unsigned blockSizeElem, unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void BinS2LNormModeTail(CostModelStats &stats, unsigned validRow, unsigned numRemainPerLine)
{
    constexpr bool strideOverFlag =
        (dstStride / blockSizeElem > REPEAT_STRIDE_MAX) || (srcStride / blockSizeElem > REPEAT_STRIDE_MAX);
    RecordTailLoopRepeats<strideOverFlag, Rows>(stats, validRow);
}

template <unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void BinS2LNormModeRowRpt(CostModelStats &stats, unsigned validRow, unsigned validCol)
{
    constexpr unsigned dstRepeatStride = dstStride / blockSizeElem;
    constexpr unsigned srcRepeatStride = srcStride / blockSizeElem;
    constexpr bool condRowRpt =
        ((Rows <= pto::REPEAT_MAX) && dstRepeatStride <= (REPEAT_STRIDE_MAX) && srcRepeatStride <= (REPEAT_STRIDE_MAX));
    if constexpr (condRowRpt) {
        RecordRowRptLoopRepeats<elementsPerRepeat>(stats, validRow, validCol);
    } else {
        unsigned numRemainPerLine = validCol;
        if constexpr (Rows > elementsPerRepeat) {
            unsigned numRepeatEachLine = validCol / elementsPerRepeat;
            numRemainPerLine = validCol % elementsPerRepeat;
            BinS2LNormModeHead(stats, validRow, numRepeatEachLine);
        }
        if (numRemainPerLine) {
            BinS2LNormModeTail<Rows, blockSizeElem, dstStride, srcStride>(stats, validRow, numRemainPerLine);
        }
    }
}

template <typename TileDataDst, typename TileDataSrc, unsigned elementsPerRepeat, unsigned blockSizeElem,
          unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void TBinSInstrNonContinuousPath(CostModelStats &stats, unsigned validRow, unsigned validCol)
{
    constexpr unsigned normColRepeat = TileDataDst::Cols / elementsPerRepeat;
    constexpr bool countMode = (normColRepeat > 1) && ((TileDataDst::Rows * normColRepeat) < PTO_SMALL_RPT) &&
                               ((TileDataSrc::Rows * normColRepeat) < PTO_SMALL_RPT);
    constexpr bool isColRpt = (TileDataDst::Rows < (normColRepeat + 1)) && (TileDataSrc::Rows < (normColRepeat + 1));
    if constexpr (countMode) {
        BinS2LCountMode<dstStride, srcStride>(stats, validRow, validCol);
    } else if constexpr (isColRpt) {
        unsigned tailElements = validCol % elementsPerRepeat;
        if (tailElements) {
            BinS2LCountMode<dstStride, srcStride>(stats, validRow, validCol);
        } else {
            BinS2LNormModeColVLAlign<elementsPerRepeat>(stats, validRow, validCol);
        }
    } else {
        BinS2LNormModeRowRpt<TileDataDst::Rows, elementsPerRepeat, blockSizeElem, dstStride, srcStride>(stats, validRow,
                                                                                                        validCol);
    }
}

template <typename TileDataDst, typename TileDataSrc, unsigned elementsPerRepeat, unsigned blockSizeElem,
          unsigned dstStride, unsigned srcStride>
PTO_INTERNAL void TBinSInstr(CostModelStats &stats, unsigned validRow, unsigned validCol)
{
    constexpr bool tileDataPass =
        ((TileDataDst::Cols == TileDataDst::ValidCol) && (TileDataSrc::Cols == TileDataSrc::ValidCol)) ||
        ((TileDataDst::Rows == 1) && (TileDataSrc::Rows == 1));
    if constexpr (tileDataPass) {
        constexpr unsigned totalRepeats =
            (TileDataDst::Rows * TileDataDst::Cols + elementsPerRepeat - 1) / elementsPerRepeat;
        constexpr bool nonVLAligned =
            (((TileDataDst::Cols % elementsPerRepeat) != 0) && (TileDataDst::Cols > elementsPerRepeat));
        constexpr bool countMode = nonVLAligned || (totalRepeats > pto::REPEAT_MAX);
        if constexpr (countMode) {
            BinS1LCountMode(stats, validRow, validCol);
        } else {
            Bin1LNormMode<elementsPerRepeat>(stats, validRow, validCol);
        }
    } else {
        if (tileDataPass)
            [[likely]]
            {
                unsigned totalRepeats = (validRow * validCol + elementsPerRepeat - 1) / elementsPerRepeat;
                bool nonVLAligned = ((validCol > elementsPerRepeat) && ((validCol % elementsPerRepeat) != 0));
                bool countMode = nonVLAligned || (totalRepeats > pto::REPEAT_MAX);
                if (countMode)
                    [[unlikely]]
                    {
                        BinS1LCountMode(stats, validRow, validCol);
                    }
                else {
                    Bin1LNormMode<elementsPerRepeat>(stats, validRow, validCol);
                }
            }
        else {
            TBinSInstrNonContinuousPath<TileDataDst, TileDataSrc, elementsPerRepeat, blockSizeElem, dstStride,
                                        srcStride>(stats, validRow, validCol);
        }
    }
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL CostModelStats TBinaryScalarOp(unsigned validRow, unsigned validCol)
{
    using T = typename TileDataDst::DType;
    CostModelStats stats;
    constexpr unsigned elementsPerRepeat = pto::REPEAT_BYTE / sizeof(T);
    constexpr unsigned blockSizeElem = pto::BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned dstStride = TileDataDst::RowStride;
    constexpr unsigned srcStride = TileDataSrc::RowStride;
    TBinSInstr<TileDataDst, TileDataSrc, elementsPerRepeat, blockSizeElem, dstStride, srcStride>(stats, validRow,
                                                                                                 validCol);
    return stats;
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL CostModelStats runBinaryScalarOp(TileDataDst &dst, TileDataSrc &src)
{
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    if ((dstValidRow != 0 && dstValidCol != 0) &&
        (dstValidRow == src.GetValidRow() && dstValidCol == src.GetValidCol())) {
        return TBinaryScalarOp<TileDataDst, TileDataSrc>(dstValidRow, dstValidCol);
    } else {
        PTO_ASSERT(false, "TADDS: dstTile validRow/validCol must be consistent with of src");
        return {};
    }
}
} // namespace pto
#endif
