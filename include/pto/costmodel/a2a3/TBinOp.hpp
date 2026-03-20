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

PTO_INTERNAL void Bin1LCountMode(CostModelStats &stats, unsigned validRow, unsigned validCol)
{
    RecordCountMode(stats);
}

PTO_INTERNAL void Bin2LCountMode(CostModelStats &stats, unsigned validRow, unsigned validCol)
{
    for (unsigned i = 0; i < validRow; i++) {
        RecordCountMode(stats);
    }
}

PTO_INTERNAL void Bin1LNormModeSmall(CostModelStats &stats, unsigned validRow, unsigned validCol)
{
    RecordRepeat(stats, static_cast<uint8_t>(validRow), false, true);
    return;
}

template <unsigned elementsPerRepeat>
PTO_INTERNAL void Bin1LNormMode(CostModelStats &stats, unsigned validRow, unsigned validCol)
{
    unsigned numElements = validRow * validCol;
    unsigned headRepeats = numElements / elementsPerRepeat;
    unsigned tailElements = numElements % elementsPerRepeat;
    RecordRepeat(stats, static_cast<uint8_t>(headRepeats));
    if (tailElements)
        [[unlikely]]
        {
            RecordRepeat(stats, 1, true);
        }
}

template <unsigned elementsPerRepeat>
PTO_INTERNAL void Bin2LNormModeColVLAlign(CostModelStats &stats, unsigned validRow, unsigned validCol)
{
    unsigned headRepeats = validCol / elementsPerRepeat;
    for (unsigned i = 0; i < validRow; i++) {
        RecordRepeat(stats, static_cast<uint8_t>(headRepeats));
    }
}

PTO_INTERNAL void Bin2LNormModeHead(CostModelStats &stats, unsigned validRow, unsigned numRepeatPerLine)
{
    if (numRepeatPerLine > 0) {
        unsigned loopCount = numRepeatPerLine / REPEAT_MAX;
        unsigned remainAfterLoop = numRepeatPerLine % REPEAT_MAX;
        for (int i = 0; i < validRow; i++) {
            if (loopCount)
                [[unlikely]]
                {
                    for (int j = 0; j < loopCount; j++) {
                        RecordRepeat(stats, REPEAT_MAX);
                    }
                }
            if (remainAfterLoop) {
                RecordRepeat(stats, static_cast<uint8_t>(remainAfterLoop));
            }
        }
    }
}

template <bool strideOverFlag, unsigned Rows>
PTO_INTERNAL void RecordTailLoopRepeats(CostModelStats &stats, unsigned validRow)
{
    unsigned loopCount = 0;
    unsigned remainAfterLoop = validRow;
    if constexpr (Rows > pto::REPEAT_MAX) {
        loopCount = validRow / REPEAT_MAX;
        for (int i = 0; i < loopCount; i++) {
            if constexpr (strideOverFlag) {
                for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                    RecordRepeat(stats, 1, true, true);
                }
            } else {
                RecordRepeat(stats, REPEAT_MAX, true, true);
            }
        }
        remainAfterLoop = validRow % REPEAT_MAX;
    }
    if (remainAfterLoop) {
        if constexpr (strideOverFlag) {
            for (unsigned j = 0; j < remainAfterLoop; j++) {
                RecordRepeat(stats, 1, true, true);
            }
        } else {
            RecordRepeat(stats, static_cast<uint8_t>(remainAfterLoop), true, true);
        }
    }
}

template <unsigned elementsPerRepeat>
PTO_INTERNAL void RecordRowRptLoopRepeats(CostModelStats &stats, unsigned validRow, unsigned validCol)
{
    unsigned loopCount = validCol / elementsPerRepeat;
    unsigned tailElements = validCol % elementsPerRepeat;
    for (unsigned i = 0; i < loopCount; i++) {
        RecordRepeat(stats, static_cast<uint8_t>(validRow), false, true);
    }
    if (tailElements) {
        RecordRepeat(stats, static_cast<uint8_t>(validRow), true, true);
    }
}

template <unsigned Rows, unsigned blockSizeElem, unsigned stride>
PTO_INTERNAL void Bin2LNormModeTail(CostModelStats &stats, unsigned validRow, unsigned numRemainPerLine)
{
    constexpr bool strideOverFlag = (stride / blockSizeElem > REPEAT_STRIDE_MAX);
    RecordTailLoopRepeats<strideOverFlag, Rows>(stats, validRow);
}

template <unsigned Rows, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL void Bin2LNormModeRowRpt(CostModelStats &stats, unsigned validRow, unsigned validCol)
{
    constexpr unsigned repeatStride = rowStride / blockSizeElem;
    constexpr bool condRowRpt = ((Rows <= pto::REPEAT_MAX) && (repeatStride <= REPEAT_STRIDE_MAX));
    if constexpr (condRowRpt) {
        RecordRowRptLoopRepeats<elementsPerRepeat>(stats, validRow, validCol);
    } else {
        unsigned numRemainPerLine = validCol;
        if constexpr (Rows > elementsPerRepeat) {
            unsigned numRepeatPerLine = validCol / elementsPerRepeat;
            numRemainPerLine = validCol % elementsPerRepeat;
            Bin2LNormModeHead(stats, validRow, numRepeatPerLine);
        }
        if (numRemainPerLine) {
            Bin2LNormModeTail<Rows, blockSizeElem, rowStride>(stats, validRow, numRemainPerLine);
        }
    }
}

template <typename TileData, unsigned elementsPerRepeat>
PTO_INTERNAL void BinaryInstrFastPath(CostModelStats &stats, unsigned validRow, unsigned validCol)
{
    constexpr unsigned totalRepeats = (TileData::Rows * TileData::Cols + elementsPerRepeat - 1) / elementsPerRepeat;
    constexpr bool nonVLAligned = (((TileData::Cols % elementsPerRepeat) != 0) && (TileData::Cols > elementsPerRepeat));
    if constexpr (nonVLAligned || (totalRepeats > pto::REPEAT_MAX)) {
        Bin1LCountMode(stats, validRow, validCol);
    } else {
        Bin1LNormMode<elementsPerRepeat>(stats, validRow, validCol);
    }
}

template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL void BinaryInstrGeneralPath(CostModelStats &stats, unsigned validRow, unsigned validCol)
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
                    Bin1LCountMode(stats, validRow, validCol);
                }
            else {
                Bin1LNormMode<elementsPerRepeat>(stats, validRow, validCol);
            }
        }
    else { // Non continuous
        constexpr unsigned normColRepeat = TileData::Cols / elementsPerRepeat;
        if constexpr ((normColRepeat > 1) && ((TileData::Rows * normColRepeat) < SMALL_RPT_BINOP)) {
            Bin2LCountMode(stats, validRow, validCol);
        } else if constexpr (TileData::Rows < (normColRepeat + 1)) {
            if ((validCol % elementsPerRepeat) > 0) {
                Bin2LCountMode(stats, validRow, validCol);
            } else {
                Bin2LNormModeColVLAlign<elementsPerRepeat>(stats, validRow, validCol);
            }
        } else {
            Bin2LNormModeRowRpt<TileData::Rows, elementsPerRepeat, blockSizeElem, rowStride>(stats, validRow, validCol);
        }
    }
}

template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned rowStride>
PTO_INTERNAL void BinaryInstr(CostModelStats &stats, unsigned validRow, unsigned validCol)
{
    // Small shape optimization
    if constexpr ((TileData::Rows <= pto::REPEAT_MAX) && (TileData::Cols < elementsPerRepeat)) {
        Bin1LNormModeSmall(stats, validRow, validCol);
        return;
    }
    // Continuous check in compile time
    if constexpr ((TileData::Cols == TileData::ValidCol) || (TileData::Rows == 1)) {
        BinaryInstrFastPath<TileData, elementsPerRepeat>(stats, validRow, validCol);
    } else {
        BinaryInstrGeneralPath<TileData, elementsPerRepeat, blockSizeElem, rowStride>(stats, validRow, validCol);
    }
}

template <unsigned elemPerBlk, unsigned dstStride, unsigned src0Stride, unsigned src1Stride>
PTO_INTERNAL void Bin2LNormModeTail(CostModelStats &stats, unsigned validRow, unsigned remain)
{
    unsigned loopCount = validRow / REPEAT_MAX;
    unsigned remainAfterLoop = validRow % REPEAT_MAX;
    constexpr bool src0StrideOverFlag = (src0Stride / elemPerBlk > REPEAT_STRIDE_MAX);
    constexpr bool src1StrideOverFlag = (src1Stride / elemPerBlk > REPEAT_STRIDE_MAX);
    constexpr bool dstStrideOverFlag = (dstStride / elemPerBlk > REPEAT_STRIDE_MAX);
    for (int i = 0; i < loopCount; i++) {
        if constexpr (src0StrideOverFlag || src1StrideOverFlag || dstStrideOverFlag) {
            for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                RecordRepeat(stats, 1, true, true);
            }
        } else {
            RecordRepeat(stats, REPEAT_MAX, true, true);
        }
    }
    remainAfterLoop = validRow % REPEAT_MAX;
    if (remainAfterLoop) {
        if constexpr (src0StrideOverFlag || src1StrideOverFlag || dstStrideOverFlag) {
            for (unsigned j = 0; j < remainAfterLoop; j++) {
                RecordRepeat(stats, 1, true, true);
            }
        } else {
            RecordRepeat(stats, static_cast<uint8_t>(remainAfterLoop), true, true);
        }
    }
}

template <unsigned elemPerRpt, unsigned elemPerBlk, unsigned dstStride, unsigned src0Stride, unsigned src1Stride>
PTO_INTERNAL void Bin2LNormModeRowRpt(CostModelStats &stats, unsigned validRow, unsigned validCol)
{
    unsigned rptPerLine = validCol / elemPerRpt;
    unsigned remain = validCol % elemPerRpt;
    Bin2LNormModeHead(stats, validRow, rptPerLine);
    if (remain) {
        Bin2LNormModeTail<elemPerBlk, dstStride, src0Stride, src1Stride>(stats, validRow, remain);
    }
}

template <unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned dstRowStride, unsigned src0RowStride,
          unsigned src1RowStride>
PTO_INTERNAL void BinaryInstr(CostModelStats &stats, unsigned validRows, unsigned validCols)
{
    Bin2LNormModeRowRpt<elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride, src1RowStride>(stats, validRows,
                                                                                                      validCols);
}

template <typename TileData, unsigned elementsPerRepeat, unsigned blockSizeElem, unsigned dstRowStride,
          unsigned src0RowStride = dstRowStride, unsigned src1RowStride = dstRowStride>
PTO_INTERNAL CostModelStats TBinaryOp(unsigned validRows, unsigned validCols)
{
    CostModelStats stats;
    if constexpr (dstRowStride == src0RowStride && dstRowStride == src1RowStride) {
        BinaryInstr<TileData, elementsPerRepeat, blockSizeElem, dstRowStride>(stats, validRows, validCols);
    } else {
        BinaryInstr<elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride, src1RowStride>(stats, validRows,
                                                                                                  validCols);
    }
    return stats;
}

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL CostModelStats runBinaryOp(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
{
    using T = typename TileDataDst::DType;
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);

    if constexpr (std::is_same_v<TileDataDst, TileDataSrc0> && std::is_same_v<TileDataDst, TileDataSrc1>) {
        constexpr unsigned dstRowStride = TileDataDst::RowStride;
        return TBinaryOp<TileDataDst, elementsPerRepeat, blockSizeElem, dstRowStride>(dst.GetValidRow(),
                                                                                      dst.GetValidCol());
    } else {
        constexpr unsigned dstRowStride = TileDataDst::RowStride;
        constexpr unsigned src0RowStride = TileDataSrc0::RowStride;
        constexpr unsigned src1RowStride = TileDataSrc1::RowStride;

        return TBinaryOp<TileDataDst, elementsPerRepeat, blockSizeElem, dstRowStride, src0RowStride, src1RowStride>(
            dst.GetValidRow(), dst.GetValidCol());
    }
}

} // namespace pto
#endif
