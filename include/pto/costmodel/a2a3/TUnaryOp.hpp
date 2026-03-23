/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TUNARYOP_HPP
#define TUNARYOP_HPP

#include <pto/common/constants.hpp>
#include "pto/costmodel/costmodel_types.hpp"

namespace pto {
#define SMALL_RPT (4)

template <typename Op>
PTO_INTERNAL void Unary1LCountMode(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    Op::UnaryInstr(stats, 0);
}

template <typename Op, unsigned nRepeatElem>
PTO_INTERNAL void Unary1LNormMode(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    unsigned nElem = validRow * validCol;
    unsigned headRepeats = nElem / nRepeatElem;
    unsigned tailElements = nElem % nRepeatElem;

    Op::UnaryInstr(stats, headRepeats);
    if (tailElements) {
        Op::UnaryInstr(stats, 1);
    }
}

template <typename Op>
PTO_INTERNAL void Unary2LCountMode(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    for (uint32_t i = 0; i < validRow; i++) {
        Op::UnaryInstr(stats, 0);
    }
}

template <typename Op, unsigned nRepeatElem>
PTO_INTERNAL void Unary2LNormModeColVLAlign(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    unsigned headRepeats = validCol / nRepeatElem;
    for (uint32_t i = 0; i < validRow; i++) {
        Op::UnaryInstr(stats, headRepeats);
    }
}

template <typename Op>
PTO_INTERNAL void Unary2LNormModeHead(std::vector<CostModelStats> &stats, unsigned validRow, unsigned nRepeatPerLine)
{
    if (nRepeatPerLine) {
        unsigned loop = nRepeatPerLine / REPEAT_MAX;
        unsigned remain = nRepeatPerLine % REPEAT_MAX;
        for (unsigned i = 0; i < validRow; i++) {
            for (unsigned j = 0; j < loop; j++) {
                Op::UnaryInstr(stats, REPEAT_MAX);
            }
            if (remain) {
                Op::UnaryInstr(stats, remain);
            }
        }
    }
}

template <typename Op, typename DstTile, typename SrcTile, unsigned blockSizeElem>
PTO_INTERNAL void Unary2LNormModeTail(std::vector<CostModelStats> &stats, unsigned validRow, unsigned nRemainPerLine)
{
    constexpr unsigned dstStride = DstTile::RowStride / blockSizeElem;
    constexpr unsigned srcStride = SrcTile::RowStride / blockSizeElem;
    unsigned loop = 0;
    unsigned remain = validRow;
    constexpr bool strideOverFlag = (dstStride > REPEAT_STRIDE_MAX || srcStride > REPEAT_STRIDE_MAX);
    if constexpr (DstTile::Rows > REPEAT_MAX || SrcTile::Rows > REPEAT_MAX) {
        loop = validRow / REPEAT_MAX;
        for (uint32_t i = 0; i < loop; i++) {
            if constexpr (strideOverFlag) {
                for (uint64_t j = 0; j < REPEAT_MAX; j++) {
                    Op::UnaryInstr(stats, 1);
                }
            } else {
                Op::UnaryInstr(stats, REPEAT_MAX);
            }
        }
        remain = validRow % REPEAT_MAX;
    }
    if (remain) {
        if constexpr (strideOverFlag) {
            for (uint32_t j = 0; j < remain; j++) {
                Op::UnaryInstr(stats, 1);
            }
        } else {
            Op::UnaryInstr(stats, remain);
        }
    }
}

template <typename Op, typename DstTile, typename SrcTile, unsigned nRepeatElem>
PTO_INTERNAL void Unary2LNormModeRowRpt(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    using T = typename DstTile::DType;
    constexpr unsigned blockSizeElem = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr unsigned dstStride = DstTile::RowStride / blockSizeElem;
    constexpr unsigned srcStride = SrcTile::RowStride / blockSizeElem;
    constexpr bool condRowRpt = ((DstTile::Rows <= REPEAT_MAX) && (dstStride <= REPEAT_STRIDE_MAX) &&
                                 (SrcTile::Rows <= REPEAT_MAX) && (srcStride <= REPEAT_STRIDE_MAX));
    if constexpr (condRowRpt) {
        unsigned loop = validCol / nRepeatElem;
        unsigned tailElements = validCol % nRepeatElem;
        for (uint32_t i = 0; i < loop; i++) {
            Op::UnaryInstr(stats, validRow);
        }
        if (tailElements) {
            Op::UnaryInstr(stats, validRow);
        }
    } else {
        unsigned nRepeatPerLine = validCol / nRepeatElem;
        unsigned remain = validCol % nRepeatElem;
        if constexpr (DstTile::Rows > nRepeatElem) {
            Unary2LNormModeHead<Op>(stats, validRow, nRepeatPerLine);
        }
        if (remain) {
            Unary2LNormModeTail<Op, DstTile, SrcTile, blockSizeElem>(stats, validRow, remain);
        }
    }
}

template <typename Op, typename DstTile, typename SrcTile, unsigned nRepeatElem>
PTO_INTERNAL void Unary2LProcess(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    constexpr unsigned normColRepeat = DstTile::Cols / nRepeatElem;
    if constexpr ((normColRepeat > 1) && ((DstTile::Rows * normColRepeat) < SMALL_RPT)) {
        Unary2LCountMode<Op>(stats, validRow, validCol);
    } else if constexpr (DstTile::Rows < (normColRepeat + 1)) {
        unsigned tailElements = validCol % nRepeatElem;
        if (tailElements) {
            Unary2LCountMode<Op>(stats, validRow, validCol);
        } else {
            Unary2LNormModeColVLAlign<Op, nRepeatElem>(stats, validRow, validCol);
        }
    } else {
        Unary2LNormModeRowRpt<Op, DstTile, SrcTile, nRepeatElem>(stats, validRow, validCol);
    }
}

template <typename Op, typename DstTile, typename SrcTile>
PTO_INTERNAL void TUnaryOp(std::vector<CostModelStats> &stats, unsigned validRow, unsigned validCol)
{
    using T = typename DstTile::DType;
    constexpr unsigned nRepeatElem = REPEAT_BYTE / sizeof(T);
    constexpr bool isCombined = ((DstTile::ValidCol == DstTile::Cols) && (SrcTile::ValidCol == SrcTile::Cols)) ||
                                ((DstTile::Rows == 1) && (SrcTile::Rows == 1));

    if constexpr (isCombined) {
        constexpr unsigned totalRepeats = (DstTile::Rows * DstTile::Cols + nRepeatElem - 1) / nRepeatElem;
        if constexpr (totalRepeats > REPEAT_MAX) {
            Unary1LCountMode<Op>(stats, validRow, validCol);
        } else {
            Unary1LNormMode<Op, nRepeatElem>(stats, validRow, DstTile::Cols);
        }
    } else {
        constexpr bool isSameShape = (DstTile::Cols == SrcTile::Cols) && (DstTile::Rows == SrcTile::Rows);
        if constexpr (isSameShape) {
            if ((validCol == DstTile::Cols) || (validRow == 1)) {
                unsigned totalRepeats = (validRow * validCol + nRepeatElem - 1) / nRepeatElem;
                if (totalRepeats > REPEAT_MAX) {
                    Unary1LCountMode<Op>(stats, validRow, validCol);
                } else {
                    Unary1LNormMode<Op, nRepeatElem>(stats, validRow, validCol);
                }
            } else {
                Unary2LProcess<Op, DstTile, SrcTile, nRepeatElem>(stats, validRow, validCol);
            }
        } else {
            Unary2LProcess<Op, DstTile, SrcTile, nRepeatElem>(stats, validRow, validCol);
        }
    }
}

template <typename Op, typename DstTile, typename SrcTile>
PTO_INTERNAL std::vector<CostModelStats> runUnaryOp(DstTile &dst, SrcTile &src)
{
    std::vector<CostModelStats> stats;
    unsigned dstValidRow = dst.GetValidRow();
    unsigned dstValidCol = dst.GetValidCol();
    if (dstValidRow == 0 || dstValidCol == 0) {
        return stats;
    }
    TUnaryOp<Op, DstTile, SrcTile>(stats, dstValidRow, dstValidCol);
    return stats;
}

} // namespace pto

#endif
