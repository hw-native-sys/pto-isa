/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TINSERT_HPP
#define TINSERT_HPP

#include <cassert>
#include "common.hpp"

namespace pto {

template <typename DstTileData, typename SrcTileData, bool applyRelu>
PTO_INTERNAL void TInsert_Impl(DstTileData &dst, SrcTileData &src, uint32_t idxRow, uint32_t idxCol)
{
    assert(src.GetValidRow() + idxRow <= dst.GetValidRow() && src.GetValidCol() + idxCol <= dst.GetValidCol());

    using D = typename DstTileData::DType;
    using S = typename SrcTileData::DType;

    for (size_t c = 0; c < src.GetValidCol(); c++) {
        const size_t subTileSrcC = c / SrcTileData::InnerCols;
        const size_t innerSrcC = c % SrcTileData::InnerCols;
        const size_t cDst = c + idxCol;
        const size_t subTileDstC = cDst / DstTileData::InnerCols;
        const size_t innerDstC = cDst % DstTileData::InnerCols;

        for (size_t r = 0; r < src.GetValidRow(); r++) {
            size_t srcTileIdx;
            size_t dstTileIdx;
            if constexpr (SrcTileData::SFractal == SLayout::NoneBox) {
                srcTileIdx = GetTileElementOffsetPlain<SrcTileData>(r, c);
            } else {
                const size_t subTileR = r / SrcTileData::InnerRows;
                const size_t innerR = r % SrcTileData::InnerRows;
                srcTileIdx = GetTileElementOffsetSubfractals<SrcTileData>(subTileR, innerR, subTileSrcC, innerSrcC);
            }
            const size_t rDst = r + idxRow;

            if constexpr (DstTileData::SFractal == SLayout::NoneBox) {
                dstTileIdx = GetTileElementOffsetPlain<DstTileData>(rDst, cDst);
            } else {
                const size_t subTileR = rDst / DstTileData::InnerRows;
                const size_t innerR = rDst % DstTileData::InnerRows;
                dstTileIdx = GetTileElementOffsetSubfractals<DstTileData>(subTileR, innerR, subTileDstC, innerDstC);
            }
            S val = src.data()[srcTileIdx];
            if constexpr (applyRelu) {
                val = ReLU(val);
            }
            dst.data()[dstTileIdx] = val;
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint32_t idxRow = 0, uint32_t idxCol = 0)
{
    TInsert_Impl<DstTileData, SrcTileData, false>(dst, src, idxRow, idxCol);
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    TInsert_Impl<DstTileData, SrcTileData, useRelu>(dst, src, indexRow, indexCol);
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow = 0,
                               uint16_t indexCol = 0)
{
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    TInsert_Impl<DstTileData, SrcTileData, useRelu>(dst, src, indexRow, indexCol);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow = 0,
                               uint16_t indexCol = 0)
{
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    TInsert_Impl<DstTileData, SrcTileData, useRelu>(dst, src, indexRow, indexCol);
}

template <auto mode, typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    TINSERT_IMPL(dst, src, indexRow, indexCol);
}
} // namespace pto
#endif // TINSERT_HPP
