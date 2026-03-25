/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TMRGSORT_COSTMODEL_HPP
#define TMRGSORT_COSTMODEL_HPP

#include "pto/costmodel/pto_isa_costmodel.hpp"

namespace pto {

// MrgSortExecutedNumList is declared in TLoad.hpp (included before this header).

// Single-src TMRGSORT: vmrgsort4(repeatTimes)
template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMRGSORT_IMPL(DstTileData &dst, SrcTileData &src, uint32_t blockLen)
{
    using T = typename DstTileData::DType;
    auto stats = runMrgSortSingleOp(dst, src, blockLen);
    dst.SetCycle(CostModel::GetInstance().VecInstPredictCycle<T>(stats));
}

// Multi-src TMRGSORT (2 sources)
template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData, bool exhausted>
PTO_INTERNAL void TMRGSORT_IMPL(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp,
                                Src0TileData &src0, Src1TileData &src1)
{
    using T = typename DstTileData::DType;
    auto stats = runMrgSortMultiOp();
    dst.SetCycle(CostModel::GetInstance().VecInstPredictCycle<T>(stats));
}

// Multi-src TMRGSORT (3 sources)
template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
          typename Src2TileData, bool exhausted>
PTO_INTERNAL void TMRGSORT_IMPL(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp,
                                Src0TileData &src0, Src1TileData &src1, Src2TileData &src2)
{
    using T = typename DstTileData::DType;
    auto stats = runMrgSortMultiOp();
    dst.SetCycle(CostModel::GetInstance().VecInstPredictCycle<T>(stats));
}

// Multi-src TMRGSORT (4 sources)
template <typename DstTileData, typename TmpTileData, typename Src0TileData, typename Src1TileData,
          typename Src2TileData, typename Src3TileData, bool exhausted>
PTO_INTERNAL void TMRGSORT_IMPL(DstTileData &dst, MrgSortExecutedNumList &executedNumList, TmpTileData &tmp,
                                Src0TileData &src0, Src1TileData &src1, Src2TileData &src2, Src3TileData &src3)
{
    using T = typename DstTileData::DType;
    auto stats = runMrgSortMultiOp();
    dst.SetCycle(CostModel::GetInstance().VecInstPredictCycle<T>(stats));
}

} // namespace pto

#endif // TMRGSORT_COSTMODEL_HPP
