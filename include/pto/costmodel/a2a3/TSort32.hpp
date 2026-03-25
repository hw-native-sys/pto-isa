/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TSORT32_COSTMODEL_HPP
#define TSORT32_COSTMODEL_HPP

#include "pto/costmodel/pto_isa_costmodel.hpp"

namespace pto {

constexpr unsigned BLOCK_SIZE = 32;

// TSORT32: vbitsort(repeatNumPerRow) + PIPE_V per row.
// See TSort32Op.hpp for cycle formula details.
template <typename DstTileData, typename SrcTileData, typename IdxTileData>
PTO_INTERNAL void TSORT32_IMPL(DstTileData &dst, SrcTileData &src, IdxTileData &idx)
{
    using T = typename DstTileData::DType;
    auto stats = runSort32Op(dst, src);
    dst.SetCycle(CostModel::GetInstance().VecInstPredictCycle<T>(stats));
}

template <typename DstTileData, typename SrcTileData, typename IdxTileData, typename TmpTileData>
PTO_INTERNAL void TSORT32_IMPL(DstTileData &dst, SrcTileData &src, IdxTileData &idx, TmpTileData &tmp)
{
    using T = typename DstTileData::DType;
    std::vector<CostModelStats> stats;
    if (src.GetValidCol() % BLOCK_SIZE > 0) {
        stats = runSort32OpWithTmp(dst, src, tmp);
    } else {
        stats = runSort32Op(dst, src);
    }
    dst.SetCycle(CostModel::GetInstance().VecInstPredictCycle<T>(stats));
}

} // namespace pto

#endif // TSORT32_COSTMODEL_HPP
