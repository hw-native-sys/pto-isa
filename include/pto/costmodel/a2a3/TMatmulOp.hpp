/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TMATMUL_OP_HPP
#define TMATMUL_OP_HPP

#include <vector>
#include "pto/costmodel/costmodel_types.hpp"

namespace pto {

// TMATMUL / TMATMUL_ACC / TMATMUL_BIAS/ TGEMV / TGEMV_ACC:
//   CCE instruction: mad(c, a, b, m, k, n, ...) — PIPE_M cube pipeline.
//   Records "mmad" with repeat=1 and the MKN dimensions.
//   m = aMatrix.GetValidRow(), k = aMatrix.GetValidCol(), n = bMatrix.GetValidCol()
template <typename TileLeft, typename TileRight>
PTO_INTERNAL std::vector<CostModelStats> runMatmulOp(TileLeft &aMatrix, TileRight &bMatrix)
{
    std::vector<CostModelStats> stats;
    int m = static_cast<int>(aMatrix.GetValidRow());
    int k = static_cast<int>(aMatrix.GetValidCol());
    int n = static_cast<int>(bMatrix.GetValidCol());
    stats.push_back(CostModelStats::MakeMmad("mad", m, k, n));
    return stats;
}

//  TGEMV_BIAS:
//   Same mad instruction but m is forced to 1 (GEMV mode).
//   k = bMatrix.GetValidRow(), n = bMatrix.GetValidCol()
template <typename TileRight, typename TileLeft>
PTO_INTERNAL std::vector<CostModelStats> runMatmulBiasOp(TileLeft &aMatrix, TileRight &bMatrix)
{
    std::vector<CostModelStats> stats;
    int m = static_cast<int>(aMatrix.GetValidRow());
    int k = static_cast<int>(bMatrix.GetValidRow());
    int n = static_cast<int>(bMatrix.GetValidCol());
    stats.push_back(CostModelStats::MakeMmad("mad", m, k, n));
    return stats;
}

} // namespace pto
#endif // TMATMUL_OP_HPP
