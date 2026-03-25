/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TSCATTER_COSTMODEL_HPP
#define TSCATTER_COSTMODEL_HPP

#include "pto/costmodel/pto_isa_costmodel.hpp"

namespace pto {

// TSCATTER: pure scalar element-wise loop — no CCE pipeline instructions.
template <typename TileDataD, typename TileDataS, typename TileDataI>
PTO_INTERNAL void TSCATTER_IMPL(TileDataD &dst, TileDataS &src, TileDataI &idx)
{
    using T = typename TileDataD::DType;
    auto stats = runScatterOp(dst, src, idx);
    dst.SetCycle(CostModel::GetInstance().VecInstPredictCycle<T>(stats));
}

} // namespace pto

#endif // TSCATTER_COSTMODEL_HPP
