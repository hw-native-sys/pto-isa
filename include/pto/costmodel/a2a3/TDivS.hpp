/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TDIVS_HPP
#define TDIVS_HPP

#include "pto/costmodel/pto_isa_costmodel.hpp"

namespace pto {

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TDIVS_IMPL(TileDataDst &dst, TileDataSrc &src, typename TileDataSrc::DType scalar)
{
    pto::CostModel::GetInstance().BinSOpPredictCycle<DivSOp, TileDataDst, TileDataSrc>("TDIVS", dst, src);
}

template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TDIVS_IMPL(TileDataDst &dst, typename TileDataSrc::DType scalar, TileDataSrc &src)
{
    pto::CostModel::GetInstance().BinSOpPredictCycle<DivSOp, TileDataDst, TileDataSrc>("TDIVS", dst, src);
}

} // namespace pto

#endif