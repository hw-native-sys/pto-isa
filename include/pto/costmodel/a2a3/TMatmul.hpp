/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TMATMUL_COSTMODEL_HPP
#define TMATMUL_COSTMODEL_HPP

#include <pto/common/type.hpp>
#include "pto/costmodel/pto_isa_costmodel.hpp"

namespace pto {

// TMATMUL and related ops use the cube pipeline (mad/mmad instruction, PIPE_M).
// MKN dimensions are recorded via runTMatmulOp / runTGemvOp.

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    pto::CostModel::GetInstance().MatmulPredictCycle<TileRes, TileLeft, TileRight>(cMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    pto::CostModel::GetInstance().MatmulPredictCycle<TileRes, TileLeft, TileRight>(cOutMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_ACC_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    TMATMUL_ACC_IMPL<Phase, TileRes, TileLeft, TileRight>(cMatrix, cMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
          typename TileBias>
PTO_INTERNAL void TMATMUL_BIAS_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData)
{
    pto::CostModel::GetInstance().MatmulPredictCycle<TileRes, TileLeft, TileRight>(cMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    pto::CostModel::GetInstance().MatmulPredictCycle<TileRes, TileLeft, TileRight>(cMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TGEMV_ACC_IMPL(TileRes &cOutMatrix, TileRes &cInMatrix, TileLeft &aMatrix, TileRight &bMatrix)
{
    pto::CostModel::GetInstance().MatmulPredictCycle<TileRes, TileLeft, TileRight>(cOutMatrix, aMatrix, bMatrix);
}

template <AccPhase Phase = AccPhase::Unspecified, typename TileRes, typename TileLeft, typename TileRight,
          typename TileBias>
PTO_INTERNAL void TGEMV_BIAS_IMPL(TileRes &cMatrix, TileLeft &aMatrix, TileRight &bMatrix, TileBias &biasData)
{
    pto::CostModel::GetInstance().MatmulBiasPredictCycle<TileRes, TileLeft, TileRight>(cMatrix, aMatrix, bMatrix);
}

} // namespace pto

#endif // TMATMUL_COSTMODEL_HPP
