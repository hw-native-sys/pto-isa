/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMOV_HPP
#define TMOV_HPP

#include "pto/costmodel/pto_isa_costmodel.hpp"

namespace pto {
template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src)
{
    pto::CostModel::GetInstance().MovOpPredictCycle<DstTileData, SrcTileData>("TMOV", dst, src);
}

// TMOV with ReluPreMode (Acc→Mat path: copy_matrix_cc_to_cbuf, PIPE_M).
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src)
{
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    pto::CostModel::GetInstance().MovModeOpPredictCycle<DstTileData, SrcTileData, quantPre, reluMode>("TMOV", dst, src);
}

// TMOV with scalar preQuantScalar (scalar-quant path: set_quant_pre + copy_matrix_cc_to_cbuf, PIPE_M).
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar)
{
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    pto::CostModel::GetInstance().MovModeOpPredictCycle<DstTileData, SrcTileData, quantPre, reluMode>("TMOV", dst, src);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp)
{
    static_assert(FpTileData::Loc == TileType::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    pto::CostModel::GetInstance().MovModeOpPredictCycle<DstTileData, SrcTileData, quantPre, reluMode>("TMOV", dst, src);
}

} // namespace pto
#endif
