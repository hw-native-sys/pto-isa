
/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TEXTRACT_HPP
#define TEXTRACT_HPP

#include "pto/costmodel/pto_isa_costmodel.hpp"

namespace pto {
template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    pto::CostModel::GetInstance().ExtractOpPredictCycle<DstTileData, SrcTileData>("TEXT", dst, src, indexRow, indexCol);
}

// relu
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    PTO_ASSERT(indexRow + DstTileData::Rows <= SrcTileData::Rows,
               "The sum of indexRow and dstRow should be less than srcRow!");
    PTO_ASSERT(indexCol + DstTileData::Cols <= SrcTileData::Cols,
               "The sum of indexCol and dstCol should be less than srcCol!");
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    pto::CostModel::GetInstance().ExtractModeOpPredictCycle<DstTileData, SrcTileData, quantPre, reluMode>(
        "TEXT", dst, src, indexRow, indexCol);
}

// scalar quant
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow = 0,
                                uint16_t indexCol = 0)
{
    PTO_ASSERT(indexRow + DstTileData::Rows <= SrcTileData::Rows,
               "The sum of indexRow and dstRow should be less than srcRow!");
    PTO_ASSERT(indexCol + DstTileData::Cols <= SrcTileData::Cols,
               "The sum of indexCol and dstCol should be less than srcCol!");
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    pto::CostModel::GetInstance().ExtractModeOpPredictCycle<DstTileData, SrcTileData, quantPre, reluMode>(
        "TEXT", dst, src, indexRow, indexCol);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow = 0,
                                uint16_t indexCol = 0)
{
    PTO_ASSERT(indexRow + DstTileData::Rows <= SrcTileData::Rows,
               "The sum of indexRow and dstRow should be less than srcRow!");
    PTO_ASSERT(indexCol + DstTileData::Cols <= SrcTileData::Cols,
               "The sum of indexCol and dstCol should be less than srcCol!");
    static_assert(FpTileData::Loc == TileType::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    pto::CostModel::GetInstance().ExtractModeOpPredictCycle<DstTileData, SrcTileData, quantPre, reluMode>(
        "TEXT", dst, src, indexRow, indexCol);
}
} // namespace pto
#endif
