/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TEXTRACT_HPP
#define TEXTRACT_HPP

#include <cassert>
#include "common.hpp"
#include <cmath>

namespace pto {

template <typename DstTileData, typename SrcTileData, QuantModeCPU_t quantMode, bool applyRelu>
PTO_INTERNAL void TExtract_Impl(DstTileData &dst, SrcTileData &src, uint32_t idxRow, uint32_t idxCol,
                                const std::vector<uint64_t> &scalars = {})
{
    assert(dst.GetValidRow() + idxRow <= src.GetValidRow() && dst.GetValidCol() + idxCol <= src.GetValidCol());

    using D = typename DstTileData::DType;
    using S = typename SrcTileData::DType;

    for (size_t c = 0; c < dst.GetValidCol(); c++) {
        for (size_t r = 0; r < dst.GetValidRow(); r++) {
            size_t srcTileIdx = GetTileElementOffset<SrcTileData>(r + idxRow, c + idxCol);
            size_t dstTileIdx = GetTileElementOffset<DstTileData>(r, c);
            if constexpr (quantMode != QuantModeCPU_t::NoQuant) {
                size_t scalarIndex = SrcTileData::isRowMajor ? c : r;
                dst.data()[dstTileIdx] =
                    quantize_element<D, S, quantMode, applyRelu>(src.data()[srcTileIdx], scalars[scalarIndex]);
            } else {
                S val = src.data()[srcTileIdx];
                if constexpr (applyRelu) {
                    val = ReLU(val);
                }
                dst.data()[dstTileIdx] = val;
            }
        }
    }
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint32_t idxRow, uint32_t idxCol)
{
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    TExtract_Impl<DstTileData, SrcTileData, QuantModeCPU_t::NoQuant, useRelu>(dst, src, idxRow, idxCol);
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint32_t idxRow, uint32_t idxCol)
{
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    TExtract_Impl<DstTileData, SrcTileData, QuantModeCPU_t::NoQuant, useRelu>(dst, src, idxRow, idxCol);
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint32_t idxRow,
                                uint32_t idxCol)
{
    constexpr QuantModeCPU_t quantPre =
        GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    std::vector<uint64_t> scalars(dst.GetValidCol(), preQuantScalar);

    TExtract_Impl<DstTileData, SrcTileData, quantPre, useRelu>(dst, src, idxRow, idxCol, scalars);
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint32_t idxRow,
                                uint32_t idxCol)
{
    constexpr QuantModeCPU_t quantPre =
        GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    std::vector<uint64_t> scalars(dst.GetValidCol(), preQuantScalar);

    TExtract_Impl<DstTileData, SrcTileData, quantPre, useRelu>(dst, src, idxRow, idxCol, scalars);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint32_t idxRow, uint32_t idxCol)
{
    constexpr QuantModeCPU_t quantPre =
        GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;

    std::vector<uint64_t> scalars(dst.GetValidCol(), 0);
    for (size_t i = 0; i < dst.GetValidCol(); i++) {
        const size_t quantTileIdx = GetTileElementOffset<FpTileData>(0, i);
        scalars[i] = fp.data()[quantTileIdx];
    }

    TExtract_Impl<DstTileData, SrcTileData, quantPre, useRelu>(dst, src, idxRow, idxCol, scalars);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint32_t idxRow, uint32_t idxCol)
{
    constexpr QuantModeCPU_t quantPre =
        GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;

    std::vector<uint64_t> scalars(dst.GetValidCol(), 0);
    for (size_t i = 0; i < dst.GetValidCol(); i++) {
        const size_t quantTileIdx = GetTileElementOffset<FpTileData>(0, i);
        scalars[i] = fp.data()[quantTileIdx];
    }

    TExtract_Impl<DstTileData, SrcTileData, quantPre, useRelu>(dst, src, idxRow, idxCol, scalars);
}

} // namespace pto
#endif // TEXTRACT_HPP
