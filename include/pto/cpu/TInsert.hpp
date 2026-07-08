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

template <typename DstTileData, typename SrcTileData, QuantMode_t quantMode, bool applyRelu>
PTO_INTERNAL void TInsert_Impl(DstTileData &dst, SrcTileData &src, uint32_t idxRow, uint32_t idxCol,
                               const std::vector<uint64_t> &scalars = {})
{
    assert(src.GetValidRow() + idxRow <= dst.GetValidRow() && src.GetValidCol() + idxCol <= dst.GetValidCol());

    using D = typename DstTileData::DType;
    using S = typename SrcTileData::DType;

    for (size_t c = 0; c < src.GetValidCol(); c++) {
        for (size_t r = 0; r < src.GetValidRow(); r++) {
            size_t srcTileIdx = GetTileElementOffset<SrcTileData>(r, c);
            size_t dstTileIdx = GetTileElementOffset<DstTileData>(r + idxRow, c + idxCol);
            size_t scalarIndex = SrcTileData::isRowMajor ? c : r;
            if constexpr (quantMode != QuantMode_t::NoQuant) {
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

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint32_t idxRow = 0, uint32_t idxCol = 0)
{
    TInsert_Impl<DstTileData, SrcTileData, QuantMode_t::NoQuant, false>(dst, src, idxRow, idxCol);
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    TInsert_Impl<DstTileData, SrcTileData, QuantMode_t::NoQuant, useRelu>(dst, src, indexRow, indexCol);
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow = 0,
                               uint16_t indexCol = 0)
{
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    constexpr QuantMode_t quantMode = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();

    size_t quantVectorSize = SrcTileData::isRowMajor ? src.GetValidCol() : src.GetValidRow();
    std::vector<uint64_t> scalars(quantVectorSize, preQuantScalar);

    TInsert_Impl<DstTileData, SrcTileData, quantMode, useRelu>(dst, src, indexRow, indexCol, scalars);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow = 0,
                               uint16_t indexCol = 0)
{
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    constexpr QuantMode_t quantMode = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();

    std::vector<uint64_t> scalars(fp.GetValidCol(), 0);
    for (size_t i = 0; i < fp.GetValidCol(); i++) {
        const size_t quantTileIndex = GetTileElementOffset<FpTileData>(0, i);
        scalars[i] = fp.data()[quantTileIndex];
    }
    TInsert_Impl<DstTileData, SrcTileData, quantMode, useRelu>(dst, src, indexRow, indexCol, scalars);
}

template <auto mode, typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    TINSERT_IMPL(dst, src, indexRow, indexCol);
}
} // namespace pto
#endif // TINSERT_HPP
