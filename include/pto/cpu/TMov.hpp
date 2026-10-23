/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TMOV_HPP
#define TMOV_HPP

#include <cassert>
#include <algorithm>
#include <pto/common/constants.hpp>
#include "pto/cpu/TExtract.hpp"
#include "pto/cpu/tile_offsets.hpp"

namespace pto {
template <typename DstTileData, typename SrcTileData, QuantMode_t quantMode, bool applyRelu, typename ScaleAt>
PTO_INTERNAL void TMovAccToMat(DstTileData& dst, SrcTileData& src, ScaleAt scaleAt)
{
    using D = typename DstTileData::DType;
    using S = typename SrcTileData::DType;
    constexpr size_t c0Size = BLOCK_BYTE_SIZE / sizeof(D);
    const size_t cols = CeilDivision(src.GetValidCol(), c0Size) * c0Size;
    // A2/A3 fixpipe moves physical Acc rows and rounds columns to a destination block.
    // Tile valid metadata does not describe this physical transfer footprint.
    assert(SrcTileData::Rows <= DstTileData::Rows && cols <= SrcTileData::Cols && cols <= DstTileData::Cols);
    for (size_t c = 0; c < cols; ++c) {
        const uint64_t scalar = scaleAt(c);
        for (size_t r = 0; r < SrcTileData::Rows; ++r) {
            dst.SetElement(r, c, ConvertStoreValue<D, S, quantMode, applyRelu>(src.GetElement(r, c), scalar));
        }
    }
}

template <typename DstTileData, typename SrcTileData, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TMOV_IMPL(DstTileData& dst, SrcTileData& src)
{
    (void)Phase;
    if constexpr (is_conv_tile_v<SrcTileData>) {
        TEXTRACT(dst, src, 0, 0);
    } else if constexpr (SrcTileData::Loc == TileType::Acc && DstTileData::Loc == TileType::Mat) {
        TMovAccToMat<DstTileData, SrcTileData, QuantMode_t::NoQuant, false>(dst, src, [](size_t) { return 0; });
    } else {
        assert(src.GetValidRow() == dst.GetValidRow() && src.GetValidCol() == dst.GetValidCol());
        for (size_t c = 0; c < src.GetValidCol(); c++) {
            for (size_t r = 0; r < src.GetValidRow(); r++) {
                dst.SetElement(r, c, src.GetElement(r, c));
            }
        }
    }
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TMOV_IMPL(DstTileData& dst, SrcTileData& src)
{
    (void)Phase;
    if constexpr (SrcTileData::Loc == TileType::Acc && DstTileData::Loc == TileType::Mat) {
        TMovAccToMat<DstTileData, SrcTileData, QuantMode_t::NoQuant, reluMode == ReluPreMode::NormalRelu>(
            dst, src, [](size_t) { return 0; });
        return;
    }
    TMOV_IMPL(dst, src);
    if constexpr (reluMode == ReluPreMode::NormalRelu) {
        const std::size_t rows = static_cast<std::size_t>(dst.GetValidRow());
        const std::size_t cols = static_cast<std::size_t>(dst.GetValidCol());
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                auto& v = dst.data()[GetTileElementOffset<DstTileData>(r, c)];
                if (v < static_cast<typename DstTileData::DType>(0)) {
                    v = static_cast<typename DstTileData::DType>(0);
                }
            }
        }
    }
}

template <
    typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TMOV_IMPL(DstTileData& dst, SrcTileData& src)
{
    (void)Phase;
    (void)mode;
    TMOV_IMPL<DstTileData, SrcTileData, reluMode>(dst, src);
}

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TMOV_IMPL(DstTileData& dst, SrcTileData& src, FpTileData& fp)
{
    (void)Phase;
    if constexpr (SrcTileData::Loc == TileType::Acc && DstTileData::Loc == TileType::Mat) {
        constexpr auto quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
        TMovAccToMat<DstTileData, SrcTileData, quantPre, reluMode == ReluPreMode::NormalRelu>(dst, src, [&](size_t c) {
            assert(fp.GetValidRow() >= 1 && c < fp.GetValidCol());
            return fp.GetElement(0, c);
        });
    } else {
        TEXTRACT_IMPL<DstTileData, SrcTileData, FpTileData, reluMode>(dst, src, fp, 0, 0);
    }
}

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
    ReluPreMode reluMode = ReluPreMode::NoRelu, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TMOV_IMPL(DstTileData& dst, SrcTileData& src, FpTileData& fp)
{
    (void)Phase;
    (void)mode;
    TEXTRACT_IMPL<DstTileData, SrcTileData, FpTileData, mode, reluMode>(dst, src, fp, 0, 0);
}

template <
    typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TMOV_IMPL(DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar)
{
    (void)Phase;
    if constexpr (SrcTileData::Loc == TileType::Acc && DstTileData::Loc == TileType::Mat) {
        constexpr auto quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
        TMovAccToMat<DstTileData, SrcTileData, quantPre, reluMode == ReluPreMode::NormalRelu>(
            dst, src, [=](size_t) { return preQuantScalar; });
    } else {
        TEXTRACT<DstTileData, SrcTileData, reluMode>(dst, src, preQuantScalar, 0, 0);
    }
}

template <
    typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TMOV_IMPL(DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar)
{
    (void)Phase;
    (void)mode;
    TEXTRACT<DstTileData, SrcTileData, reluMode>(dst, src, preQuantScalar, 0, 0);
}
} // namespace pto
#endif // TMOV_HPP
