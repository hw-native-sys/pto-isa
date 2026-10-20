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
// A5 X->ZZ consumes contiguous exponent bytes, independently of the source
// tile row stride. Each destination fractal contains sixteen rows and two scales.
template <int grp_axis, typename DstTileData, typename SrcTileData, typename TmpTileData>
PTO_INTERNAL void TMovMxScaleToZzCPU(DstTileData& dst, SrcTileData& src, TmpTileData& tmp)
{
    (void)tmp;
    using T = typename DstTileData::DType;
    static_assert(grp_axis == 0 || grp_axis == 1, "X2ZZ group axis must be 0 or 1.");
    static_assert(
        std::is_same_v<T, uint8_t> || std::is_same_v<T, hifloat8_t> || std::is_same_v<T, float8_e8m0_t>,
        "X2ZZ requires a supported one-byte exponent carrier.");
    static_assert(
        std::is_same_v<T, typename SrcTileData::DType> && std::is_same_v<T, typename TmpTileData::DType>,
        "X2ZZ source, destination and temporary dtypes must match.");
    static_assert(
        DstTileData::Loc == TileType::Vec && SrcTileData::Loc == TileType::Vec && TmpTileData::Loc == TileType::Vec,
        "X2ZZ source, destination and temporary tiles must be Vec.");
    static_assert(
        SrcTileData::isRowMajor && SrcTileData::SFractal == SLayout::NoneBox,
        "X2ZZ source must have RowMajor/NoneBox layout.");
    static_assert(
        DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor && DstTileData::SFractalSize == 32,
        "X2ZZ destination must have RowMajor/RowMajor/32 layout.");
    const size_t rows = grp_axis == 0 ? src.GetValidCol() : dst.GetValidRow();
    const size_t cols = grp_axis == 0 ? src.GetValidRow() : dst.GetValidCol();
    const size_t paddedRows = ((rows + 15) / 16) * 16;
    assert(cols % 2 == 0 && paddedRows * cols <= DstTileData::Rows * DstTileData::Cols);
    assert(paddedRows * cols <= SrcTileData::Rows * SrcTileData::Cols);
    if constexpr (grp_axis == 0) {
        assert(rows % 16 == 0);
    }
    auto* srcBytes = reinterpret_cast<uint8_t*>(src.data());
    auto* dstBytes = reinterpret_cast<uint8_t*>(dst.data());
    if constexpr (grp_axis == 1) {
        // Native ND->ZZ also clears the source's inactive row padding.
        std::fill(srcBytes + rows * cols, srcBytes + paddedRows * cols, uint8_t{0});
    }
    for (size_t r = 0; r < paddedRows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            const size_t srcOffset = grp_axis == 0 ? c * rows + r : r * cols + c;
            const size_t dstOffset = (r / 16) * cols * 16 + (c / 2) * 32 + (r % 16) * 2 + c % 2;
            dstBytes[dstOffset] = srcBytes[srcOffset];
        }
    }
}

template <typename DstTileData, typename SrcTileData, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TMOV_IMPL(DstTileData& dst, SrcTileData& src)
{
    (void)Phase;
    if constexpr (is_conv_tile_v<SrcTileData>) {
        TEXTRACT(dst, src, 0, 0);
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
    if constexpr (
        reluMode == ReluPreMode::NoRelu && DstTileData::Loc == TileType::Vec && SrcTileData::Loc == TileType::Vec &&
        FpTileData::Loc == TileType::Vec && SrcTileData::isRowMajor && SrcTileData::SFractal == SLayout::NoneBox &&
        DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor && DstTileData::SFractalSize == 32 &&
        (std::is_same_v<typename DstTileData::DType, uint8_t> ||
         std::is_same_v<typename DstTileData::DType, hifloat8_t> ||
         std::is_same_v<typename DstTileData::DType, float8_e8m0_t>) &&
        std::is_same_v<typename DstTileData::DType, typename SrcTileData::DType> &&
        std::is_same_v<typename DstTileData::DType, typename FpTileData::DType>) {
        TMovMxScaleToZzCPU<1>(dst, src, fp);
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
    TEXTRACT<DstTileData, SrcTileData, reluMode>(dst, src, preQuantScalar, 0, 0);
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
template <
    int grp_axis, typename DstTileData, typename SrcTileData, typename TmpTileData,
    std::enable_if_t<(TmpTileData::Loc != TileType::Scaling), int> = 0>
PTO_INTERNAL void TMOV_IMPL(DstTileData& dst, SrcTileData& src, TmpTileData& tmp)
{
    TMovMxScaleToZzCPU<grp_axis>(dst, src, tmp);
}

} // namespace pto
#endif // TMOV_HPP
