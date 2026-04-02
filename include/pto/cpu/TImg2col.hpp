/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TIMG2COL_CPU_HPP
#define TIMG2COL_CPU_HPP

#include <algorithm>
#include <cstdint>
#include "pto/cpu/tile_offsets.hpp"

namespace pto {
namespace cpu_img2col {
template <typename ConvTileData>
inline size_t GetInputOffset(const ConvTileData &src, int64_t n, int64_t d, int64_t c1, int64_t h, int64_t w,
                             int64_t c0)
{
    const int64_t c0Size = src.GetShape(ConvTileData::totalDimCount - 1);
    if constexpr (ConvTileData::layout == Layout::NC1HWC0) {
        const int64_t c1Size = src.GetShape(1);
        const int64_t hSize = src.GetShape(2);
        const int64_t wSize = src.GetShape(3);
        return static_cast<size_t>(((((n * c1Size + c1) * hSize + h) * wSize + w) * c0Size) + c0);
    } else {
        const int64_t dSize = src.GetShape(1);
        const int64_t c1Size = src.GetShape(2);
        const int64_t hSize = src.GetShape(3);
        const int64_t wSize = src.GetShape(4);
        return static_cast<size_t>((((((n * dSize + d) * c1Size + c1) * hSize + h) * wSize + w) * c0Size) + c0);
    }
}

template <typename TileData, typename ConvTileData>
inline void Timg2colCheck()
{
    static_assert(ConvTileData::Loc == TileType::Mat, "TImg2col: Source TileType only support Mat.");
    static_assert(TileData::Loc == TileType::Left, "TImg2col: Destination TileType only support Left.");
    static_assert((ConvTileData::layout == Layout::NC1HWC0) || (ConvTileData::layout == Layout::NDC1HWC0),
                  "TImg2col: Source layout only support NC1HWC0/NDC1HWC0.");
    static_assert(TileData::SFractal == SLayout::RowMajor && !TileData::isRowMajor,
                  "TImg2col: Destination layout only support SLayout RowMajor + BLayout ColMajor.");
    static_assert(std::is_same_v<typename ConvTileData::DType, typename TileData::DType>,
                  "TImg2col: Destination and source tile data types must match.");
}
} // namespace cpu_img2col

template <typename ConvTileData>
struct Img2ColParams {
    int64_t fmapN;
    int64_t fmapD;
    int64_t fmapC1;
    int64_t fmapH;
    int64_t fmapW;
    int64_t fmapC0;
    int64_t strideH;
    int64_t strideW;
    int64_t dilationH;
    int64_t dilationW;
    int64_t filterH;
    int64_t filterW;
    int64_t padLeft;
    int64_t padRight;
    int64_t padTop;
    int64_t padBottom;
    int64_t channelSize;
    int64_t outH;
    int64_t outW;
};

template <typename ConvTileData>
PTO_INTERNAL Img2ColParams<ConvTileData> ExtractImg2ColParams(const ConvTileData &src)
{
    Img2ColParams<ConvTileData> params;

    params.fmapN = src.GetShape(0);
    params.fmapD = (ConvTileData::layout == Layout::NDC1HWC0) ? src.GetShape(1) : 1;
    params.fmapC1 = (ConvTileData::layout == Layout::NDC1HWC0) ? src.GetShape(2) : src.GetShape(1);
    params.fmapH = (ConvTileData::layout == Layout::NDC1HWC0) ? src.GetShape(3) : src.GetShape(2);
    params.fmapW = (ConvTileData::layout == Layout::NDC1HWC0) ? src.GetShape(4) : src.GetShape(3);
    params.fmapC0 = src.GetShape(ConvTileData::totalDimCount - 1);
    params.strideH = src.GetStrideH();
    params.strideW = src.GetStrideW();
    params.dilationH = src.GetDilationH();
    params.dilationW = src.GetDilationW();
    params.filterH = src.GetFilterH();
    params.filterW = src.GetFilterW();
    params.padLeft = src.GetPadList(0);
    params.padRight = src.GetPadList(1);
    params.padTop = src.GetPadList(2);
    params.padBottom = src.GetPadList(3);
    params.channelSize = src.GetChannelSize() > 0 ? src.GetChannelSize() : params.fmapC1 * params.fmapC0;
    params.outH = (params.fmapH + params.padTop + params.padBottom - params.dilationH * (params.filterH - 1) - 1) /
                      params.strideH +
                  1;
    params.outW = (params.fmapW + params.padLeft + params.padRight - params.dilationW * (params.filterW - 1) - 1) /
                      params.strideW +
                  1;

    return params;
}

template <typename TileData, typename ConvTileData, SetFmatrixMode FmatrixMode = SetFmatrixMode::FMATRIX_A_MANUAL>
PTO_INTERNAL void TIMG2COL_IMPL(TileData &dst, ConvTileData &src, uint16_t posM, uint16_t posK)
{
    (void)FmatrixMode;
    cpu_img2col::Timg2colCheck<TileData, ConvTileData>();

    const auto params = ExtractImg2ColParams(src);
    const int64_t mPerBatch = params.fmapD * params.outH * params.outW;
    const auto padValue = src.GetPadValue();

    for (int r = 0; r < dst.GetValidRow(); ++r) {
        const int64_t mIndex = static_cast<int64_t>(posM) + r;
        const int64_t nIndex = mIndex / mPerBatch;
        const int64_t remAfterN = mIndex % mPerBatch;
        const int64_t dIndex = remAfterN / (params.outH * params.outW);
        const int64_t remAfterD = remAfterN % (params.outH * params.outW);
        const int64_t outRow = remAfterD / params.outW;
        const int64_t outCol = remAfterD % params.outW;

        for (int c = 0; c < dst.GetValidCol(); ++c) {
            const int64_t kIndex = static_cast<int64_t>(posK) + c;
            const int64_t channelIndex = kIndex / (params.filterH * params.filterW);
            const int64_t kernelOffset = kIndex % (params.filterH * params.filterW);
            const int64_t kernelH = kernelOffset / params.filterW;
            const int64_t kernelW = kernelOffset % params.filterW;

            auto value =
                CalculateValue(src, params, nIndex, dIndex, channelIndex, kernelH, kernelW, outRow, outCol, padValue);

            dst.data()[GetTileElementOffset<TileData>(r, c)] = value;
        }
    }
}

template <typename ConvTileData>
PTO_INTERNAL auto CalculateValue(const ConvTileData &src, const Img2ColParams<ConvTileData> &params, int64_t nIndex,
                                 int64_t dIndex, int64_t channelIndex, int64_t kernelH, int64_t kernelW, int64_t outRow,
                                 int64_t outCol, const typename ConvTileData::DType &padValue)
{
    auto value = padValue;

    if (nIndex < params.fmapN && channelIndex < params.channelSize) {
        const int64_t c1Index = channelIndex / params.fmapC0;
        const int64_t c0Index = channelIndex % params.fmapC0;
        const int64_t inputH = outRow * params.strideH + kernelH * params.dilationH - params.padTop;
        const int64_t inputW = outCol * params.strideW + kernelW * params.dilationW - params.padLeft;

        if (inputH >= 0 && inputH < params.fmapH && inputW >= 0 && inputW < params.fmapW) {
            const size_t srcOffset = cpu_img2col::GetInputOffset(src, nIndex, dIndex, c1Index, inputH, inputW, c0Index);
            value = src.data()[srcOffset];
        }
    }

    return value;
}

} // namespace pto

#endif
