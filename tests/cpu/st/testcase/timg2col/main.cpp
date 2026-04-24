/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <vector>
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace {
template <typename ConvTileData>
size_t Nc1hwc0Offset(int64_t n, int64_t c1, int64_t h, int64_t w, int64_t c0)
{
    const int64_t c1Count = ConvTileData::staticShape[1];
    const int64_t hCount = ConvTileData::staticShape[2];
    const int64_t wCount = ConvTileData::staticShape[3];
    const int64_t c0Count = ConvTileData::staticShape[4];
    return static_cast<size_t>(((((n * c1Count + c1) * hCount + h) * wCount + w) * c0Count) + c0);
}

template <typename TileData, typename ConvTileData>
std::vector<typename TileData::DType> BuildExpected(const ConvTileData &src, uint16_t posM, uint16_t posK)
{
    const int64_t fmapN = ConvTileData::staticShape[0];
    const int64_t fmapC1 = ConvTileData::staticShape[1];
    const int64_t fmapH = ConvTileData::staticShape[2];
    const int64_t fmapW = ConvTileData::staticShape[3];
    const int64_t fmapC0 = ConvTileData::staticShape[4];
    const int64_t strideH = src.GetStrideH();
    const int64_t strideW = src.GetStrideW();
    const int64_t dilationH = src.GetDilationH();
    const int64_t dilationW = src.GetDilationW();
    const int64_t filterH = src.GetFilterH();
    const int64_t filterW = src.GetFilterW();
    const int64_t padLeft = src.GetPadList(0);
    const int64_t padTop = src.GetPadList(2);
    const int64_t outH = (fmapH + src.GetPadList(2) + src.GetPadList(3) - dilationH * (filterH - 1) - 1) / strideH + 1;
    const int64_t outW = (fmapW + src.GetPadList(0) + src.GetPadList(1) - dilationW * (filterW - 1) - 1) / strideW + 1;

    std::vector<typename TileData::DType> expected(TileData::Numel, 0);
    for (int r = 0; r < TileData::ValidRow; ++r) {
        const int64_t mIndex = posM + r;
        const int64_t nIndex = mIndex / (outH * outW);
        const int64_t rem = mIndex % (outH * outW);
        const int64_t outRow = rem / outW;
        const int64_t outCol = rem % outW;
        for (int c = 0; c < TileData::ValidCol; ++c) {
            const int64_t kIndex = posK + c;
            const int64_t channelIndex = kIndex / (filterH * filterW);
            const int64_t kernelOffset = kIndex % (filterH * filterW);
            const int64_t kernelH = kernelOffset / filterW;
            const int64_t kernelW = kernelOffset % filterW;
            typename TileData::DType value = src.GetPadValue();
            if (channelIndex < src.GetChannelSize()) {
                const int64_t c1 = channelIndex / fmapC0;
                const int64_t c0 = channelIndex % fmapC0;
                const int64_t inputH = outRow * strideH + kernelH * dilationH - padTop;
                const int64_t inputW = outCol * strideW + kernelW * dilationW - padLeft;
                if (nIndex < fmapN && inputH >= 0 && inputH < fmapH && inputW >= 0 && inputW < fmapW && c1 < fmapC1) {
                    value = src.data()[Nc1hwc0Offset<ConvTileData>(nIndex, c1, inputH, inputW, c0)];
                }
            }
            expected[GetTileElementOffset<TileData>(r, c)] = value;
        }
    }
    return expected;
}
} // namespace

TEST(TImg2colCpuSimTest, ManualMetadataPathMatchesReferenceWithPadding)
{
    using SrcTile = ConvTile<TileType::Mat, float, 1 * 1 * 3 * 4 * 8, Layout::NC1HWC0, ConvTileShape<1, 1, 3, 4, 8>>;
    using DstTile = TileLeft<float, 16, 16, 4, 16>;
    std::vector<float> storage(SrcTile::bufferSize, 0.0f);
    SrcTile src;
    DstTile dst;
    src.data() = storage.data();
    TASSIGN(dst, 0);

    for (int n = 0; n < 1; ++n) {
        for (int h = 0; h < 3; ++h) {
            for (int w = 0; w < 4; ++w) {
                for (int c0 = 0; c0 < 8; ++c0) {
                    src.data()[Nc1hwc0Offset<SrcTile>(n, 0, h, w, c0)] = static_cast<float>(100 * h + 10 * w + c0);
                }
            }
        }
    }
    uint8_t padList[] = {1, 0, 1, 0};
    src.SetPadListArray(padList);
    src.SetFmapH(3);
    src.SetFmapW(4);
    src.SetFilterH(2);
    src.SetFilterW(2);
    src.SetStrideH(1);
    src.SetStrideW(1);
    src.SetDilationH(1);
    src.SetDilationW(1);
    src.SetChannelSize(8);
    src.SetPadValue(-1.0f);
    src.SetRepeatStride(0);
    src.SetRepeatTime(1);
    src.SetRepeatMode(0);
    src.SetDstStride(1);

    SETFMATRIX<SrcTile, SetFmatrixMode::FMATRIX_B_MANUAL>(src);
    SET_IMG2COL_PADDING<SrcTile, SetFmatrixMode::FMATRIX_B_MANUAL>(src);
    SET_IMG2COL_RPT<SrcTile, SetFmatrixMode::FMATRIX_B_MANUAL>(src);
    TIMG2COL<DstTile, SrcTile, SetFmatrixMode::FMATRIX_B_MANUAL>(dst, src, 1, 8);

    const auto expected = BuildExpected<DstTile>(src, 1, 8);
    for (int i = 0; i < DstTile::Numel; ++i) {
        EXPECT_FLOAT_EQ(dst.data()[i], expected[i]);
    }
}

TEST(TImg2colCpuSimTest, AutoMetadataPathMatchesReferenceForSplitKChunk)
{
    using SrcTile = ConvTile<TileType::Mat, int8_t, 1 * 1 * 4 * 4 * 32, Layout::NC1HWC0, ConvTileShape<1, 1, 4, 4, 32>>;
    using DstTile = TileLeft<int8_t, 16, 32, 4, 32>;
    std::vector<int8_t> storage(SrcTile::bufferSize, 0);
    SrcTile src;
    DstTile dst;
    src.data() = storage.data();
    TASSIGN(dst, 0);

    for (int h = 0; h < 4; ++h) {
        for (int w = 0; w < 4; ++w) {
            for (int c0 = 0; c0 < 32; ++c0) {
                src.data()[Nc1hwc0Offset<SrcTile>(0, 0, h, w, c0)] = static_cast<int8_t>((h * 4 + w + c0) % 97);
            }
        }
    }
    uint8_t padList[] = {0, 0, 0, 0};
    src.SetPadListArray(padList);
    src.SetFmapH(4);
    src.SetFmapW(4);
    src.SetFilterH(2);
    src.SetFilterW(2);
    src.SetStrideH(2);
    src.SetStrideW(2);
    src.SetDilationH(1);
    src.SetDilationW(1);
    src.SetChannelSize(32);
    src.SetPadValue(0);
    src.SetRepeatStride(0);
    src.SetRepeatTime(1);
    src.SetRepeatMode(0);
    src.SetDstStride(1);

    SETFMATRIX<SrcTile, SetFmatrixMode::FMATRIX_B_AUTO>(src);
    SET_IMG2COL_PADDING<SrcTile, SetFmatrixMode::FMATRIX_B_AUTO>(src);
    SET_IMG2COL_RPT<SrcTile, SetFmatrixMode::FMATRIX_B_AUTO>(src);
    TIMG2COL<DstTile, SrcTile, SetFmatrixMode::FMATRIX_B_AUTO>(dst, src, 0, 32);

    const auto expected = BuildExpected<DstTile>(src, 0, 32);
    for (int i = 0; i < DstTile::Numel; ++i) {
        EXPECT_EQ(dst.data()[i], expected[i]);
    }
}
