/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_CONV_TILE_TRANSFER_HPP
#define PTO_CPU_CONV_TILE_TRANSFER_HPP

#include "common.hpp"

namespace pto {
template <typename ConvTileData>
PTO_INTERNAL size_t GetConvTileSpatialRow(ConvTileData& dst, size_t n, size_t d, size_t h, size_t w)
{
    if constexpr (ConvTileData::layout == Layout::NCHW) {
        const size_t H = dst.GetShape(2);
        const size_t W = dst.GetShape(3);
        return (n * H + h) * W + w;
    } else {
        const size_t D = dst.GetShape(2);
        const size_t H = dst.GetShape(3);
        const size_t W = dst.GetShape(4);
        return ((n * D + d) * H + h) * W + w;
    }
}

template <typename ConvTileData>
PTO_INTERNAL size_t GetConvTileNCHWOffset(ConvTileData& dst, size_t n, size_t c, size_t d, size_t h, size_t w)
{
    const size_t C = dst.GetShape(1);
    if constexpr (ConvTileData::layout == Layout::NCHW) {
        const size_t H = dst.GetShape(2);
        const size_t W = dst.GetShape(3);
        return ((n * C + c) * H + h) * W + w;
    } else {
        const size_t D = dst.GetShape(2);
        const size_t H = dst.GetShape(3);
        const size_t W = dst.GetShape(4);
        return (((n * C + c) * D + d) * H + h) * W + w;
    }
}

template <typename DstTileData, typename SrcTileData, QuantMode_t quantMode>
PTO_INTERNAL void ValidateAccToConvTileTransfer()
{
    using D = typename DstTileData::DType;
    using S = typename SrcTileData::DType;

    static_assert(is_conv_tile_v<DstTileData>, "Conv transfer: destination must be ConvTile.");
    static_assert(
        DstTileData::Loc == TileType::Vec && SrcTileData::Loc == TileType::Acc,
        "Conv transfer: source must be Acc and destination must be Vec.");
    static_assert(
        !SrcTileData::isRowMajor && SrcTileData::SFractal == SLayout::RowMajor,
        "Conv transfer: source Acc must use NZ layout.");
    static_assert(
        (DstTileData::layout == Layout::NCHW && DstTileData::totalDimCount == 4) ||
            (DstTileData::layout == Layout::NCDHW && DstTileData::totalDimCount == 5),
        "Conv transfer: destination must use NCHW or NCDHW layout.");
    static_assert(
        std::is_same_v<S, float> || std::is_same_v<S, int32_t>,
        "Conv transfer: source Acc type must be float or int32_t.");

    if constexpr (quantMode == QuantMode_t::NoQuant) {
        static_assert(
            (std::is_same_v<S, float> &&
             (std::is_same_v<D, float> || std::is_same_v<D, half> || std::is_same_v<D, bfloat16_t>)) ||
                (std::is_same_v<S, int32_t> && std::is_same_v<D, int32_t>),
            "Unsupported non-quantized Acc-to-ConvTile dtype pair.");
    } else {
        constexpr bool validFloatDst = std::is_same_v<D, float> || std::is_same_v<D, half> ||
                                       std::is_same_v<D, bfloat16_t> || std::is_same_v<D, int8_t> ||
                                       std::is_same_v<D, uint8_t> || std::is_same_v<D, hifloat8_t> ||
                                       std::is_same_v<D, float8_e4m3_t>;

        constexpr bool validIntDst = std::is_same_v<D, int32_t> || std::is_same_v<D, int8_t> ||
                                     std::is_same_v<D, uint8_t> || std::is_same_v<D, half> ||
                                     std::is_same_v<D, bfloat16_t>;

        static_assert(
            (std::is_same_v<S, float> && validFloatDst) || (std::is_same_v<S, int32_t> && validIntDst),
            "Unsupported quantized Acc-to-ConvTile dtype pair.");
    }
}

template <
    bool IsInsert, typename DstTileData, typename SrcTileData, QuantMode_t quantMode, bool applyRelu,
    bool scalarQuant = false>
PTO_INTERNAL void AccToConvTileTransfer(
    DstTileData& dst, SrcTileData& src, uint32_t indexRow, uint32_t indexCol, const std::vector<uint64_t>& scalars = {})
{
    using D = typename DstTileData::DType;
    using S = typename SrcTileData::DType;

    ValidateAccToConvTileTransfer<DstTileData, SrcTileData, quantMode>();

    const size_t N = dst.GetShape(0);
    const size_t C = dst.GetShape(1);

    size_t depth = 1;
    size_t H;
    size_t W;
    size_t validN = dst.GetValidShape(0);
    size_t validC = dst.GetValidShape(1);
    size_t validD = 1;
    size_t validH;
    size_t validW;

    if constexpr (DstTileData::layout == Layout::NCHW) {
        H = dst.GetShape(2);
        W = dst.GetShape(3);
        validH = dst.GetValidShape(2);
        validW = dst.GetValidShape(3);
    } else {
        depth = dst.GetShape(2);
        H = dst.GetShape(3);
        W = dst.GetShape(4);

        validD = dst.GetValidShape(2);
        validH = dst.GetValidShape(3);
        validW = dst.GetValidShape(4);
    }

    for (int i = 0; i < DstTileData::totalDimCount; ++i) {
        PTO_CPU_ASSERT(
            dst.GetValidShape(i) <= dst.GetShape(i), "Conv transfer: valid shape exceeds destination shape.");
    }

    auto copyElement = [&](size_t srcRow, size_t srcCol, size_t dstOffset, size_t localCol) {
        if constexpr (quantMode != QuantMode_t::NoQuant) {
            const size_t scaleCol = (IsInsert || scalarQuant) ? localCol : srcCol;
            PTO_CPU_ASSERT(scaleCol < scalars.size(), "Conv transfer: quantization scale index is out of range.");
            dst.SetElement(
                dstOffset,
                quantize_element<D, S, quantMode, applyRelu>(src.GetElement(srcRow, srcCol), scalars[scaleCol]));
        } else {
            S value = src.GetElement(srcRow, srcCol);
            if constexpr (applyRelu) {
                value = ReLU(value);
            }
            dst.SetElement(dstOffset, value);
        }
    };

    if constexpr (!IsInsert) {
        // Read a window from the NZ Acc tile and write it into NCHW/NCDHW UB.
        if (validN == 0 || validC == 0 || validH == 0 || validW == 0 || validD == 0) {
            return;
        }
        const size_t lastSrcRow =
            static_cast<size_t>(indexRow) + GetConvTileSpatialRow(dst, validN - 1, validD - 1, validH - 1, validW - 1);
        PTO_CPU_ASSERT(lastSrcRow < src.GetValidRow(), "TEXTRACT Conv: source valid row is insufficient.");
        PTO_CPU_ASSERT(
            static_cast<size_t>(indexCol) + validC <= src.GetValidCol(),
            "TEXTRACT Conv: source valid col is insufficient.");

        for (size_t n = 0; n < validN; ++n) {
            for (size_t d = 0; d < validD; ++d) {
                for (size_t h = 0; h < validH; ++h) {
                    for (size_t w = 0; w < validW; ++w) {
                        const size_t srcRow = indexRow + GetConvTileSpatialRow(dst, n, d, h, w);
                        for (size_t c = 0; c < validC; ++c) {
                            const size_t srcCol = indexCol + c;
                            const size_t dstOffset = GetConvTileNCHWOffset(dst, n, c, d, h, w);
                            copyElement(srcRow, srcCol, dstOffset, c);
                        }
                    }
                }
            }
        }
    } else {
        // Insert the NZ Acc window into the NCHW/NCDHW UB ConvTile.
        const size_t totalSpatialRows = N * depth * H * W;
        PTO_CPU_ASSERT(
            static_cast<size_t>(indexRow) + src.GetValidRow() <= totalSpatialRows,
            "TINSERT Conv: source rows exceed destination.");
        PTO_CPU_ASSERT(
            static_cast<size_t>(indexCol) + src.GetValidCol() <= validC,
            "TINSERT Conv: source columns exceed destination.");

        for (size_t r = 0; r < src.GetValidRow(); ++r) {
            const size_t dstRow = indexRow + r;
            const size_t spatialPerN = depth * H * W;
            const size_t n = dstRow / spatialPerN;
            size_t spatial = dstRow % spatialPerN;
            const size_t d = spatial / (H * W);
            spatial %= H * W;
            const size_t h = spatial / W;
            const size_t w = spatial % W;

            PTO_CPU_ASSERT(
                n < validN && d < validD && h < validH && w < validW,
                "TINSERT Conv: destination index is outside valid shape.");
            for (size_t c = 0; c < src.GetValidCol(); ++c) {
                const size_t dstCol = indexCol + c;
                const size_t dstOffset = GetConvTileNCHWOffset(dst, n, dstCol, d, h, w);
                copyElement(r, c, dstOffset, c);
            }
        }
    }
}

} // namespace pto
#endif
