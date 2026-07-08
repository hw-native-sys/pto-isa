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
#include "../common/utils.hpp"
#include <cmath>

namespace pto {

template <typename DstTileData, typename SrcTileData, QuantMode_t quantMode, bool applyRelu>
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
            if constexpr (quantMode != QuantMode_t::NoQuant) {
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

template <typename T>
__tf__ PTO_INTERNAL void copy_fractal_shapes(T *dst, T *src, uint16_t indexC, uint16_t indexR, uint16_t dstC,
                                             uint16_t dstR, uint16_t srcStride, uint16_t dstStride)
{
    // All indices and strides represent number of fractals
    constexpr const uint16_t BLOCK_SIZE_ELEM = CUBE_BLOCK_SIZE / sizeof(T);
    src = src + indexR * srcStride * BLOCK_SIZE_ELEM;
    const size_t kCopySize = dstC * BLOCK_SIZE_ELEM;
    cpu::parallel_for_rows(dstR, dstR * dstC * BLOCK_SIZE_ELEM, [&](std::size_t r) {
        T *srcBase = src + (r * srcStride + indexC) * BLOCK_SIZE_ELEM;
        T *dstBase = dst + r * dstStride * BLOCK_SIZE_ELEM;
        std::copy(srcBase, srcBase + kCopySize, dstBase);
    });
}

template <typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void TExtractToBConv(typename DstTileData::TileDType __out__ dst,
                                         typename SrcTileData::TileDType __in__ src, uint16_t srcCol,
                                         uint16_t dstValidRow, uint16_t dstValidCol, uint16_t indexRow,
                                         uint16_t indexCol)
{
    using DataType = typename SrcTileData::DType;
    constexpr int c0Size = BLOCK_BYTE_SIZE / sizeof(DataType);

    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __cb__ DataType *dstAddr = (__cb__ DataType *)__cce_get_tile_ptr(dst);
    uint16_t dstValidColAlign = CeilDivision(dstValidCol, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    uint16_t dstValidRowAlign = CeilDivision(dstValidRow, c0Size) * c0Size;

    uint16_t mStartPosition = indexCol >> SHIFT_BLOCK_LEN;
    uint16_t kStartPosition = (indexRow * sizeof(DataType)) >> SHIFT_BLOCK_BYTE;
    uint16_t mStep = dstValidColAlign >> SHIFT_BLOCK_LEN;
    uint16_t kStep = (dstValidRowAlign * sizeof(DataType)) >> SHIFT_BLOCK_BYTE;
    uint16_t srcStride = srcCol >> SHIFT_BLOCK_LEN;
    uint16_t dstStride = dstValidColAlign >> SHIFT_BLOCK_LEN;

    copy_fractal_shapes<DataType>(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TextractConvTileCheck(DstTileData &dst, SrcTileData &src)
{
    static_assert(std::is_same_v<typename DstTileData::DType, int8_t> ||
                      std::is_same_v<typename DstTileData::DType, uint8_t> ||
                      std::is_same_v<typename DstTileData::DType, int16_t> ||
                      std::is_same_v<typename DstTileData::DType, uint16_t> ||
                      std::is_same_v<typename DstTileData::DType, int32_t> ||
                      std::is_same_v<typename DstTileData::DType, uint32_t> ||
                      std::is_same_v<typename DstTileData::DType, half> ||
                      std::is_same_v<typename DstTileData::DType, bfloat16_t> ||
                      std::is_same_v<typename DstTileData::DType, float>,
                  "Fix: Data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/bfloat16_t/float!");
    static_assert(SrcTileData::Loc == pto::TileType::Mat, "Fix: Src TileType must be Mat!");
    static_assert(DstTileData::Loc == pto::TileType::Right, "Fix: Dst TileType must be Right!");
    static_assert(sizeof(typename DstTileData::DType) == sizeof(typename SrcTileData::DType),
                  "Fix: Source dtype must be same with dst dtype!");

    static_assert((SrcTileData::layout == Layout::FRACTAL_Z) || (SrcTileData::layout == Layout::FRACTAL_Z_3D),
                  "TExtract: Source layout only support FRACTAL_Z or FRACTAL_Z_3D.");
    static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
                  "TExtract: Destination layout only support SLayout is ColMajor ang BLayout is RowMajor.");
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_CONVTILE_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol)
{
    TextractConvTileCheck<DstTileData, SrcTileData>(dst, src);
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename SrcTileData::DType);
    if constexpr (SrcTileData::totalDimCount == 4) { // ConvTile layout is [C1HW,N/16,16,C0]
        static_assert(SrcTileData::staticShape[2] == FRACTAL_NZ_ROW && SrcTileData::staticShape[3] == c0ElemCount,
                      "Fix: The SrcTileData last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
        uint16_t srcCol = src.GetShape(1) * src.GetShape(2);
        TExtractToBConv<DstTileData, SrcTileData>(dst.data(), src.data(), srcCol, dst.GetValidRow(), dst.GetValidCol(),
                                                  indexRow, indexCol);
    } else { //  [C1,H,W,N,C0]
        TExtractToBConv<DstTileData, SrcTileData>(dst.data(), src.data(), src.GetShape(3), dst.GetValidRow(),
                                                  dst.GetValidCol(), indexRow, indexCol);
    }
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t idxRow, uint16_t idxCol)
{
    if constexpr (is_conv_tile_v<SrcTileData>) {
        TEXTRACT_CONVTILE_IMPL(dst, src, idxRow, idxCol);
    } else {
        constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
        TExtract_Impl<DstTileData, SrcTileData, QuantMode_t::NoQuant, useRelu>(dst, src, idxRow, idxCol);
    }
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint32_t idxRow, uint32_t idxCol)
{
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    TExtract_Impl<DstTileData, SrcTileData, QuantMode_t::NoQuant, useRelu>(dst, src, idxRow, idxCol);
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint32_t idxRow,
                                uint32_t idxCol)
{
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    std::vector<uint64_t> scalars(dst.GetValidCol(), preQuantScalar);

    TExtract_Impl<DstTileData, SrcTileData, quantPre, useRelu>(dst, src, idxRow, idxCol, scalars);
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint32_t idxRow,
                                uint32_t idxCol)
{
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    std::vector<uint64_t> scalars(dst.GetValidCol(), preQuantScalar);

    TExtract_Impl<DstTileData, SrcTileData, quantPre, useRelu>(dst, src, idxRow, idxCol, scalars);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint32_t idxRow, uint32_t idxCol)
{
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
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
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
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
