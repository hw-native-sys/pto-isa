/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TEXTRACTOP_HPP
#define TEXTRACTOP_HPP
#include "common.hpp"

namespace pto {

template <typename DstTileData, typename SrcTileData, bool Transpose>
AICORE void TExtractToA(std::vector<CostModelStats> &stats, uint16_t indexRow, uint16_t indexCol)
{
    using SrcType = std::conditional_t<(sizeof(typename SrcTileData::DType) == 2), half, typename SrcTileData::DType>;
    constexpr int32_t srcR = SrcTileData::Rows;
    constexpr int32_t srcC = SrcTileData::Cols;
    constexpr int32_t dstR = DstTileData::Rows;
    constexpr int32_t dstC = DstTileData::Cols;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(SrcType);
    constexpr int32_t fractalSize = (sizeof(SrcType) == 1) ? 32 : 16;

    if constexpr (!Transpose) {
        // srcRow/srcCol/dstRow/dstCol对齐校验
        static_assert((srcR % 16) == 0, "srcRow must be aligned to 16");
        static_assert((srcC % c0Size) == 0, "srcCol must be aligned to C0Size");
        static_assert((dstR % 16) == 0, "dstRow must be aligned to 16");
        static_assert((dstC % c0Size) == 0, "dstCol must be aligned to C0Size");
        PTO_ASSERT((indexRow % 16) == 0, "indexRow must be aligned to 16");
        PTO_ASSERT((indexCol % c0Size) == 0, "indexCol must be aligned to C0Size");
    } else {
        // L1->L0A:load_cbuf_to_ca_transpose
        static_assert((srcR % fractalSize) == 0, "srcRow must be aligned");
        static_assert((srcC % fractalSize) == 0, "srcCol must be aligned");
        static_assert((dstR % fractalSize) == 0, "dstRow must be aligned");
        static_assert((dstC % fractalSize) == 0, "dstCol must be aligned");
        PTO_ASSERT((indexRow % fractalSize) == 0, "indexRow must be aligned");
        PTO_ASSERT((indexCol % fractalSize) == 0, "indexCol must be aligned");
    }
}

template <typename DstTileData, typename SrcTileData>
AICORE void TExtractToAVector(std::vector<CostModelStats> &stats, uint16_t indexRow, uint16_t indexCol,
                              uint16_t dstValidCol)
{
    using DataType = typename SrcTileData::DType;

    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr int32_t dstCol = DstTileData::Cols;
    constexpr int32_t fractalSize = CUBE_BLOCK_SIZE / sizeof(DataType);

    static_assert((srcCol % fractalSize) == 0, "srcCol * sizeof(DataType) must be aligned to 512B");
    static_assert((dstCol % fractalSize) == 0, "dstCol * sizeof(DataType) must be aligned to 512B");
    PTO_ASSERT((indexCol % fractalSize) == 0, "indexCol * sizeof(DataType) must be aligned to 512B");
}

template <typename DstTileData, typename SrcTileData, bool Transpose>
AICORE void TExtractToB(std::vector<CostModelStats> &stats, uint16_t indexRow, uint16_t indexCol)
{
    using SrcType = std::conditional_t<(sizeof(typename SrcTileData::DType) == 2), half, typename SrcTileData::DType>;
    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr int32_t dstRow = DstTileData::Rows;
    constexpr int32_t dstCol = DstTileData::Cols;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(SrcType);
    constexpr int32_t fractalSize = (sizeof(SrcType) == 1) ? 32 : 16;

    if constexpr (!Transpose) {
        static_assert((srcRow % c0Size) == 0, "srcRow must be aligned to C0Size");
        static_assert((srcCol % 16) == 0, "srcCol must be aligned to 16");
        static_assert((dstRow % c0Size) == 0, "dstRow must be aligned to C0Size");
        static_assert((dstCol % 16) == 0, "dstCol must be aligned to 16");
        PTO_ASSERT((indexRow % c0Size) == 0, "indexRow must be aligned to c0Size");
        PTO_ASSERT((indexCol % 16) == 0, "indexCol must be aligned to 16");
    } else {
        static_assert((srcRow % fractalSize) == 0, "srcRow must be aligned");
        static_assert((srcCol % fractalSize) == 0, "srcCol must be aligned");
        static_assert((dstRow % fractalSize) == 0, "dstRow must be aligned");
        static_assert((dstCol % fractalSize) == 0, "dstCol must be aligned");
        PTO_ASSERT((indexRow % fractalSize) == 0, "indexRow must be aligned");
        PTO_ASSERT((indexCol % fractalSize) == 0, "indexCol must be aligned");
    }
}

template <typename DstTileData, typename SrcTileData, bool Transpose>
AICORE void TExtractToACompact(std::vector<CostModelStats> &stats, uint16_t indexRow, uint16_t indexCol,
                               uint16_t dstValidRow, uint16_t dstValidCol, bool isKAligned)
{
    using SrcType = std::conditional_t<(sizeof(typename SrcTileData::DType) == 2), half, typename SrcTileData::DType>;
    using DstType = std::conditional_t<(sizeof(typename DstTileData::DType) == 2), half, typename DstTileData::DType>;

    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(SrcType);
    constexpr int32_t fractalSize = (sizeof(SrcType) == 1) ? 32 : 16;
    if constexpr (!Transpose) {
        // srcRow/srcCol/dstRow/dstCol check
        static_assert((SrcTileData::Rows % 16) == 0, "srcRow must be aligned to 16");
        static_assert((SrcTileData::Cols % c0Size) == 0, "srcCol must be aligned to C0Size");
        PTO_ASSERT((indexRow % 16) == 0, "indexRow must be aligned to 16");
        PTO_ASSERT((indexCol % c0Size) == 0, "indexCol must be aligned to C0Size");
    } else {
        // L1->L0A:load_cbuf_to_ca_transpose
        static_assert((SrcTileData::Rows % fractalSize) == 0, "srcRow must be aligned");
        static_assert((SrcTileData::Cols % fractalSize) == 0, "srcCol must be aligned");
        PTO_ASSERT((indexRow % fractalSize) == 0, "indexRow must be aligned");
        PTO_ASSERT((indexCol % fractalSize) == 0, "indexCol must be aligned");
    }
}

template <typename DstTileData, typename SrcTileData, bool Transpose>
AICORE void TExtractToBCompact(std::vector<CostModelStats> &stats, uint16_t indexRow, uint16_t indexCol,
                               uint16_t dstValidRow, uint16_t dstValidCol)
{
    using SrcType = std::conditional_t<(sizeof(typename SrcTileData::DType) == 2), half, typename SrcTileData::DType>;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(SrcType);
    constexpr int32_t fractalSize = (sizeof(SrcType) == 1) ? 32 : 16;
    static_assert((DstTileData::Rows % c0Size) == 0, "dstRow must be aligned to C0Size");
    static_assert((DstTileData::Cols % 16) == 0, "dstCol must be aligned to 16");
    if constexpr (!Transpose) {
        static_assert((SrcTileData::Rows % c0Size) == 0, "srcRow must be aligned to C0Size");
        static_assert((SrcTileData::Cols % 16) == 0, "srcCol must be aligned to 16");
        PTO_ASSERT((indexRow % c0Size) == 0, "indexRow must be aligned to c0Size");
        PTO_ASSERT((indexCol % 16) == 0, "indexCol must be aligned to 16");
    } else {
        static_assert((SrcTileData::Rows % fractalSize) == 0, "srcRow must be aligned");
        static_assert((SrcTileData::Cols % fractalSize) == 0, "srcCol must be aligned");
        PTO_ASSERT((indexRow % fractalSize) == 0, "indexRow must be aligned");
        PTO_ASSERT((indexCol % fractalSize) == 0, "indexCol must be aligned");
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_CONVTILE_IMPL(std::vector<CostModelStats> &stats, DstTileData &dst, SrcTileData &src,
                                         uint16_t indexRow, uint16_t indexCol)
{
    static_assert(SrcTileData::Loc == pto::TileType::Mat, "Fix: Src TileType must be Mat!");
    static_assert(DstTileData::Loc == pto::TileType::Right, "Fix: Dst TileType must be Right!");
    static_assert(sizeof(typename DstTileData::DType) == sizeof(typename SrcTileData::DType),
                  "Fix: Source dtype must be same with dst dtype!");

    static_assert((SrcTileData::layout == Layout::FRACTAL_Z) || (SrcTileData::layout == Layout::FRACTAL_Z_3D),
                  "TExtract: Source layout only support FRACTAL_Z or FRACTAL_Z_3D.");
    static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
                  "TExtract: Destination layout only support SLayout is ColMajor ang BLayout is RowMajor.");
    static_assert(std::is_same<typename DstTileData::DType, int8_t>::value ||
                      std::is_same<typename DstTileData::DType, half>::value ||
                      std::is_same<typename DstTileData::DType, bfloat16_t>::value ||
                      std::is_same<typename DstTileData::DType, float>::value,
                  "TExtract: Invalid data type.");

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename SrcTileData::DType);
    if constexpr (SrcTileData::totalDimCount == 4) { // ConvTile layout is [C1HW,N/16,16,C0]
        int srcCol = src.GetShape(1) * src.GetShape(2);
        TExtractToBConv<DstTileData, SrcTileData>(stats, srcCol, dst.GetValidRow(), dst.GetValidCol(), indexRow,
                                                  indexCol);
    } else { //  [C1,H,W,N,C0]
        TExtractToBConv<DstTileData, SrcTileData>(stats, src.GetShape(3), dst.GetValidRow(), dst.GetValidCol(),
                                                  indexRow, indexCol);
    }
}

template <typename DstTileData, typename SrcTileData>
AICORE void TExtractToLeft(std::vector<CostModelStats> &stats, DstTileData &dst, uint16_t indexRow, uint16_t indexCol)
{
    static_assert((SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor) ||
                      (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor) ||
                      (SrcTileData::Rows == 1 && SrcTileData::isRowMajor),
                  "TExtract: SrcTile Invalid Fractal");
    static_assert(DstTileData::SFractal == SLayout::RowMajor && DstTileData::isRowMajor,
                  "TExtract: LeftTile Invalid Fractal.");
    if constexpr (SrcTileData::Rows == 1 && SrcTileData::isRowMajor) {
        TExtractToAVector<DstTileData, SrcTileData>(stats, indexRow, indexCol, dst.GetValidCol());
    } else if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
        if constexpr (DstTileData::Compact == CompactMode::Normal) {
            TExtractToACompact<DstTileData, SrcTileData, false>(stats, indexRow, indexCol, dst.GetValidRow(),
                                                                dst.GetValidCol(), dst.GetKAligned());
        } else {
            TExtractToA<DstTileData, SrcTileData, false>(stats, indexRow, indexCol);
        }
    } else {
        if constexpr (DstTileData::Compact == CompactMode::Normal) {
            TExtractToACompact<DstTileData, SrcTileData, true>(stats, indexRow, indexCol, dst.GetValidRow(),
                                                               dst.GetValidCol(), dst.GetKAligned());
        } else {
            TExtractToA<DstTileData, SrcTileData, true>(stats, indexRow, indexCol);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
AICORE void TExtractToRight(std::vector<CostModelStats> &stats, DstTileData &dst, uint16_t indexRow, uint16_t indexCol)
{
    static_assert((SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor) ||
                      (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor),
                  "TExtract: SrcTile Invalid Fractal");
    static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
                  "TExtract: RightTile Invalid Fractal.");
    if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
        if constexpr (DstTileData::Compact == CompactMode::Normal) {
            TExtractToBCompact<DstTileData, SrcTileData, false>(stats, indexRow, indexCol, dst.GetValidRow(),
                                                                dst.GetValidCol());
        } else {
            TExtractToB<DstTileData, SrcTileData, false>(stats, indexRow, indexCol);
        }
    } else {
        if constexpr (DstTileData::Compact == CompactMode::Normal) {
            TExtractToBCompact<DstTileData, SrcTileData, true>(stats, indexRow, indexCol, dst.GetValidRow(),
                                                               dst.GetValidCol());
        } else {
            TExtractToB<DstTileData, SrcTileData, true>(stats, indexRow, indexCol);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_TILE_IMPL(std::vector<CostModelStats> &stats, DstTileData &dst, SrcTileData &src,
                                     uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    PTO_ASSERT(indexRow + DstTileData::Rows <= SrcTileData::Rows,
               "The sum of indexRow and dstRow should be less than srcRow!");
    PTO_ASSERT(indexCol + DstTileData::Cols <= SrcTileData::Cols,
               "The sum of indexCol and dstCol should be less than srcCol!");
    if constexpr (DstTileData::Loc == TileType::Left) {
        TExtractToLeft<DstTileData, SrcTileData>(stats, dst, indexRow, indexCol);
    } else if constexpr (DstTileData::Loc == TileType::Right) {
        TExtractToRight<DstTileData, SrcTileData>(stats, dst, indexRow, indexCol);
    } else if constexpr (SrcTileData::Loc == TileType::Acc && DstTileData::Loc == TileType::Mat) {
        constexpr QuantMode_t quantPre =
            GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
        TExtractAccToMat<DstTileData, SrcTileData, quantPre, ReluPreMode::NoRelu>(
            stats, dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void runTExtractOp(std::vector<CostModelStats> &stats, DstTileData &dst, SrcTileData &src,
                                uint16_t indexRow, uint16_t indexCol)
{
    if constexpr (is_conv_tile_v<SrcTileData>) {
        TEXTRACT_CONVTILE_IMPL(stats, dst, src, indexRow, indexCol);
    } else {
        TEXTRACT_TILE_IMPL(stats, dst, src, indexRow, indexCol);
    }
}
} // namespace pto
#endif
