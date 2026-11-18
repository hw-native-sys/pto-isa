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
#include "ConvTileTransfer.hpp"
#include "../common/utils.hpp"
#include <cmath>

namespace pto {

template <typename DstTileData, typename SrcTileData, QuantMode_t quantMode, bool applyRelu, bool scalarQuant = false>
PTO_INTERNAL void TExtract_Impl(
    DstTileData& dst, SrcTileData& src, uint32_t idxRow, uint32_t idxCol, const std::vector<uint64_t>& scalars = {})
{
    assert(dst.GetValidRow() + idxRow <= src.GetValidRow() && dst.GetValidCol() + idxCol <= src.GetValidCol());

    using D = typename DstTileData::DType;
    using S = typename SrcTileData::DType;

    for (size_t c = 0; c < dst.GetValidCol(); c++) {
        // Scalar scales are broadcast over destination columns; vector scales follow source columns.
        const size_t scaleCol = scalarQuant ? c : c + idxCol;
        if constexpr (quantMode != QuantMode_t::NoQuant) {
            assert(scaleCol < scalars.size());
        }
        for (size_t r = 0; r < dst.GetValidRow(); r++) {
            if constexpr (quantMode != QuantMode_t::NoQuant) {
                dst.SetElement(
                    r, c,
                    quantize_element<D, S, quantMode, applyRelu>(
                        src.GetElement(r + idxRow, c + idxCol), scalars[scaleCol]));
            } else {
                S val = src.GetElement(r + idxRow, c + idxCol);
                if constexpr (applyRelu) {
                    val = ReLU(val);
                }
                dst.SetElement(r, c, val);
            }
        }
    }
}

template <typename T>
__tf__ PTO_INTERNAL void copy_fractal_shapes(
    T* dst, T* src, uint16_t indexC, uint16_t indexR, uint16_t dstC, uint16_t dstR, uint16_t srcStride,
    uint16_t dstStride)
{
    // All indices and strides represent number of fractals
    constexpr const uint16_t BLOCK_SIZE_ELEM = CUBE_BLOCK_SIZE / sizeof(T);
    src = src + indexR * srcStride * BLOCK_SIZE_ELEM;
    const size_t kCopySize = dstC * BLOCK_SIZE_ELEM;
    cpu::parallel_for_rows(dstR, dstR * dstC * BLOCK_SIZE_ELEM, [&](std::size_t r) {
        T* srcBase = src + (r * srcStride + indexC) * BLOCK_SIZE_ELEM;
        T* dstBase = dst + r * dstStride * BLOCK_SIZE_ELEM;
        std::copy(srcBase, srcBase + kCopySize, dstBase);
    });
}

template <typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void TExtractToBConv(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t srcCol,
    uint16_t dstValidRow, uint16_t dstValidCol, uint16_t indexRow, uint16_t indexCol)
{
    using DataType = typename SrcTileData::DType;
    constexpr int c0Size = BLOCK_BYTE_SIZE / sizeof(DataType);

    __cbuf__ DataType* srcAddr = (__cbuf__ DataType*)__cce_get_tile_ptr(src);
    __cb__ DataType* dstAddr = (__cb__ DataType*)__cce_get_tile_ptr(dst);
    uint16_t dstValidColAlign = CeilDivision(dstValidCol, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    uint16_t dstValidRowAlign = CeilDivision(dstValidRow, c0Size) * c0Size;

    uint16_t mStartPosition = indexCol >> SHIFT_BLOCK_LEN;
    uint16_t kStartPosition = (indexRow * sizeof(DataType)) >> SHIFT_BLOCK_BYTE;
    uint16_t mStep = dstValidColAlign >> SHIFT_BLOCK_LEN;
    uint16_t kStep = (dstValidRowAlign * sizeof(DataType)) >> SHIFT_BLOCK_BYTE;
    uint16_t srcStride = srcCol >> SHIFT_BLOCK_LEN;
    uint16_t dstStride = DstTileData::Cols >> SHIFT_BLOCK_LEN;

    copy_fractal_shapes<DataType>(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride);
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TextractConvTileCheck(DstTileData& dst, SrcTileData& src)
{
    static_assert(
        std::is_same_v<typename DstTileData::DType, int8_t> || std::is_same_v<typename DstTileData::DType, uint8_t> ||
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
    static_assert(
        sizeof(typename DstTileData::DType) == sizeof(typename SrcTileData::DType),
        "Fix: Source dtype must be same with dst dtype!");

    static_assert(
        (SrcTileData::layout == Layout::FRACTAL_Z) || (SrcTileData::layout == Layout::FRACTAL_Z_3D),
        "TExtract: Source layout only support FRACTAL_Z or FRACTAL_Z_3D.");
    static_assert(
        DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
        "TExtract: Destination layout only support SLayout is ColMajor ang BLayout is RowMajor.");
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_CONVTILE_IMPL(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol)
{
    TextractConvTileCheck<DstTileData, SrcTileData>(dst, src);
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename SrcTileData::DType);
    if constexpr (SrcTileData::totalDimCount == 4) { // ConvTile layout is [C1HW,N/16,16,C0]
        static_assert(
            SrcTileData::staticShape[2] == FRACTAL_NZ_ROW && SrcTileData::staticShape[3] == c0ElemCount,
            "Fix: The SrcTileData last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
        uint16_t srcCol = src.GetShape(1) * src.GetShape(2);
        TExtractToBConv<DstTileData, SrcTileData>(
            dst.data(), src.data(), srcCol, dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
    } else { //  [C1,H,W,N,C0]
        TExtractToBConv<DstTileData, SrcTileData>(
            dst.data(), src.data(), src.GetShape(3), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
    }
}

template <
    typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData& dst, SrcTileData& src, uint16_t idxRow, uint16_t idxCol)
{
    (void)Phase;
    if constexpr (is_conv_tile_v<DstTileData> && SrcTileData::Loc == TileType::Acc) {
        constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
        AccToConvTileTransfer<false, DstTileData, SrcTileData, QuantMode_t::NoQuant, useRelu>(dst, src, idxRow, idxCol);
    } else if constexpr (is_conv_tile_v<SrcTileData>) {
        TEXTRACT_CONVTILE_IMPL(dst, src, idxRow, idxCol);
    } else {
        constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
        TExtract_Impl<DstTileData, SrcTileData, QuantMode_t::NoQuant, useRelu>(dst, src, idxRow, idxCol);
    }
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData& dst, SrcTileData& src, uint32_t idxRow, uint32_t idxCol)
{
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    if constexpr (is_conv_tile_v<DstTileData> && SrcTileData::Loc == TileType::Acc) {
        AccToConvTileTransfer<false, DstTileData, SrcTileData, QuantMode_t::NoQuant, useRelu>(dst, src, idxRow, idxCol);
    } else {
        TExtract_Impl<DstTileData, SrcTileData, QuantMode_t::NoQuant, useRelu>(dst, src, idxRow, idxCol);
    }
}

template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TEXTRACT_IMPL(
    DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, uint32_t idxRow, uint32_t idxCol)
{
    (void)Phase;
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    if constexpr (is_conv_tile_v<DstTileData> && SrcTileData::Loc == TileType::Acc) {
        std::vector<uint64_t> scalars(dst.GetValidShape(1), preQuantScalar);
        AccToConvTileTransfer<false, DstTileData, SrcTileData, quantPre, useRelu, true>(
            dst, src, idxRow, idxCol, scalars);
    } else {
        std::vector<uint64_t> scalars(dst.GetValidCol(), preQuantScalar);
        TExtract_Impl<DstTileData, SrcTileData, quantPre, useRelu, true>(dst, src, idxRow, idxCol, scalars);
    }
}

template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(
    DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, uint32_t idxRow, uint32_t idxCol)
{
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;
    if constexpr (is_conv_tile_v<DstTileData> && SrcTileData::Loc == TileType::Acc) {
        std::vector<uint64_t> scalars(dst.GetValidShape(1), preQuantScalar);
        AccToConvTileTransfer<false, DstTileData, SrcTileData, quantPre, useRelu, true>(
            dst, src, idxRow, idxCol, scalars);
    } else {
        std::vector<uint64_t> scalars(dst.GetValidCol(), preQuantScalar);
        TExtract_Impl<DstTileData, SrcTileData, quantPre, useRelu, true>(dst, src, idxRow, idxCol, scalars);
    }
}

template <
    typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData& dst, SrcTileData& src, FpTileData& fp, uint32_t idxRow, uint32_t idxCol)
{
    (void)Phase;
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;

    std::vector<uint64_t> scalars(fp.GetValidCol(), 0);
    for (size_t i = 0; i < fp.GetValidCol(); i++) {
        const size_t quantTileIdx = GetTileElementOffset<FpTileData>(0, i);
        scalars[i] = fp.data()[quantTileIdx];
    }
    if constexpr (is_conv_tile_v<DstTileData> && SrcTileData::Loc == TileType::Acc) {
        AccToConvTileTransfer<false, DstTileData, SrcTileData, quantPre, useRelu>(dst, src, idxRow, idxCol, scalars);
    } else {
        TExtract_Impl<DstTileData, SrcTileData, quantPre, useRelu>(dst, src, idxRow, idxCol, scalars);
    }
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData& dst, SrcTileData& src, FpTileData& fp, uint32_t idxRow, uint32_t idxCol)
{
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    constexpr bool useRelu = reluMode == ReluPreMode::NormalRelu;

    std::vector<uint64_t> scalars(fp.GetValidCol(), 0);
    for (size_t i = 0; i < fp.GetValidCol(); i++) {
        const size_t quantTileIdx = GetTileElementOffset<FpTileData>(0, i);
        scalars[i] = fp.data()[quantTileIdx];
    }
    if constexpr (is_conv_tile_v<DstTileData> && SrcTileData::Loc == TileType::Acc) {
        AccToConvTileTransfer<false, DstTileData, SrcTileData, quantPre, useRelu>(dst, src, idxRow, idxCol, scalars);
    } else {
        TExtract_Impl<DstTileData, SrcTileData, quantPre, useRelu>(dst, src, idxRow, idxCol, scalars);
    }
}

// Structural checks for the two-destination ND->2xNZ overload. The CPU simulator is
// layout-aware through GetElement/SetElement (tile_offsets.hpp maps the logical (r, c)
// to the NZ physical offset), so the copy itself is element-wise; these checks pin the
// interface to the same ND source / NZ destination contract as the A5/A2A3 backends
// (see docs/isa/TEXTRACT.md, "ND -> 2xNZ extraction path").
template <typename T, typename DstTileData, typename SrcTileData>
PTO_INTERNAL void CheckTExtractNdToNzCpu()
{
    static_assert(
        SrcTileData::Loc == TileType::Vec && DstTileData::Loc == TileType::Vec,
        "TEXTRACT ND->2xNZ (CPU): Source and destinations must be Vec (UB) tiles.");
    static_assert(
        SrcTileData::isRowMajor && SrcTileData::SFractal == SLayout::NoneBox,
        "TEXTRACT ND->2xNZ (CPU): Source must be ND (RowMajor, NoneBox).");
    static_assert(
        !DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor,
        "TEXTRACT ND->2xNZ (CPU): Destination must be NZ (ColMajor, RowMajor fractal).");
    static_assert(
        std::is_same_v<typename DstTileData::DType, typename SrcTileData::DType>,
        "TEXTRACT ND->2xNZ (CPU): Source and destination data types must match.");
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / sizeof(T);
    static_assert(DstTileData::Cols % c0Size == 0, "TEXTRACT ND->2xNZ (CPU): Destination cols must be c0-aligned.");
    static_assert(
        (SrcTileData::RowStride * sizeof(T)) % BLOCK_BYTE_SIZE == 0,
        "TEXTRACT ND->2xNZ (CPU): Source row stride must be 32-byte aligned.");
}

// Two-destination ND->2xNZ extraction: extracts two independent ND sub-windows from a
// single ND source and writes each as a separate NZ destination. Each window k copies
// dst_k[r][c] = src[indexRow_k + r][indexCol_k + c] for the valid window, leaving the
// rest of the destination tile untouched (callers zero the padding, e.g. ZeroDst).
template <typename Dst0TileData, typename Dst1TileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_ND2XNZ_IMPL(
    Dst0TileData& dst0, Dst1TileData& dst1, SrcTileData& src, uint16_t indexRow0, uint16_t indexCol0,
    uint16_t indexRow1, uint16_t indexCol1)
{
    using T = typename SrcTileData::DType;
    CheckTExtractNdToNzCpu<T, Dst0TileData, SrcTileData>();
    CheckTExtractNdToNzCpu<T, Dst1TileData, SrcTileData>();

    const uint32_t validRow0 = dst0.GetValidRow();
    const uint32_t validCol0 = dst0.GetValidCol();
    const uint32_t validRow1 = dst1.GetValidRow();
    const uint32_t validCol1 = dst1.GetValidCol();

    PTO_CPU_ASSERT(
        indexRow0 + validRow0 <= SrcTileData::Rows,
        "TEXTRACT ND->2xNZ (CPU): window0 indexRow + validRow exceeds srcRows!");
    PTO_CPU_ASSERT(
        indexCol0 + validCol0 <= SrcTileData::Cols,
        "TEXTRACT ND->2xNZ (CPU): window0 indexCol + validCol exceeds srcCols!");
    PTO_CPU_ASSERT(
        indexRow1 + validRow1 <= SrcTileData::Rows,
        "TEXTRACT ND->2xNZ (CPU): window1 indexRow + validRow exceeds srcRows!");
    PTO_CPU_ASSERT(
        indexCol1 + validCol1 <= SrcTileData::Cols,
        "TEXTRACT ND->2xNZ (CPU): window1 indexCol + validCol exceeds srcCols!");

    for (size_t c = 0; c < validCol0; c++) {
        for (size_t r = 0; r < validRow0; r++) {
            dst0.SetElement(r, c, src.GetElement(r + indexRow0, c + indexCol0));
        }
    }
    for (size_t c = 0; c < validCol1; c++) {
        for (size_t r = 0; r < validRow1; r++) {
            dst1.SetElement(r, c, src.GetElement(r + indexRow1, c + indexCol1));
        }
    }
}

} // namespace pto
#endif // TEXTRACT_HPP
