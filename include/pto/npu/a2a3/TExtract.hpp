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
#include "common.hpp"
#include "pto/common/arch/memory/textract_common.hpp"

namespace pto {

template <typename DstTileData, typename SrcTileData, typename DstType, typename SrcType>
PTO_INTERNAL void CheckTExtract()
{
    static_assert(
        (SrcTileData::Loc == TileType::Acc) || std::is_same<DstType, SrcType>::value,
        "TExtract: Destination and Source tile data types must be the same.");
    static_assert(
        std::is_same<DstType, int8_t>::value || std::is_same<DstType, half>::value ||
            std::is_same<DstType, bfloat16_t>::value || std::is_same<DstType, float>::value,
        "TExtract: Invalid data type.");
}

template <
    typename DstTileData, typename SrcTileData, QuantMode_t QuantPre, ReluPreMode reluMode,
    STPhase Phase = STPhase::Unspecified>
__tf__ AICORE void TExtractAccToMat(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint16_t indexRow, uint16_t indexCol)
{
    using SrcType = typename SrcTileData::DType;
    using DstType = typename DstTileData::DType;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(DstType);
    uint32_t srcOffset = SrcTileData::Rows * ACC_C0_SIZE * (indexCol / ACC_C0_SIZE) +
                         (indexRow * ACC_C0_SIZE + (indexCol % ACC_C0_SIZE));
    __cc__ SrcType* srcAddr = (__cc__ SrcType*)__cce_get_tile_ptr(src) + srcOffset;
    __cbuf__ DstType* dstAddr = (__cbuf__ DstType*)__cce_get_tile_ptr(dst);

    constexpr uint32_t dstStrideD = DstTileData::Rows;
    constexpr uint16_t srcStride = SrcTileData::Rows;
    uint16_t nSize = CeilDivision(validCol, c0Size) * c0Size;
    constexpr uint8_t unitFlagCtrl = static_cast<uint8_t>(Phase);
    pto_copy_matrix_cc_to_cbuf(
        dstAddr, srcAddr, 0, nSize, validRow, dstStrideD, srcStride, unitFlagCtrl, QuantPre,
        static_cast<uint8_t>(reluMode), false, false);
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractToAVector(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol, uint16_t dstValidCol)
{
    using DataType = typename SrcTileData::DType;
    __cbuf__ DataType* srcAddr = (__cbuf__ DataType*)__cce_get_tile_ptr(src);
    __ca__ DataType* dstAddr = (__ca__ DataType*)__cce_get_tile_ptr(dst);

    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr int32_t dstCol = DstTileData::Cols;
    constexpr int32_t fractalSize = CUBE_BLOCK_SIZE / sizeof(DataType);

    static_assert((srcCol % fractalSize) == 0, "srcCol * sizeof(DataType) must be aligned to 512B");
    static_assert((dstCol % fractalSize) == 0, "dstCol * sizeof(DataType) must be aligned to 512B");
    PTO_ASSERT((indexCol % fractalSize) == 0, "indexCol * sizeof(DataType) must be aligned to 512B");

    int32_t kAlign = (dstValidCol + fractalSize - 1) & ~(fractalSize - 1);
    uint16_t baseIdx = indexCol * sizeof(DataType) >> SHIFT_FRACTAL_BYTE;
    uint8_t repeatTimes = kAlign / fractalSize;
    pto_load_cbuf_to_ca(dstAddr, srcAddr, baseIdx, repeatTimes, 1, 0);
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL constexpr bool IsTExtractMatToLeftSmallM()
{
    using T = typename SrcTileData::DType;
    return SrcTileData::Loc == TileType::Mat && DstTileData::Loc == TileType::Left &&
           (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) &&
           std::is_same_v<T, typename DstTileData::DType> && !SrcTileData::isRowMajor &&
           SrcTileData::SFractal == SLayout::RowMajor && SrcTileData::SFractalSize == CUBE_BLOCK_SIZE &&
           DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor &&
           DstTileData::SFractalSize == CUBE_BLOCK_SIZE && SrcTileData::Rows % FRACTAL_NZ_ROW == 0 &&
           DstTileData::Rows == FRACTAL_NZ_ROW &&
           (DstTileData::Compact == CompactMode::Null || DstTileData::Compact == CompactMode::Normal) &&
           (DstTileData::ValidRow == DYNAMIC || (DstTileData::ValidRow > 0 && DstTileData::ValidRow < FRACTAL_NZ_ROW));
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractMatToLeftSmallM(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol, uint32_t validRow, uint32_t validCol, uint32_t srcValidRow, uint32_t srcValidCol)
{
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / sizeof(half);
    constexpr uint32_t fractalElements = CUBE_BLOCK_SIZE / sizeof(half);
    constexpr uint16_t srcStride = SrcTileData::Rows / FRACTAL_NZ_ROW;
    static_assert(SrcTileData::Cols % c0Size == 0, "srcCol must be aligned to C0Size");
    static_assert(DstTileData::Cols % c0Size == 0, "dstCol must be aligned to C0Size");
    __cbuf__ half* srcAddr = (__cbuf__ half*)__cce_get_tile_ptr(src);
    __ca__ half* dstAddr = (__ca__ half*)__cce_get_tile_ptr(dst);
    // Keep dynamic selection within one tile function so auto mode sees one write to Left.
    if constexpr (DstTileData::ValidRow == DYNAMIC) {
        if (validRow == 0 || validRow >= FRACTAL_NZ_ROW) {
            PTO_ASSERT(
                indexRow + DstTileData::Rows <= SrcTileData::Rows,
                "The sum of indexRow and dstRow should be less than srcRow!");
            PTO_ASSERT(
                indexCol + DstTileData::Cols <= SrcTileData::Cols,
                "The sum of indexCol and dstCol should be less than srcCol!");
            PTO_ASSERT(indexRow % FRACTAL_NZ_ROW == 0, "indexRow must be aligned to 16");
            PTO_ASSERT(indexCol % c0Size == 0, "indexCol must be aligned to C0Size");
            if constexpr (DstTileData::Compact == CompactMode::Normal) {
                TExtractToANonTransposeCompact<half, half, SrcTileData::Rows, SrcTileData::Cols>(
                    dstAddr, srcAddr, indexRow, indexCol, CeilDivision(validRow, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW,
                    CeilDivision(validCol, c0Size) * c0Size);
            } else {
                TExtractToANonTranspose<
                    half, half, SrcTileData::Rows, SrcTileData::Cols, DstTileData::Rows, DstTileData::Cols>(
                    dstAddr, srcAddr, indexRow, indexCol);
            }
            return;
        }
    }
    PTO_ASSERT(validRow > 0 && validRow < FRACTAL_NZ_ROW, "TEXTRACT Mat->Left: valid rows must be in [1, 15].");
    PTO_ASSERT(validCol > 0 && validCol <= DstTileData::Cols, "TEXTRACT Mat->Left: invalid valid columns.");
    PTO_ASSERT(indexRow + validRow <= srcValidRow, "TEXTRACT Mat->Left: window exceeds source valid rows.");
    PTO_ASSERT(indexCol + validCol <= srcValidCol, "TEXTRACT Mat->Left: window exceeds source valid columns.");
    PTO_ASSERT(indexCol % c0Size == 0, "TEXTRACT Mat->Left: indexCol must be C0-aligned.");
    uint32_t copyCols =
        DstTileData::Compact == CompactMode::Normal ? CeilDivision(validCol, c0Size) * c0Size : DstTileData::Cols;
    PTO_ASSERT(indexCol + copyCols <= SrcTileData::Cols, "TEXTRACT Mat->Left: copy exceeds source physical columns.");
    // load2d reads a full 512B fractal even when only its first few rows are valid.
    uint64_t srcEndBytes = static_cast<uint64_t>(indexCol + copyCols - c0Size) * SrcTileData::Rows * sizeof(half) +
                           static_cast<uint64_t>(indexRow) * BLOCK_BYTE_SIZE + CUBE_BLOCK_SIZE;
    PTO_ASSERT(
        srcEndBytes <= static_cast<uint64_t>(SrcTileData::Numel) * sizeof(half),
        "TEXTRACT Mat->Left: full-fractal read exceeds source storage; reserve physical column padding.");
    PTO_ASSERT(copyCols <= DstTileData::Cols, "TEXTRACT Mat->Left: copy exceeds destination storage.");
    uint32_t srcOffset = static_cast<uint32_t>(indexCol / c0Size) * SrcTileData::Rows * c0Size +
                         static_cast<uint32_t>(indexRow) * c0Size;
    srcAddr += srcOffset;
    uint32_t remaining = copyCols / c0Size;
    if constexpr (DstTileData::Cols / c0Size <= REPEAT_MAX) {
        pto_load_cbuf_to_ca(dstAddr, srcAddr, 0, static_cast<uint8_t>(remaining), srcStride, 0);
        return;
    }
    while (remaining > 0) {
        uint8_t repeat = remaining > REPEAT_MAX ? static_cast<uint8_t>(REPEAT_MAX) : static_cast<uint8_t>(remaining);
        pto_load_cbuf_to_ca(dstAddr, srcAddr, 0, repeat, srcStride, 0);
        srcAddr += static_cast<uint32_t>(repeat) * srcStride * fractalElements;
        dstAddr += static_cast<uint32_t>(repeat) * fractalElements;
        remaining -= repeat;
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_TILE_IMPL(DstTileData& dst, SrcTileData& src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    CheckTExtract<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType>();
    if constexpr (IsTExtractMatToLeftSmallM<DstTileData, SrcTileData>()) {
        TExtractMatToLeftSmallM<DstTileData, SrcTileData>(
            dst.data(), src.data(), indexRow, indexCol, dst.GetValidRow(), dst.GetValidCol(), src.GetValidRow(),
            src.GetValidCol());
        return;
    }
    PTO_ASSERT(
        indexRow + DstTileData::Rows <= SrcTileData::Rows,
        "The sum of indexRow and dstRow should be less than srcRow!");
    PTO_ASSERT(
        indexCol + DstTileData::Cols <= SrcTileData::Cols,
        "The sum of indexCol and dstCol should be less than srcCol!");
    if constexpr (DstTileData::Loc == TileType::Left) {
        TExtractToLeft<DstTileData, SrcTileData>(dst, src, indexRow, indexCol);
    } else if constexpr (DstTileData::Loc == TileType::Right) {
        TExtractToRight<DstTileData, SrcTileData>(dst, src, indexRow, indexCol);
    } else if constexpr (SrcTileData::Loc == TileType::Acc && DstTileData::Loc == TileType::Mat) {
        CheckTMovAccToMat<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
        constexpr QuantMode_t quantPre =
            GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
        TExtractAccToMat<DstTileData, SrcTileData, quantPre, ReluPreMode::NoRelu>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractVecToVecNDAligned(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint32_t indexRow, uint32_t indexCol)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;
    constexpr uint32_t dstRowStride = DstTileData::RowStride;
    __ubuf__ T* srcStart = srcAddr + indexRow * srcRowStride + indexCol;
    uint32_t rowBytes = static_cast<uint32_t>(validCol) * sizeof(T);
    if (validCol == dstRowStride && validCol == srcRowStride) {
        uint32_t totalBytes = static_cast<uint32_t>(validRow) * rowBytes;
        uint16_t burstLen = static_cast<uint16_t>(totalBytes / BLOCK_BYTE_SIZE);
        pto_copy_ubuf_to_ubuf((__ubuf__ void*)dstAddr, (__ubuf__ void*)srcStart, 1, burstLen, 0, 0);
    } else {
        uint16_t rowBurst = static_cast<uint16_t>(rowBytes / BLOCK_BYTE_SIZE);
        uint16_t srcGap = static_cast<uint16_t>((srcRowStride - validCol) * sizeof(T) / BLOCK_BYTE_SIZE);
        uint16_t dstGap = static_cast<uint16_t>((dstRowStride - validCol) * sizeof(T) / BLOCK_BYTE_SIZE);
        pto_copy_ubuf_to_ubuf((__ubuf__ void*)dstAddr, (__ubuf__ void*)srcStart, validRow, rowBurst, srcGap, dstGap);
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractVecToVecNDUnaligned(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint32_t indexRow, uint32_t indexCol)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;
    constexpr uint32_t dstRowStride = DstTileData::RowStride;
    __ubuf__ T* srcStart = srcAddr + indexRow * srcRowStride + indexCol;
    uint32_t totalBytes = static_cast<uint32_t>(validCol) * sizeof(T);
    uint32_t alignedBytes = (totalBytes / BLOCK_BYTE_SIZE) * BLOCK_BYTE_SIZE;
    uint32_t tailBytes = totalBytes - alignedBytes;
    if (alignedBytes > 0) {
        uint16_t burstLen = static_cast<uint16_t>(alignedBytes / BLOCK_BYTE_SIZE);
        uint16_t srcGap = static_cast<uint16_t>((srcRowStride * sizeof(T) - alignedBytes) / BLOCK_BYTE_SIZE);
        uint16_t dstGap = static_cast<uint16_t>((dstRowStride * sizeof(T) - alignedBytes) / BLOCK_BYTE_SIZE);
        pto_copy_ubuf_to_ubuf((__ubuf__ void*)dstAddr, (__ubuf__ void*)srcStart, validRow, burstLen, srcGap, dstGap);
    }
    if (tailBytes > 0) {
        uint32_t alignedElems = alignedBytes / sizeof(T);
        __ubuf__ uint16_t* srcTail = (__ubuf__ uint16_t*)(srcStart + alignedElems);
        __ubuf__ uint16_t* dstTail = (__ubuf__ uint16_t*)(dstAddr + alignedElems);
        uint64_t tailU16 = static_cast<uint64_t>(tailBytes / sizeof(uint16_t));
        constexpr uint16_t srcRptU16 = static_cast<uint16_t>(srcRowStride * sizeof(T) / BLOCK_BYTE_SIZE);
        constexpr uint16_t dstRptU16 = static_cast<uint16_t>(dstRowStride * sizeof(T) / BLOCK_BYTE_SIZE);
        constexpr uint32_t srcStrideU16 = srcRowStride * sizeof(T) / sizeof(uint16_t);
        constexpr uint32_t dstStrideU16 = dstRowStride * sizeof(T) / sizeof(uint16_t);
        set_mask_count();
        set_vector_mask(0, tailU16);
        uint16_t remainRows = validRow;
        while (remainRows > 0) {
            uint8_t chunk =
                remainRows > REPEAT_MAX ? static_cast<uint8_t>(REPEAT_MAX) : static_cast<uint8_t>(remainRows);
            vcopy(dstTail, srcTail, chunk, 1, 1, dstRptU16, srcRptU16);
            remainRows -= chunk;
            srcTail += static_cast<uint32_t>(chunk) * srcStrideU16;
            dstTail += static_cast<uint32_t>(chunk) * dstStrideU16;
        }
        set_mask_norm();
        set_vector_mask(-1, -1);
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractVecToVecNZAligned(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint32_t indexRow, uint32_t indexCol)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    constexpr uint32_t typeSize = sizeof(T);
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / typeSize;
    constexpr uint32_t srcRows = SrcTileData::Rows;
    constexpr uint32_t dstRows = DstTileData::Rows;
    uint16_t burstNum = static_cast<uint16_t>(validCol / c0Size);
    uint16_t burstLen = static_cast<uint16_t>((validRow * c0Size * typeSize) / BLOCK_BYTE_SIZE);
    uint32_t srcOffset = (indexCol / c0Size) * srcRows * c0Size + indexRow * c0Size;
    uint16_t srcGap = static_cast<uint16_t>(srcRows - validRow);
    uint16_t dstGap = static_cast<uint16_t>(dstRows - validRow);
    pto_copy_ubuf_to_ubuf(
        (__ubuf__ void*)dstAddr, (__ubuf__ void*)(srcAddr + srcOffset), burstNum, burstLen, srcGap, dstGap);
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractVecToVecNZUnaligned(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint32_t indexRow, uint32_t indexCol)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    constexpr uint32_t typeSize = sizeof(T);
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / typeSize;
    constexpr uint32_t srcRows = SrcTileData::Rows;
    constexpr uint32_t dstRows = DstTileData::Rows;
    uint16_t fullStripes = static_cast<uint16_t>(validCol / c0Size);
    uint16_t tailCols = static_cast<uint16_t>(validCol % c0Size);
    uint32_t srcOffsetBase = (indexCol / c0Size) * srcRows * c0Size + indexRow * c0Size;
    if (fullStripes > 0) {
        uint16_t burstLen = validRow;
        uint16_t srcGap = static_cast<uint16_t>(srcRows - validRow);
        uint16_t dstGap = static_cast<uint16_t>(dstRows - validRow);
        pto_copy_ubuf_to_ubuf(
            (__ubuf__ void*)dstAddr, (__ubuf__ void*)(srcAddr + srcOffsetBase), fullStripes, burstLen, srcGap, dstGap);
    }
    if (tailCols > 0) {
        uint32_t srcTailElems = srcOffsetBase + static_cast<uint32_t>(fullStripes) * srcRows * c0Size;
        uint32_t dstTailElems = static_cast<uint32_t>(fullStripes) * dstRows * c0Size;
        __ubuf__ uint16_t* srcTail = (__ubuf__ uint16_t*)(srcAddr + srcTailElems);
        __ubuf__ uint16_t* dstTail = (__ubuf__ uint16_t*)(dstAddr + dstTailElems);
        uint64_t tailU16 = static_cast<uint64_t>(tailCols) * typeSize / sizeof(uint16_t);
        constexpr uint16_t rptStride = static_cast<uint16_t>(c0Size * typeSize / BLOCK_BYTE_SIZE);
        constexpr uint32_t strideU16 = c0Size * typeSize / sizeof(uint16_t);
        set_mask_count();
        set_vector_mask(0, tailU16);
        uint16_t remainRows = validRow;
        while (remainRows > 0) {
            uint8_t chunk =
                remainRows > REPEAT_MAX ? static_cast<uint8_t>(REPEAT_MAX) : static_cast<uint8_t>(remainRows);
            vcopy(dstTail, srcTail, chunk, 1, 1, rptStride, rptStride);
            remainRows -= chunk;
            srcTail += static_cast<uint32_t>(chunk) * strideU16;
            dstTail += static_cast<uint32_t>(chunk) * strideU16;
        }
        set_mask_norm();
        set_vector_mask(-1, -1);
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TExtractVecToVecNDDispatch(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol)
{
    using T = typename DstTileData::DType;
    CheckTExtractVecToVecND<DstTileData, SrcTileData>();
    uint32_t idxRow = static_cast<uint32_t>(indexRow);
    uint32_t idxCol = static_cast<uint32_t>(indexCol);
    if constexpr (DstTileData::ValidRow == 1 && DstTileData::ValidCol == 1) {
        PTO_ASSERT(idxRow < SrcTileData::Rows, "TEXTRACT ND Vec->Vec : indexRow exceeds srcRows!");
        PTO_ASSERT(idxCol < SrcTileData::Cols, "TEXTRACT ND Vec->Vec : indexCol exceeds srcCols!");
        TExtractVecToVecNDScalar<T, DstTileData, SrcTileData>(dst.data(), src.data(), idxRow, idxCol);
    } else {
        PTO_ASSERT(
            idxCol * sizeof(T) % BLOCK_BYTE_SIZE == 0,
            "TEXTRACT ND Vec->Vec : indexCol bytes must be 32-byte aligned (A3 limitation).");
        PTO_ASSERT(
            idxRow + DstTileData::ValidRow <= SrcTileData::Rows,
            "TEXTRACT ND Vec->Vec : indexRow + dstValidRow exceeds source rows!");
        PTO_ASSERT(
            idxCol + DstTileData::ValidCol <= SrcTileData::Cols,
            "TEXTRACT ND Vec->Vec : indexCol + dstValidCol exceeds source cols!");
        uint16_t validRow = static_cast<uint16_t>(dst.GetValidRow());
        uint16_t validCol = static_cast<uint16_t>(dst.GetValidCol());
        if constexpr ((DstTileData::ValidCol * sizeof(T)) % BLOCK_BYTE_SIZE == 0) {
            TExtractVecToVecNDAligned<T, DstTileData, SrcTileData>(
                dst.data(), src.data(), validRow, validCol, idxRow, idxCol);
        } else {
            TExtractVecToVecNDUnaligned<T, DstTileData, SrcTileData>(
                dst.data(), src.data(), validRow, validCol, idxRow, idxCol);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TExtractVecToVecNZDispatch(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol)
{
    using T = typename DstTileData::DType;
    constexpr uint32_t kC0Size = BLOCK_BYTE_SIZE / sizeof(T);
    CheckTExtractVecToVecNZ<DstTileData, SrcTileData>();
    uint32_t idxRow = static_cast<uint32_t>(indexRow);
    uint32_t idxCol = static_cast<uint32_t>(indexCol);
    if constexpr (DstTileData::ValidRow == 1 && DstTileData::ValidCol == 1) {
        PTO_ASSERT(idxRow < SrcTileData::Rows, "TEXTRACT NZ Vec->Vec : indexRow exceeds srcRows!");
        PTO_ASSERT(idxCol < SrcTileData::Cols, "TEXTRACT NZ Vec->Vec : indexCol exceeds srcCols!");
        TExtractVecToVecNZScalar<T, DstTileData, SrcTileData>(dst.data(), src.data(), idxRow, idxCol);
    } else {
        PTO_ASSERT(idxRow % FRACTAL_NZ_ROW == 0, "TEXTRACT NZ Vec->Vec : indexRow must be 16-aligned (A3 limitation).");
        PTO_ASSERT(idxCol % kC0Size == 0, "TEXTRACT NZ Vec->Vec : indexCol must be c0Size-aligned (A3 limitation).");
        uint16_t validRow = static_cast<uint16_t>(dst.GetValidRow());
        uint16_t validCol = static_cast<uint16_t>(dst.GetValidCol());
        PTO_ASSERT(
            idxRow + validRow <= SrcTileData::Rows, "TEXTRACT NZ Vec->Vec : indexRow + validRow exceeds source rows!");
        PTO_ASSERT(
            idxCol + validCol <= SrcTileData::Cols, "TEXTRACT NZ Vec->Vec : indexCol + validCol exceeds source cols!");
        if constexpr ((DstTileData::ValidCol % kC0Size) == 0) {
            TExtractVecToVecNZAligned<T, DstTileData, SrcTileData>(
                dst.data(), src.data(), validRow, validCol, idxRow, idxCol);
        } else {
            TExtractVecToVecNZUnaligned<T, DstTileData, SrcTileData>(
                dst.data(), src.data(), validRow, validCol, idxRow, idxCol);
        }
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractNdToNzScalar(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol, uint16_t validRow, uint16_t validCol)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    constexpr uint32_t typeSize = sizeof(T);
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / typeSize;
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;
    constexpr uint32_t dstRows = DstTileData::Rows;
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    for (uint16_t r = 0; r < validRow; ++r) {
        for (uint16_t c = 0; c < validCol; ++c) {
            uint32_t cb = static_cast<uint32_t>(c) / c0Size;
            uint32_t cc = static_cast<uint32_t>(c) % c0Size;
            uint32_t dstOff = cb * dstRows * c0Size + static_cast<uint32_t>(r) * c0Size + cc;
            uint32_t srcOff =
                (static_cast<uint32_t>(indexRow) + r) * srcRowStride + static_cast<uint32_t>(indexCol) + c;
            dstAddr[dstOff] = srcAddr[srcOff];
        }
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractNdToNz(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol, uint16_t validRow, uint16_t validCol)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    constexpr uint32_t typeSize = sizeof(T);
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / typeSize;
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;
    constexpr uint32_t dstRows = DstTileData::Rows;
    __ubuf__ T* srcStart = srcAddr + static_cast<uint32_t>(indexRow) * srcRowStride + static_cast<uint32_t>(indexCol);
    uint16_t numColBlocks = static_cast<uint16_t>(CeilDivision(static_cast<uint32_t>(validCol), c0Size));
    constexpr uint16_t srcRepeatStride = static_cast<uint16_t>(srcRowStride * typeSize / BLOCK_BYTE_SIZE);
    constexpr uint16_t dstRepeatStride = static_cast<uint16_t>(c0Size * typeSize / BLOCK_BYTE_SIZE);
    constexpr uint32_t srcStrideU16 = srcRowStride * typeSize / sizeof(uint16_t);
    constexpr uint32_t dstStrideU16 = c0Size * typeSize / sizeof(uint16_t);

    set_mask_count();
    for (uint16_t cb = 0; cb < numColBlocks; ++cb) {
        uint32_t colCount = static_cast<uint32_t>(validCol) - static_cast<uint32_t>(cb) * c0Size;
        if (colCount > c0Size) {
            colCount = c0Size;
        }
        uint64_t maskU16 = static_cast<uint64_t>(colCount) * typeSize / sizeof(uint16_t);
        __ubuf__ uint16_t* srcCb = (__ubuf__ uint16_t*)(srcStart + static_cast<uint32_t>(cb) * c0Size);
        __ubuf__ uint16_t* dstCb = (__ubuf__ uint16_t*)(dstAddr + static_cast<uint32_t>(cb) * dstRows * c0Size);
        set_vector_mask(0, maskU16);
        uint16_t remainRows = validRow;
        uint32_t done = 0;
        while (remainRows > 0) {
            uint8_t chunk =
                remainRows > REPEAT_MAX ? static_cast<uint8_t>(REPEAT_MAX) : static_cast<uint8_t>(remainRows);
            vcopy(
                dstCb + done * dstStrideU16, srcCb + done * srcStrideU16, chunk, 1, 1, dstRepeatStride,
                srcRepeatStride);
            remainRows -= chunk;
            done += chunk;
        }
    }
    set_mask_norm();
    set_vector_mask(-1, -1);
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractNdToNzWiden(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol, uint16_t validRow, uint16_t validCol)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    __ubuf__ half* tmpHalf = (__ubuf__ half*)(TMP_UB_OFFSET);
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;
    constexpr uint32_t dstRows = DstTileData::Rows;
    __ubuf__ T* srcStart = srcAddr + static_cast<uint32_t>(indexRow) * srcRowStride + static_cast<uint32_t>(indexCol);
    uint16_t numColBlocks = static_cast<uint16_t>(CeilDivision(static_cast<uint32_t>(validCol), c0Size));
    constexpr uint16_t srcRepeatStride = static_cast<uint16_t>(srcRowStride * sizeof(T) / BLOCK_BYTE_SIZE);
    constexpr uint16_t tmpRepeatStride = static_cast<uint16_t>(c0Size * sizeof(half) / BLOCK_BYTE_SIZE);
    constexpr uint16_t dstRepeatStride = static_cast<uint16_t>(c0Size * sizeof(T) / BLOCK_BYTE_SIZE);
    constexpr uint16_t maxChunkRows = static_cast<uint16_t>(TMP_UB_SIZE / (c0Size * sizeof(half)));

    set_mask_norm();
    for (uint16_t cb = 0; cb < numColBlocks; ++cb) {
        uint32_t colCount = static_cast<uint32_t>(validCol) - static_cast<uint32_t>(cb) * c0Size;
        if (colCount > c0Size) {
            colCount = c0Size;
        }
        __ubuf__ T* srcCb = srcStart + static_cast<uint32_t>(cb) * c0Size;
        __ubuf__ T* dstCb = dstAddr + static_cast<uint32_t>(cb) * dstRows * c0Size;
        SetContinuousMask(colCount);
        uint16_t remainRows = validRow;
        uint32_t done = 0;
        while (remainRows > 0) {
            uint8_t chunk =
                remainRows > maxChunkRows ? static_cast<uint8_t>(maxChunkRows) : static_cast<uint8_t>(remainRows);
            vconv_s82f16(tmpHalf, srcCb + done * srcRowStride, chunk, 1, 1, tmpRepeatStride, srcRepeatStride);
            pipe_barrier(PIPE_V);
            vconv_f162s8z(dstCb + done * c0Size, tmpHalf, chunk, 1, 1, dstRepeatStride, tmpRepeatStride);
            pipe_barrier(PIPE_V);
            remainRows -= chunk;
            done += chunk;
        }
    }
    set_vector_mask(-1, -1);
}

template <typename T, typename DstTileData, typename SrcTileData>
PTO_INTERNAL void DispatchNdToNz(
    DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol, uint16_t validRow, uint16_t validCol)
{
    bool colAligned = ((static_cast<uint32_t>(indexCol) * sizeof(T)) % BLOCK_BYTE_SIZE) == 0;
    if ((validRow == 1 && validCol == 1) || !colAligned) {
        TExtractNdToNzScalar<T, DstTileData, SrcTileData>(
            dst.data(), src.data(), indexRow, indexCol, validRow, validCol);
    } else if constexpr (sizeof(T) == 1) {
        if ((static_cast<uint32_t>(validCol) * sizeof(T)) % sizeof(uint16_t) == 0) {
            TExtractNdToNz<T, DstTileData, SrcTileData>(dst.data(), src.data(), indexRow, indexCol, validRow, validCol);
        } else {
            TExtractNdToNzWiden<T, DstTileData, SrcTileData>(
                dst.data(), src.data(), indexRow, indexCol, validRow, validCol);
        }
    } else {
        TExtractNdToNz<T, DstTileData, SrcTileData>(dst.data(), src.data(), indexRow, indexCol, validRow, validCol);
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
PTO_INTERNAL void CheckTExtractNdToNz()
{
    static_assert(
        SrcTileData::Loc == TileType::Vec && DstTileData::Loc == TileType::Vec,
        "TEXTRACT A2A3 ND->2xNZ : Source and destinations must be Vec (UB) tiles.");
    static_assert(
        SrcTileData::isRowMajor && (SrcTileData::SFractal == SLayout::NoneBox),
        "TEXTRACT A2A3 ND->2xNZ : Source must be ND (RowMajor, NoneBox).");
    static_assert(
        !DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor),
        "TEXTRACT A2A3 ND->2xNZ : Destination must be NZ (ColMajor, RowMajor fractal).");
    static_assert(
        DstTileData::Compact != CompactMode::RowPlusOne,
        "TEXTRACT A2A3ND->2xNZ : A2A3 supports plain NZ output only (no NZ+1).");
    static_assert(
        std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
        "TEXTRACT A2A3 ND->2xNZ : Source and destination data types must match.");
    static_assert(
        std::is_same<T, int8_t>::value || std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value ||
            std::is_same<T, float>::value || std::is_same<T, int32_t>::value,
        "TEXTRACT A2A3 ND->2xNZ : Unsupported data type.");
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / sizeof(T);
    static_assert(DstTileData::Cols % c0Size == 0, "TEXTRACT ND->2xNZ : Destination cols must be c0-aligned.");
    static_assert(
        (SrcTileData::RowStride * sizeof(T)) % BLOCK_BYTE_SIZE == 0,
        "TEXTRACT A2A3 ND->2xNZ : Source row stride must be 32-byte aligned.");
}

template <typename Dst0TileData, typename Dst1TileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_ND2XNZ_IMPL(
    Dst0TileData& dst0, Dst1TileData& dst1, SrcTileData& src, uint16_t indexRow0, uint16_t indexCol0,
    uint16_t indexRow1, uint16_t indexCol1)
{
    using T = typename SrcTileData::DType;
    CheckTExtractNdToNz<T, Dst0TileData, SrcTileData>();
    CheckTExtractNdToNz<T, Dst1TileData, SrcTileData>();

    uint16_t validRow1 = static_cast<uint16_t>(dst1.GetValidRow());
    uint16_t validCol1 = static_cast<uint16_t>(dst1.GetValidCol());
    uint16_t validRow0 = static_cast<uint16_t>(dst0.GetValidRow());
    uint16_t validCol0 = static_cast<uint16_t>(dst0.GetValidCol());

    PTO_ASSERT(
        indexRow0 + validRow0 <= SrcTileData::Rows,
        "TEXTRACT A2A3 ND->2xNZ : window0 indexRow + validRow exceeds srcRows!");
    PTO_ASSERT(
        indexCol0 + validCol0 <= SrcTileData::Cols,
        "TEXTRACT A2A3 ND->2xNZ : window0 indexCol + validCol exceeds srcCols!");
    PTO_ASSERT(
        indexRow1 + validRow1 <= SrcTileData::Rows,
        "TEXTRACT A2A3 ND->2xNZ : window1 indexRow + validRow exceeds srcRows!");
    PTO_ASSERT(
        indexCol1 + validCol1 <= SrcTileData::Cols,
        "TEXTRACT A2A3 ND->2xNZ : window1 indexCol + validCol exceeds srcCols!");

    DispatchNdToNz<T, Dst0TileData, SrcTileData>(dst0, src, indexRow0, indexCol0, validRow0, validCol0);
    DispatchNdToNz<T, Dst1TileData, SrcTileData>(dst1, src, indexRow1, indexCol1, validRow1, validCol1);
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData& dst, SrcTileData& src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    if constexpr (DstTileData::Loc == TileType::Vec && SrcTileData::Loc == TileType::Vec) {
        CheckTExtractVecToVecCommon<DstTileData, SrcTileData>();
        if constexpr (DstTileData::isRowMajor && SrcTileData::isRowMajor) {
            TExtractVecToVecNDDispatch<DstTileData, SrcTileData>(dst, src, indexRow, indexCol);
        } else if constexpr (
            !DstTileData::isRowMajor && !SrcTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor &&
            SrcTileData::SFractal == SLayout::RowMajor) {
            TExtractVecToVecNZDispatch<DstTileData, SrcTileData>(dst, src, indexRow, indexCol);
        } else {
            static_assert(
                DstTileData::isRowMajor == SrcTileData::isRowMajor,
                "TEXTRACT Vec->Vec : Source and destination layout must match (both ND or both NZ).");
            static_assert(
                DstTileData::SFractal == SrcTileData::SFractal,
                "TEXTRACT Vec->Vec : Source and destination SFractal must match.");
        }
    } else if constexpr (is_conv_tile_v<SrcTileData>) {
        TEXTRACT_CONVTILE_IMPL(dst, src, indexRow, indexCol);
    } else {
        TEXTRACT_TILE_IMPL(dst, src, indexRow, indexCol);
    }
}

// relu
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode, STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData& dst, SrcTileData& src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    PTO_ASSERT(
        indexRow + DstTileData::Rows <= SrcTileData::Rows,
        "The sum of indexRow and dstRow should be less than srcRow!");
    PTO_ASSERT(
        indexCol + DstTileData::Cols <= SrcTileData::Cols,
        "The sum of indexCol and dstCol should be less than srcCol!");
    CheckTMovAccToMat<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    TExtractAccToMat<DstTileData, SrcTileData, quantPre, reluMode, Phase>(
        dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
}

// scalar quant
template <
    typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TEXTRACT_IMPL(
    DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    PTO_ASSERT(
        indexRow + DstTileData::Rows <= SrcTileData::Rows,
        "The sum of indexRow and dstRow should be less than srcRow!");
    PTO_ASSERT(
        indexCol + DstTileData::Cols <= SrcTileData::Cols,
        "The sum of indexCol and dstCol should be less than srcCol!");
    CheckTMovAccToMat<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    set_quant_pre(preQuantScalar);
    TExtractAccToMat<DstTileData, SrcTileData, quantPre, reluMode, Phase>(
        dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
}

// vector quant
template <
    typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu,
    STPhase Phase = STPhase::Unspecified>
PTO_INTERNAL void TEXTRACT_IMPL(
    DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    PTO_ASSERT(
        indexRow + DstTileData::Rows <= SrcTileData::Rows,
        "The sum of indexRow and dstRow should be less than srcRow!");
    PTO_ASSERT(
        indexCol + DstTileData::Cols <= SrcTileData::Cols,
        "The sum of indexCol and dstCol should be less than srcCol!");
    CheckTMovAccToMat<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
    static_assert(FpTileData::Loc == TileType::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    SetFPC<FpTileData>(fp.data(), indexCol);
    TExtractAccToMat<DstTileData, SrcTileData, quantPre, reluMode, Phase>(
        dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
}
} // namespace pto
#endif // TEXTRACT_HPP
