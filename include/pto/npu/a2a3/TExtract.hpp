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
    static_assert((SrcTileData::Loc == TileType::Acc) || std::is_same<DstType, SrcType>::value,
                  "TExtract: Destination and Source tile data types must be the same.");
    static_assert(std::is_same<DstType, int8_t>::value || std::is_same<DstType, half>::value ||
                      std::is_same<DstType, bfloat16_t>::value || std::is_same<DstType, float>::value,
                  "TExtract: Invalid data type.");
}

template <typename DstTileData, typename SrcTileData, QuantMode_t QuantPre, ReluPreMode reluMode>
__tf__ AICORE void TExtractAccToMat(typename DstTileData::TileDType __out__ dst,
                                    typename SrcTileData::TileDType __in__ src, uint16_t validRow, uint16_t validCol,
                                    uint16_t indexRow, uint16_t indexCol)
{
    using SrcType = typename SrcTileData::DType;
    using DstType = typename DstTileData::DType;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(DstType);
    uint32_t srcOffset = SrcTileData::Rows * ACC_C0_SIZE * (indexCol / ACC_C0_SIZE) +
                         (indexRow * ACC_C0_SIZE + (indexCol % ACC_C0_SIZE));
    __cc__ SrcType *srcAddr = (__cc__ SrcType *)__cce_get_tile_ptr(src) + srcOffset;
    __cbuf__ DstType *dstAddr = (__cbuf__ DstType *)__cce_get_tile_ptr(dst);

    constexpr uint32_t dstStrideD = DstTileData::Rows;
    constexpr uint16_t srcStride = SrcTileData::Rows;
    uint16_t nSize = CeilDivision(validCol, c0Size) * c0Size;
    pto_copy_matrix_cc_to_cbuf(dstAddr, srcAddr, 0, nSize, validRow, dstStrideD, srcStride, 0, QuantPre,
                               static_cast<uint8_t>(reluMode), false, false);
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_TILE_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    CheckTExtract<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType>();
    PTO_ASSERT(indexRow + DstTileData::Rows <= SrcTileData::Rows,
               "The sum of indexRow and dstRow should be less than srcRow!");
    PTO_ASSERT(indexCol + DstTileData::Cols <= SrcTileData::Cols,
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
__tf__ AICORE void TExtractVecToVecNDAligned(typename DstTileData::TileDType __out__ dst,
                                             typename SrcTileData::TileDType __in__ src, uint16_t validRow,
                                             uint16_t validCol, uint32_t indexRow, uint32_t indexCol)
{
    __ubuf__ T *dstAddr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;
    constexpr uint32_t dstRowStride = DstTileData::RowStride;
    __ubuf__ T *srcStart = srcAddr + indexRow * srcRowStride + indexCol;
    uint32_t rowBytes = static_cast<uint32_t>(validCol) * sizeof(T);
    if (validCol == dstRowStride && validCol == srcRowStride) {
        uint32_t totalBytes = static_cast<uint32_t>(validRow) * rowBytes;
        uint16_t burstLen = static_cast<uint16_t>(totalBytes / BLOCK_BYTE_SIZE);
        pto_copy_ubuf_to_ubuf((__ubuf__ void *)dstAddr, (__ubuf__ void *)srcStart, 1, burstLen, 0, 0);
    } else {
        uint16_t rowBurst = static_cast<uint16_t>(rowBytes / BLOCK_BYTE_SIZE);
        uint16_t srcGap = static_cast<uint16_t>((srcRowStride - validCol) * sizeof(T) / BLOCK_BYTE_SIZE);
        uint16_t dstGap = static_cast<uint16_t>((dstRowStride - validCol) * sizeof(T) / BLOCK_BYTE_SIZE);
        pto_copy_ubuf_to_ubuf((__ubuf__ void *)dstAddr, (__ubuf__ void *)srcStart, validRow, rowBurst, srcGap, dstGap);
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractVecToVecNDUnaligned(typename DstTileData::TileDType __out__ dst,
                                               typename SrcTileData::TileDType __in__ src, uint16_t validRow,
                                               uint16_t validCol, uint32_t indexRow, uint32_t indexCol)
{
    __ubuf__ T *dstAddr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;
    constexpr uint32_t dstRowStride = DstTileData::RowStride;
    __ubuf__ T *srcStart = srcAddr + indexRow * srcRowStride + indexCol;
    uint32_t totalBytes = static_cast<uint32_t>(validCol) * sizeof(T);
    uint32_t alignedBytes = (totalBytes / BLOCK_BYTE_SIZE) * BLOCK_BYTE_SIZE;
    uint32_t tailBytes = totalBytes - alignedBytes;
    if (alignedBytes > 0) {
        uint16_t burstLen = static_cast<uint16_t>(alignedBytes / BLOCK_BYTE_SIZE);
        uint16_t srcGap = static_cast<uint16_t>((srcRowStride * sizeof(T) - alignedBytes) / BLOCK_BYTE_SIZE);
        uint16_t dstGap = static_cast<uint16_t>((dstRowStride * sizeof(T) - alignedBytes) / BLOCK_BYTE_SIZE);
        pto_copy_ubuf_to_ubuf((__ubuf__ void *)dstAddr, (__ubuf__ void *)srcStart, validRow, burstLen, srcGap, dstGap);
    }
    if (tailBytes > 0) {
        uint32_t alignedElems = alignedBytes / sizeof(T);
        __ubuf__ uint16_t *srcTail = (__ubuf__ uint16_t *)(srcStart + alignedElems);
        __ubuf__ uint16_t *dstTail = (__ubuf__ uint16_t *)(dstAddr + alignedElems);
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
__tf__ AICORE void TExtractVecToVecNZAligned(typename DstTileData::TileDType __out__ dst,
                                             typename SrcTileData::TileDType __in__ src, uint16_t validRow,
                                             uint16_t validCol, uint32_t indexRow, uint32_t indexCol)
{
    __ubuf__ T *dstAddr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    constexpr uint32_t typeSize = sizeof(T);
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / typeSize;
    constexpr uint32_t srcRows = SrcTileData::Rows;
    constexpr uint32_t dstRows = DstTileData::Rows;
    uint16_t burstNum = static_cast<uint16_t>(validCol / c0Size);
    uint16_t burstLen = static_cast<uint16_t>((validRow * c0Size * typeSize) / BLOCK_BYTE_SIZE);
    uint32_t srcOffset = (indexCol / c0Size) * srcRows * c0Size + indexRow * c0Size;
    uint16_t srcGap = static_cast<uint16_t>(srcRows - validRow);
    uint16_t dstGap = static_cast<uint16_t>(dstRows - validRow);
    pto_copy_ubuf_to_ubuf((__ubuf__ void *)dstAddr, (__ubuf__ void *)(srcAddr + srcOffset), burstNum, burstLen, srcGap,
                          dstGap);
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractVecToVecNZUnaligned(typename DstTileData::TileDType __out__ dst,
                                               typename SrcTileData::TileDType __in__ src, uint16_t validRow,
                                               uint16_t validCol, uint32_t indexRow, uint32_t indexCol)
{
    __ubuf__ T *dstAddr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src);
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
        pto_copy_ubuf_to_ubuf((__ubuf__ void *)dstAddr, (__ubuf__ void *)(srcAddr + srcOffsetBase), fullStripes,
                              burstLen, srcGap, dstGap);
    }
    if (tailCols > 0) {
        uint32_t srcTailElems = srcOffsetBase + static_cast<uint32_t>(fullStripes) * srcRows * c0Size;
        uint32_t dstTailElems = static_cast<uint32_t>(fullStripes) * dstRows * c0Size;
        __ubuf__ uint16_t *srcTail = (__ubuf__ uint16_t *)(srcAddr + srcTailElems);
        __ubuf__ uint16_t *dstTail = (__ubuf__ uint16_t *)(dstAddr + dstTailElems);
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
PTO_INTERNAL void TExtractVecToVecNDDispatch(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol)
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
        PTO_ASSERT(idxCol * sizeof(T) % BLOCK_BYTE_SIZE == 0,
                   "TEXTRACT ND Vec->Vec : indexCol bytes must be 32-byte aligned (A3 limitation).");
        PTO_ASSERT(idxRow + DstTileData::ValidRow <= SrcTileData::Rows,
                   "TEXTRACT ND Vec->Vec : indexRow + dstValidRow exceeds source rows!");
        PTO_ASSERT(idxCol + DstTileData::ValidCol <= SrcTileData::Cols,
                   "TEXTRACT ND Vec->Vec : indexCol + dstValidCol exceeds source cols!");
        uint16_t validRow = static_cast<uint16_t>(dst.GetValidRow());
        uint16_t validCol = static_cast<uint16_t>(dst.GetValidCol());
        if constexpr ((DstTileData::ValidCol * sizeof(T)) % BLOCK_BYTE_SIZE == 0) {
            TExtractVecToVecNDAligned<T, DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol, idxRow,
                                                                   idxCol);
        } else {
            TExtractVecToVecNDUnaligned<T, DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol, idxRow,
                                                                     idxCol);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TExtractVecToVecNZDispatch(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol)
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
        PTO_ASSERT(idxRow + validRow <= SrcTileData::Rows,
                   "TEXTRACT NZ Vec->Vec : indexRow + validRow exceeds source rows!");
        PTO_ASSERT(idxCol + validCol <= SrcTileData::Cols,
                   "TEXTRACT NZ Vec->Vec : indexCol + validCol exceeds source cols!");
        if constexpr ((DstTileData::ValidCol % kC0Size) == 0) {
            TExtractVecToVecNZAligned<T, DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol, idxRow,
                                                                   idxCol);
        } else {
            TExtractVecToVecNZUnaligned<T, DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol, idxRow,
                                                                     idxCol);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    if constexpr (DstTileData::Loc == TileType::Vec && SrcTileData::Loc == TileType::Vec) {
        CheckTExtractVecToVecCommon<DstTileData, SrcTileData>();
        if constexpr (DstTileData::isRowMajor && SrcTileData::isRowMajor) {
            TExtractVecToVecNDDispatch<DstTileData, SrcTileData>(dst, src, indexRow, indexCol);
        } else if constexpr (!DstTileData::isRowMajor && !SrcTileData::isRowMajor &&
                             DstTileData::SFractal == SLayout::RowMajor && SrcTileData::SFractal == SLayout::RowMajor) {
            TExtractVecToVecNZDispatch<DstTileData, SrcTileData>(dst, src, indexRow, indexCol);
        } else {
            static_assert(DstTileData::isRowMajor == SrcTileData::isRowMajor,
                          "TEXTRACT Vec->Vec : Source and destination layout must match (both ND or both NZ).");
            static_assert(DstTileData::SFractal == SrcTileData::SFractal,
                          "TEXTRACT Vec->Vec : Source and destination SFractal must match.");
        }
    } else if constexpr (is_conv_tile_v<SrcTileData>) {
        TEXTRACT_CONVTILE_IMPL(dst, src, indexRow, indexCol);
    } else {
        TEXTRACT_TILE_IMPL(dst, src, indexRow, indexCol);
    }
}

// relu
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    PTO_ASSERT(indexRow + DstTileData::Rows <= SrcTileData::Rows,
               "The sum of indexRow and dstRow should be less than srcRow!");
    PTO_ASSERT(indexCol + DstTileData::Cols <= SrcTileData::Cols,
               "The sum of indexCol and dstCol should be less than srcCol!");
    CheckTMovAccToMat<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    TExtractAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), dst.GetValidRow(),
                                                                   dst.GetValidCol(), indexRow, indexCol);
}

// scalar quant
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow = 0,
                                uint16_t indexCol = 0)
{
    PTO_ASSERT(indexRow + DstTileData::Rows <= SrcTileData::Rows,
               "The sum of indexRow and dstRow should be less than srcRow!");
    PTO_ASSERT(indexCol + DstTileData::Cols <= SrcTileData::Cols,
               "The sum of indexCol and dstCol should be less than srcCol!");
    CheckTMovAccToMat<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    set_quant_pre(preQuantScalar);
    TExtractAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), dst.GetValidRow(),
                                                                   dst.GetValidCol(), indexRow, indexCol);
}

// vector quant
template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow = 0,
                                uint16_t indexCol = 0)
{
    PTO_ASSERT(indexRow + DstTileData::Rows <= SrcTileData::Rows,
               "The sum of indexRow and dstRow should be less than srcRow!");
    PTO_ASSERT(indexCol + DstTileData::Cols <= SrcTileData::Cols,
               "The sum of indexCol and dstCol should be less than srcCol!");
    CheckTMovAccToMat<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
    static_assert(FpTileData::Loc == TileType::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    SetFPC<FpTileData>(fp.data(), indexCol);
    TExtractAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), dst.GetValidRow(),
                                                                   dst.GetValidCol(), indexRow, indexCol);
}
} // namespace pto
#endif // TEXTRACT_HPP
