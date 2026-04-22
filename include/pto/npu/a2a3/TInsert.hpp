/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TINSERT_HPP
#define TINSERT_HPP
#include "common.hpp"

namespace pto {
template <typename DstTileData, typename SrcTileData, QuantMode_t QuantPre, ReluPreMode reluMode>
__tf__ PTO_INTERNAL void TInsertAccToMat(typename DstTileData::TileDType __out__ dst,
                                         typename SrcTileData::TileDType __in__ src, uint16_t validRow,
                                         uint16_t validCol, uint16_t indexRow, uint16_t indexCol)
{
    using SrcType = typename SrcTileData::DType;
    using DstType = typename DstTileData::DType;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(DstType);
    uint32_t dstOffset = DstTileData::Rows * c0Size * (indexCol / c0Size) + (indexRow * c0Size + (indexCol % c0Size));
    __cc__ SrcType *srcAddr = (__cc__ SrcType *)__cce_get_tile_ptr(src);
    __cbuf__ DstType *dstAddr = (__cbuf__ DstType *)__cce_get_tile_ptr(dst) + dstOffset;

    constexpr uint32_t dstStrideD = DstTileData::Rows;
    constexpr uint16_t srcStride = SrcTileData::Rows;
    uint16_t nSize = CeilDivision(validCol, c0Size) * c0Size;
    copy_matrix_cc_to_cbuf(dstAddr, srcAddr, 0, nSize, SrcTileData::Rows, dstStrideD, srcStride, 0, QuantPre, reluMode,
                           false, false);
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TInsertVecToVecNDScalar(typename DstTileData::TileDType __out__ dst,
                                           typename SrcTileData::TileDType __in__ src, uint32_t indexRow,
                                           uint32_t indexCol)
{
    __ubuf__ T *dstAddr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    constexpr uint32_t dstRowStride = DstTileData::RowStride;
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    dstAddr[indexRow * dstRowStride + indexCol] = srcAddr[0];
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TInsertVecToVecNDAligned(typename DstTileData::TileDType __out__ dst,
                                            typename SrcTileData::TileDType __in__ src, uint16_t validRow,
                                            uint16_t validCol, uint32_t indexRow, uint32_t indexCol)
{
    __ubuf__ T *dstAddr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;
    constexpr uint32_t dstRowStride = DstTileData::RowStride;
    __ubuf__ T *dstStart = dstAddr + indexRow * dstRowStride + indexCol;
    uint32_t rowBytes = static_cast<uint32_t>(validCol) * sizeof(T);
    if (validCol == dstRowStride && validCol == srcRowStride) {
        uint32_t totalBytes = static_cast<uint32_t>(validRow) * rowBytes;
        uint16_t burstLen = static_cast<uint16_t>(totalBytes / BLOCK_BYTE_SIZE);
        pto_copy_ubuf_to_ubuf((__ubuf__ void *)dstStart, (__ubuf__ void *)srcAddr, 1, burstLen, 0, 0);
    } else {
        uint16_t rowBurst = static_cast<uint16_t>(rowBytes / BLOCK_BYTE_SIZE);
        uint16_t srcGap = static_cast<uint16_t>((srcRowStride - validCol) * sizeof(T) / BLOCK_BYTE_SIZE);
        uint16_t dstGap = static_cast<uint16_t>((dstRowStride - validCol) * sizeof(T) / BLOCK_BYTE_SIZE);
        pto_copy_ubuf_to_ubuf((__ubuf__ void *)dstStart, (__ubuf__ void *)srcAddr, validRow, rowBurst, srcGap, dstGap);
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TInsertVecToVecNDUnaligned(typename DstTileData::TileDType __out__ dst,
                                              typename SrcTileData::TileDType __in__ src, uint16_t validRow,
                                              uint16_t validCol, uint32_t indexRow, uint32_t indexCol)
{
    __ubuf__ T *dstAddr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;
    constexpr uint32_t dstRowStride = DstTileData::RowStride;
    __ubuf__ T *dstStart = dstAddr + indexRow * dstRowStride + indexCol;
    uint32_t totalBytes = static_cast<uint32_t>(validCol) * sizeof(T);
    uint32_t alignedBytes = (totalBytes / BLOCK_BYTE_SIZE) * BLOCK_BYTE_SIZE;
    uint32_t tailBytes = totalBytes - alignedBytes;
    if (alignedBytes > 0) {
        uint16_t burstLen = static_cast<uint16_t>(alignedBytes / BLOCK_BYTE_SIZE);
        uint16_t srcGap = static_cast<uint16_t>((srcRowStride * sizeof(T) - alignedBytes) / BLOCK_BYTE_SIZE);
        uint16_t dstGap = static_cast<uint16_t>((dstRowStride * sizeof(T) - alignedBytes) / BLOCK_BYTE_SIZE);
        pto_copy_ubuf_to_ubuf((__ubuf__ void *)dstStart, (__ubuf__ void *)srcAddr, validRow, burstLen, srcGap, dstGap);
    }
    if (tailBytes > 0) {
        uint32_t alignedElems = alignedBytes / sizeof(T);
        __ubuf__ uint16_t *srcTail = (__ubuf__ uint16_t *)(srcAddr + alignedElems);
        __ubuf__ uint16_t *dstTail = (__ubuf__ uint16_t *)(dstStart + alignedElems);
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
__tf__ AICORE void TInsertVecToVecNZScalar(typename DstTileData::TileDType __out__ dst,
                                           typename SrcTileData::TileDType __in__ src, uint32_t indexRow,
                                           uint32_t indexCol)
{
    __ubuf__ T *dstAddr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr uint32_t dstRows = DstTileData::Rows;
    uint32_t dstOffset = (indexCol / c0Size) * dstRows * c0Size + indexRow * c0Size + (indexCol % c0Size);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    dstAddr[dstOffset] = srcAddr[0];
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TInsertVecToVecNZAligned(typename DstTileData::TileDType __out__ dst,
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
    uint32_t dstOffset = (indexCol / c0Size) * dstRows * c0Size + indexRow * c0Size;
    uint16_t srcGap = static_cast<uint16_t>(srcRows - validRow);
    uint16_t dstGap = static_cast<uint16_t>(dstRows - validRow);
    pto_copy_ubuf_to_ubuf((__ubuf__ void *)(dstAddr + dstOffset), (__ubuf__ void *)srcAddr, burstNum, burstLen, srcGap,
                          dstGap);
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TInsertVecToVecNZUnaligned(typename DstTileData::TileDType __out__ dst,
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
    uint32_t dstOffsetBase = (indexCol / c0Size) * dstRows * c0Size + indexRow * c0Size;
    if (fullStripes > 0) {
        uint16_t burstLen = validRow;
        uint16_t srcGap = static_cast<uint16_t>(srcRows - validRow);
        uint16_t dstGap = static_cast<uint16_t>(dstRows - validRow);
        pto_copy_ubuf_to_ubuf((__ubuf__ void *)(dstAddr + dstOffsetBase), (__ubuf__ void *)srcAddr, fullStripes,
                              burstLen, srcGap, dstGap);
    }
    if (tailCols > 0) {
        uint32_t srcTailElems = static_cast<uint32_t>(fullStripes) * srcRows * c0Size;
        uint32_t dstTailElems = dstOffsetBase + static_cast<uint32_t>(fullStripes) * dstRows * c0Size;
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
PTO_INTERNAL void CheckTInsertVecToVecCommon()
{
    using DstT = typename DstTileData::DType;
    using SrcT = typename SrcTileData::DType;
    static_assert(std::is_same<DstT, SrcT>::value, "TINSERT Vec->Vec : Source and destination data types must match.");
    static_assert(std::is_same<DstT, int8_t>::value || std::is_same<DstT, uint8_t>::value ||
                      std::is_same<DstT, int16_t>::value || std::is_same<DstT, uint16_t>::value ||
                      std::is_same<DstT, half>::value || std::is_same<DstT, bfloat16_t>::value ||
                      std::is_same<DstT, float>::value || std::is_same<DstT, int32_t>::value ||
                      std::is_same<DstT, uint32_t>::value,
                  "TINSERT Vec->Vec : Unsupported data type for A3.");
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void CheckTInsertVecToVecND()
{
    using T = typename DstTileData::DType;
    static_assert(SrcTileData::Rows <= DstTileData::Rows,
                  "TINSERT ND Vec->Vec : Source Rows must not exceed destination Rows.");
    static_assert(SrcTileData::Cols <= DstTileData::Cols,
                  "TINSERT ND Vec->Vec : Source Cols must not exceed destination Cols.");
    static_assert(SrcTileData::RowStride * sizeof(T) % BLOCK_BYTE_SIZE == 0,
                  "TINSERT ND Vec->Vec : SrcTile RowStride bytes must be 32-byte aligned.");
    static_assert(DstTileData::RowStride * sizeof(T) % BLOCK_BYTE_SIZE == 0,
                  "TINSERT ND Vec->Vec : DstTile RowStride bytes must be 32-byte aligned.");
    if constexpr (!(SrcTileData::ValidRow == 1 && SrcTileData::ValidCol == 1)) {
        static_assert((SrcTileData::ValidCol * sizeof(T)) % sizeof(uint16_t) == 0,
                      "TINSERT ND Vec->Vec : SrcTile ValidCol bytes must be at least 2-byte aligned.");
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void CheckTInsertVecToVecNZ()
{
    using T = typename DstTileData::DType;
    constexpr uint32_t kC0Size = BLOCK_BYTE_SIZE / sizeof(T);
    static_assert(SrcTileData::Rows <= DstTileData::Rows,
                  "TINSERT NZ Vec->Vec : Source Rows must not exceed destination Rows.");
    static_assert(SrcTileData::Cols <= DstTileData::Cols,
                  "TINSERT NZ Vec->Vec : Source Cols must not exceed destination Cols.");
    static_assert(SrcTileData::Rows % FRACTAL_NZ_ROW == 0, "TINSERT NZ Vec->Vec : SrcTile Rows must be 16-aligned.");
    static_assert(DstTileData::Rows % FRACTAL_NZ_ROW == 0, "TINSERT NZ Vec->Vec : DstTile Rows must be 16-aligned.");
    static_assert(SrcTileData::Cols % kC0Size == 0, "TINSERT NZ Vec->Vec : SrcTile Cols must be c0Size-aligned.");
    static_assert(DstTileData::Cols % kC0Size == 0, "TINSERT NZ Vec->Vec : DstTile Cols must be c0Size-aligned.");
    if constexpr (!(SrcTileData::ValidRow == 1 && SrcTileData::ValidCol == 1)) {
        static_assert((SrcTileData::ValidCol * sizeof(T)) % sizeof(uint16_t) == 0,
                      "TINSERT NZ Vec->Vec : SrcTile ValidCol bytes must be at least 2-byte aligned.");
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TInsertVecToVecNDDispatch(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol)
{
    using T = typename DstTileData::DType;
    CheckTInsertVecToVecND<DstTileData, SrcTileData>();
    uint32_t idxRow = static_cast<uint32_t>(indexRow);
    uint32_t idxCol = static_cast<uint32_t>(indexCol);
    if constexpr (SrcTileData::ValidRow == 1 && SrcTileData::ValidCol == 1) {
        PTO_ASSERT(idxRow < DstTileData::Rows, "TINSERT ND Vec->Vec : indexRow exceeds dstRows!");
        PTO_ASSERT(idxCol < DstTileData::Cols, "TINSERT ND Vec->Vec : indexCol exceeds dstCols!");
        TInsertVecToVecNDScalar<T, DstTileData, SrcTileData>(dst.data(), src.data(), idxRow, idxCol);
    } else {
        PTO_ASSERT(idxCol * sizeof(T) % BLOCK_BYTE_SIZE == 0,
                   "TINSERT ND Vec->Vec : indexCol bytes must be 32-byte aligned (A3 limitation).");
        PTO_ASSERT(idxRow + SrcTileData::ValidRow <= DstTileData::Rows,
                   "TINSERT ND Vec->Vec : indexRow + srcValidRow exceeds destination rows!");
        PTO_ASSERT(idxCol + SrcTileData::ValidCol <= DstTileData::Cols,
                   "TINSERT ND Vec->Vec : indexCol + srcValidCol exceeds destination cols!");
        uint16_t validRow = static_cast<uint16_t>(src.GetValidRow());
        uint16_t validCol = static_cast<uint16_t>(src.GetValidCol());
        if constexpr ((SrcTileData::ValidCol * sizeof(T)) % BLOCK_BYTE_SIZE == 0) {
            TInsertVecToVecNDAligned<T, DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol, idxRow,
                                                                  idxCol);
        } else {
            TInsertVecToVecNDUnaligned<T, DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol, idxRow,
                                                                    idxCol);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TInsertVecToVecNZDispatch(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol)
{
    using T = typename DstTileData::DType;
    constexpr uint32_t kC0Size = BLOCK_BYTE_SIZE / sizeof(T);
    CheckTInsertVecToVecNZ<DstTileData, SrcTileData>();
    uint32_t idxRow = static_cast<uint32_t>(indexRow);
    uint32_t idxCol = static_cast<uint32_t>(indexCol);
    if constexpr (SrcTileData::ValidRow == 1 && SrcTileData::ValidCol == 1) {
        PTO_ASSERT(idxRow < DstTileData::Rows, "TINSERT NZ Vec->Vec : indexRow exceeds dstRows!");
        PTO_ASSERT(idxCol < DstTileData::Cols, "TINSERT NZ Vec->Vec : indexCol exceeds dstCols!");
        TInsertVecToVecNZScalar<T, DstTileData, SrcTileData>(dst.data(), src.data(), idxRow, idxCol);
    } else {
        PTO_ASSERT(idxRow % FRACTAL_NZ_ROW == 0, "TINSERT NZ Vec->Vec : indexRow must be 16-aligned (A3 limitation).");
        PTO_ASSERT(idxCol % kC0Size == 0, "TINSERT NZ Vec->Vec : indexCol must be c0Size-aligned (A3 limitation).");
        uint16_t validRow = static_cast<uint16_t>(src.GetValidRow());
        uint16_t validCol = static_cast<uint16_t>(src.GetValidCol());
        PTO_ASSERT(idxRow + validRow <= DstTileData::Rows,
                   "TINSERT NZ Vec->Vec : indexRow + validRow exceeds destination rows!");
        PTO_ASSERT(idxCol + validCol <= DstTileData::Cols,
                   "TINSERT NZ Vec->Vec : indexCol + validCol exceeds destination cols!");
        if constexpr ((SrcTileData::ValidCol % kC0Size) == 0) {
            TInsertVecToVecNZAligned<T, DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol, idxRow,
                                                                  idxCol);
        } else {
            TInsertVecToVecNZUnaligned<T, DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol, idxRow,
                                                                    idxCol);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    if constexpr (DstTileData::Loc == TileType::Vec && SrcTileData::Loc == TileType::Vec) {
        CheckTInsertVecToVecCommon<DstTileData, SrcTileData>();
        if constexpr (DstTileData::isRowMajor && SrcTileData::isRowMajor) {
            TInsertVecToVecNDDispatch<DstTileData, SrcTileData>(dst, src, indexRow, indexCol);
        } else if constexpr (!DstTileData::isRowMajor && !SrcTileData::isRowMajor &&
                             DstTileData::SFractal == SLayout::RowMajor && SrcTileData::SFractal == SLayout::RowMajor) {
            TInsertVecToVecNZDispatch<DstTileData, SrcTileData>(dst, src, indexRow, indexCol);
        } else {
            static_assert(DstTileData::isRowMajor == SrcTileData::isRowMajor,
                          "TINSERT Vec->Vec : Source and destination layout must match (both ND or both NZ).");
            static_assert(DstTileData::SFractal == SrcTileData::SFractal,
                          "TINSERT Vec->Vec : Source and destination SFractal must match.");
        }
    } else {
        CheckTMovAccToMat<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
        PTO_ASSERT(indexRow + SrcTileData::Rows <= DstTileData::Rows,
                   "The sum of indexRow and srcRow should be less than dstRow!");
        PTO_ASSERT(indexCol + SrcTileData::Cols <= DstTileData::Cols,
                   "The sum of indexCol and srcCol should be less than dstCol!");
        constexpr QuantMode_t quantPre =
            GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
        TInsertAccToMat<DstTileData, SrcTileData, quantPre, ReluPreMode::NoRelu>(
            dst.data(), src.data(), src.GetValidRow(), src.GetValidCol(), indexRow, indexCol);
    }
}

// relu
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    CheckTMovAccToMat<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    PTO_ASSERT(indexRow + SrcTileData::Rows <= DstTileData::Rows,
               "The sum of indexRow and srcRow should be less than dstRow!");
    PTO_ASSERT(indexCol + SrcTileData::Cols <= DstTileData::Cols,
               "The sum of indexCol and srcCol should be less than dstCol!");
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    TInsertAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), src.GetValidRow(),
                                                                  src.GetValidCol(), indexRow, indexCol);
}

// scalar quant
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow = 0,
                               uint16_t indexCol = 0)
{
    CheckTMovAccToMat<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
    PTO_ASSERT(indexRow + SrcTileData::Rows <= DstTileData::Rows,
               "The sum of indexRow and srcRow should be less than dstRow!");
    PTO_ASSERT(indexCol + SrcTileData::Cols <= DstTileData::Cols,
               "The sum of indexCol and srcCol should be less than dstCol!");
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    set_quant_pre(preQuantScalar);
    TInsertAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), src.GetValidRow(),
                                                                  src.GetValidCol(), indexRow, indexCol);
}

// vector quant
template <typename FpTileData>
__tf__ PTO_INTERNAL void SetFPCInsert(typename FpTileData::TileDType __in__ fp)
{
    using FpType = typename FpTileData::DType;
    __fbuf__ FpType *dstAddrFp = (__fbuf__ FpType *)__cce_get_tile_ptr(fp);
    uint64_t deqTensorAddr = ((uint64_t)dstAddrFp >> static_cast<uint64_t>(7))
                             << 8; // fpc[15:8] means Quant_PRE_ADDR, uint of 128(2^7)bytes
    set_fpc(deqTensorAddr);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow = 0,
                               uint16_t indexCol = 0)
{
    CheckTMovAccToMat<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
    PTO_ASSERT(indexRow + SrcTileData::Rows <= DstTileData::Rows,
               "The sum of indexRow and srcRow should be less than dstRow!");
    PTO_ASSERT(indexCol + SrcTileData::Cols <= DstTileData::Cols,
               "The sum of indexCol and srcCol should be less than dstCol!");
    static_assert(FpTileData::Loc == TileType::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    SetFPCInsert<FpTileData>(fp.data());
    TInsertAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), src.GetValidRow(),
                                                                  src.GetValidCol(), indexRow, indexCol);
}
} // namespace pto
#endif // TInsert_HPP