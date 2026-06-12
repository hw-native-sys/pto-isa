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
        constexpr uint32_t srcStrideU16 = srcRowStride * sizeof(T) / sizeof(uint16_t);
        constexpr uint32_t dstStrideU16 = dstRowStride * sizeof(T) / sizeof(uint16_t);
        constexpr uint16_t srcRptU16 = static_cast<uint16_t>(srcRowStride * sizeof(T) / BLOCK_BYTE_SIZE);
        constexpr uint16_t dstRptU16 = static_cast<uint16_t>(dstRowStride * sizeof(T) / BLOCK_BYTE_SIZE);
        uint64_t tailU16 = static_cast<uint64_t>(tailBytes / sizeof(uint16_t));
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

#include "pto/common/arch/memory/tinsert_common.hpp"

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
    PTO_ASSERT(indexCol + SrcTileData::Cols <= DstTileData::Cols,
               "The sum of indexCol and srcCol should be less than dstCol!");
    PTO_ASSERT(indexRow + SrcTileData::Rows <= DstTileData::Rows,
               "The sum of indexRow and srcRow should be less than dstRow!");
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    set_quant_pre(preQuantScalar);
    TInsertAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), src.GetValidRow(),
                                                                  src.GetValidCol(), indexRow, indexCol);
}

// vector quant
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
#endif // TINSERT_HPP
