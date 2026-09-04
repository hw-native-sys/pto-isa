/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TINSERT_HPP_KIRINDEV0000
#define TINSERT_HPP_KIRINDEV0000
#include "pto/npu/kirin9030/common.hpp"
#include "pto/npu/kirin9030/utils.hpp"
#include "pto/common/arch/register/tinsert_common.hpp"

namespace pto {

template <typename T>
__tf__ PTO_INTERNAL void CopyUbufToCbuf(
    __cbuf__ T* dst, __ubuf__ T* src, uint8_t sid, uint16_t nBurst, uint16_t lenBurst, uint16_t srcGap, uint16_t dstGap)
{
    constexpr uint32_t CBUF_UB_BURST_UNIT = 32;
    __cbuf__ uint8_t* dstP = reinterpret_cast<__cbuf__ uint8_t*>(dst);
    __ubuf__ uint8_t* srcP = reinterpret_cast<__ubuf__ uint8_t*>(src);
    uint32_t srcStep = (lenBurst + srcGap) * CBUF_UB_BURST_UNIT;
    uint32_t dstStep = (lenBurst + dstGap) * CBUF_UB_BURST_UNIT;
    for (uint16_t i = 0; i < nBurst; ++i) {
        copy_ubuf_to_cbuf(
            reinterpret_cast<__cbuf__ void*>(dstP + i * dstStep), reinterpret_cast<__ubuf__ void*>(srcP + i * srcStep),
            sid, 1, lenBurst, 0, 0);
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void TInsertImpl(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint16_t dstRow, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    __cbuf__ T* dstAddr = (__cbuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    uint16_t burstNum, burstLen, srcGap, dstGap;
    uint32_t dstOffset;
    ComputeNZBlockParams<T, DstTileData, SrcTileData>(
        validRow, validCol, dstRow, burstNum, burstLen, srcGap, dstGap, dstOffset, indexRow, indexCol);
    __cbuf__ T* dstAddr2 = dstAddr + dstOffset;
    CopyUbufToCbuf(dstAddr2, srcAddr, 0, burstNum, burstLen, srcGap, dstGap);
}

template <uint32_t SplitCount, typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void TInsertSplitImpl(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    __cbuf__ T* dstAddr = (__cbuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);

    constexpr uint32_t typeSize = sizeof(T);
    constexpr bool isFp4Type = std::is_same_v<T, float4_e2m1x2_t> || std::is_same_v<T, float4_e1m2x2_t>;
    uint32_t c0Size = BLOCK_BYTE_SIZE / typeSize;
    constexpr uint32_t nzRow = FRACTAL_NZ_ROW;

    uint32_t byteValidCol = isFp4Type ? validCol / 2 : validCol;
    uint32_t byteIndexCol = isFp4Type ? indexCol / 2 : indexCol;
    uint32_t alignedRow = CeilDivision(validRow, nzRow) * nzRow;
    uint16_t totalBurstNum = static_cast<uint16_t>(CeilDivision(byteValidCol, c0Size));
    uint16_t burstLen = (alignedRow * c0Size * typeSize) / BLOCK_BYTE_SIZE;
    uint16_t partBurstNum = totalBurstNum / SplitCount;
    uint16_t lastBurstNum = totalBurstNum - partBurstNum * (SplitCount - 1);
    uint32_t srcStrideRows;
    if constexpr (SrcTileData::Compact == CompactMode::Null) {
        srcStrideRows = SrcTileData::Rows;
    } else if constexpr (SrcTileData::Compact == CompactMode::RowPlusOne) {
        srcStrideRows = alignedRow + 1;
    } else {
        srcStrideRows = alignedRow;
    }
    uint16_t srcGap = static_cast<uint16_t>(srcStrideRows - alignedRow);
    uint16_t dstGap = static_cast<uint16_t>(DstTileData::Rows - alignedRow);
    uint32_t srcBlockSize = (burstLen + srcGap) * BLOCK_BYTE_SIZE / typeSize;
    uint32_t dstBlockSize = DstTileData::Rows * c0Size;

    uint32_t colBlockOffset = (byteIndexCol / c0Size) * DstTileData::Rows * c0Size;
    uint32_t rowOffset = indexRow * c0Size + (byteIndexCol % c0Size);
    uint32_t dstOffset = colBlockOffset + rowOffset;

    __cbuf__ T* dstAddr0 = dstAddr + dstOffset;
    CopyUbufToCbuf(dstAddr0, srcAddr, 0, partBurstNum, burstLen, srcGap, dstGap);

    if constexpr (SplitCount >= 2) {
        __ubuf__ T* src1 = srcAddr + partBurstNum * srcBlockSize;
        __cbuf__ T* dst1 = dstAddr0 + partBurstNum * dstBlockSize;
        uint16_t burst1Num = (SplitCount == 2) ? lastBurstNum : partBurstNum;
        CopyUbufToCbuf(dst1, src1, 0, burst1Num, burstLen, srcGap, dstGap);
    }

    if constexpr (SplitCount >= 4) {
        __ubuf__ T* src2 = srcAddr + 2 * partBurstNum * srcBlockSize;
        __cbuf__ T* dst2 = dstAddr0 + 2 * partBurstNum * dstBlockSize;
        CopyUbufToCbuf(dst2, src2, 0, partBurstNum, burstLen, srcGap, dstGap);

        __ubuf__ T* src3 = srcAddr + 3 * partBurstNum * srcBlockSize;
        __cbuf__ T* dst3 = dstAddr0 + 3 * partBurstNum * dstBlockSize;
        CopyUbufToCbuf(dst3, src3, 0, lastBurstNum, burstLen, srcGap, dstGap);
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void TInsertNDImpl(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint16_t dstCols, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    __cbuf__ T* dstAddr = (__cbuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);

    uint32_t dstOffset = indexRow * dstCols + indexCol;
    __cbuf__ T* dstStart = dstAddr + dstOffset;

    uint32_t totalBytes = static_cast<uint32_t>(validRow) * static_cast<uint32_t>(validCol) * sizeof(T);

    if (validCol == SrcTileData::Cols && validCol == dstCols && totalBytes >= BLOCK_BYTE_SIZE) {
        uint16_t burstLen = static_cast<uint16_t>(totalBytes / BLOCK_BYTE_SIZE);
        CopyUbufToCbuf(dstStart, srcAddr, 0, 1, burstLen, 0, 0);
    } else if (static_cast<uint32_t>(validCol) * sizeof(T) >= BLOCK_BYTE_SIZE) {
        uint16_t rowBurstLen = static_cast<uint16_t>((validCol * sizeof(T)) / BLOCK_BYTE_SIZE);
        uint16_t srcRowGap =
            static_cast<uint16_t>((SrcTileData::Cols * sizeof(T) - validCol * sizeof(T)) / BLOCK_BYTE_SIZE);
        uint16_t dstRowGap = static_cast<uint16_t>((dstCols * sizeof(T) - validCol * sizeof(T)) / BLOCK_BYTE_SIZE);
        CopyUbufToCbuf(dstStart, srcAddr, 0, validRow, rowBurstLen, srcRowGap, dstRowGap);
    } else {
        uint16_t burstLen = static_cast<uint16_t>(CeilDivision(totalBytes, static_cast<uint32_t>(BLOCK_BYTE_SIZE)));
        CopyUbufToCbuf(dstStart, srcAddr, 0, 1, burstLen, 0, 0);
    }
}

// vlds+vstus path: strides or indexCol NOT 32B-aligned.
template <typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL OP_NAME(TINSERT) OP_TYPE(element_wise) void TInsertVecToVecNDVectorImpl(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol, uint16_t validRow, uint16_t validCol, VFImplKind version)
{
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    constexpr uint32_t dstRowStride = DstTileData::RowStride;
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;
    constexpr uint32_t elementsPerRepeat = CCE_VL / sizeof(T);
    constexpr uint32_t kValidCol = SrcTileData::ValidCol;
    constexpr uint16_t kFullRepeats = static_cast<uint16_t>(kValidCol / elementsPerRepeat);
    constexpr uint32_t kRemainder = kValidCol % elementsPerRepeat;

    __VEC_SCOPE__
    {
        RegTensor<T> vreg;
        UnalignReg ureg;

        for (uint16_t i = 0; i < validRow; ++i) {
            uint32_t srcRowOff = static_cast<uint32_t>(i) * srcRowStride;
            __ubuf__ T* pdst = dstAddr + (indexRow + static_cast<uint32_t>(i)) * dstRowStride + indexCol;
            for (uint16_t j = 0; j < kFullRepeats; ++j) {
                vlds(vreg, srcAddr, srcRowOff + static_cast<uint32_t>(j) * elementsPerRepeat, NORM);
                vstus(ureg, elementsPerRepeat, vreg, pdst, POST_UPDATE);
            }
            if constexpr (kRemainder > 0) {
                vlds(vreg, srcAddr, srcRowOff + static_cast<uint32_t>(kFullRepeats) * elementsPerRepeat, NORM);
                vstus(ureg, kRemainder, vreg, pdst, POST_UPDATE);
            }
            vstas(ureg, pdst, 0, POST_UPDATE);
        }
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TInsertVecToMatImpl(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol)
{
    uint16_t validRow = static_cast<uint16_t>(src.GetValidRow());
    uint16_t validCol = static_cast<uint16_t>(src.GetValidCol());

    if constexpr (SrcTileData::isRowMajor) {
        uint16_t dstCols = static_cast<uint16_t>(DstTileData::Cols);
        TInsertNDImpl<T, DstTileData, SrcTileData>(
            dst.data(), src.data(), validRow, validCol, dstCols, indexRow, indexCol);
    } else if constexpr (!SrcTileData::isRowMajor && (SrcTileData::SFractal == SLayout::RowMajor)) {
        constexpr uint16_t dstRow =
            static_cast<uint16_t>(DstTileData::BFractal == BLayout::ColMajor ? DstTileData::Rows : DstTileData::Cols);
        PTO_ASSERT(indexRow + validRow <= dstRow, "TINSERT NZ : indexRow + validRow exceeds destination rows!");
        TInsertImpl<T, DstTileData, SrcTileData>(
            dst.data(), src.data(), validRow, validCol, dstRow, indexRow, indexCol);
    }
}

template <typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void TInsertCbufToUb(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint16_t indexRow, uint16_t indexCol)
{
    using dstType = typename DstTileData::DType;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(dstType);
    constexpr uint32_t innerRows = SrcTileData::InnerRows;
    constexpr uint32_t innerCols = SrcTileData::InnerCols;
    constexpr uint32_t innerNumel = SrcTileData::InnerNumel;
    constexpr uint32_t blockNumRow = SrcTileData::Rows / innerRows;
    constexpr uint32_t copyBytes = c0Size * sizeof(dstType);

    __ubuf__ dstType* dstAddr = (__ubuf__ dstType*)__cce_get_tile_ptr(dst);
    __cbuf__ dstType* srcData = (__cbuf__ dstType*)__cce_get_tile_ptr(src);
    __ubuf__ uint8_t* dstBytes = reinterpret_cast<__ubuf__ uint8_t*>(dstAddr);
    __cbuf__ uint8_t* srcBytes = reinterpret_cast<__cbuf__ uint8_t*>(srcData);

    uint16_t alignedValidCol = CeilAlignment(validCol, c0Size);

    SetFixpNzPara(1, 1, 1, 0);
    for (uint32_t row = 0; row < static_cast<uint32_t>(validRow); ++row) {
        uint32_t srcBlockRow = row / innerRows;
        uint32_t srcInnerRow = row % innerRows;
        for (uint32_t colBlk = 0; colBlk < alignedValidCol / c0Size; ++colBlk) {
            uint32_t srcCol = colBlk * c0Size;
            uint32_t srcBlockCol = srcCol / innerCols;
            uint32_t srcInnerCol = srcCol % innerCols;
            uint32_t srcElemOff =
                (blockNumRow * srcBlockCol + srcBlockRow) * innerNumel + srcInnerRow * innerCols + srcInnerCol;
            uint32_t srcByteOff = srcElemOff * sizeof(dstType);
            uint32_t dstByteOff =
                ((row + indexRow) * DstTileData::Cols + (colBlk * c0Size + indexCol)) * sizeof(dstType);
            fix_cbuf_to_ubuf(
                dstBytes + dstByteOff, srcBytes + srcByteOff, copyBytes, copyBytes, fixp_trans_mode_t::NORMAL_DMA,
                static_cast<uint64_t>(0), 1);
        }
    }
    SetFixpNzPara(0, 0, 0, 0);
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TINSERT_IMPL(DstTileData& dst, SrcTileData& src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    if constexpr (SrcTileData::Loc == TileType::Acc || SrcTileData::Loc == TileType::Mat) {
        if constexpr (DstTileData::Loc == TileType::Mat) {
            TMovCbufToCbufAcc<DstTileData, SrcTileData>(dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol());
        } else {
            if (indexRow == 0 && indexCol == 0) {
                TMovCbufToUb<DstTileData, SrcTileData>(dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol());
            } else {
                TInsertCbufToUb<DstTileData, SrcTileData>(
                    dst.data(), src.data(), src.GetValidRow(), src.GetValidCol(), indexRow, indexCol);
            }
        }
    } else {
        using T = typename SrcTileData::DType;
        static_assert(
            std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
            "TINSERT : Source and destination data types must match");
        static_assert(
            (std::is_same<T, half>::value) || (std::is_same<T, bfloat16_t>::value) || (std::is_same<T, float>::value) ||
                (std::is_same<T, int32_t>::value) || (std::is_same<T, float8_e4m3_t>::value) ||
                (std::is_same<T, float8_e5m2_t>::value) || (std::is_same<T, hifloat8_t>::value) ||
                (std::is_same<T, int8_t>::value) || (std::is_same<T, float8_e8m0_t>::value) ||
                (std::is_same<T, float4_e2m1x2_t>::value) || (std::is_same<T, float4_e1m2x2_t>::value),
            "TINSERT : Unsupported data type.");

        if constexpr (DstTileData::Loc == TileType::Vec && SrcTileData::Loc == TileType::Vec) {
            TInsertVecToVecImpl<T>(dst, src, indexRow, indexCol);
        } else if constexpr (DstTileData::Loc == TileType::Mat && SrcTileData::Loc == TileType::Vec) {
            TInsertVecToMatImpl<T>(dst, src, indexRow, indexCol);
        }
    }
}

template <TInsertMode mode, typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TINSERT_IMPL(DstTileData& dst, SrcTileData& src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    using T = typename SrcTileData::DType;
    static_assert(
        std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
        "TINSERT : Source and destination data types must match");
    static_assert(DstTileData::Loc == TileType::Mat, "TINSERT : Destination must be Mat tile (L1/cbuf)");
    static_assert(SrcTileData::Loc == TileType::Vec, "TINSERT : Source must be Vec tile (UB/ubuf)");
    static_assert(
        !SrcTileData::isRowMajor && (SrcTileData::SFractal == SLayout::RowMajor),
        "TINSERT NZ : Source must be NZ format (column-major, RowMajor fractal)");
    static_assert(
        (std::is_same<T, half>::value) || (std::is_same<T, bfloat16_t>::value) || (std::is_same<T, float>::value) ||
            (std::is_same<T, int32_t>::value) || (std::is_same<T, float8_e4m3_t>::value) ||
            (std::is_same<T, float8_e5m2_t>::value) || (std::is_same<T, hifloat8_t>::value) ||
            (std::is_same<T, int8_t>::value) || (std::is_same<T, float8_e8m0_t>::value) ||
            (std::is_same<T, float4_e2m1x2_t>::value) || (std::is_same<T, float4_e1m2x2_t>::value),
        "TINSERT NZ : Unsupported data type.");

    uint16_t validRow = static_cast<uint16_t>(src.GetValidRow());
    uint16_t validCol = static_cast<uint16_t>(src.GetValidCol());
    PTO_ASSERT(indexRow + validRow <= DstTileData::Rows, "TINSERT : indexRow + validRow exceeds destination rows!");
    PTO_ASSERT(indexCol + validCol <= DstTileData::Cols, "TINSERT : indexCol + validCol exceeds destination cols!");

    if constexpr (mode == TInsertMode::SPLIT2) {
        TInsertSplitImpl<2, T, DstTileData, SrcTileData>(
            dst.data(), src.data(), validRow, validCol, indexRow, indexCol);
    } else if constexpr (mode == TInsertMode::SPLIT4) {
        TInsertSplitImpl<4, T, DstTileData, SrcTileData>(
            dst.data(), src.data(), validRow, validCol, indexRow, indexCol);
    }
}

} // namespace pto
#endif // TINSERT_HPP_KIRINDEV0000
