/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

/**
 * @file TExtract.hpp
 * @brief TEXTRACT Implementation for NPU KirinDev0000 Architecture
 *
 * Common parts live in pto/common/arch/register/textract_common.hpp; this shell
 * provides the kirinDev0000-specific paths: no L0A buffer (declaration-only L0A
 * helpers), TMov-based Mat/Acc extract, and the tile-level dispatcher.
 */

#ifndef TEXTRACT_KIRINDEV0000_HPP
#define TEXTRACT_KIRINDEV0000_HPP
#include "pto/npu/kirin9030/common.hpp"
#include "pto/common/arch/register/textract_common.hpp"

namespace pto {

//---------------------------------------------------------------------------------------
// kirinDev0000-specific helpers
//---------------------------------------------------------------------------------------

__tf__ PTO_INTERNAL void SetFixpNzPara(
    uint16_t loop4Size, uint16_t loop2SrcStride, uint16_t loop3SrcStride, uint16_t loop4SrcStride)
{
    uint64_t fixpNzPara = static_cast<uint64_t>(loop4Size) | (static_cast<uint64_t>(loop2SrcStride) << 16) |
                          (static_cast<uint64_t>(loop3SrcStride) << 32) | (static_cast<uint64_t>(loop4SrcStride) << 48);
    set_fixp_nz_para(fixpNzPara);
}

// TMov-based paths used by TEXTRACT_TILE_IMPL below. Defined in
// kirinDev0000/TMov.hpp. Declared here because calls with explicit template
// arguments require visibility at template definition point (no ADL).
template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TMovCbufToUb(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol);

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TMovCbufToCbufAcc(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol);

// kirinDev0000 has no L0A buffer: the L0A load helpers are declaration-only here.
// This keeps kirin9030/TMov.hpp templates compilable; instantiation on kirinDev0000
// triggers a link error, the intended behaviour for unsupported paths.
template <typename DstTile, typename SrcTile, bool Transpose>
__tf__ PTO_INTERNAL void TExtractToA(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol);

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TExtractToAVector(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol, uint16_t dstValidCol);

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TExtractToACompact(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol, uint16_t madM, uint16_t madK);

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TExtractToATransCompact(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol, uint16_t madM, uint16_t madK);

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TExtractCbufToUb(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint16_t indexRow, uint16_t indexCol)
{
    using dstType = typename DstTile::DType;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(dstType);
    constexpr uint32_t innerRows = SrcTile::InnerRows;
    constexpr uint32_t innerCols = SrcTile::InnerCols;
    constexpr uint32_t innerNumel = SrcTile::InnerNumel;
    constexpr uint32_t blockNumRow = SrcTile::Rows / innerRows;
    constexpr uint32_t copyBytes = c0Size * sizeof(dstType);

    __ubuf__ dstType* dstAddr = (__ubuf__ dstType*)__cce_get_tile_ptr(dst);
    __cbuf__ dstType* srcData = (__cbuf__ dstType*)__cce_get_tile_ptr(src);
    __ubuf__ uint8_t* dstBytes = reinterpret_cast<__ubuf__ uint8_t*>(dstAddr);
    __cbuf__ uint8_t* srcBytes = reinterpret_cast<__cbuf__ uint8_t*>(srcData);

    uint16_t alignedValidCol = CeilAlignment(validCol, c0Size);

    SetFixpNzPara(1, 1, 1, 0);
    for (uint32_t row = 0; row < static_cast<uint32_t>(validRow); ++row) {
        uint32_t srcRow = row + indexRow;
        uint32_t srcBlockRow = srcRow / innerRows;
        uint32_t srcInnerRow = srcRow % innerRows;
        for (uint32_t colBlk = 0; colBlk < alignedValidCol / c0Size; ++colBlk) {
            uint32_t srcCol = colBlk * c0Size + indexCol;
            uint32_t srcBlockCol = srcCol / innerCols;
            uint32_t srcInnerCol = srcCol % innerCols;
            uint32_t srcElemOff =
                (blockNumRow * srcBlockCol + srcBlockRow) * innerNumel + srcInnerRow * innerCols + srcInnerCol;
            uint32_t srcByteOff = srcElemOff * sizeof(dstType);
            uint32_t dstByteOff = (row * DstTile::Cols + colBlk * c0Size) * sizeof(dstType);
            fix_cbuf_to_ubuf(
                dstBytes + dstByteOff, srcBytes + srcByteOff, copyBytes, copyBytes, fixp_trans_mode_t::NORMAL_DMA,
                static_cast<uint64_t>(0), 1);
        }
    }
    SetFixpNzPara(0, 0, 0, 0);
}

//---------------------------------------------------------------------------------------
// kirinDev0000 shell of the common interface contracts
//---------------------------------------------------------------------------------------

// One 32-byte-unit burst per block: kirinDev0000 hardware does not support the
// multi-burst form of copy_ubuf_to_cbuf.
template <typename T>
__tf__ PTO_INTERNAL void TCopyUbToCbufFractal(
    __cbuf__ T* dstPtr, __ubuf__ T* srcPtr, uint16_t blockCout, uint16_t blockLen, uint16_t srcStride)
{
    constexpr uint32_t CBUF_UB_BURST_UNIT = 32;
    __cbuf__ uint8_t* dstP = reinterpret_cast<__cbuf__ uint8_t*>(dstPtr);
    __ubuf__ uint8_t* srcP = reinterpret_cast<__ubuf__ uint8_t*>(srcPtr);
    uint32_t srcStep = (blockLen + srcStride) * CBUF_UB_BURST_UNIT;
    uint32_t dstStep = blockLen * CBUF_UB_BURST_UNIT;
    for (uint16_t i = 0; i < blockCout; ++i) {
        copy_ubuf_to_cbuf(
            reinterpret_cast<__cbuf__ void*>(dstP + i * dstStep), reinterpret_cast<__ubuf__ void*>(srcP + i * srcStep),
            0, 1, blockLen, 0, 0);
    }
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TEXTRACT_TILE_IMPL(DstTile& dst, SrcTile& src, uint16_t indexRow, uint16_t indexCol)
{
    static_assert(
        is_textract_supported_type<typename DstTile::DType>,
        "TExtract: Unsupported data type! Supported types: (u)int8_t, (u)int16_t, int32, half");
    static_assert(
        std::is_same_v<typename DstTile::DType, typename SrcTile::DType>,
        "TExtract: Destination and Source tile data types must be the same");

    if constexpr (DstTile::Loc == TileType::Right) {
        TExtractToRight(dst, src, indexRow, indexCol);
    } else if constexpr (SrcTile::Loc == TileType::Vec && DstTile::Loc == TileType::Mat) {
        TExtractVecToMat<DstTile, SrcTile>(
            dst.data(), src.data(), indexRow, indexCol, src.GetValidRow(), src.GetValidCol(), dst.GetValidRow(),
            dst.GetValidCol());
    } else if constexpr (SrcTile::Loc == TileType::Acc || SrcTile::Loc == TileType::Mat) {
        if constexpr (DstTile::Loc == TileType::Mat) {
            TMovCbufToCbufAcc<DstTile, SrcTile>(dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol());
        } else {
            if (indexRow == 0 && indexCol == 0) {
                TMovCbufToUb<DstTile, SrcTile>(dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol());
            } else {
                TExtractCbufToUb<DstTile, SrcTile>(
                    dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
            }
        }
    } else {
        static_assert(
            sizeof(typename DstTile::DType) == 0, "TExtract: unsupported tile type combination on kirinDev0000.");
    }
}

} // namespace pto
#endif // TEXTRACT_KIRINDEV0000_HPP
