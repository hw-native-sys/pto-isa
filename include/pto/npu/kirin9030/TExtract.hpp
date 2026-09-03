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
 * @brief TEXTRACT Implementation for NPU Kirin9030 Architecture
 *
 * Common parts live in pto/common/arch/register/textract_common.hpp; this shell
 * provides the kirin9030-specific L0A load paths, Acc extract with quantization,
 * and the tile-level dispatcher.
 */

#ifndef TEXTRACT_HPP
#define TEXTRACT_HPP
#include "common.hpp"
#include "pto/common/arch/register/textract_common.hpp"

namespace pto {

//---------------------------------------------------------------------------------------
// CBUF -> L0A (Left operand) helpers
//---------------------------------------------------------------------------------------

template <typename DstTile, typename SrcTile, bool Transpose>
__tf__ PTO_INTERNAL void TExtractToA(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol)
{
    constexpr int32_t srcRow = SrcTile::Rows;
    constexpr int32_t srcCol = SrcTile::Cols;
    constexpr int32_t dstRow = DstTile::Rows;
    constexpr int32_t dstCol = DstTile::Cols;
    using DataType = typename SrcTile::DType;
    constexpr int typeSize = sizeof(DataType);
    __cbuf__ DataType* srcAddr = (__cbuf__ DataType*)__cce_get_tile_ptr(src);
    __ca__ DataType* dstAddr = (__ca__ DataType*)__cce_get_tile_ptr(dst);
    constexpr int c0Size = BLOCK_BYTE_SIZE / typeSize;

    if constexpr (!Transpose) {
        uint16_t mStartPosition = indexRow >> SHIFT_BLOCK_LEN;
        uint16_t kStartPosition = (indexCol * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint8_t mStep = dstRow >> SHIFT_BLOCK_LEN;
        constexpr uint8_t kStep = (dstCol * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint16_t srcStride = srcRow >> SHIFT_BLOCK_LEN;
        constexpr uint16_t dstStride = dstRow >> SHIFT_BLOCK_LEN;

        load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 0);
    } else {
        static_assert((srcRow % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "srcRow must be aligned");
        static_assert((srcCol % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "srcCol must be aligned");
        static_assert((dstRow % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "dstRow must be aligned");
        static_assert((dstCol % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "dstCol must be aligned");

        uint16_t mStartPosition = indexCol >> SHIFT_BLOCK_LEN;
        uint16_t kStartPosition = (indexRow * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint8_t mStep = dstCol >> SHIFT_BLOCK_LEN;
        constexpr uint8_t kStep = (dstRow * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint16_t srcStride = srcCol >> SHIFT_BLOCK_LEN;
        constexpr uint16_t dstStride = dstRow >> SHIFT_BLOCK_LEN;

        load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
    }
}

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TExtractToAVector(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol, uint16_t dstValidCol)
{
    using DataType = typename SrcTile::DType;
    constexpr int typeSize = sizeof(DataType);
    constexpr int32_t blockSize = BLOCK_BYTE_SIZE / typeSize;
    constexpr int32_t fractalSize = CUBE_BLOCK_SIZE / typeSize;
    int32_t kAlign = (dstValidCol + fractalSize - 1) & ~(fractalSize - 1);

    static_assert((SrcTile::Cols % blockSize) == 0, "srcCol * sizeof(DataType) must be aligned to 32B");
    static_assert((DstTile::Cols % fractalSize) == 0, "dstCol * sizeof(DataType) must be aligned to 512B");
    PTO_ASSERT((indexCol % blockSize) == 0, "indexCol * sizeof(DataType) must be aligned to 32B");

    __cbuf__ DataType* srcAddr = ((__cbuf__ DataType*)__cce_get_tile_ptr(src)) + indexCol;
    __ca__ DataType* dstAddr = (__ca__ DataType*)__cce_get_tile_ptr(dst);
    uint8_t kStep = kAlign / fractalSize;
    load_cbuf_to_ca(dstAddr, srcAddr, 0, 0, 1, kStep, 1, 1, 0);
}

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TExtractToACompact(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol, uint16_t madM, uint16_t madK)
{
    using DataType = typename SrcTile::DType;
    constexpr int typeSize = sizeof(DataType);
    __cbuf__ DataType* srcAddr = (__cbuf__ DataType*)__cce_get_tile_ptr(src);
    __ca__ DataType* dstAddr = (__ca__ DataType*)__cce_get_tile_ptr(dst);
    constexpr int c0Size = BLOCK_BYTE_SIZE / typeSize;
    uint16_t madMAlign = CeilDivision(madM, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    uint16_t madKAlign = CeilDivision(madK, c0Size) * c0Size;

    uint16_t mStartPosition = indexRow >> SHIFT_BLOCK_LEN;
    uint16_t kStartPosition = (indexCol * typeSize) >> SHIFT_BLOCK_BYTE;
    uint8_t mStep = madMAlign >> SHIFT_BLOCK_LEN;
    uint8_t kStep = (madKAlign * typeSize) >> SHIFT_BLOCK_BYTE;
    constexpr uint16_t srcStride = SrcTile::Rows >> SHIFT_BLOCK_LEN;
    uint16_t dstStride = madMAlign >> SHIFT_BLOCK_LEN;
    load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 0);
}

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TExtractToATransCompact(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol, uint16_t madM, uint16_t madK)
{
    using DataType = typename SrcTile::DType;
    constexpr int typeSize = sizeof(DataType);
    constexpr int c0Size = BLOCK_BYTE_SIZE / typeSize;
    __cbuf__ DataType* srcAddr = (__cbuf__ DataType*)__cce_get_tile_ptr(src);
    __ca__ DataType* dstAddr = (__ca__ DataType*)__cce_get_tile_ptr(dst);

    uint16_t alignNum = max(FRACTAL_NZ_ROW, c0Size);
    uint16_t madMAlign = CeilDivision(madM, alignNum) * alignNum;
    uint16_t madKAlign = CeilDivision(madK, alignNum) * alignNum;

    uint16_t mStartPosition = indexCol >> SHIFT_BLOCK_LEN;
    uint16_t kStartPosition = (indexRow * typeSize) >> SHIFT_BLOCK_BYTE;
    uint8_t mStep = madKAlign >> SHIFT_BLOCK_LEN;
    uint8_t kStep = (madMAlign * typeSize) >> SHIFT_BLOCK_BYTE;
    constexpr uint16_t srcStride = SrcTile::Cols >> SHIFT_BLOCK_LEN;
    uint16_t dstStride = madMAlign >> SHIFT_BLOCK_LEN;
    if constexpr (typeSize == 1) { // b8
        uint16_t dstAddrStride = CeilDivision(madM, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW * BLOCK_BYTE_SIZE;
        uint16_t mLoop = mStep >> SHIFT_M_STEP_B8;
        mStep = M_STEP_MIN_VAL_B8;
        for (uint16_t idx = 0; idx < mLoop; ++idx) {
            load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
            dstAddr += dstAddrStride;
            mStartPosition += M_STEP_MIN_VAL_B8;
        }
    } else { // b16/b32
        load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
    }
}

//---------------------------------------------------------------------------------------
// Acc (CC) extract with quantization
//---------------------------------------------------------------------------------------

template <typename DstTile, typename SrcTile, QuantMode_t QuantPre, ReluPreMode reluMode>
__tf__ PTO_INTERNAL void TExtractAccToMat(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint16_t indexRow, uint16_t indexCol)
{
    using dstType = typename DstTile::DType;
    using srcType = typename SrcTile::DType;
    constexpr bool channelSplitEnable = (!DstTile::isRowMajor && (DstTile::SFractal == SLayout::RowMajor)) &&
                                        (std::is_same_v<dstType, float>) && (DstTile::SFractalSize == CUBE_BLOCK_SIZE);
    constexpr int32_t c0Size = (!channelSplitEnable) && (DstTile::SFractalSize == 2 * CUBE_BLOCK_SIZE) ?
                                   2 * C0_SIZE_BYTE / sizeof(dstType) :
                                   C0_SIZE_BYTE / sizeof(dstType);
    constexpr uint32_t dstStride = DstTile::Rows * c0Size;
    uint16_t nSize = CeilDivision(validCol, c0Size) * c0Size;
    uint32_t srcOffset =
        SrcTile::Rows * ACC_C0_SIZE * (indexCol / ACC_C0_SIZE) + (indexRow * ACC_C0_SIZE + (indexCol % ACC_C0_SIZE));
    __cbuf__ dstType* dstAddr = (__cbuf__ dstType*)__cce_get_tile_ptr(dst);
    __cc__ srcType* srcData = (__cc__ srcType*)__cce_get_tile_ptr(src) + srcOffset;

    pto_copy_matrix_cc_to_cbuf(
        dstAddr, srcData, 0, nSize, validRow, dstStride, SrcTile::Rows, 0, 0, 0, QuantPre,
        static_cast<uint8_t>(reluMode), false, false, 0, 0, false, false, 0, false, false, false, false, false, false);
}

template <typename DstTile, typename SrcTile, AccToVecMode mode, QuantMode_t quantPre, ReluPreMode reluMode>
__tf__ PTO_INTERNAL void TExtractAccToVec(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint16_t srcValidRow, uint16_t indexRow, uint16_t indexCol)
{
    using dstType = typename DstTile::DType;
    using srcType = typename SrcTile::DType;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(dstType);
    constexpr bool subBlockId = (mode == AccToVecMode::SingleModeVec1);
    constexpr uint8_t dualDstCtl = GetDualDstCtl<DstTile, SrcTile, mode, quantPre>();
    constexpr uint32_t dstStride = DstTile::Cols;
    static_assert(
        ((dstStride * sizeof(dstType) % C0_SIZE_BYTE == 0) && ((dstStride) > 0)),
        "Dst Tile Cols * sizeof(dstT) must be multiples of 32 and not 0 when nz2nd.");
    constexpr uint16_t ndNum = 1;
    constexpr uint16_t dstNdStride = 0;
    constexpr uint16_t srcNdStride = 0;
    constexpr uint64_t loop3Para = static_cast<uint64_t>(dstNdStride) << 32 | static_cast<uint64_t>(srcNdStride) << 16 |
                                   static_cast<uint64_t>(ndNum);
    set_loop3_para(loop3Para);
    __ubuf__ dstType* dstAddr = (__ubuf__ dstType*)__cce_get_tile_ptr(dst);
    auto srcStride = SrcTile::Rows;
    uint32_t srcOffset =
        SrcTile::Rows * ACC_C0_SIZE * (indexCol / ACC_C0_SIZE) + (indexRow * ACC_C0_SIZE + (indexCol % ACC_C0_SIZE));
    if constexpr (SrcTile::Compact == CompactMode::Normal) {
        srcStride = (srcValidRow + BLOCK_LEN - 1) / BLOCK_LEN * BLOCK_LEN;
        srcOffset =
            srcStride * ACC_C0_SIZE * (indexCol / ACC_C0_SIZE) + (indexRow * ACC_C0_SIZE + (indexCol % ACC_C0_SIZE));
    }
    validCol = (validCol + c0Size - 1) / c0Size * c0Size;
    __cc__ srcType* srcData = (__cc__ srcType*)__cce_get_tile_ptr(src) + srcOffset;
    pto_copy_matrix_cc_to_ub(
        dstAddr, srcData, 0, validCol, validRow, dstStride, srcStride, 0, false, 0, 0, quantPre,
        static_cast<uint8_t>(reluMode), false, true, 0, 0, false, false, 0, false, false, false, false, false, false);
}

template <typename DstTile, typename SrcTile>
AICORE void TExtractToLeft(DstTile& dst, SrcTile& src, uint16_t indexRow, uint16_t indexCol)
{
    static_assert(
        (SrcTile::SFractal == SLayout::ColMajor && SrcTile::isRowMajor) ||
            (SrcTile::SFractal == SLayout::RowMajor && !SrcTile::isRowMajor) ||
            (SrcTile::Rows == 1 && SrcTile::isRowMajor),
        "TExtract: SrcTile Invalid Fractal");
    static_assert(DstTile::SFractal == SLayout::RowMajor && !DstTile::isRowMajor, "TExtract: DstTile Invalid Fractal");
    CheckTExtractToL0<typename DstTile::DType, typename SrcTile::DType>();
    if constexpr (SrcTile::Rows == 1 && SrcTile::isRowMajor) {
        TExtractToAVector<DstTile, SrcTile>(dst.data(), src.data(), indexRow, indexCol, dst.GetValidCol());
    } else if constexpr (DstTile::SFractal == SrcTile::SFractal) {
        if constexpr (DstTile::Compact == CompactMode::Normal) {
            TExtractToACompact<DstTile, SrcTile>(
                dst.data(), src.data(), indexRow, indexCol, dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToA<DstTile, SrcTile, false>(dst.data(), src.data(), indexRow, indexCol);
        }
    } else {
        if constexpr (DstTile::Compact == CompactMode::Normal || sizeof(typename SrcTile::DType) == 1) {
            TExtractToATransCompact<DstTile, SrcTile>(
                dst.data(), src.data(), indexRow, indexCol, dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToA<DstTile, SrcTile, true>(dst.data(), src.data(), indexRow, indexCol);
        }
    }
}

//---------------------------------------------------------------------------------------
// Kirin9030 shell of the common interface contracts
//---------------------------------------------------------------------------------------

// Single multi-burst copy: kirin9030 hardware supports it directly.
template <typename T>
__tf__ PTO_INTERNAL void TCopyUbToCbufFractal(
    __cbuf__ T* dstPtr, __ubuf__ T* srcPtr, uint16_t blockCout, uint16_t blockLen, uint16_t srcStride)
{
    copy_ubuf_to_cbuf(dstPtr, srcPtr, 0, blockCout, blockLen, srcStride, 0);
}

template <typename DstTile, typename SrcTile>
PTO_INTERNAL void TEXTRACT_TILE_IMPL(DstTile& dst, SrcTile& src, uint16_t indexRow, uint16_t indexCol)
{
    static_assert(
        is_textract_supported_type<typename DstTile::DType>,
        "TExtract: Unsupported data type! Supported types: (u)int8_t, (u)int16_t, int32, half");
    static_assert(
        (SrcTile::Loc == TileType::Acc) || std::is_same_v<typename DstTile::DType, typename SrcTile::DType>,
        "TExtract: Destination and Source tile data types must be the same");

    if constexpr (DstTile::Loc == TileType::Left) {
        TExtractToLeft(dst, src, indexRow, indexCol);
    } else if constexpr (DstTile::Loc == TileType::Right) {
        TExtractToRight(dst, src, indexRow, indexCol);
    } else if constexpr (SrcTile::Loc == TileType::Vec && DstTile::Loc == TileType::Mat) {
        TExtractVecToMat<DstTile, SrcTile>(
            dst.data(), src.data(), indexRow, indexCol, src.GetValidRow(), src.GetValidCol(), dst.GetValidRow(),
            dst.GetValidCol());
    } else if constexpr (DstTile::Loc == TileType::ScaleLeft) {
        static_assert(sizeof(DstTile::DType) == 0, "TExtract: ScaleLeft tile type is not supported yet.");
    } else if constexpr (DstTile::Loc == TileType::ScaleRight) {
        static_assert(sizeof(DstTile::DType) == 0, "TExtract: ScaleRight tile type is not supported yet.");
    } else if constexpr (
        SrcTile::Loc == TileType::Acc && (DstTile::Loc == TileType::Mat || DstTile::Loc == TileType::Vec)) {
        static_assert(
            (!DstTile::isRowMajor && DstTile::SFractal == SLayout::RowMajor) ||
                (DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox),
            "Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor) or (BFractal: RowMajor, "
            "SFractal: NoneBox).");
        CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType>();
        constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
        if constexpr ((DstTile::Loc == TileType::Mat)) {
            TExtractAccToMat<DstTile, SrcTile, quantPre, ReluPreMode::NoRelu>(
                dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
        } else {
            TExtractAccToVec<DstTile, SrcTile, AccToVecMode::SingleModeVec0, quantPre, ReluPreMode::NoRelu>(
                dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), src.GetValidRow(), indexRow, indexCol);
        }
    }
}

//---------------------------------------------------------------------------------------
// Acc extract with relu / scalar-quant / vector-quant overloads
//---------------------------------------------------------------------------------------

// vector quant
template <typename FpTile>
__tf__ PTO_INTERNAL void SetFPC(typename FpTile::TileDType __in__ fp, uint16_t indexCol)
{
    __fbuf__ typename FpTile::DType* dstAddrFp = (__fbuf__ typename FpTile::DType*)__cce_get_tile_ptr(fp) + indexCol;
    uint64_t deqTensorAddr = ((uint64_t)dstAddrFp >> static_cast<uint64_t>(7)) << 8;
    set_fpc(deqTensorAddr);
}

// relu
template <typename DstTile, typename SrcTile, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTile& dst, SrcTile& src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    static_assert(
        (DstTile::Loc == TileType::Mat || DstTile::Loc == TileType::Vec),
        "Destination TileType only support Mat and Vec.");
    CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType>();
    static_assert(
        (!DstTile::isRowMajor && DstTile::SFractal == SLayout::RowMajor) ||
            (DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox),
        "Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor) or (BFractal: RowMajor, "
        "SFractal: NoneBox).");
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
    if constexpr ((DstTile::Loc == TileType::Mat)) {
        TExtractAccToMat<DstTile, SrcTile, quantPre, reluMode>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
    } else {
        TExtractAccToVec<DstTile, SrcTile, AccToVecMode::SingleModeVec0, quantPre, ReluPreMode::NoRelu>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), src.GetValidRow(), indexRow, indexCol);
    }
}

template <typename DstTile, typename SrcTile, AccToVecMode mode, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTile& dst, SrcTile& src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    static_assert((DstTile::Loc == TileType::Vec), "Destination TileType only support Mat and Vec.");
    CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType>();
    static_assert(
        (DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox),
        "Dst fractal format should be (BFractal: RowMajor, SFractal: NoneBox).");
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
    TExtractAccToVec<DstTile, SrcTile, mode, quantPre, reluMode>(
        dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), src.GetValidRow(), indexRow, indexCol);
}

// scalar quant
template <typename DstTile, typename SrcTile, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(
    DstTile& dst, SrcTile& src, uint64_t preQuantScalar, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType, true>();
    static_assert(
        (DstTile::Loc == TileType::Mat || DstTile::Loc == TileType::Vec), "Destination TileType only support Mat.");
    static_assert(
        (!DstTile::isRowMajor && DstTile::SFractal == SLayout::RowMajor) ||
            (DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox),
        "Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor) or (BFractal: RowMajor, "
        "SFractal: NoneBox).");
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
    set_quant_pre(preQuantScalar);
    if constexpr ((DstTile::Loc == TileType::Mat)) {
        TExtractAccToMat<DstTile, SrcTile, quantPre, reluMode>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
    } else {
        TExtractAccToVec<DstTile, SrcTile, AccToVecMode::SingleModeVec0, quantPre, reluMode>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), src.GetValidRow(), indexRow, indexCol);
    }
}

template <typename DstTile, typename SrcTile, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(
    DstTile& dst, SrcTile& src, uint64_t preQuantScalar, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType, true>();
    static_assert((DstTile::Loc == TileType::Vec), "Destination TileType only support Mat.");
    static_assert(
        (DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox),
        "Dst fractal format should be (BFractal: RowMajor, SFractal: NoneBox).");
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
    set_quant_pre(preQuantScalar);
    TExtractAccToVec<DstTile, SrcTile, mode, quantPre, reluMode>(
        dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), src.GetValidRow(), indexRow, indexCol);
}

// fp
template <typename DstTile, typename SrcTile, typename FpTile, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(DstTile& dst, SrcTile& src, FpTile& fp, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType, true>();
    static_assert(
        (DstTile::Loc == TileType::Mat || DstTile::Loc == TileType::Vec),
        "Destination TileType only support Mat and Vec.");
    static_assert(
        (!DstTile::isRowMajor && DstTile::SFractal == SLayout::RowMajor) ||
            (DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox),
        "Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor) or (BFractal: RowMajor, "
        "SFractal: NoneBox).");
    static_assert(FpTile::Loc == TileType::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
    SetFPC<FpTile>(fp.data(), indexCol);
    if constexpr ((DstTile::Loc == TileType::Mat)) {
        TExtractAccToMat<DstTile, SrcTile, quantPre, reluMode>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
    } else {
        TExtractAccToVec<DstTile, SrcTile, AccToVecMode::SingleModeVec0, quantPre, reluMode>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), src.GetValidRow(), indexRow, indexCol);
    }
}

template <
    typename DstTile, typename SrcTile, typename FpTile, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(DstTile& dst, SrcTile& src, FpTile& fp, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    CheckTMovAccValid<DstTile, SrcTile, typename DstTile::DType, typename SrcTile::DType, true>();
    static_assert((DstTile::Loc == TileType::Vec), "Destination TileType only support Mat and Vec.");
    static_assert(
        (DstTile::isRowMajor && DstTile::SFractal == SLayout::NoneBox),
        "Dst fractal format should be (BFractal: RowMajor, SFractal: NoneBox).");
    static_assert(FpTile::Loc == TileType::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTile::DType, typename DstTile::DType>();
    SetFPC<FpTile>(fp.data(), indexCol);

    TExtractAccToVec<DstTile, SrcTile, mode, quantPre, reluMode>(
        dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), src.GetValidRow(), indexRow, indexCol);
}
} // namespace pto
#endif // TEXTRACT_HPP
