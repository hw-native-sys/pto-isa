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
        std::is_same<DstType, int8_t>::value || std::is_same<DstType, uint8_t>::value ||
            std::is_same<DstType, half>::value || std::is_same<DstType, int16_t>::value ||
            std::is_same<DstType, uint16_t>::value || std::is_same<DstType, int32_t>::value ||
            std::is_same<DstType, float>::value,
        "TExtract: Unsupported data type! Supported types: (u)int8_t, (u)int16_t, int32, half, float");
}
template <typename DstTileData, typename SrcTileData, QuantMode_t QuantPre, ReluPreMode reluMode>
__tf__ AICORE void TExtractAccToMat(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint16_t indexRow, uint16_t indexCol)
{
    using dstType = typename DstTileData::DType;
    using srcType = typename SrcTileData::DType;

    using Cfg = TMovConfig<DstTileData>;

    constexpr uint32_t dstStride = Cfg::accDstStride;
    static_assert(
        ((dstStride * sizeof(dstType) % C0_SIZE_BYTE == 0) && (dstStride > 0)),
        "Dst Tile Cols * sizeof(dstType) must be multiples of 32 and not 0 when nz2nd/nz2nz");

    if constexpr (Cfg::isNz2Nz) {
        validRow = SrcTileData::Rows;
        validCol = CeilAlignment(validCol, Cfg::baseC0Size);
    }

    uint32_t srcOffset = SrcTileData::Rows * ACC_C0_SIZE * (indexCol / ACC_C0_SIZE) +
                         (indexRow * ACC_C0_SIZE + (indexCol % ACC_C0_SIZE));
    if constexpr (Cfg::isNz2Nd) {
        SetLoop3Para();
    }
    auto srcStride = CeilAlignment(validRow, BLOCK_LEN);
    __cbuf__ dstType* dstAddr = (__cbuf__ dstType*)__cce_get_tile_ptr(dst);
    __cc__ srcType* srcData = (__cc__ srcType*)__cce_get_tile_ptr(src) + srcOffset;

    pto_copy_matrix_cc_to_cbuf(
        dstAddr, srcData, 0, validCol, validRow, dstStride, srcStride, 0, QuantPre, static_cast<uint8_t>(reluMode),
        false, Cfg::isNz2Nd);
}

template <typename DstTileData, typename SrcTileData, QuantMode_t QuantPre, ReluPreMode reluMode>
__tf__ AICORE void TExtractAccToVec(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint16_t indexRow, uint16_t indexCol)
{
    using dstType = typename DstTileData::DType;
    using srcType = typename SrcTileData::DType;

    using Cfg = TMovConfig<DstTileData>;

    constexpr uint32_t dstStride = Cfg::accDstStride;
    static_assert(
        ((dstStride * sizeof(dstType) % C0_SIZE_BYTE == 0) && (dstStride > 0)),
        "Dst Tile Cols * sizeof(dstT) must be multiples of 32 and not 0 when nz2nd/nz2nz.");

    if constexpr (Cfg::isNz2Nz) {
        validRow = SrcTileData::Rows;
        validCol = CeilAlignment(validCol, Cfg::baseC0Size);
    }

    uint32_t srcOffset = SrcTileData::Rows * ACC_C0_SIZE * (indexCol / ACC_C0_SIZE) +
                         (indexRow * ACC_C0_SIZE + (indexCol % ACC_C0_SIZE));
    if constexpr (Cfg::isNz2Nd) {
        SetLoop3Para();
    }
    auto srcStride = CeilAlignment(validRow, BLOCK_LEN);
    __ubuf__ dstType* dstAddr = (__ubuf__ dstType*)__cce_get_tile_ptr(dst);
    __cc__ srcType* srcData = (__cc__ srcType*)__cce_get_tile_ptr(src) + srcOffset;

    pto_copy_matrix_cc_to_ub(
        dstAddr, srcData, 0, validCol, validRow, dstStride, srcStride, 0, false, 0, 0, QuantPre,
        static_cast<uint8_t>(reluMode), false, Cfg::isNz2Nd, 0, 0, false, false, 0, false, false, false, false, false,
        false);
}

template <typename DstTile, typename SrcTile>
__tf__ PTO_INTERNAL void TExtractVecToMat(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src, uint16_t indexRow,
    uint16_t indexCol, uint32_t srcValidRow, uint32_t srcValidCol, uint32_t dstValidRow, uint32_t dstValidCol)
{
    using T = typename SrcTile::DType;

    static_assert(
        sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4, "TExtract Vec->Mat: Data type size must be 1/2/4 bytes.");

    static_assert(
        SrcTile::Loc == TileType::Vec && DstTile::Loc == TileType::Mat, "TExtract Vec->Mat: Only support Vec->Mat.");

    static_assert(
        (SrcTile::isRowMajor && SrcTile::SFractal == SLayout::NoneBox && DstTile::isRowMajor &&
         DstTile::SFractal == SLayout::NoneBox) ||
            (!SrcTile::isRowMajor && SrcTile::SFractal == SLayout::RowMajor && !DstTile::isRowMajor &&
             DstTile::SFractal == SLayout::RowMajor),
        "TExtract Vec->Mat: Only support ND->ND or NZ->NZ on kirinX90.");

    using T = typename SrcTile::DType;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(T);
    uint32_t offset = SrcTile::Rows * c0Size * (indexCol / c0Size) + (indexRow * c0Size + (indexCol % c0Size));
    if constexpr (SrcTile::isRowMajor && (SrcTile::SFractal == SLayout::NoneBox)) {
        offset = indexRow * srcValidCol + indexCol;
    }

    __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src) + offset;
    __cbuf__ T* dstPtr = (__cbuf__ T*)__cce_get_tile_ptr(dst);
    if constexpr (SrcTile::isRowMajor && (SrcTile::SFractal == SLayout::NoneBox)) {
        uint16_t blockLen = dstValidRow * dstValidCol * sizeof(T) / BLOCK_BYTE_SIZE;
        // dst, src, sid, nBurst, lenBurst, srcStride, dstStride
        copy_ubuf_to_cbuf(dstPtr, srcPtr, 0, 1, blockLen, 0, 0);
    } else if constexpr (!SrcTile::isRowMajor && (SrcTile::SFractal == SLayout::RowMajor)) {
        uint16_t blockCout = CeilDivision(dstValidCol, c0Size);
        uint32_t alignRow = (dstValidRow + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
        uint16_t blockLen = alignRow * c0Size * sizeof(T) / BLOCK_BYTE_SIZE;
        constexpr uint16_t srcStride = SrcTile::Rows - DstTile::Rows;
        copy_ubuf_to_cbuf(dstPtr, srcPtr, 0, blockCout, blockLen, srcStride, 0);
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_TILE_IMPL(DstTileData& dst, SrcTileData& src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    CheckTExtract<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType>();
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
    } else if constexpr (SrcTileData::Loc == TileType::Vec && DstTileData::Loc == TileType::Mat) {
        TExtractVecToMat<DstTileData, SrcTileData>(
            dst.data(), src.data(), indexRow, indexCol, src.GetValidRow(), src.GetValidCol(), dst.GetValidRow(),
            dst.GetValidCol());
    } else if constexpr (SrcTileData::Loc == TileType::Acc) {
        CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
        constexpr QuantMode_t quantPre =
            GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
        if constexpr (DstTileData::Loc == TileType::Mat) {
            TExtractAccToMat<DstTileData, SrcTileData, quantPre, ReluPreMode::NoRelu>(
                dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
        } else {
            TExtractAccToVec<DstTileData, SrcTileData, quantPre, ReluPreMode::NoRelu>(
                dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
        }
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractVecToVecND(
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
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();

    if (alignedBytes > 0) {
        uint16_t burstLen = static_cast<uint16_t>(alignedBytes / BLOCK_BYTE_SIZE);
        uint16_t srcGap = static_cast<uint16_t>((srcRowStride * sizeof(T) - alignedBytes) / BLOCK_BYTE_SIZE);
        uint16_t dstGap = static_cast<uint16_t>((dstRowStride * sizeof(T) - alignedBytes) / BLOCK_BYTE_SIZE);
        pto_copy_ubuf_to_ubuf((__ubuf__ void*)dstAddr, (__ubuf__ void*)srcStart, validRow, burstLen, srcGap, dstGap);
    }

    if (tailBytes > 0) {
        __VEC_SCOPE__
        {
            RegTensor<T> vreg;
            MaskReg preg;
            uint32_t alignedNums = BLOCK_BYTE_SIZE / sizeof(T);
            uint32_t offset = validCol / alignedNums * alignedNums;
            uint32_t remainsEles = validCol - offset;
            preg = CreatePredicate<T>(remainsEles);

            for (uint16_t i = 0; i < validRow; i++) {
                vlds(vreg, srcStart, i * srcRowStride + offset, NORM);
                vsts(vreg, dstAddr, i * dstRowStride + offset, distValue, preg);
            }
        }
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractVecToVecNZ(
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

    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    if (fullStripes > 0) {
        uint16_t burstLen = validRow;
        uint16_t srcGap = static_cast<uint16_t>(srcRows - validRow);
        uint16_t dstGap = static_cast<uint16_t>(dstRows - validRow);
        pto_copy_ubuf_to_ubuf(
            (__ubuf__ void*)dstAddr, (__ubuf__ void*)(srcAddr + srcOffsetBase), fullStripes, burstLen, srcGap, dstGap);
    }
    if (tailCols > 0) {
        uint32_t alignedNums = BLOCK_BYTE_SIZE / sizeof(T);
        uint32_t offset = srcRows * (validCol / alignedNums * alignedNums);
        uint32_t remainsEles = validCol - validCol / alignedNums * alignedNums;
        __VEC_SCOPE__
        {
            RegTensor<T> vreg;
            MaskReg preg;

            preg = CreatePredicate<T>(remainsEles);

            for (uint16_t i = 0; i < validRow; i++) {
                vlds(vreg, srcAddr + srcOffsetBase + offset, i * alignedNums, NORM);
                vsts(vreg, dstAddr + offset, i * alignedNums, distValue, preg);
            }
        }
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
        TExtractVecToVecND<T, DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol, idxRow, idxCol);
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
        TExtractVecToVecNZ<T, DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol, idxRow, idxCol);
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData& dst, SrcTileData& src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    if constexpr (DstTileData::Loc == TileType::Vec && SrcTileData::Loc == TileType::Vec) {
        CheckTExtractVecToVecCommon<DstTileData, SrcTileData>();
        if constexpr (
            DstTileData::isRowMajor && SrcTileData::isRowMajor || DstTileData::SFractal == SLayout::NoneBox ||
            SrcTileData::SFractal == SLayout::NoneBox) {
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
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData& dst, SrcTileData& src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    PTO_ASSERT(
        indexRow + DstTileData::Rows <= SrcTileData::Rows,
        "The sum of indexRow and dstRow should be less than srcRow!");
    PTO_ASSERT(
        indexCol + DstTileData::Cols <= SrcTileData::Cols,
        "The sum of indexCol and dstCol should be less than srcCol!");
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    if constexpr (DstTileData::Loc == TileType::Mat) {
        TExtractAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
    } else {
        TExtractAccToVec<DstTileData, SrcTileData, quantPre, reluMode>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
    }
}

// scalar quant
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(
    DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    PTO_ASSERT(
        indexRow + DstTileData::Rows <= SrcTileData::Rows,
        "The sum of indexRow and dstRow should be less than srcRow!");
    PTO_ASSERT(
        indexCol + DstTileData::Cols <= SrcTileData::Cols,
        "The sum of indexCol and dstCol should be less than srcCol!");
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    set_quant_pre(preQuantScalar);
    if constexpr (DstTileData::Loc == TileType::Mat) {
        TExtractAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
    } else {
        TExtractAccToVec<DstTileData, SrcTileData, quantPre, reluMode>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
    }
}

// vector quant
template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(
    DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    PTO_ASSERT(
        indexRow + DstTileData::Rows <= SrcTileData::Rows,
        "The sum of indexRow and dstRow should be less than srcRow!");
    PTO_ASSERT(
        indexCol + DstTileData::Cols <= SrcTileData::Cols,
        "The sum of indexCol and dstCol should be less than srcCol!");
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
    static_assert(FpTileData::Loc == TileType::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    SetFPC<FpTileData>(fp.data(), indexCol);
    if constexpr (DstTileData::Loc == TileType::Mat) {
        TExtractAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
    } else {
        TExtractAccToVec<DstTileData, SrcTileData, quantPre, reluMode>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
    }
}

// relu with AccToVecMode
template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData& dst, SrcTileData& src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    static_assert(mode == AccToVecMode::SingleModeVec0, "Only SingleModeVec0 is supported.");
    TEXTRACT_IMPL<DstTileData, SrcTileData, reluMode>(dst, src, indexRow, indexCol);
}

// scalar quant with AccToVecMode
template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(
    DstTileData& dst, SrcTileData& src, uint64_t preQuantScalar, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    static_assert(mode == AccToVecMode::SingleModeVec0, "Only SingleModeVec0 is supported.");
    TEXTRACT_IMPL<DstTileData, SrcTileData, reluMode>(dst, src, preQuantScalar, indexRow, indexCol);
}

// vector quant with AccToVecMode
template <
    typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
    ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(
    DstTileData& dst, SrcTileData& src, FpTileData& fp, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    static_assert(mode == AccToVecMode::SingleModeVec0, "Only SingleModeVec0 is supported.");
    TEXTRACT_IMPL<DstTileData, SrcTileData, FpTileData, reluMode>(dst, src, fp, indexRow, indexCol);
}
} // namespace pto
#endif // TEXTRACT_HPP
