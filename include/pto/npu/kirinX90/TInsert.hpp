/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TINSERT_HPP_KIRINX90
#define TINSERT_HPP_KIRINX90
#include "common.hpp"

namespace pto {

#ifndef TINSERT_MODE_DEFINED
#define TINSERT_MODE_DEFINED
enum class TInsertMode : uint8_t
{
    SPLIT2 = 2,
    SPLIT4 = 3,
};
#endif

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TInsertVecToVecNDUnaligned(typename DstTileData::TileDType __out__ dst,
                                              typename SrcTileData::TileDType __in__ src, uint16_t validRow,
                                              uint16_t validCol, uint32_t indexRow, uint32_t indexCol)
{
    __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ T *dstAddr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    constexpr uint32_t dstRowStride = DstTileData::RowStride;
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;
    constexpr uint32_t elementsPerRepeat = REPEAT_BYTE / sizeof(T);
    constexpr uint32_t kValidCol = SrcTileData::ValidCol;
    constexpr uint16_t kFullRepeats = static_cast<uint16_t>(kValidCol / elementsPerRepeat);
    constexpr uint32_t kRemainder = kValidCol % elementsPerRepeat;

    __VEC_SCOPE__
    {
        RegTensor<T> vreg;
        UnalignReg ureg;

        for (uint16_t i = 0; i < validRow; ++i) {
            uint32_t srcRowOff = static_cast<uint32_t>(i) * srcRowStride;
            __ubuf__ T *pdst = dstAddr + (indexRow + static_cast<uint32_t>(i)) * dstRowStride + indexCol;
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
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    if (fullStripes > 0) {
        uint16_t burstLen = validRow;
        uint16_t srcGap = static_cast<uint16_t>(srcRows - validRow);
        uint16_t dstGap = static_cast<uint16_t>(dstRows - validRow);
        pto_copy_ubuf_to_ubuf((__ubuf__ void *)(dstAddr + dstOffsetBase), (__ubuf__ void *)srcAddr, fullStripes,
                              burstLen, srcGap, dstGap);
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
                vlds(vreg, srcAddr + offset, i * alignedNums, NORM);
                vsts(vreg, dstAddr + dstOffsetBase + offset, i * alignedNums, distValue, preg);
            }
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL constexpr uint32_t GetTmovAccDstStride()
{
    if constexpr (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox) {
        return DstTileData::Cols;
    }

    constexpr bool channelSplitEnable = (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor)) &&
                                        (std::is_same_v<typename DstTileData::DType, float>) &&
                                        (DstTileData::SFractalSize == 512);
    constexpr uint32_t c0Size = (!channelSplitEnable) &&
                                        (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor)) &&
                                        (DstTileData::SFractalSize == 1024) ?
                                    2 * C0_SIZE_BYTE / sizeof(typename DstTileData::DType) :
                                    C0_SIZE_BYTE / sizeof(typename DstTileData::DType);
    return DstTileData::Rows * c0Size;
}

template <typename DstTileData, typename SrcTileData, QuantMode_t QuantPre, ReluPreMode reluMode>
__tf__ PTO_INTERNAL void TInsertAccToVec(typename DstTileData::TileDType __out__ dst,
                                         typename SrcTileData::TileDType __in__ src, uint16_t validRow,
                                         uint16_t validCol, uint16_t indexRow, uint16_t indexCol)
{
    using dstType = typename DstTileData::DType;
    constexpr bool enableNz2Nd = (DstTileData::isRowMajor && DstTileData::SFractal == SLayout::NoneBox);
    constexpr bool enableNz2Nz = (!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor);
    constexpr bool channelSplitEnable =
        enableNz2Nz && (std::is_same_v<dstType, float>) && (DstTileData::SFractalSize == CUBE_BLOCK_SIZE);
    constexpr uint32_t dstStride = GetTmovAccDstStride<DstTileData, SrcTileData>();

    uint32_t dstOffset;
    if constexpr (enableNz2Nd) {
        dstOffset = static_cast<uint32_t>(indexRow) * DstTileData::Cols + indexCol;
        constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(dstType);
        validCol = (validCol + c0Size - 1) / c0Size * c0Size;
    } else {
        constexpr int32_t c0Size = (!channelSplitEnable) && (DstTileData::SFractalSize == 2 * CUBE_BLOCK_SIZE) ?
                                       2 * C0_SIZE_BYTE / sizeof(dstType) :
                                       C0_SIZE_BYTE / sizeof(dstType);
        dstOffset = DstTileData::Rows * c0Size * (indexCol / c0Size) + (indexRow * c0Size + (indexCol % c0Size));
    }

    if constexpr (enableNz2Nz) {
        constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(dstType);
        validRow = SrcTileData::Rows;

        if constexpr (std::is_same_v<dstType, float>) {
            constexpr int32_t align = channelSplitEnable ? c0Size : FRACTAL_NZ_ROW;
            validCol = CeilDivision(static_cast<uint32_t>(validCol), static_cast<uint32_t>(align)) * align;
        } else {
            validCol = CeilDivision(static_cast<uint32_t>(validCol), static_cast<uint32_t>(c0Size)) * c0Size;
        }
    }

    if constexpr (enableNz2Nd) {
        SetLoop3Para();
    }

    auto srcStride = (validRow + BLOCK_LEN - 1) / BLOCK_LEN * BLOCK_LEN;
    __ubuf__ dstType *dstAddr = (__ubuf__ dstType *)__cce_get_tile_ptr(dst) + dstOffset;
    __cc__ typename SrcTileData::DType *srcData = (__cc__ typename SrcTileData::DType *)__cce_get_tile_ptr(src);

    copy_matrix_cc_to_ub(dstAddr, srcData, 0, validCol, validRow, dstStride, srcStride, 0, 0, QuantPre, reluMode, false,
                         enableNz2Nd, false);
}

template <typename DstTileData, typename SrcTileData, QuantMode_t QuantPre, ReluPreMode reluMode>
__tf__ PTO_INTERNAL void TInsertAccToMat(typename DstTileData::TileDType __out__ dst,
                                         typename SrcTileData::TileDType __in__ src, uint16_t validRow,
                                         uint16_t validCol, uint16_t indexRow, uint16_t indexCol)
{
    using SrcType = typename SrcTileData::DType;
    using DstType = typename DstTileData::DType;

    constexpr bool channelSplitEnable = (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor)) &&
                                        (std::is_same_v<DstType, float>) &&
                                        (DstTileData::SFractalSize == CUBE_BLOCK_SIZE);
    constexpr int32_t baseC0Size = C0_SIZE_BYTE / sizeof(DstType);
    constexpr int32_t c0Size =
        (!channelSplitEnable) && (DstTileData::SFractalSize == 2 * CUBE_BLOCK_SIZE) ? 2 * baseC0Size : baseC0Size;
    uint32_t dstOffset = DstTileData::Rows * c0Size * (indexCol / c0Size) + (indexRow * c0Size + (indexCol % c0Size));
    __cc__ SrcType *srcAddr = (__cc__ SrcType *)__cce_get_tile_ptr(src);
    __cbuf__ DstType *dstAddr = (__cbuf__ DstType *)__cce_get_tile_ptr(dst) + dstOffset;

    constexpr uint32_t dstStrideD = DstTileData::Rows * c0Size;
    constexpr uint16_t srcStride = SrcTileData::Rows;
    uint16_t nSize = CeilDivision(validCol, c0Size) * c0Size;
    copy_matrix_cc_to_cbuf(dstAddr, srcAddr, 0, nSize, SrcTileData::Rows, dstStrideD, srcStride, 0, 0, QuantPre,
                           reluMode, channelSplitEnable, false, false);
}

#include "pto/common/arch/memory/tinsert_common.hpp"
template <typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void TInsertNDImpl(typename DstTileData::TileDType __out__ dst,
                                       typename SrcTileData::TileDType __in__ src, uint16_t validRow, uint16_t validCol,
                                       uint16_t dstCols, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    __cbuf__ T *dstAddr = (__cbuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src);

    uint32_t dstOffset = indexRow * dstCols + indexCol;
    __cbuf__ T *dstStart = dstAddr + dstOffset;

    uint32_t totalBytes = static_cast<uint32_t>(validRow) * static_cast<uint32_t>(validCol) * sizeof(T);

    if (validCol == SrcTileData::Cols && validCol == dstCols && totalBytes >= BLOCK_BYTE_SIZE) {
        uint16_t burstLen = static_cast<uint16_t>(totalBytes / BLOCK_BYTE_SIZE);
        copy_ubuf_to_cbuf(dstStart, srcAddr, 0, 1, burstLen, 0, 0);
    } else if (static_cast<uint32_t>(validCol) * sizeof(T) >= BLOCK_BYTE_SIZE) {
        uint16_t rowBurstLen = static_cast<uint16_t>((validCol * sizeof(T)) / BLOCK_BYTE_SIZE);
        uint16_t srcRowGap =
            static_cast<uint16_t>((SrcTileData::Cols * sizeof(T) - validCol * sizeof(T)) / BLOCK_BYTE_SIZE);
        uint16_t dstRowGap = static_cast<uint16_t>((dstCols * sizeof(T) - validCol * sizeof(T)) / BLOCK_BYTE_SIZE);
        copy_ubuf_to_cbuf(dstStart, srcAddr, 0, validRow, rowBurstLen, srcRowGap, dstRowGap);
    } else {
        uint16_t burstLen = static_cast<uint16_t>(CeilDivision(totalBytes, static_cast<uint32_t>(BLOCK_BYTE_SIZE)));
        copy_ubuf_to_cbuf(dstStart, srcAddr, 0, 1, burstLen, 0, 0);
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
PTO_INTERNAL void ComputeNZBlockParams(uint32_t validRow, uint32_t validCol, uint32_t dstRow, uint16_t &burstNum,
                                       uint16_t &burstLen, uint16_t &srcGap, uint16_t &dstGap, uint32_t &dstOffset,
                                       uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    constexpr uint32_t typeSize = sizeof(T);
    constexpr bool isFp4Type = std::is_same_v<T, float4_e2m1x2_t> || std::is_same_v<T, float4_e1m2x2_t>;
    uint32_t c0Size = BLOCK_BYTE_SIZE / typeSize;
    uint32_t byteValidCol = isFp4Type ? validCol / 2 : validCol;
    uint32_t byteIndexCol = isFp4Type ? indexCol / 2 : indexCol;
    burstNum = static_cast<uint16_t>(CeilDivision(byteValidCol, c0Size));
    burstLen = (validRow * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;
    uint32_t colBlockOffset = (byteIndexCol / c0Size) * dstRow * c0Size;
    uint32_t rowOffset = indexRow * c0Size + (byteIndexCol % c0Size);
    dstOffset = colBlockOffset + rowOffset;
    uint32_t srcStrideRows;
    if constexpr (SrcTileData::Compact == CompactMode::Null) {
        srcStrideRows = SrcTileData::Rows;
    } else if constexpr (SrcTileData::Compact == CompactMode::RowPlusOne) {
        srcStrideRows = CeilDivision(validRow, static_cast<uint32_t>(FRACTAL_NZ_ROW)) * FRACTAL_NZ_ROW + 1;
    } else {
        srcStrideRows = CeilDivision(validRow, static_cast<uint32_t>(FRACTAL_NZ_ROW)) * FRACTAL_NZ_ROW;
    }
    srcGap = static_cast<uint16_t>(srcStrideRows - validRow);
    dstGap = static_cast<uint16_t>(dstRow - validRow);
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void TInsertImpl(typename DstTileData::TileDType __out__ dst,
                                     typename SrcTileData::TileDType __in__ src, uint16_t validRow, uint16_t validCol,
                                     uint16_t dstRow, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    __cbuf__ T *dstAddr = (__cbuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    uint16_t burstNum, burstLen, srcGap, dstGap;
    uint32_t dstOffset;
    ComputeNZBlockParams<T, DstTileData, SrcTileData>(validRow, validCol, dstRow, burstNum, burstLen, srcGap, dstGap,
                                                      dstOffset, indexRow, indexCol);
    __cbuf__ T *dstAddr2 = dstAddr + dstOffset;
    copy_ubuf_to_cbuf(dstAddr2, srcAddr, 0, burstNum, burstLen, srcGap, dstGap);
}

template <typename T, typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TInsertVecToMatImpl(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol)
{
    uint16_t validRow = static_cast<uint16_t>(src.GetValidRow());
    uint16_t validCol = static_cast<uint16_t>(src.GetValidCol());
    PTO_ASSERT(indexRow + validRow <= DstTileData::Rows, "TINSERT : indexRow + validRow exceeds destination rows!");
    PTO_ASSERT(indexCol + validCol <= DstTileData::Cols, "TINSERT : indexCol + validCol exceeds destination cols!");

    if constexpr (SrcTileData::isRowMajor) {
        uint16_t dstCols = static_cast<uint16_t>(DstTileData::Cols);
        TInsertNDImpl<T, DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol, dstCols, indexRow,
                                                   indexCol);
    } else if constexpr (!SrcTileData::isRowMajor && (SrcTileData::SFractal == SLayout::RowMajor)) {
        constexpr uint16_t dstRow =
            static_cast<uint16_t>(DstTileData::BFractal == BLayout::ColMajor ? DstTileData::Rows : DstTileData::Cols);
        PTO_ASSERT(indexRow + validRow <= dstRow, "TINSERT NZ : indexRow + validRow exceeds destination rows!");
        TInsertImpl<T, DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol, dstRow, indexRow,
                                                 indexCol);
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    if constexpr (DstTileData::Loc == TileType::Vec && SrcTileData::Loc == TileType::Vec) {
        CheckTInsertVecToVecCommon<DstTileData, SrcTileData>();
        if constexpr (DstTileData::isRowMajor && SrcTileData::isRowMajor &&
                      (DstTileData::SFractal == SLayout::NoneBox) && (SrcTileData::SFractal == SLayout::NoneBox)) {
            TInsertVecToVecNDDispatch<DstTileData, SrcTileData>(dst, src, indexRow, indexCol);
        } else if constexpr (!DstTileData::isRowMajor && !SrcTileData::isRowMajor &&
                             DstTileData::SFractal == SLayout::RowMajor && SrcTileData::SFractal == SLayout::RowMajor) {
            using T = typename DstTileData::DType;
            constexpr uint32_t kC0Size = BLOCK_BYTE_SIZE / sizeof(T);
            uint16_t validRow = static_cast<uint16_t>(src.GetValidRow());
            uint16_t validCol = static_cast<uint16_t>(src.GetValidCol());
            PTO_ASSERT(indexRow % FRACTAL_NZ_ROW == 0, "TINSERT NZ Vec->Vec : indexRow must be 16-aligned.");
            PTO_ASSERT(indexCol % kC0Size == 0, "TINSERT NZ Vec->Vec : indexCol must be c0Size-aligned.");
            PTO_ASSERT(indexRow + validRow <= DstTileData::Rows,
                       "TINSERT NZ Vec->Vec : indexRow + validRow exceeds dstRows.");
            PTO_ASSERT(indexCol + validCol <= DstTileData::Cols,
                       "TINSERT NZ Vec->Vec : indexCol + validCol exceeds dstCols.");
            uint16_t burstNum, burstLen, srcGap, dstGap;
            uint32_t dstOffset;
            ComputeNZBlockParams<T, DstTileData, SrcTileData>(validRow, validCol,
                                                              static_cast<uint32_t>(DstTileData::Rows), burstNum,
                                                              burstLen, srcGap, dstGap, dstOffset, indexRow, indexCol);
            __ubuf__ T *dstAddr = (__ubuf__ T *)__cce_get_tile_ptr(dst.data());
            __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src.data());
            pto_copy_ubuf_to_ubuf((__ubuf__ void *)(dstAddr + dstOffset), (__ubuf__ void *)srcAddr, burstNum, burstLen,
                                  srcGap, dstGap);
        } else {
            static_assert(DstTileData::isRowMajor == SrcTileData::isRowMajor,
                          "TINSERT Vec->Vec : Source and destination layout must match (both ND or both NZ).");
            static_assert(DstTileData::SFractal == SrcTileData::SFractal,
                          "TINSERT Vec->Vec : Source and destination SFractal must match.");
        }
    } else if constexpr (DstTileData::Loc == TileType::Mat && SrcTileData::Loc == TileType::Vec) {
        using T = typename SrcTileData::DType;

        static_assert(std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
                      "TINSERT : Source and destination data types must match");
        static_assert((std::is_same<T, half>::value) || (std::is_same<T, bfloat16_t>::value) ||
                          (std::is_same<T, float>::value) || (std::is_same<T, int32_t>::value) ||
                          (std::is_same<T, float8_e4m3_t>::value) || (std::is_same<T, float8_e5m2_t>::value) ||
                          (std::is_same<T, hifloat8_t>::value) || (std::is_same<T, int8_t>::value) ||
                          (std::is_same<T, float8_e8m0_t>::value) || (std::is_same<T, float4_e2m1x2_t>::value) ||
                          (std::is_same<T, float4_e1m2x2_t>::value),
                      "TINSERT : Unsupported data type.");
        TInsertVecToMatImpl<T>(dst, src, indexRow, indexCol);

    } else {
        CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
        PTO_ASSERT(indexRow + SrcTileData::Rows <= DstTileData::Rows,
                   "The sum of indexRow and srcRow should be less than dstRow!");
        PTO_ASSERT(indexCol + SrcTileData::Cols <= DstTileData::Cols,
                   "The sum of indexCol and srcCol should be less than dstCol!");
        constexpr QuantMode_t quantPre =
            GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
        if constexpr (DstTileData::Loc == TileType::Mat) {
            TInsertAccToMat<DstTileData, SrcTileData, quantPre, ReluPreMode::NoRelu>(
                dst.data(), src.data(), src.GetValidRow(), src.GetValidCol(), indexRow, indexCol);
        } else {
            TInsertAccToVec<DstTileData, SrcTileData, quantPre, ReluPreMode::NoRelu>(
                dst.data(), src.data(), src.GetValidRow(), src.GetValidCol(), indexRow, indexCol);
        }
    }
}

// relu
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, false>();
    PTO_ASSERT(indexRow + SrcTileData::Rows <= DstTileData::Rows,
               "The sum of indexRow and srcRow should be less than dstRow!");
    PTO_ASSERT(indexCol + SrcTileData::Cols <= DstTileData::Cols,
               "The sum of indexCol and srcCol should be less than dstCol!");
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    if constexpr (DstTileData::Loc == TileType::Mat) {
        TInsertAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), src.GetValidRow(),
                                                                      src.GetValidCol(), indexRow, indexCol);
    } else {
        TInsertAccToVec<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), src.GetValidRow(),
                                                                      src.GetValidCol(), indexRow, indexCol);
    }
}

// scalar quant
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow = 0,
                               uint16_t indexCol = 0)
{
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    PTO_ASSERT(indexRow + SrcTileData::Rows <= DstTileData::Rows,
               "The sum of indexRow and srcRow should be less than dstRow!");
    PTO_ASSERT(indexCol + SrcTileData::Cols <= DstTileData::Cols,
               "The sum of indexCol and srcCol should be less than dstCol!");
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    set_quant_pre(preQuantScalar);
    if constexpr (DstTileData::Loc == TileType::Mat) {
        TInsertAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), src.GetValidRow(),
                                                                      src.GetValidCol(), indexRow, indexCol);
    } else {
        TInsertAccToVec<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), src.GetValidRow(),
                                                                      src.GetValidCol(), indexRow, indexCol);
    }
}

// vector quant
template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow = 0,
                               uint16_t indexCol = 0)
{
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    PTO_ASSERT(indexRow + SrcTileData::Rows <= DstTileData::Rows,
               "The sum of indexRow and srcRow should be less than dstRow!");
    PTO_ASSERT(indexCol + SrcTileData::Cols <= DstTileData::Cols,
               "The sum of indexCol and srcCol should be less than dstCol!");
    static_assert(FpTileData::Loc == TileType::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    SetFPCInsert<FpTileData>(fp.data());
    if constexpr (DstTileData::Loc == TileType::Mat) {
        TInsertAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), src.GetValidRow(),
                                                                      src.GetValidCol(), indexRow, indexCol);
    } else {
        TInsertAccToVec<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), src.GetValidRow(),
                                                                      src.GetValidCol(), indexRow, indexCol);
    }
}

// relu with AccToVecMode
template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    static_assert(mode == AccToVecMode::SingleModeVec0, "Only SingleModeVec0 is supported.");
    TINSERT_IMPL<DstTileData, SrcTileData, reluMode>(dst, src, indexRow, indexCol);
}

// scalar quant with AccToVecMode
template <typename DstTileData, typename SrcTileData, AccToVecMode mode, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow = 0,
                               uint16_t indexCol = 0)
{
    static_assert(mode == AccToVecMode::SingleModeVec0, "Only SingleModeVec0 is supported.");
    TINSERT_IMPL<DstTileData, SrcTileData, reluMode>(dst, src, preQuantScalar, indexRow, indexCol);
}

// vector quant with AccToVecMode
template <typename DstTileData, typename SrcTileData, typename FpTileData, AccToVecMode mode,
          ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow = 0,
                               uint16_t indexCol = 0)
{
    static_assert(mode == AccToVecMode::SingleModeVec0, "Only SingleModeVec0 is supported.");
    TINSERT_IMPL<DstTileData, SrcTileData, FpTileData, reluMode>(dst, src, fp, indexRow, indexCol);
}

template <uint32_t SplitCount, typename T, typename DstTileData, typename SrcTileData>
__tf__ PTO_INTERNAL void TInsertSplitImpl(typename DstTileData::TileDType __out__ dst,
                                          typename SrcTileData::TileDType __in__ src, uint16_t validRow,
                                          uint16_t validCol, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    __cbuf__ T *dstAddr = (__cbuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src);

    constexpr uint32_t typeSize = sizeof(T);
    constexpr bool isFp4Type = false;
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

    __cbuf__ T *dstAddr0 = dstAddr + dstOffset;
    copy_ubuf_to_cbuf(dstAddr0, srcAddr, 0, partBurstNum, burstLen, srcGap, dstGap);

    if constexpr (SplitCount >= 2) {
        __ubuf__ T *src1 = srcAddr + partBurstNum * srcBlockSize;
        __cbuf__ T *dst1 = dstAddr0 + partBurstNum * dstBlockSize;
        uint16_t burst1Num = (SplitCount == 2) ? lastBurstNum : partBurstNum;
        copy_ubuf_to_cbuf(dst1, src1, 0, burst1Num, burstLen, srcGap, dstGap);
    }

    if constexpr (SplitCount >= 4) {
        __ubuf__ T *src2 = srcAddr + 2 * partBurstNum * srcBlockSize;
        __cbuf__ T *dst2 = dstAddr0 + 2 * partBurstNum * dstBlockSize;
        copy_ubuf_to_cbuf(dst2, src2, 0, partBurstNum, burstLen, srcGap, dstGap);

        __ubuf__ T *src3 = srcAddr + 3 * partBurstNum * srcBlockSize;
        __cbuf__ T *dst3 = dstAddr0 + 3 * partBurstNum * dstBlockSize;
        copy_ubuf_to_cbuf(dst3, src3, 0, lastBurstNum, burstLen, srcGap, dstGap);
    }
}

template <TInsertMode mode, typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    using T = typename SrcTileData::DType;
    static_assert(std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
                  "TINSERT : Source and destination data types must match");
    static_assert(DstTileData::Loc == TileType::Mat, "TINSERT : Destination must be Mat tile (L1/cbuf)");
    static_assert(SrcTileData::Loc == TileType::Vec, "TINSERT : Source must be Vec tile (UB/ubuf)");
    static_assert(!SrcTileData::isRowMajor && (SrcTileData::SFractal == SLayout::RowMajor),
                  "TINSERT NZ : Source must be NZ format (column-major, RowMajor fractal)");
    static_assert((std::is_same<T, half>::value) || (std::is_same<T, float>::value) ||
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
        TInsertSplitImpl<2, T, DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol, indexRow,
                                                         indexCol);
    } else if constexpr (mode == TInsertMode::SPLIT4) {
        TInsertSplitImpl<4, T, DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol, indexRow,
                                                         indexCol);
    }
}

} // namespace pto
#endif // TINSERT_HPP
