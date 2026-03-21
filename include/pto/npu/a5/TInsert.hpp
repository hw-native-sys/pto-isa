/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#ifndef COPY_CC_TO_CUBF
#define COPY_CC_TO_CUBF(dst, src, nSize, srcRow, dstStride, srcStride, QuantPre, reluMode, channelSplitEnable) \
    copy_matrix_cc_to_cbuf(dst, src, 0, nSize, srcRow, dstStride, srcStride, 0, 0, 0, QuantPre, reluMode,      \
                           channelSplitEnable, false, 0, 0, false, false, 0, false, false, false, false, false, false)
#endif
namespace pto {

#ifndef TINSERT_MODE_DEFINED
#define TINSERT_MODE_DEFINED
enum class TInsertMode : uint8_t
{
    NZ = 0,
    NZ_PLUS_1 = 1,
    SPLIT2_NZ_PLUS_1 = 2,
    SPLIT4_NZ_PLUS_1 = 3,
    ND = 4,
};
#endif

template <typename DstTileData, typename SrcTileData, QuantMode_t QuantPre, ReluPreMode reluMode>
__tf__ PTO_INTERNAL void TInsertAccToMat(typename DstTileData::TileDType __out__ dst,
                                         typename SrcTileData::TileDType __in__ src, uint16_t validRow,
                                         uint16_t validCol, uint16_t indexRow, uint16_t indexCol)
{
    using dstType = typename DstTileData::DType;
    constexpr bool channelSplitEnable =
        (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor)) &&
        (std::is_same_v<dstType, float>)&&(DstTileData::SFractalSize == CUBE_BLOCK_SIZE);
    constexpr int32_t c0Size = (!channelSplitEnable) && (DstTileData::SFractalSize == 2 * CUBE_BLOCK_SIZE) ?
                                   2 * C0_SIZE_BYTE / sizeof(dstType) :
                                   C0_SIZE_BYTE / sizeof(dstType);
    uint32_t dstOffset = DstTileData::Rows * c0Size * (indexCol / c0Size) + (indexRow * c0Size + (indexCol % c0Size));
    constexpr uint32_t dstStride = DstTileData::Rows * c0Size;
    uint16_t nSize = CeilDivision(validCol, c0Size) * c0Size;
    __cbuf__ dstType *dstAddr = (__cbuf__ dstType *)__cce_get_tile_ptr(dst) + dstOffset;
    __cc__ typename SrcTileData::DType *srcData = (__cc__ typename SrcTileData::DType *)__cce_get_tile_ptr(src);

    COPY_CC_TO_CUBF(dstAddr, srcData, nSize, SrcTileData::Rows, dstStride, SrcTileData::Rows, QuantPre, reluMode,
                    channelSplitEnable);
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    static_assert((DstTileData::Loc == TileType::Mat), "Destination TileType only support Mat.");
    static_assert((!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor),
                  "Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor).");
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType>();
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    TInsertAccToMat<DstTileData, SrcTileData, quantPre, ReluPreMode::NoRelu>(dst.data(), src.data(), src.GetValidRow(),
                                                                             src.GetValidCol(), indexRow, indexCol);
}

template <typename FpTileData>
__tf__ PTO_INTERNAL void SetFPCInsert(typename FpTileData::TileDType __in__ fp)
{
    __fbuf__ typename FpTileData::DType *dstAddrFp = (__fbuf__ typename FpTileData::DType *)__cce_get_tile_ptr(fp);
    uint64_t deqTensorAddr = ((uint64_t)dstAddrFp >> static_cast<uint64_t>(7)) << 8;
    set_fpc(deqTensorAddr);
}

// relu
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType>();
    static_assert((DstTileData::Loc == TileType::Mat), "Destination TileType only support Mat.");
    static_assert((!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor),
                  "Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor).");
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    TInsertAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), src.GetValidRow(),
                                                                  src.GetValidCol(), indexRow, indexCol);
}

// scalar quant
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow = 0,
                               uint16_t indexCol = 0)
{
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    static_assert((DstTileData::Loc == TileType::Mat), "Destination TileType only support Mat.");
    static_assert((!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor),
                  "Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor).");
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
    static_assert((DstTileData::Loc == TileType::Mat), "Destination TileType only support Mat.");
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    static_assert((!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor),
                  "Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor).");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    SetFPCInsert<FpTileData>(fp.data());
    TInsertAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), src.GetValidRow(),
                                                                  src.GetValidCol(), indexRow, indexCol);
}

template <typename T, typename DstTileData, typename SrcTileData>
AICORE inline void ComputeNZBlockParams(uint32_t validRow, uint32_t validCol, uint32_t dstRow, TInsertMode mode,
                                        uint16_t &burstNum, uint16_t &burstLen, uint16_t &srcGap, uint16_t &dstGap,
                                        uint32_t &dstOffset, uint32_t indexRow = 0, uint32_t indexCol = 0)
{
    constexpr uint32_t typeSize = sizeof(T);
    uint32_t c0Size = BLOCK_BYTE_SIZE / typeSize;
    constexpr uint32_t nzRow = FRACTAL_NZ_ROW;
    burstNum = CeilDivision(validCol, c0Size);
    uint32_t alignedRow = CeilDivision(validRow, nzRow) * nzRow;
    burstLen = (alignedRow * c0Size * sizeof(T)) / BLOCK_BYTE_SIZE;
    uint32_t colBlockOffset = (indexCol / c0Size) * dstRow * c0Size;
    uint32_t rowOffset = indexRow * c0Size + (indexCol % c0Size);
    dstOffset = colBlockOffset + rowOffset;
    switch (mode) {
        case TInsertMode::NZ:
            srcGap = 0;
            dstGap = static_cast<uint16_t>(dstRow - validRow);
            break;
        case TInsertMode::NZ_PLUS_1:
        case TInsertMode::SPLIT2_NZ_PLUS_1:
        case TInsertMode::SPLIT4_NZ_PLUS_1:
            srcGap = 1;
            dstGap = static_cast<uint16_t>(dstRow - validRow);
            break;
        default:
            srcGap = 1;
            dstGap = static_cast<uint16_t>(dstRow - validRow);
            break;
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TInsertImpl(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
                               TInsertMode mode, uint16_t validRow, uint16_t validCol, uint16_t dstRow,
                               uint32_t indexRow = 0, uint32_t indexCol = 0)
{
    __cbuf__ T *dstAddr = (__cbuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    uint16_t burstNum, burstLen, srcGap, dstGap;
    uint32_t dstOffset;
    ComputeNZBlockParams<T, DstTileData, SrcTileData>(validRow, validCol, dstRow, mode, burstNum, burstLen, srcGap,
                                                      dstGap, dstOffset, indexRow, indexCol);
    __cbuf__ T *dstAddr2 = dstAddr + dstOffset;
    copy_ubuf_to_cbuf(dstAddr2, srcAddr, 0, burstNum, burstLen, srcGap, dstGap);
}

template <uint32_t SplitCount, typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TInsertSplitImpl(typename DstTileData::TileDType __out__ dst,
                                    typename SrcTileData::TileDType __in__ src, TInsertMode mode, uint16_t validRow,
                                    uint16_t validCol, uint32_t indexRow = 0, uint32_t indexCol = 0)
{
    __cbuf__ T *dstAddr = (__cbuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src);

    constexpr uint32_t typeSize = sizeof(T);
    uint32_t c0Size = BLOCK_BYTE_SIZE / typeSize;
    constexpr uint32_t nzRow = FRACTAL_NZ_ROW;

    uint32_t alignedRow = CeilDivision(validRow, nzRow) * nzRow;
    uint16_t totalBurstNum = CeilDivision(validCol, c0Size);
    uint16_t burstLen = (alignedRow * c0Size * typeSize) / BLOCK_BYTE_SIZE;
    uint16_t partBurstNum = totalBurstNum / SplitCount;
    uint32_t srcBlockSize = (burstLen + 1) * BLOCK_BYTE_SIZE / typeSize;
    uint32_t dstBlockSize = DstTileData::Rows * c0Size;

    uint32_t colBlockOffset = (indexCol / c0Size) * DstTileData::Rows * c0Size;
    uint32_t rowOffset = indexRow * c0Size + (indexCol % c0Size);
    uint32_t dstOffset = colBlockOffset + rowOffset;

    __cbuf__ T *dstAddr0 = dstAddr + dstOffset;
    copy_ubuf_to_cbuf(dstAddr0, srcAddr, 0, partBurstNum, burstLen, 1, 0);

    if constexpr (SplitCount >= 2) {
        __ubuf__ T *src1 = srcAddr + partBurstNum * srcBlockSize;
        __cbuf__ T *dst1 = dstAddr0 + partBurstNum * dstBlockSize;
        copy_ubuf_to_cbuf(dst1, src1, 0, partBurstNum, burstLen, 1, 0);
    }

    if constexpr (SplitCount >= 4) {
        __ubuf__ T *src2 = srcAddr + 2 * partBurstNum * srcBlockSize;
        __cbuf__ T *dst2 = dstAddr0 + 2 * partBurstNum * dstBlockSize;
        copy_ubuf_to_cbuf(dst2, src2, 0, partBurstNum, burstLen, 1, 0);

        __ubuf__ T *src3 = srcAddr + 3 * partBurstNum * srcBlockSize;
        __cbuf__ T *dst3 = dstAddr0 + 3 * partBurstNum * dstBlockSize;
        copy_ubuf_to_cbuf(dst3, src3, 0, partBurstNum, burstLen, 1, 0);
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TInsertNDImpl(typename DstTileData::TileDType __out__ dst,
                                 typename SrcTileData::TileDType __in__ src, uint16_t validRow, uint16_t validCol,
                                 uint16_t dstCols, uint32_t indexRow = 0, uint32_t indexCol = 0)
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

template <TInsertMode mode = TInsertMode::NZ, typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TINSERT_IMPL(DstTileData &dst, SrcTileData &src, uint32_t indexRow = 0, uint32_t indexCol = 0)
{
    using T = typename SrcTileData::DType;
    static_assert(DstTileData::Loc == TileType::Mat, "TINSERT : Destination must be Mat tile (L1/cbuf)");
    static_assert(SrcTileData::Loc == TileType::Vec, "TINSERT : Source must be Vec tile (UB/ubuf)");
    static_assert(std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
                  "TINSERT : Source and destination data types must match");

    PTO_ASSERT(indexRow + SrcTileData::Rows <= DstTileData::Rows,
               "TINSERT : The sum of indexRow and srcRow should be less than dstRow!");
    PTO_ASSERT(indexCol + SrcTileData::Cols <= DstTileData::Cols,
               "TINSERT : The sum of indexCol and srcCol should be less than dstCol!");

    uint16_t validRow = static_cast<uint16_t>(src.GetValidRow());
    uint16_t validCol = static_cast<uint16_t>(src.GetValidCol());

    if constexpr (mode == TInsertMode::ND) {
        static_assert(SrcTileData::isRowMajor, "TINSERT ND : Source must be RowMajor (ND format)");
        static_assert((std::is_same<T, half>::value) || (std::is_same<T, bfloat16_t>::value) ||
                          (std::is_same<T, float>::value) || (std::is_same<T, int32_t>::value) ||
                          (std::is_same<T, float8_e4m3_t>::value) || (std::is_same<T, float8_e5m2_t>::value) ||
                          (std::is_same<T, hifloat8_t>::value) || (std::is_same<T, int8_t>::value) ||
                          (std::is_same<T, float8_e8m0_t>::value),
                      "TINSERT ND : Unsupported data type.");
        uint16_t dstCols = static_cast<uint16_t>(DstTileData::Cols);
        TInsertNDImpl<T, DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol, dstCols, indexRow,
                                                   indexCol);
    } else {
        static_assert(!SrcTileData::isRowMajor && (SrcTileData::SFractal == SLayout::RowMajor),
                      "TINSERT NZ : Source must be NZ format (column-major, RowMajor fractal)");
        static_assert((std::is_same<T, half>::value) || (std::is_same<T, bfloat16_t>::value) ||
                          (std::is_same<T, float>::value) || (std::is_same<T, int32_t>::value) ||
                          (std::is_same<T, float8_e4m3_t>::value) || (std::is_same<T, float8_e5m2_t>::value) ||
                          (std::is_same<T, hifloat8_t>::value) || (std::is_same<T, int8_t>::value) ||
                          (std::is_same<T, float8_e8m0_t>::value),
                      "TINSERT NZ : Unsupported data type.");

        uint16_t dstRow = static_cast<uint16_t>(dst.GetValidRow());

        if constexpr (mode == TInsertMode::SPLIT2_NZ_PLUS_1) {
            TInsertSplitImpl<2, T, DstTileData, SrcTileData>(dst.data(), src.data(), mode, validRow, validCol, indexRow,
                                                             indexCol);
        } else if constexpr (mode == TInsertMode::SPLIT4_NZ_PLUS_1) {
            TInsertSplitImpl<4, T, DstTileData, SrcTileData>(dst.data(), src.data(), mode, validRow, validCol, indexRow,
                                                             indexCol);
        } else {
            TInsertImpl<T, DstTileData, SrcTileData>(dst.data(), src.data(), mode, validRow, validCol, dstRow, indexRow,
                                                     indexCol);
        }
    }
}

} // namespace pto
#ifdef COPY_CC_TO_CUBF
#undef COPY_CC_TO_CUBF
#endif
#endif // TInsert_HPP