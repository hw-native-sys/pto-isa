/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TEXTRACT_COMMON_MEMORY
#define TEXTRACT_COMMON_MEMORY
#include <pto/common/utils.hpp>

namespace pto {

template <typename DstType, typename SrcType, int32_t srcRow, int32_t srcCol, int32_t dstRow, int32_t dstCol>
PTO_INTERNAL void TExtractToANonTranspose(__ca__ DstType *dstAddr, __cbuf__ SrcType *srcAddr, uint16_t indexRow,
                                          uint16_t indexCol)
{
    constexpr int config = srcRow | (1u << 16);
    set_fmatrix(config);
    img2colv2_cbuf_to_ca(dstAddr, srcAddr, dstCol, dstRow, indexCol, indexRow, 1, 1, 1, 1, 1, 1, false, false, false,
                         false, srcCol);
}

template <typename DstType, typename SrcType, int32_t srcRow, int32_t srcCol, int32_t dstRow, int32_t dstCol>
PTO_INTERNAL void TExtractToATranspose(__ca__ DstType *dstAddr, __cbuf__ SrcType *srcAddr, uint16_t indexRow,
                                       uint16_t indexCol)
{
    if constexpr (sizeof(SrcType) == 1) {
        constexpr uint16_t fractNum = 2;
        constexpr uint16_t srcColNum = srcCol >> (SHIFT_BLOCK_LEN + fractNum - 1);
        constexpr uint16_t dstColNum = dstCol * sizeof(SrcType) >> SHIFT_BLOCK_BYTE;
        constexpr uint16_t dstRowNum = dstRow >> (SHIFT_BLOCK_LEN + fractNum - 1);
        uint16_t dstGap = 0;
        uint16_t dstFracGap = 0;
        uint16_t startIdx0 = (indexCol >> (SHIFT_BLOCK_LEN + fractNum - 1)) +
                             (indexRow * srcColNum * sizeof(SrcType) >> SHIFT_BLOCK_BYTE);
        if constexpr (dstRowNum >= dstColNum) {
            dstGap = fractNum * dstColNum - 1;
            dstFracGap = dstColNum - 1;
            for (uint16_t i = 0; i < dstColNum; i++) {
                load_cbuf_to_ca_transpose(dstAddr, srcAddr, startIdx0 + i, dstRowNum, srcColNum, dstGap, false,
                                          dstFracGap);
                dstAddr += CUBE_BLOCK_SIZE;
            }
        } else {
            dstFracGap = dstColNum - 1;
            for (uint16_t i = 0; i < dstRowNum; i++) {
                load_cbuf_to_ca_transpose(dstAddr, srcAddr, startIdx0 + i * srcColNum, dstColNum, 1, 0, false,
                                          dstFracGap);
                dstAddr += dstColNum * CUBE_BLOCK_SIZE * fractNum;
            }
        }
    } else {
        constexpr int config = srcCol | (1u << 16);
        set_fmatrix(config);
        img2colv2_cbuf_to_ca(dstAddr, srcAddr, dstRow, dstCol, indexRow, indexCol, 1, 1, 1, 1, 1, 1, false, false, true,
                             false, srcRow);
    }
}
template <typename DstTileData, typename SrcTileData, bool Transpose>
__tf__ AICORE void TExtractToA(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
                               uint16_t indexRow, uint16_t indexCol)
{
    using SrcType = std::conditional_t<(sizeof(typename SrcTileData::DType) == 2), half, typename SrcTileData::DType>;
    using DstType = std::conditional_t<(sizeof(typename DstTileData::DType) == 2), half, typename DstTileData::DType>;
    __cbuf__ SrcType *srcAddr = (__cbuf__ SrcType *)__cce_get_tile_ptr(src);
    __ca__ DstType *dstAddr = (__ca__ DstType *)__cce_get_tile_ptr(dst);

    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr int32_t dstRow = DstTileData::Rows;
    constexpr int32_t dstCol = DstTileData::Cols;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(SrcType);
    constexpr int32_t fractalSize = (sizeof(SrcType) == 1) ? 32 : 16;

    if constexpr (!Transpose) {
        static_assert((srcRow % 16) == 0, "srcRow must be aligned to 16");
        static_assert((srcCol % c0Size) == 0, "srcCol must be aligned to C0Size");
        static_assert((dstRow % 16) == 0, "dstRow must be aligned to 16");
        static_assert((dstCol % c0Size) == 0, "dstCol must be aligned to C0Size");
        PTO_ASSERT((indexRow % 16) == 0, "indexRow must be aligned to 16");
        PTO_ASSERT((indexCol % c0Size) == 0, "indexCol must be aligned to C0Size");
        TExtractToANonTranspose<SrcType, DstType, srcRow, srcCol, dstRow, dstCol>(dstAddr, srcAddr, indexRow, indexCol);
    } else {
        static_assert((srcRow % fractalSize) == 0, "srcRow must be aligned");
        static_assert((srcCol % fractalSize) == 0, "srcCol must be aligned");
        static_assert((dstRow % fractalSize) == 0, "dstRow must be aligned");
        static_assert((dstCol % fractalSize) == 0, "dstCol must be aligned");
        PTO_ASSERT((indexRow % fractalSize) == 0, "indexRow must be aligned");
        PTO_ASSERT((indexCol % fractalSize) == 0, "indexCol must be aligned");
        TExtractToATranspose<SrcType, DstType, srcRow, srcCol, dstRow, dstCol>(dstAddr, srcAddr, indexRow, indexCol);
    }
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractToAVector(typename DstTileData::TileDType __out__ dst,
                                     typename SrcTileData::TileDType __in__ src, uint16_t indexRow, uint16_t indexCol,
                                     uint16_t dstValidCol)
{
    using DataType = typename SrcTileData::DType;
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __ca__ DataType *dstAddr = (__ca__ DataType *)__cce_get_tile_ptr(dst);

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

template <typename DstType, typename SrcType, int32_t srcRow, int32_t srcCol, int32_t dstRow, int32_t dstCol>
PTO_INTERNAL void TExtractToBNonTranspose(__cb__ DstType *dstAddr, __cbuf__ SrcType *srcAddr, uint16_t indexRow,
                                          uint16_t indexCol)
{
    uint16_t dstGap = 0;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(SrcType);
    constexpr uint16_t dstRowNum = (dstRow * sizeof(DstType)) >> SHIFT_BLOCK_BYTE;
    constexpr uint16_t dstColNum = dstCol >> SHIFT_BLOCK_LEN;
    constexpr uint16_t srcColNum = srcCol >> SHIFT_BLOCK_LEN;
    constexpr uint16_t srcRowNum = (srcRow * sizeof(SrcType)) >> SHIFT_BLOCK_BYTE;
    uint16_t blockNum = CUBE_BLOCK_SIZE / sizeof(SrcType);
    uint16_t startIdx0 = (indexRow * sizeof(SrcType) * srcColNum >> SHIFT_BLOCK_BYTE) + (indexCol >> SHIFT_BLOCK_LEN);
    if constexpr (dstRowNum >= dstColNum) {
        dstGap = dstColNum - 1;
        for (uint16_t i = 0; i < dstColNum; i++) {
            pto_load_cbuf_to_cb(dstAddr, srcAddr, startIdx0 + i, dstRowNum, srcColNum, dstGap);
            dstAddr += blockNum;
        }
    } else {
        for (uint16_t i = 0; i < dstRowNum; i++) {
            pto_load_cbuf_to_cb(dstAddr, srcAddr, startIdx0 + i * srcColNum, dstColNum, 1, 0);
            dstAddr += dstCol * c0Size;
        }
    }
}

template <typename DstType, typename SrcType, int32_t srcRow, int32_t srcCol, int32_t dstRow, int32_t dstCol>
PTO_INTERNAL void TExtractToBTranspose(__cb__ DstType *dstAddr, __cbuf__ SrcType *srcAddr, uint16_t indexRow,
                                       uint16_t indexCol)
{
    if constexpr (sizeof(SrcType) == 1) {
        constexpr uint16_t fractNum = 2;
        constexpr uint16_t srcColNum = srcCol * sizeof(SrcType) >> SHIFT_BLOCK_BYTE;
        constexpr uint16_t srcRowNum = srcRow >> (SHIFT_BLOCK_LEN + fractNum - 1);
        constexpr uint16_t dstColNum = dstCol >> (SHIFT_BLOCK_LEN + fractNum - 1);
        constexpr uint16_t dstRowNum = dstRow * sizeof(DstType) >> SHIFT_BLOCK_BYTE;
        uint16_t dstGap = 0;
        uint16_t startIdx0 = (indexRow >> (SHIFT_BLOCK_LEN + fractNum - 1)) +
                             (indexCol * sizeof(SrcType) * srcRowNum >> SHIFT_BLOCK_BYTE);
        if constexpr (dstRowNum >= dstColNum) {
            dstGap = fractNum * dstColNum - 1;
            for (uint16_t i = 0; i < dstColNum; i++) {
                load_cbuf_to_cb_transpose(dstAddr, srcAddr, startIdx0 + i * srcRowNum, dstRowNum, 1, dstGap, false, 0);
                dstAddr += fractNum * CUBE_BLOCK_SIZE;
            }
        } else {
            dstGap = fractNum - 1;
            for (uint16_t i = 0; i < dstRowNum; i++) {
                load_cbuf_to_cb_transpose(dstAddr, srcAddr, startIdx0 + i, dstColNum, srcRowNum, dstGap, false, 0);
                dstAddr += dstColNum * fractNum * CUBE_BLOCK_SIZE;
            }
        }
    } else {
        constexpr int config = srcRow | (1u << 16);
        set_fmatrix_b(config);
        img2colv2_cbuf_to_cb(dstAddr, srcAddr, dstCol, dstRow, indexCol, indexRow, 1, 1, 1, 1, 1, 1, false, false,
                             false, true, srcCol);
    }
}

template <typename DstTileData, typename SrcTileData, bool Transpose>
__tf__ AICORE void TExtractToB(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
                               uint16_t indexRow, uint16_t indexCol)
{
    using SrcType = std::conditional_t<(sizeof(typename SrcTileData::DType) == 2), half, typename SrcTileData::DType>;
    using DstType = std::conditional_t<(sizeof(typename DstTileData::DType) == 2), half, typename DstTileData::DType>;
    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr int32_t dstRow = DstTileData::Rows;
    constexpr int32_t dstCol = DstTileData::Cols;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(SrcType);
    constexpr int32_t fractalSize = (sizeof(SrcType) == 1) ? 32 : 16;
    __cbuf__ SrcType *srcAddr = (__cbuf__ SrcType *)__cce_get_tile_ptr(src);
    __cb__ DstType *dstAddr = (__cb__ DstType *)__cce_get_tile_ptr(dst);
    if constexpr (!Transpose) {
        static_assert((srcRow % c0Size) == 0, "srcRow must be aligned to C0Size");
        static_assert((srcCol % 16) == 0, "srcCol must be aligned to 16");
        static_assert((dstRow % c0Size) == 0, "dstRow must be aligned to C0Size");
        static_assert((dstCol % 16) == 0, "dstCol must be aligned to 16");
        PTO_ASSERT((indexRow % c0Size) == 0, "indexRow must be aligned to c0Size");
        PTO_ASSERT((indexCol % 16) == 0, "indexCol must be aligned to 16");
        TExtractToBNonTranspose<SrcType, DstType, srcRow, srcCol, dstRow, dstCol>(dstAddr, srcAddr, indexRow, indexCol);
    } else {
        static_assert((srcRow % fractalSize) == 0, "srcRow must be aligned");
        static_assert((srcCol % fractalSize) == 0, "srcCol must be aligned");
        static_assert((dstRow % fractalSize) == 0, "dstRow must be aligned");
        static_assert((dstCol % fractalSize) == 0, "dstCol must be aligned");
        PTO_ASSERT((indexRow % fractalSize) == 0, "indexRow must be aligned");
        PTO_ASSERT((indexCol % fractalSize) == 0, "indexCol must be aligned");
        TExtractToBTranspose<SrcType, DstType, srcRow, srcCol, dstRow, dstCol>(dstAddr, srcAddr, indexRow, indexCol);
    }
}

template <typename DstType, typename SrcType, int32_t srcRow, int32_t srcCol>
PTO_INTERNAL void TExtractToANonTransposeCompact(__ca__ DstType *dstAddr, __cbuf__ SrcType *srcAddr, uint16_t indexRow,
                                                 uint16_t indexCol, uint16_t dstValidRowAlign,
                                                 uint16_t dstValidColAlign)
{
    constexpr int config = srcRow | (1u << 16);
    set_fmatrix(config);
    img2colv2_cbuf_to_ca(dstAddr, srcAddr, dstValidColAlign, dstValidRowAlign, indexCol, indexRow, 1, 1, 1, 1, 1, 1,
                         false, false, false, false, srcCol);
}

template <typename DstType, typename SrcType, int32_t srcRow, int32_t srcCol>
PTO_INTERNAL void TExtractToATransposeCompact(__ca__ DstType *dstAddr, __cbuf__ SrcType *srcAddr, uint16_t indexRow,
                                              uint16_t indexCol, uint16_t dstValidRowAlign, uint16_t dstValidColAlign)
{
    if constexpr (sizeof(SrcType) == 1) {
        constexpr uint16_t fractNum = 2;
        constexpr uint16_t srcColNum = srcCol >> (SHIFT_BLOCK_LEN + fractNum - 1);
        uint16_t dstColNum = dstValidColAlign >> SHIFT_BLOCK_BYTE;
        uint16_t dstRowNum = dstValidRowAlign >> (SHIFT_BLOCK_LEN + fractNum - 1);
        uint16_t dstGap = 0;
        uint16_t dstFracGap = 0;
        uint16_t startIdx0 = (indexCol >> (SHIFT_BLOCK_LEN + fractNum - 1)) +
                             (indexRow * srcColNum * sizeof(SrcType) >> SHIFT_BLOCK_BYTE);
        if (dstRowNum >= dstColNum) {
            dstGap = fractNum * dstColNum - 1;
            dstFracGap = dstColNum - 1;
            for (uint16_t i = 0; i < dstColNum; i++) {
                load_cbuf_to_ca_transpose(dstAddr, srcAddr, startIdx0 + i, dstRowNum, srcColNum, dstGap, false,
                                          dstFracGap);
                dstAddr = dstAddr + CUBE_BLOCK_SIZE;
            }
        } else {
            dstFracGap = dstColNum - 1;
            for (uint16_t i = 0; i < dstRowNum; i++) {
                load_cbuf_to_ca_transpose(dstAddr, srcAddr, startIdx0 + i * srcColNum, dstColNum, 1, 0, false,
                                          dstFracGap);
                dstAddr = dstAddr + dstColNum * CUBE_BLOCK_SIZE * fractNum;
            }
        }
    } else {
        constexpr int config = srcCol | (1u << 16);
        set_fmatrix(config);
        img2colv2_cbuf_to_ca(dstAddr, srcAddr, dstValidRowAlign, dstValidColAlign, indexRow, indexCol, 1, 1, 1, 1, 1, 1,
                             false, false, true, false, srcRow);
    }
}

template <typename DstTileData, typename SrcTileData, bool Transpose>
__tf__ AICORE void TExtractToACompact(typename DstTileData::TileDType __out__ dst,
                                      typename SrcTileData::TileDType __in__ src, uint16_t indexRow, uint16_t indexCol,
                                      uint16_t dstValidRow, uint16_t dstValidCol, bool isKAligned)
{
    using SrcType = std::conditional_t<(sizeof(typename SrcTileData::DType) == 2), half, typename SrcTileData::DType>;
    using DstType = std::conditional_t<(sizeof(typename DstTileData::DType) == 2), half, typename DstTileData::DType>;
    __cbuf__ SrcType *srcAddr = (__cbuf__ SrcType *)__cce_get_tile_ptr(src);
    __ca__ DstType *dstAddr = (__ca__ DstType *)__cce_get_tile_ptr(dst);

    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(SrcType);
    constexpr int32_t fractalSize = (sizeof(SrcType) == 1) ? 32 : 16;
    if constexpr (!Transpose) {
        static_assert((SrcTileData::Rows % 16) == 0, "srcRow must be aligned to 16");
        static_assert((SrcTileData::Cols % c0Size) == 0, "srcCol must be aligned to C0Size");
        PTO_ASSERT((indexRow % 16) == 0, "indexRow must be aligned to 16");
        PTO_ASSERT((indexCol % c0Size) == 0, "indexCol must be aligned to C0Size");
        uint16_t dstValidRowAlign = CeilDivision(dstValidRow, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
        uint16_t dstValidColAlign = CeilDivision(dstValidCol, c0Size) * c0Size;
        if (isKAligned && (std::is_same<DstType, float>::value)) {
            dstValidColAlign = CeilDivision(dstValidCol, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
        }
        TExtractToANonTransposeCompact<SrcType, DstType, SrcTileData::Rows, SrcTileData::Cols>(
            dstAddr, srcAddr, indexRow, indexCol, dstValidRowAlign, dstValidColAlign);
    } else {
        static_assert((SrcTileData::Rows % fractalSize) == 0, "srcRow must be aligned");
        static_assert((SrcTileData::Cols % fractalSize) == 0, "srcCol must be aligned");
        PTO_ASSERT((indexRow % fractalSize) == 0, "indexRow must be aligned");
        PTO_ASSERT((indexCol % fractalSize) == 0, "indexCol must be aligned");
        uint16_t dstValidRowAlign = CeilDivision(dstValidRow, fractalSize) * fractalSize;
        uint16_t dstValidColAlign = CeilDivision(dstValidCol, fractalSize) * fractalSize;
        TExtractToATransposeCompact<SrcType, DstType, SrcTileData::Rows, SrcTileData::Cols>(
            dstAddr, srcAddr, indexRow, indexCol, dstValidRowAlign, dstValidColAlign);
    }
}

template <typename DstType, typename SrcType, int32_t srcRow, int32_t srcCol>
PTO_INTERNAL void TExtractToBNonTransposeCompact(__cb__ DstType *dstAddr, __cbuf__ SrcType *srcAddr, uint16_t indexRow,
                                                 uint16_t indexCol, uint16_t dstValidRowAlign,
                                                 uint16_t dstValidColAlign)
{
    uint16_t dstGap = 0;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(SrcType);
    uint16_t dstRowNum = (dstValidRowAlign * sizeof(DstType)) >> SHIFT_BLOCK_BYTE;
    uint16_t dstColNum = dstValidColAlign >> SHIFT_BLOCK_LEN;
    constexpr uint16_t srcColNum = srcCol >> SHIFT_BLOCK_LEN;
    constexpr uint16_t srcRowNum = (srcRow * sizeof(SrcType)) >> SHIFT_BLOCK_BYTE;
    uint16_t blockNum = CUBE_BLOCK_SIZE / sizeof(SrcType);
    uint16_t startIdx0 = (indexRow * sizeof(SrcType) * srcColNum >> SHIFT_BLOCK_BYTE) + (indexCol >> SHIFT_BLOCK_LEN);
    if (dstRowNum >= dstColNum) {
        dstGap = dstColNum - 1;
        for (uint16_t i = 0; i < dstColNum; i++) {
            pto_load_cbuf_to_cb(dstAddr, srcAddr, startIdx0 + i, dstRowNum, srcColNum, dstGap);
            dstAddr += blockNum;
        }
    } else {
        for (uint16_t i = 0; i < dstRowNum; i++) {
            pto_load_cbuf_to_cb(dstAddr, srcAddr, startIdx0 + i * srcColNum, dstColNum, 1, 0);
            dstAddr += dstValidColAlign * c0Size;
        }
    }
}

template <typename DstType, typename SrcType, int32_t srcRow, int32_t srcCol>
PTO_INTERNAL void TExtractToBTransposeCompact(__cb__ DstType *dstAddr, __cbuf__ SrcType *srcAddr, uint16_t indexRow,
                                              uint16_t indexCol, uint16_t dstValidRowAlign, uint16_t dstValidColAlign,
                                              uint16_t dstValidCol)
{
    if constexpr (sizeof(SrcType) == 1) {
        constexpr uint16_t fractNum = 2;
        constexpr uint16_t srcColNum = srcCol * sizeof(SrcType) >> SHIFT_BLOCK_BYTE;
        constexpr uint16_t srcRowNum = srcRow >> (SHIFT_BLOCK_LEN + fractNum - 1);
        uint16_t dstColNum = dstValidColAlign >> (SHIFT_BLOCK_LEN + fractNum - 1);
        uint16_t dstRowNum = dstValidRowAlign * sizeof(DstType) >> SHIFT_BLOCK_BYTE;
        uint16_t dstGap = fractNum - 1;
        uint16_t startIdx0 = (indexRow >> (SHIFT_BLOCK_LEN + fractNum - 1)) +
                             (indexCol * sizeof(SrcType) * srcRowNum >> SHIFT_BLOCK_BYTE);
        uint16_t dstAddrStride = CeilDivision(dstValidCol, FRACTAL_NZ_ROW) * CUBE_BLOCK_SIZE;
        for (uint16_t i = 0; i < dstRowNum; i++) {
            load_cbuf_to_cb_transpose(dstAddr, srcAddr, startIdx0 + i, dstColNum, srcRowNum, dstGap, false, 0);
            dstAddr += dstAddrStride;
        }
    } else {
        constexpr int config = srcRow | (1u << 16);
        set_fmatrix_b(config);
        img2colv2_cbuf_to_cb(dstAddr, srcAddr, dstValidColAlign, dstValidRowAlign, indexCol, indexRow, 1, 1, 1, 1, 1, 1,
                             false, false, false, true, srcCol);
    }
}

template <typename DstTileData, typename SrcTileData, bool Transpose>
__tf__ AICORE void TExtractToBCompact(typename DstTileData::TileDType __out__ dst,
                                      typename SrcTileData::TileDType __in__ src, uint16_t indexRow, uint16_t indexCol,
                                      uint16_t dstValidRow, uint16_t dstValidCol)
{
    using SrcType = std::conditional_t<(sizeof(typename SrcTileData::DType) == 2), half, typename SrcTileData::DType>;
    using DstType = std::conditional_t<(sizeof(typename DstTileData::DType) == 2), half, typename DstTileData::DType>;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(SrcType);
    constexpr int32_t fractalSize = (sizeof(SrcType) == 1) ? 32 : 16;
    __cbuf__ SrcType *srcAddr = (__cbuf__ SrcType *)__cce_get_tile_ptr(src);
    __cb__ DstType *dstAddr = (__cb__ DstType *)__cce_get_tile_ptr(dst);
    static_assert((DstTileData::Rows % c0Size) == 0, "dstRow must be aligned to C0Size");
    static_assert((DstTileData::Cols % 16) == 0, "dstCol must be aligned to 16");
    if constexpr (!Transpose) {
        static_assert((SrcTileData::Rows % c0Size) == 0, "srcRow must be aligned to C0Size");
        static_assert((SrcTileData::Cols % 16) == 0, "srcCol must be aligned to 16");
        PTO_ASSERT((indexRow % c0Size) == 0, "indexRow must be aligned to c0Size");
        PTO_ASSERT((indexCol % 16) == 0, "indexCol must be aligned to 16");
        uint16_t dstValidRowAlign = CeilDivision(dstValidRow, c0Size) * c0Size;
        uint16_t dstValidColAlign = CeilDivision(dstValidCol, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
        TExtractToBNonTransposeCompact<SrcType, DstType, SrcTileData::Rows, SrcTileData::Cols>(
            dstAddr, srcAddr, indexRow, indexCol, dstValidRowAlign, dstValidColAlign);
    } else {
        static_assert((SrcTileData::Rows % fractalSize) == 0, "srcRow must be aligned");
        static_assert((SrcTileData::Cols % fractalSize) == 0, "srcCol must be aligned");
        PTO_ASSERT((indexRow % fractalSize) == 0, "indexRow must be aligned");
        PTO_ASSERT((indexCol % fractalSize) == 0, "indexCol must be aligned");
        uint16_t dstValidRowAlign = CeilDivision(dstValidRow, fractalSize) * fractalSize;
        uint16_t dstValidColAlign = CeilDivision(dstValidCol, fractalSize) * fractalSize;
        TExtractToBTransposeCompact<SrcType, DstType, SrcTileData::Rows, SrcTileData::Cols>(
            dstAddr, srcAddr, indexRow, indexCol, dstValidRowAlign, dstValidColAlign, dstValidCol);
    }
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractToBConv(typename DstTileData::TileDType __out__ dst,
                                   typename SrcTileData::TileDType __in__ src, uint16_t srcCol, uint16_t dstValidRow,
                                   uint16_t dstValidCol, uint16_t indexRow, uint16_t indexCol)
{
    using SrcType = typename SrcTileData::DType;
    using DstType = typename DstTileData::DType;
    __cbuf__ SrcType *srcAddr = (__cbuf__ SrcType *)__cce_get_tile_ptr(src);
    __cb__ DstType *dstAddr = (__cb__ DstType *)__cce_get_tile_ptr(dst);

    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(SrcType);
    uint16_t dstValidRowAlign = CeilDivision(dstValidRow, c0Size) * c0Size;
    uint16_t dstValidColAlign = CeilDivision(dstValidCol, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    uint16_t dstRowNum = (dstValidRowAlign * sizeof(DstType)) >> SHIFT_BLOCK_BYTE;
    uint16_t dstColNum = dstValidColAlign >> SHIFT_BLOCK_LEN;
    uint16_t srcColNum = srcCol >> SHIFT_BLOCK_LEN;
    uint16_t blockNum = CUBE_BLOCK_SIZE / sizeof(SrcType);
    uint16_t startIdx0 = (indexRow * sizeof(SrcType) * srcColNum >> SHIFT_BLOCK_BYTE) + (indexCol >> SHIFT_BLOCK_LEN);
    for (uint16_t i = 0; i < dstRowNum; i++) {
        pto_load_cbuf_to_cb(dstAddr, srcAddr, startIdx0 + i * srcColNum, dstColNum, 1, 0);
        dstAddr += dstValidColAlign * c0Size;
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_CONVTILE_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol)
{
    static_assert(SrcTileData::Loc == pto::TileType::Mat, "Fix: Src TileType must be Mat!");
    static_assert(DstTileData::Loc == pto::TileType::Right, "Fix: Dst TileType must be Right!");
    static_assert(sizeof(typename DstTileData::DType) == sizeof(typename SrcTileData::DType),
                  "Fix: Source dtype must be same with dst dtype!");

    static_assert((SrcTileData::layout == Layout::FRACTAL_Z) || (SrcTileData::layout == Layout::FRACTAL_Z_3D),
                  "TExtract: Source layout only support FRACTAL_Z or FRACTAL_Z_3D.");
    static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
                  "TExtract: Destination layout only support SLayout is ColMajor ang BLayout is RowMajor.");
    static_assert(std::is_same<typename DstTileData::DType, int8_t>::value ||
                      std::is_same<typename DstTileData::DType, half>::value ||
                      std::is_same<typename DstTileData::DType, bfloat16_t>::value ||
                      std::is_same<typename DstTileData::DType, float>::value,
                  "TExtract: Invalid data type.");

    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename SrcTileData::DType);
    if constexpr (SrcTileData::totalDimCount == 4) {
        int srcCol = src.GetShape(1) * src.GetShape(2);
        TExtractToBConv<DstTileData, SrcTileData>(dst.data(), src.data(), srcCol, dst.GetValidRow(), dst.GetValidCol(),
                                                  indexRow, indexCol);
    } else {
        TExtractToBConv<DstTileData, SrcTileData>(dst.data(), src.data(), src.GetShape(3), dst.GetValidRow(),
                                                  dst.GetValidCol(), indexRow, indexCol);
    }
}

template <typename DstTileData, typename SrcTileData>
AICORE void TExtractToLeft(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol)
{
    static_assert((SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor) ||
                      (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor) ||
                      (SrcTileData::Rows == 1 && SrcTileData::isRowMajor),
                  "TExtract: SrcTile Invalid Fractal");
    static_assert(DstTileData::SFractal == SLayout::RowMajor && DstTileData::isRowMajor,
                  "TExtract: LeftTile Invalid Fractal.");
    if constexpr (SrcTileData::Rows == 1 && SrcTileData::isRowMajor) {
        TExtractToAVector<DstTileData, SrcTileData>(dst.data(), src.data(), indexRow, indexCol, dst.GetValidCol());
    } else if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
        if constexpr (DstTileData::Compact == CompactMode::Normal) {
            TExtractToACompact<DstTileData, SrcTileData, false>(
                dst.data(), src.data(), indexRow, indexCol, dst.GetValidRow(), dst.GetValidCol(), dst.GetKAligned());
        } else {
            TExtractToA<DstTileData, SrcTileData, false>(dst.data(), src.data(), indexRow, indexCol);
        }
    } else {
        if constexpr (DstTileData::Compact == CompactMode::Normal) {
            TExtractToACompact<DstTileData, SrcTileData, true>(dst.data(), src.data(), indexRow, indexCol,
                                                               dst.GetValidRow(), dst.GetValidCol(), dst.GetKAligned());
        } else {
            TExtractToA<DstTileData, SrcTileData, true>(dst.data(), src.data(), indexRow, indexCol);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
AICORE void TExtractToRight(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol)
{
    static_assert((SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor) ||
                      (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor),
                  "TExtract: SrcTile Invalid Fractal");
    static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
                  "TExtract: RightTile Invalid Fractal.");
    if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
        if constexpr (DstTileData::Compact == CompactMode::Normal) {
            TExtractToBCompact<DstTileData, SrcTileData, false>(dst.data(), src.data(), indexRow, indexCol,
                                                                dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToB<DstTileData, SrcTileData, false>(dst.data(), src.data(), indexRow, indexCol);
        }
    } else {
        if constexpr (DstTileData::Compact == CompactMode::Normal) {
            TExtractToBCompact<DstTileData, SrcTileData, true>(dst.data(), src.data(), indexRow, indexCol,
                                                               dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToB<DstTileData, SrcTileData, true>(dst.data(), src.data(), indexRow, indexCol);
        }
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractVecToVecNDScalar(typename DstTileData::TileDType __out__ dst,
                                            typename SrcTileData::TileDType __in__ src, uint32_t indexRow,
                                            uint32_t indexCol)
{
    __ubuf__ T *dstAddr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    dstAddr[0] = srcAddr[indexRow * srcRowStride + indexCol];
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractVecToVecNZScalar(typename DstTileData::TileDType __out__ dst,
                                            typename SrcTileData::TileDType __in__ src, uint32_t indexRow,
                                            uint32_t indexCol)
{
    __ubuf__ T *dstAddr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / sizeof(T);
    constexpr uint32_t srcRows = SrcTileData::Rows;
    uint32_t srcOffset = (indexCol / c0Size) * srcRows * c0Size + indexRow * c0Size + (indexCol % c0Size);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    dstAddr[0] = srcAddr[srcOffset];
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void CheckTExtractVecToVecCommon()
{
    using DstT = typename DstTileData::DType;
    using SrcT = typename SrcTileData::DType;
    static_assert(std::is_same<DstT, SrcT>::value, "TEXTRACT Vec->Vec : Source and destination data types must match.");
    static_assert(std::is_same<DstT, int8_t>::value || std::is_same<DstT, uint8_t>::value ||
                      std::is_same<DstT, int16_t>::value || std::is_same<DstT, uint16_t>::value ||
                      std::is_same<DstT, half>::value || std::is_same<DstT, bfloat16_t>::value ||
                      std::is_same<DstT, float>::value || std::is_same<DstT, int32_t>::value ||
                      std::is_same<DstT, uint32_t>::value,
                  "TEXTRACT Vec->Vec : Unsupported data type.");
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void CheckTExtractVecToVecND()
{
    using T = typename DstTileData::DType;
    static_assert(DstTileData::Rows <= SrcTileData::Rows,
                  "TEXTRACT ND Vec->Vec : Destination Rows must not exceed source Rows.");
    static_assert(DstTileData::Cols <= SrcTileData::Cols,
                  "TEXTRACT ND Vec->Vec : Destination Cols must not exceed source Cols.");
    static_assert(SrcTileData::RowStride * sizeof(T) % BLOCK_BYTE_SIZE == 0,
                  "TEXTRACT ND Vec->Vec : SrcTile RowStride bytes must be 32-byte aligned.");
    static_assert(DstTileData::RowStride * sizeof(T) % BLOCK_BYTE_SIZE == 0,
                  "TEXTRACT ND Vec->Vec : DstTile RowStride bytes must be 32-byte aligned.");
    if constexpr (!(DstTileData::ValidRow == 1 && DstTileData::ValidCol == 1)) {
        static_assert((DstTileData::ValidCol * sizeof(T)) % sizeof(uint16_t) == 0,
                      "TEXTRACT ND Vec->Vec : DstTile ValidCol bytes must be at least 2-byte aligned.");
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void CheckTExtractVecToVecNZ()
{
    using T = typename DstTileData::DType;
    constexpr uint32_t kC0Size = BLOCK_BYTE_SIZE / sizeof(T);
    static_assert(DstTileData::Rows <= SrcTileData::Rows,
                  "TEXTRACT NZ Vec->Vec : Destination Rows must not exceed source Rows.");
    static_assert(DstTileData::Cols <= SrcTileData::Cols,
                  "TEXTRACT NZ Vec->Vec : Destination Cols must not exceed source Cols.");
    static_assert(SrcTileData::Rows % FRACTAL_NZ_ROW == 0, "TEXTRACT NZ Vec->Vec : SrcTile Rows must be 16-aligned.");
    static_assert(DstTileData::Rows % FRACTAL_NZ_ROW == 0, "TEXTRACT NZ Vec->Vec : DstTile Rows must be 16-aligned.");
    static_assert(SrcTileData::Cols % kC0Size == 0, "TEXTRACT NZ Vec->Vec : SrcTile Cols must be c0Size-aligned.");
    static_assert(DstTileData::Cols % kC0Size == 0, "TEXTRACT NZ Vec->Vec : DstTile Cols must be c0Size-aligned.");
    if constexpr (!(DstTileData::ValidRow == 1 && DstTileData::ValidCol == 1)) {
        static_assert((DstTileData::ValidCol * sizeof(T)) % sizeof(uint16_t) == 0,
                      "TEXTRACT NZ Vec->Vec : DstTile ValidCol bytes must be at least 2-byte aligned.");
    }
}

template <typename FpTileData>
__tf__ PTO_INTERNAL void SetFPC(typename FpTileData::TileDType __in__ fp, uint16_t indexCol)
{
    using FpType = typename FpTileData::DType;
    __fbuf__ FpType *dstAddrFp = (__fbuf__ FpType *)__cce_get_tile_ptr(fp) + indexCol;
    uint64_t deqTensorAddr = ((uint64_t)dstAddrFp >> static_cast<uint64_t>(7)) << 8;
    set_fpc(deqTensorAddr);
}

} // namespace pto
#endif
