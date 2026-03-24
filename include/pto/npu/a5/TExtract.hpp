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
#include "utils.hpp"

namespace pto {

constexpr const int KHALF = 2;             // for b4 data
constexpr const int M_STEP_MIN_VAL_B8 = 2; // m_step per loop for fp8
constexpr const int SHIFT_M_STEP_B8 = 1;   // 2^1 = 2
constexpr const int M_STEP_MIN_VAL_B4 = 4; // m_step per loop for fp4
constexpr const int SHIFT_M_STEP_B4 = 2;   // 2^2 = 4

constexpr const int SHIFT_MX_COL = 1; // 2^1 = 2
constexpr const int SHIFT_MX_ROW = 4; // 2^4 = 16
constexpr const int CO_SIZE_SCALE = 2;
constexpr const int SCALE_CUBE_BLOCK_SIZE = 32;

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractToAmx(typename DstTileData::TileDType __out__ dst,
                                 typename SrcTileData::TileDType __in__ src, uint16_t indexRow, uint16_t indexCol,
                                 uint16_t validRow, uint16_t validCol)
{
    static_assert((SrcTileData::SFractal == SLayout::RowMajor && SrcTileData::isRowMajor),
                  "TMov_mx: SrcTile Invalid Fractal.");
    static_assert((DstTileData::SFractal == SLayout::RowMajor && DstTileData::isRowMajor),
                  "TMov_mx: DstTile Invalid Fractal.");

    using DataType = typename DstTileData::DType;
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    uint64_t dstAddr = (uint64_t)__cce_get_tile_ptr(dst);
    uint16_t rowStartPosition = indexRow >> SHIFT_MX_ROW;
    uint16_t colStartPosition = (indexCol * sizeof(DataType)) >> SHIFT_MX_COL;

    if constexpr (DstTileData::Rows == 1) {
        uint8_t shiftCol = CeilDivision(validCol * sizeof(DataType), SCALE_CUBE_BLOCK_SIZE) * CO_SIZE_SCALE;
        uint8_t colStep = (shiftCol * sizeof(DataType)) >> SHIFT_MX_COL;
        uint16_t srcStride = shiftCol >> SHIFT_MX_COL;
        uint16_t dstStride = shiftCol >> SHIFT_MX_COL;

        load_cbuf_to_ca_mx(dstAddr, static_cast<__cbuf__ void *>(srcAddr), rowStartPosition, colStartPosition, 1,
                           colStep, srcStride, dstStride);
    } else if constexpr (DstTileData::Compact == CompactMode::Normal) {
        uint16_t validRowAlign = CeilDivision(validRow, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
        uint8_t rowStep = validRowAlign >> SHIFT_MX_ROW;
        uint8_t colStep = (validCol * sizeof(DataType)) >> SHIFT_MX_COL;
        constexpr uint16_t srcStride = SrcTileData::Cols >> SHIFT_MX_COL;
        uint16_t dstStride = validCol >> SHIFT_MX_COL;

        load_cbuf_to_ca_mx(dstAddr, static_cast<__cbuf__ void *>(srcAddr), rowStartPosition, colStartPosition, rowStep,
                           colStep, srcStride, dstStride);
    } else {
        constexpr uint8_t rowStep = DstTileData::Rows >> SHIFT_MX_ROW;
        constexpr uint8_t colStep = (DstTileData::Cols * sizeof(DataType)) >> SHIFT_MX_COL;
        constexpr uint16_t srcStride = SrcTileData::Cols >> SHIFT_MX_COL;
        constexpr uint16_t dstStride = DstTileData::Cols >> SHIFT_MX_COL;

        load_cbuf_to_ca_mx(dstAddr, static_cast<__cbuf__ void *>(srcAddr), rowStartPosition, colStartPosition, rowStep,
                           colStep, srcStride, dstStride);
    }
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractToBmx(typename DstTileData::TileDType __out__ dst,
                                 typename SrcTileData::TileDType __in__ src, uint16_t indexRow, uint16_t indexCol,
                                 uint16_t validRow, uint16_t validCol)
{
    static_assert((SrcTileData::SFractal == SLayout::ColMajor && !SrcTileData::isRowMajor),
                  "TMov_mx: SrcTile Invalid Fractal.");
    static_assert((DstTileData::SFractal == SLayout::ColMajor && !DstTileData::isRowMajor),
                  "TMov_mx: DstTile Invalid Fractal.");

    using DataType = typename DstTileData::DType;
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    uint64_t dstAddr = (uint64_t)__cce_get_tile_ptr(dst);
    uint16_t rowStartPosition = indexCol >> SHIFT_MX_ROW;
    uint16_t colStartPosition = (indexRow * sizeof(DataType)) >> SHIFT_MX_COL;

    if constexpr (DstTileData::Compact == CompactMode::Normal) {
        uint16_t validColAlign = CeilDivision(validCol, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
        uint8_t rowStep = validColAlign >> SHIFT_MX_ROW;
        uint8_t colStep = (validRow * sizeof(DataType)) >> SHIFT_MX_COL;
        constexpr uint16_t srcStride = SrcTileData::Rows >> SHIFT_MX_COL;
        uint16_t dstStride = validRow >> SHIFT_MX_COL;

        load_cbuf_to_cb_mx(dstAddr, reinterpret_cast<__cbuf__ void *>(srcAddr), rowStartPosition, colStartPosition,
                           rowStep, colStep, srcStride, dstStride);
    } else {
        constexpr uint8_t rowStep = DstTileData::Cols >> SHIFT_MX_ROW;
        constexpr uint8_t colStep = (DstTileData::Rows * sizeof(DataType)) >> SHIFT_MX_COL;
        constexpr uint16_t srcStride = SrcTileData::Rows >> SHIFT_MX_COL;
        constexpr uint16_t dstStride = DstTileData::Rows >> SHIFT_MX_COL;

        load_cbuf_to_cb_mx(dstAddr, reinterpret_cast<__cbuf__ void *>(srcAddr), rowStartPosition, colStartPosition,
                           rowStep, colStep, srcStride, dstStride);
    }
}

template <typename DstTileData, typename SrcTileData, bool Transpose, bool isFp4Type>
__tf__ AICORE void TExtractToA(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
                               uint16_t indexRow, uint16_t indexCol)
{
    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr int32_t dstRow = DstTileData::Rows;
    constexpr int32_t dstCol = DstTileData::Cols;
    using DataType = typename SrcTileData::DType;
    constexpr int typeSize = sizeof(DataType);
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __ca__ DataType *dstAddr = (__ca__ DataType *)__cce_get_tile_ptr(dst);
    constexpr int c0Size = isFp4Type ? BLOCK_BYTE_SIZE * KHALF / typeSize : BLOCK_BYTE_SIZE / typeSize;

    if constexpr (!Transpose) {
        uint16_t mStartPosition = indexRow >> SHIFT_BLOCK_LEN;
        uint16_t kStartPosition = (indexCol * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint8_t mStep = dstRow >> SHIFT_BLOCK_LEN;
        constexpr uint8_t kStep = (dstCol * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint16_t srcStride = srcRow >> SHIFT_BLOCK_LEN;
        constexpr uint16_t dstStride = dstRow >> SHIFT_BLOCK_LEN;

        if constexpr (isFp4Type) {
            load_cbuf_to_ca_s4(dstAddr, srcAddr, mStartPosition, kStartPosition / KHALF, mStep, kStep / KHALF,
                               srcStride, dstStride, 0);
        } else {
            load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 0);
        }
    } else {
        static_assert((srcRow % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0,
                      "srcRow must be aligned"); // fp16, fp32 should be aligned to 16
        static_assert((srcCol % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "srcCol must be aligned");
        static_assert((dstRow % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "dstRow must be aligned");
        static_assert((dstCol % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "dstCol must be aligned");

        uint16_t mStartPosition = indexCol >> SHIFT_BLOCK_LEN;
        uint16_t kStartPosition = (indexRow * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint8_t mStep = dstCol >> SHIFT_BLOCK_LEN;
        constexpr uint8_t kStep = (dstRow * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint16_t srcStride = srcCol >> SHIFT_BLOCK_LEN;
        constexpr uint16_t dstStride = dstRow >> SHIFT_BLOCK_LEN;

        if constexpr (isFp4Type) {
            load_cbuf_to_ca_s4(dstAddr, srcAddr, mStartPosition, kStartPosition / KHALF, mStep, kStep / KHALF,
                               srcStride, dstStride, 1);
        } else {
            load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
        }
    }
}

template <typename DstTileData, typename SrcTileData, bool isFp4Type>
__tf__ AICORE void TExtractToAVector(typename DstTileData::TileDType __out__ dst,
                                     typename SrcTileData::TileDType __in__ src, uint16_t indexRow, uint16_t indexCol,
                                     uint16_t dstValidCol)
{
    using DataType = typename SrcTileData::DType;
    constexpr int typeSize = sizeof(DataType);
    constexpr int32_t fractalSize = isFp4Type ? CUBE_BLOCK_SIZE * KHALF : CUBE_BLOCK_SIZE / typeSize;
    int32_t kAlign = (dstValidCol + fractalSize - 1) & ~(fractalSize - 1);

    static_assert((SrcTileData::Cols % fractalSize) == 0, "srcCol * sizeof(DataType) must be aligned to 512B");
    static_assert((DstTileData::Cols % fractalSize) == 0, "dstCol * sizeof(DataType) must be aligned to 512B");
    PTO_ASSERT((indexCol % fractalSize) == 0, "indexCol * sizeof(DataType) must be aligned to 512B");

    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __ca__ DataType *dstAddr = (__ca__ DataType *)__cce_get_tile_ptr(dst);
    uint16_t kStartPosition = (indexCol * typeSize) >> SHIFT_FRACTAL_BYTE;
    uint8_t kStep = kAlign / fractalSize;
    if constexpr (isFp4Type) {
        load_cbuf_to_ca_s4(dstAddr, srcAddr, 0, kStartPosition / KHALF, 1, kStep, 1, 1, 0);
    } else {
        load_cbuf_to_ca(dstAddr, srcAddr, 0, kStartPosition, 1, kStep, 1, 1, 0);
    }
}

template <typename DstTileData, typename SrcTileData, bool isFp4Type>
__tf__ AICORE void TExtractToACompact(typename DstTileData::TileDType __out__ dst,
                                      typename SrcTileData::TileDType __in__ src, uint16_t indexRow, uint16_t indexCol,
                                      uint16_t madM, uint16_t madK)
{
    using DataType = typename SrcTileData::DType;
    constexpr int typeSize = sizeof(DataType);
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __ca__ DataType *dstAddr = (__ca__ DataType *)__cce_get_tile_ptr(dst);
    constexpr int c0Size = isFp4Type ? BLOCK_BYTE_SIZE * KHALF / typeSize : BLOCK_BYTE_SIZE / typeSize;
    uint16_t madMAlign = CeilDivision(madM, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    uint16_t madKAlign = CeilDivision(madK, c0Size) * c0Size;

    uint16_t mStartPosition = indexRow >> SHIFT_BLOCK_LEN;
    uint16_t kStartPosition = (indexCol * typeSize) >> SHIFT_BLOCK_BYTE;
    uint8_t mStep = madMAlign >> SHIFT_BLOCK_LEN;
    uint8_t kStep = (madKAlign * typeSize) >> SHIFT_BLOCK_BYTE;
    constexpr uint16_t srcStride = SrcTileData::Rows >> SHIFT_BLOCK_LEN;
    uint16_t dstStride = madMAlign >> SHIFT_BLOCK_LEN;
    if constexpr (isFp4Type) {
        load_cbuf_to_ca_s4(dstAddr, srcAddr, mStartPosition, kStartPosition / KHALF, mStep, kStep / KHALF, srcStride,
                           dstStride, 0);
    } else {
        load_cbuf_to_ca(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 0);
    }
}

template <typename DstTileData, typename SrcTileData, bool isFp4Type>
__tf__ AICORE void TExtractToATransCompact(typename DstTileData::TileDType __out__ dst,
                                           typename SrcTileData::TileDType __in__ src, uint16_t indexRow,
                                           uint16_t indexCol, uint16_t madM, uint16_t madK)
{
    using DataType = typename SrcTileData::DType;
    constexpr int typeSize = sizeof(DataType);
    constexpr int c0Size = isFp4Type ? BLOCK_BYTE_SIZE * KHALF / typeSize : BLOCK_BYTE_SIZE / typeSize;
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __ca__ DataType *dstAddr = (__ca__ DataType *)__cce_get_tile_ptr(dst);

    uint16_t alignNum = max(FRACTAL_NZ_ROW, c0Size);
    uint16_t madMAlign = CeilDivision(madM, alignNum) * alignNum;
    uint16_t madKAlign = CeilDivision(madK, alignNum) * alignNum;

    uint16_t mStartPosition = indexCol >> SHIFT_BLOCK_LEN;
    uint16_t kStartPosition = (indexRow * typeSize) >> SHIFT_BLOCK_BYTE;
    uint8_t mStep = madKAlign >> SHIFT_BLOCK_LEN;
    uint8_t kStep = (madMAlign * typeSize) >> SHIFT_BLOCK_BYTE;
    constexpr uint16_t srcStride = SrcTileData::Cols >> SHIFT_BLOCK_LEN;
    uint16_t dstStride = madMAlign >> SHIFT_BLOCK_LEN;
    if constexpr (isFp4Type) { // b4
        uint16_t dstAddrStride = CeilDivision(madM, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW * BLOCK_BYTE_SIZE;
        uint16_t mLoop = mStep >> SHIFT_M_STEP_B4;
        mStep = M_STEP_MIN_VAL_B4;
        for (uint16_t idx = 0; idx < mLoop; ++idx) {
            load_cbuf_to_ca_s4(dstAddr, srcAddr, mStartPosition, kStartPosition / KHALF, mStep, kStep / KHALF,
                               srcStride, dstStride, 1);
            dstAddr += dstAddrStride;
            mStartPosition += M_STEP_MIN_VAL_B4;
        }
    } else if constexpr (typeSize == 1) { // b8
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

template <typename DstTileData, typename SrcTileData, bool Transpose, bool isFp4Type>
__tf__ AICORE void TExtractToB(typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src,
                               uint16_t indexRow, uint16_t indexCol)
{
    using DataType = typename SrcTileData::DType;
    constexpr int typeSize = sizeof(DataType);
    constexpr int32_t srcRow = SrcTileData::Rows;
    constexpr int32_t srcCol = SrcTileData::Cols;
    constexpr int32_t dstRow = DstTileData::Rows;
    constexpr int32_t dstCol = DstTileData::Cols;
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __cb__ DataType *dstAddr = (__cb__ DataType *)__cce_get_tile_ptr(dst);
    constexpr int c0Size = isFp4Type ? BLOCK_BYTE_SIZE * KHALF / typeSize : BLOCK_BYTE_SIZE / typeSize;

    if constexpr (!Transpose) {
        uint16_t mStartPosition = indexCol >> SHIFT_BLOCK_LEN;
        uint16_t kStartPosition = (indexRow * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint8_t mStep = dstCol >> SHIFT_BLOCK_LEN;
        constexpr uint8_t kStep = (dstRow * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint16_t srcStride = srcCol >> SHIFT_BLOCK_LEN;
        constexpr uint16_t dstStride = dstCol >> SHIFT_BLOCK_LEN;
        if constexpr (isFp4Type) {
            load_cbuf_to_cb_s4(dstAddr, srcAddr, mStartPosition, kStartPosition / KHALF, mStep, kStep / KHALF,
                               srcStride, dstStride, 0);
        } else {
            load_cbuf_to_cb(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 0);
        }
    } else {
        static_assert((srcRow % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0,
                      "srcRow must be aligned"); // fp16, fp32 should be aligned to 16
        static_assert((srcCol % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "srcCol must be aligned");
        static_assert((dstRow % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "dstRow must be aligned");
        static_assert((dstCol % (typeSize == 1 ? c0Size : FRACTAL_NZ_ROW)) == 0, "dstCol must be aligned");

        uint16_t mStartPosition = indexRow >> SHIFT_BLOCK_LEN;
        uint16_t kStartPosition = (indexCol * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint8_t mStep = dstRow >> SHIFT_BLOCK_LEN;
        constexpr uint8_t kStep = (dstCol * typeSize) >> SHIFT_BLOCK_BYTE;
        constexpr uint16_t srcStride = srcRow >> SHIFT_BLOCK_LEN;
        constexpr uint16_t dstStride = dstCol >> SHIFT_BLOCK_LEN;
        if constexpr (isFp4Type) {
            load_cbuf_to_cb_s4(dstAddr, srcAddr, mStartPosition, kStartPosition / KHALF, mStep, kStep / KHALF,
                               srcStride, dstStride, 1);
        } else {
            load_cbuf_to_cb(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
        }
    }
}

template <typename DstTileData, typename SrcTileData, bool isFp4Type>
__tf__ AICORE void TExtractToBCompact(typename DstTileData::TileDType __out__ dst,
                                      typename SrcTileData::TileDType __in__ src, uint16_t indexRow, uint16_t indexCol,
                                      uint16_t madK, uint16_t madN)
{
    using DataType = typename SrcTileData::DType;
    constexpr int typeSize = sizeof(DataType);
    constexpr int c0Size = isFp4Type ? BLOCK_BYTE_SIZE * KHALF / typeSize : BLOCK_BYTE_SIZE / typeSize;

    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __cb__ DataType *dstAddr = (__cb__ DataType *)__cce_get_tile_ptr(dst);
    uint16_t madNAlign = CeilDivision(madN, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    uint16_t madKAlign = CeilDivision(madK, c0Size) * c0Size;

    uint16_t mStartPosition = indexCol >> SHIFT_BLOCK_LEN;
    uint16_t kStartPosition = (indexRow * typeSize) >> SHIFT_BLOCK_BYTE;
    uint8_t mStep = madNAlign >> SHIFT_BLOCK_LEN;
    uint8_t kStep = (madKAlign * typeSize) >> SHIFT_BLOCK_BYTE;
    constexpr uint16_t srcStride = SrcTileData::Cols >> SHIFT_BLOCK_LEN;
    uint16_t dstStride = madNAlign >> SHIFT_BLOCK_LEN;
    if constexpr (isFp4Type) {
        load_cbuf_to_cb_s4(dstAddr, srcAddr, mStartPosition, kStartPosition / KHALF, mStep, kStep / KHALF, srcStride,
                           dstStride, 0);
    } else {
        load_cbuf_to_cb(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 0);
    }
}

template <typename DstTileData, typename SrcTileData, bool isFp4Type>
__tf__ AICORE void TExtractToBTransCompact(typename DstTileData::TileDType __out__ dst,
                                           typename SrcTileData::TileDType __in__ src, uint16_t indexRow,
                                           uint16_t indexCol, uint16_t madK, uint16_t madN)
{
    using DataType = typename SrcTileData::DType;
    constexpr int typeSize = sizeof(DataType);
    constexpr int c0Size = isFp4Type ? BLOCK_BYTE_SIZE * KHALF / typeSize : BLOCK_BYTE_SIZE / typeSize;
    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __cb__ DataType *dstAddr = (__cb__ DataType *)__cce_get_tile_ptr(dst);

    uint16_t alignNum = max(FRACTAL_NZ_ROW, c0Size);
    uint16_t madNAlign = CeilDivision(madN, alignNum) * alignNum;
    uint16_t madKAlign = CeilDivision(madK, alignNum) * alignNum;

    uint16_t mStartPosition = indexRow >> SHIFT_BLOCK_LEN;
    uint16_t kStartPosition = (indexCol * typeSize) >> SHIFT_BLOCK_BYTE;
    uint8_t mStep = madKAlign >> SHIFT_BLOCK_LEN;
    uint8_t kStep = (madNAlign * typeSize) >> SHIFT_BLOCK_BYTE;
    constexpr uint16_t srcStride = SrcTileData::Rows >> SHIFT_BLOCK_LEN;
    uint16_t dstStride = madNAlign >> SHIFT_BLOCK_LEN;
    if constexpr (isFp4Type) { // b4
        uint16_t dstAddrStride = CeilDivision(madN, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW * BLOCK_BYTE_SIZE;
        uint16_t nLoop = mStep >> SHIFT_M_STEP_B4;
        mStep = M_STEP_MIN_VAL_B4;
        for (uint16_t idx = 0; idx < nLoop; ++idx) {
            load_cbuf_to_cb_s4(dstAddr, srcAddr, mStartPosition, kStartPosition / KHALF, mStep, kStep / KHALF,
                               srcStride, dstStride, 1);
            dstAddr += dstAddrStride;
            mStartPosition += M_STEP_MIN_VAL_B4;
        }
    } else if constexpr (typeSize == 1) { // b8
        uint16_t dstAddrStride = CeilDivision(madN, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW * BLOCK_BYTE_SIZE;
        uint16_t nLoop = mStep >> SHIFT_M_STEP_B8;
        mStep = M_STEP_MIN_VAL_B8;
        for (uint16_t idx = 0; idx < nLoop; ++idx) {
            load_cbuf_to_cb(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
            dstAddr += dstAddrStride;
            mStartPosition += M_STEP_MIN_VAL_B8;
        }
    } else { // b16/b32
        load_cbuf_to_cb(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 1);
    }
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractVecToMat(typename DstTileData::TileDType __out__ dst,
                                    typename SrcTileData::TileDType __in__ src, uint16_t indexRow, uint16_t indexCol,
                                    uint32_t srcValidRow, uint32_t srcValidCol, uint32_t dstValidRow,
                                    uint32_t dstValidCol)
{
    using T = typename SrcTileData::DType;
    constexpr int32_t c0Size = BLOCK_BYTE_SIZE / sizeof(T);
    uint32_t offset = SrcTileData::Rows * c0Size * (indexCol / c0Size) + (indexRow * c0Size + (indexCol % c0Size));
    if constexpr (SrcTileData::isRowMajor && (SrcTileData::SFractal == SLayout::NoneBox)) {
        offset = indexRow * srcValidCol + indexCol;
    }

    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src) + offset;
    __cbuf__ T *dstPtr = (__cbuf__ T *)__cce_get_tile_ptr(dst);
    if constexpr (SrcTileData::isRowMajor && (SrcTileData::SFractal == SLayout::NoneBox)) {
        uint16_t blockLen = dstValidRow * dstValidCol * sizeof(T) / BLOCK_BYTE_SIZE;
        // dst, src, sid, nBurst, lenBurst, srcStride, dstStride
        copy_ubuf_to_cbuf(dstPtr, srcPtr, 0, 1, blockLen, 0, 0);
    } else if constexpr (!SrcTileData::isRowMajor && (SrcTileData::SFractal == SLayout::RowMajor)) {
        uint16_t blockCout = CeilDivision(dstValidCol, c0Size);
        uint32_t alignRow = (dstValidRow + FRACTAL_NZ_ROW - 1) / FRACTAL_NZ_ROW * FRACTAL_NZ_ROW;
        uint16_t blockLen = alignRow * c0Size * sizeof(T) / BLOCK_BYTE_SIZE;
        constexpr uint16_t srcStride = SrcTileData::Rows - DstTileData::Rows;
        copy_ubuf_to_cbuf(dstPtr, srcPtr, 0, blockCout, blockLen, srcStride, 0);
    }
}

template <typename DstTileData, typename SrcTileData, QuantMode_t QuantPre, ReluPreMode reluMode>
__tf__ AICORE void TExtractAccToMat(typename DstTileData::TileDType __out__ dst,
                                    typename SrcTileData::TileDType __in__ src, uint16_t validRow, uint16_t validCol,
                                    uint16_t indexRow, uint16_t indexCol)
{
    using dstType = typename DstTileData::DType;
    using srcType = typename SrcTileData::DType;
    constexpr bool channelSplitEnable =
        (!DstTileData::isRowMajor && (DstTileData::SFractal == SLayout::RowMajor)) &&
        (std::is_same_v<dstType, float>)&&(DstTileData::SFractalSize == CUBE_BLOCK_SIZE);
    constexpr int32_t c0Size = (!channelSplitEnable) && (DstTileData::SFractalSize == 2 * CUBE_BLOCK_SIZE) ?
                                   2 * C0_SIZE_BYTE / sizeof(dstType) :
                                   C0_SIZE_BYTE / sizeof(dstType);
    constexpr uint32_t dstStride = DstTileData::Rows * c0Size;
    uint16_t nSize = CeilDivision(validCol, c0Size) * c0Size;
    uint32_t srcOffset = SrcTileData::Rows * ACC_C0_SIZE * (indexCol / ACC_C0_SIZE) +
                         (indexRow * ACC_C0_SIZE + (indexCol % ACC_C0_SIZE));
    __cbuf__ dstType *dstAddr = (__cbuf__ dstType *)__cce_get_tile_ptr(dst);
    __cc__ srcType *srcData = (__cc__ srcType *)__cce_get_tile_ptr(src) + srcOffset;

    copy_matrix_cc_to_cbuf(dstAddr, srcData, 0, nSize, validRow, dstStride, SrcTileData::Rows, 0, 0, 0, QuantPre,
                           reluMode, channelSplitEnable, false, 0, 0, false, false, 0, false, false, false, false,
                           false, false);
}

template <typename T>
constexpr bool is_textract_supported_type =
    std::disjunction_v<std::is_same<T, int8_t>, std::is_same<T, float8_e4m3_t>, std::is_same<T, float8_e5m2_t>,
                       std::is_same<T, hifloat8_t>, std::is_same<T, half>, std::is_same<T, bfloat16_t>,
                       std::is_same<T, float>, std::is_same<T, float4_e2m1x2_t>, std::is_same<T, float4_e1m2x2_t>,
                       std::is_same<T, float8_e8m0_t>>;

template <typename DstTileData, typename SrcTileData>
AICORE void TExtractToLeft(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol)
{
    static_assert((SrcTileData::SFractal == SLayout::ColMajor && SrcTileData::isRowMajor) ||
                      (SrcTileData::SFractal == SLayout::RowMajor && !SrcTileData::isRowMajor) ||
                      (SrcTileData::Rows == 1 && SrcTileData::isRowMajor),
                  "TExtract: SrcTile Invalid Fractal");
    static_assert(DstTileData::SFractal == SLayout::RowMajor && !DstTileData::isRowMajor,
                  "TExtract: DstTile Invalid Fractal");
    constexpr bool isFp4Type = std::is_same<typename SrcTileData::DType, float4_e2m1x2_t>::value ||
                               std::is_same<typename SrcTileData::DType, float4_e1m2x2_t>::value;
    if constexpr (SrcTileData::Rows == 1 && SrcTileData::isRowMajor) {
        TExtractToAVector<DstTileData, SrcTileData, isFp4Type>(dst.data(), src.data(), indexRow, indexCol,
                                                               dst.GetValidCol());
    } else if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
        if constexpr (DstTileData::Compact == CompactMode::Normal) {
            TExtractToACompact<DstTileData, SrcTileData, isFp4Type>(dst.data(), src.data(), indexRow, indexCol,
                                                                    dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToA<DstTileData, SrcTileData, false, isFp4Type>(dst.data(), src.data(), indexRow, indexCol);
        }
    } else {
        if constexpr (DstTileData::Compact == CompactMode::Normal || sizeof(typename SrcTileData::DType) == 1) {
            TExtractToATransCompact<DstTileData, SrcTileData, isFp4Type>(dst.data(), src.data(), indexRow, indexCol,
                                                                         dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToA<DstTileData, SrcTileData, true, isFp4Type>(dst.data(), src.data(), indexRow, indexCol);
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
                  "TExtract: DstTile Invalid Fractal");
    constexpr bool isFp4Type = std::is_same<typename SrcTileData::DType, float4_e2m1x2_t>::value ||
                               std::is_same<typename SrcTileData::DType, float4_e1m2x2_t>::value;
    if constexpr (DstTileData::SFractal == SrcTileData::SFractal) {
        if constexpr (DstTileData::Compact == CompactMode::Normal) {
            TExtractToBCompact<DstTileData, SrcTileData, isFp4Type>(dst.data(), src.data(), indexRow, indexCol,
                                                                    dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToB<DstTileData, SrcTileData, false, isFp4Type>(dst.data(), src.data(), indexRow, indexCol);
        }
    } else {
        if constexpr (DstTileData::Compact == CompactMode::Normal || sizeof(typename SrcTileData::DType) == 1) {
            TExtractToBTransCompact<DstTileData, SrcTileData, isFp4Type>(dst.data(), src.data(), indexRow, indexCol,
                                                                         dst.GetValidRow(), dst.GetValidCol());
        } else {
            TExtractToB<DstTileData, SrcTileData, true, isFp4Type>(dst.data(), src.data(), indexRow, indexCol);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_TILE_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol)
{
    static_assert(is_textract_supported_type<typename DstTileData::DType>,
                  "TExtract: Unsupported data type! Supported types: int8_t, hifloat8_t, fp8_e5m2_t, fp8_e4m3fn_t, \
        half, bfloat16_t, float, float4_e2m1x2_t, float4_e1m2x2_t, float8_e8m0_t");
    static_assert((SrcTileData::Loc == TileType::Acc) ||
                      std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
                  "TExtract: Destination and Source tile data types must be the same");

    if constexpr (DstTileData::Loc == TileType::Left) {
        TExtractToLeft(dst, src, indexRow, indexCol);
    } else if constexpr (DstTileData::Loc == TileType::Right) {
        TExtractToRight(dst, src, indexRow, indexCol);
    } else if constexpr (SrcTileData::Loc == TileType::Vec && DstTileData::Loc == TileType::Mat) {
        TExtractVecToMat<DstTileData, SrcTileData>(dst.data(), src.data(), indexRow, indexCol, src.GetValidRow(),
                                                   src.GetValidCol(), dst.GetValidRow(), dst.GetValidCol());
    } else if constexpr (DstTileData::Loc == TileType::ScaleLeft) {
        TExtractToAmx<DstTileData, SrcTileData>(dst.data(), src.data(), indexRow, indexCol, dst.GetValidRow(),
                                                dst.GetValidCol());
    } else if constexpr (DstTileData::Loc == TileType::ScaleRight) {
        TExtractToBmx<DstTileData, SrcTileData>(dst.data(), src.data(), indexRow, indexCol, dst.GetValidRow(),
                                                dst.GetValidCol());
    } else if constexpr (SrcTileData::Loc == TileType::Acc && DstTileData::Loc == TileType::Mat) {
        static_assert((!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor),
                      "Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor).");
        CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType>();
        constexpr QuantMode_t quantPre =
            GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
        TExtractAccToMat<DstTileData, SrcTileData, quantPre, ReluPreMode::NoRelu>(
            dst.data(), src.data(), dst.GetValidRow(), dst.GetValidCol(), indexRow, indexCol);
    }
}

template <typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractToBConv(typename DstTileData::TileDType __out__ dst,
                                   typename SrcTileData::TileDType __in__ src, uint16_t srcCol, uint16_t dstValidRow,
                                   uint16_t dstValidCol, uint16_t indexRow, uint16_t indexCol)
{
    using DataType = typename SrcTileData::DType;
    constexpr int c0Size = BLOCK_BYTE_SIZE / sizeof(DataType);

    __cbuf__ DataType *srcAddr = (__cbuf__ DataType *)__cce_get_tile_ptr(src);
    __cb__ DataType *dstAddr = (__cb__ DataType *)__cce_get_tile_ptr(dst);
    uint16_t dstValidColAlign = CeilDivision(dstValidCol, FRACTAL_NZ_ROW) * FRACTAL_NZ_ROW;
    uint16_t dstValidRowAlign = CeilDivision(dstValidRow, c0Size) * c0Size;

    uint16_t mStartPosition = indexCol >> SHIFT_BLOCK_LEN;
    uint16_t kStartPosition = (indexRow * sizeof(DataType)) >> SHIFT_BLOCK_BYTE;
    uint8_t mStep = dstValidColAlign >> SHIFT_BLOCK_LEN;
    uint8_t kStep = (dstValidRowAlign * sizeof(DataType)) >> SHIFT_BLOCK_BYTE;
    uint16_t srcStride = srcCol >> SHIFT_BLOCK_LEN;
    uint16_t dstStride = dstValidColAlign >> SHIFT_BLOCK_LEN;
    load_cbuf_to_cb(dstAddr, srcAddr, mStartPosition, kStartPosition, mStep, kStep, srcStride, dstStride, 0);
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TextractConvTileCheck(DstTileData &dst, SrcTileData &src)
{
    static_assert(std::is_same_v<typename DstTileData::DType, int8_t> ||
                      std::is_same_v<typename DstTileData::DType, uint8_t> ||
                      std::is_same_v<typename DstTileData::DType, int16_t> ||
                      std::is_same_v<typename DstTileData::DType, uint16_t> ||
                      std::is_same_v<typename DstTileData::DType, int32_t> ||
                      std::is_same_v<typename DstTileData::DType, uint32_t> ||
                      std::is_same_v<typename DstTileData::DType, half> ||
                      std::is_same_v<typename DstTileData::DType, bfloat16_t> ||
                      std::is_same_v<typename DstTileData::DType, float>,
                  "Fix: Data type must be int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/bfloat16_t/float!");
    static_assert(SrcTileData::Loc == pto::TileType::Mat, "Fix: Src TileType must be Mat!");
    static_assert(DstTileData::Loc == pto::TileType::Right, "Fix: Dst TileType must be Right!");
    static_assert(sizeof(typename DstTileData::DType) == sizeof(typename SrcTileData::DType),
                  "Fix: Source dtype must be same with dst dtype!");

    static_assert((SrcTileData::layout == Layout::FRACTAL_Z) || (SrcTileData::layout == Layout::FRACTAL_Z_3D),
                  "TExtract: Source layout only support FRACTAL_Z or FRACTAL_Z_3D.");
    static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
                  "TExtract: Destination layout only support SLayout is ColMajor ang BLayout is RowMajor.");
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_CONVTILE_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow, uint16_t indexCol)
{
    TextractConvTileCheck<DstTileData, SrcTileData>(dst, src);
    constexpr uint32_t c0ElemCount = C0_SIZE_BYTE / sizeof(typename SrcTileData::DType);
    if constexpr (SrcTileData::totalDimCount == 4) { // ConvTile layout is [C1HW,N/16,16,C0]
        static_assert(SrcTileData::staticShape[2] == FRACTAL_NZ_ROW && SrcTileData::staticShape[3] == c0ElemCount,
                      "Fix: The SrcTileData last 2 dim must be static and satisfy [16, 32 / sizeof(DataType)]");
        uint16_t srcCol = src.GetShape(1) * src.GetShape(2);
        TExtractToBConv<DstTileData, SrcTileData>(dst.data(), src.data(), srcCol, dst.GetValidRow(), dst.GetValidCol(),
                                                  indexRow, indexCol);
    } else { //  [C1,H,W,N,C0]
        TExtractToBConv<DstTileData, SrcTileData>(dst.data(), src.data(), src.GetShape(3), dst.GetValidRow(),
                                                  dst.GetValidCol(), indexRow, indexCol);
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractVecToVecNDImpl(typename DstTileData::TileDType __out__ dst,
                                          typename SrcTileData::TileDType __in__ src, uint16_t validRow,
                                          uint16_t validCol, uint32_t indexRow, uint32_t indexCol)
{
    __ubuf__ T *dstAddr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    constexpr uint32_t dstRowStride = DstTileData::RowStride;
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;

    __ubuf__ T *srcStart = srcAddr + indexRow * srcRowStride + indexCol;
    uint32_t rowBytes = static_cast<uint32_t>(validCol) * sizeof(T);
    uint32_t totalBytes = static_cast<uint32_t>(validRow) * rowBytes;
    uint16_t rowBurstLen = static_cast<uint16_t>(rowBytes / BLOCK_BYTE_SIZE);

    if (validCol == dstRowStride && validCol == srcRowStride && totalBytes >= BLOCK_BYTE_SIZE) {
        uint16_t burstLen = static_cast<uint16_t>(totalBytes / BLOCK_BYTE_SIZE);
        copy_ubuf_to_ubuf((__ubuf__ void *)dstAddr, (__ubuf__ void *)srcStart, 0, 1, burstLen, 0, 0);
    } else {
        uint16_t srcGap = static_cast<uint16_t>((srcRowStride - validCol) * sizeof(T) / BLOCK_BYTE_SIZE);
        uint16_t dstGap = static_cast<uint16_t>((dstRowStride - validCol) * sizeof(T) / BLOCK_BYTE_SIZE);
        copy_ubuf_to_ubuf((__ubuf__ void *)dstAddr, (__ubuf__ void *)srcStart, 0, validRow, rowBurstLen, srcGap,
                          dstGap);
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractVecToVecNDAlignedImpl(typename DstTileData::TileDType __out__ dst,
                                                 typename SrcTileData::TileDType __in__ src, uint16_t validRow,
                                                 uint16_t validCol, uint32_t indexRow, uint32_t indexCol)
{
    __ubuf__ T *dstAddr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    constexpr uint32_t dstRowStride = DstTileData::RowStride;
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;
    constexpr uint32_t elementsPerRepeat = REPEAT_BYTE / sizeof(T);

    __VEC_SCOPE__
    {
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        RegTensor<T> vreg;
        MaskReg preg;
        uint16_t repeatTimes = CeilDivision(static_cast<uint32_t>(validCol), elementsPerRepeat);

        for (uint16_t i = 0; i < validRow; ++i) {
            uint32_t sreg = static_cast<uint32_t>(validCol);
            uint32_t srcRowOff = (indexRow + static_cast<uint32_t>(i)) * srcRowStride + indexCol;
            uint32_t dstRowOff = static_cast<uint32_t>(i) * dstRowStride;
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                preg = CreatePredicate<T>(sreg);
                vlds(vreg, srcAddr, srcRowOff + static_cast<uint32_t>(j) * elementsPerRepeat, NORM);
                vsts(vreg, dstAddr, dstRowOff + static_cast<uint32_t>(j) * elementsPerRepeat, distValue, preg);
            }
        }
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractVecToVecNDVectorImpl(typename DstTileData::TileDType __out__ dst,
                                                typename SrcTileData::TileDType __in__ src, uint16_t validRow,
                                                uint16_t validCol, uint32_t indexRow, uint32_t indexCol)
{
    __ubuf__ T *dstAddr = (__ubuf__ T *)__cce_get_tile_ptr(dst);
    __ubuf__ T *srcAddr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    constexpr uint32_t dstRowStride = DstTileData::RowStride;
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;
    constexpr uint32_t elementsPerRepeat = REPEAT_BYTE / sizeof(T);

    __VEC_SCOPE__
    {
        constexpr auto distValue =
            std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
        RegTensor<T> vreg;
        UnalignReg ureg;
        MaskReg preg;
        uint16_t repeatTimes = CeilDivision(static_cast<uint32_t>(validCol), elementsPerRepeat);

        for (uint16_t i = 0; i < validRow; ++i) {
            uint32_t sreg = static_cast<uint32_t>(validCol);
            __ubuf__ T *psrc = srcAddr + (indexRow + static_cast<uint32_t>(i)) * srcRowStride + indexCol;
            uint32_t dstRowOff = static_cast<uint32_t>(i) * dstRowStride;
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                preg = CreatePredicate<T>(sreg);
                vldas(ureg, psrc + static_cast<uint32_t>(j) * elementsPerRepeat);
                vldus(vreg, ureg, psrc + static_cast<uint32_t>(j) * elementsPerRepeat);
                vsts(vreg, dstAddr, dstRowOff + static_cast<uint32_t>(j) * elementsPerRepeat, distValue, preg);
            }
        }
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TExtractVecToVecNDScalarImpl(typename DstTileData::TileDType __out__ dst,
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
PTO_INTERNAL void TExtractVecToVecNDDispatch(DstTileData &dst, SrcTileData &src, uint32_t indexRow, uint32_t indexCol)
{
    uint16_t validRow = static_cast<uint16_t>(dst.GetValidRow());
    uint16_t validCol = static_cast<uint16_t>(dst.GetValidCol());

    PTO_ASSERT(indexRow + DstTileData::ValidRow <= SrcTileData::Rows,
               "TEXTRACT ND_VEC : indexRow + dstValidRows exceeds srcRows!");
    PTO_ASSERT(indexCol + DstTileData::ValidCol <= SrcTileData::Cols,
               "TEXTRACT ND_VEC : indexCol + dstValidCols exceeds srcCols!");

    constexpr bool kStridesAligned = (SrcTileData::RowStride * sizeof(T) % BLOCK_BYTE_SIZE == 0) &&
                                     (DstTileData::RowStride * sizeof(T) % BLOCK_BYTE_SIZE == 0);
    constexpr bool kValidColAligned = (DstTileData::ValidCol * sizeof(T) % BLOCK_BYTE_SIZE == 0);

    if constexpr (kStridesAligned) {
        if (indexCol * sizeof(T) % BLOCK_BYTE_SIZE == 0) {
            if constexpr (kValidColAligned) {
                TExtractVecToVecNDImpl<T, DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol,
                                                                    indexRow, indexCol);
            } else {
                TExtractVecToVecNDAlignedImpl<T, DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol,
                                                                           indexRow, indexCol);
            }
        } else {
            TExtractVecToVecNDVectorImpl<T, DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol,
                                                                      indexRow, indexCol);
        }
    } else {
        TExtractVecToVecNDVectorImpl<T, DstTileData, SrcTileData>(dst.data(), src.data(), validRow, validCol, indexRow,
                                                                  indexCol);
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    if constexpr (DstTileData::Loc == TileType::Vec && SrcTileData::Loc == TileType::Vec) {
        using T = typename DstTileData::DType;
        static_assert(std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
                      "TEXTRACT : Source and destination data types must match");
        static_assert(DstTileData::isRowMajor, "TEXTRACT : Destination must be RowMajor (ND format)");
        static_assert(SrcTileData::isRowMajor, "TEXTRACT : Source must be RowMajor (ND format)");
        static_assert(DstTileData::Rows <= SrcTileData::Rows,
                      "TEXTRACT : Destination rows must not exceed source rows");
        static_assert(DstTileData::Cols <= SrcTileData::Cols,
                      "TEXTRACT : Destination cols must not exceed source cols");
        static_assert((std::is_same<T, half>::value) || (std::is_same<T, bfloat16_t>::value) ||
                          (std::is_same<T, float>::value) || (std::is_same<T, int32_t>::value) ||
                          (std::is_same<T, float8_e4m3_t>::value) || (std::is_same<T, float8_e5m2_t>::value) ||
                          (std::is_same<T, hifloat8_t>::value) || (std::is_same<T, int8_t>::value) ||
                          (std::is_same<T, float8_e8m0_t>::value),
                      "TEXTRACT : Unsupported data type.");
        uint32_t idxRow = static_cast<uint32_t>(indexRow);
        uint32_t idxCol = static_cast<uint32_t>(indexCol);
        if constexpr (DstTileData::ValidRow == 1 && DstTileData::ValidCol == 1) {
            PTO_ASSERT(idxRow < SrcTileData::Rows, "TEXTRACT : indexRow exceeds srcRows!");
            PTO_ASSERT(idxCol < SrcTileData::Cols, "TEXTRACT : indexCol exceeds srcCols!");
            TExtractVecToVecNDScalarImpl<T, DstTileData, SrcTileData>(dst.data(), src.data(), idxRow, idxCol);
        } else {
            TExtractVecToVecNDDispatch<T>(dst, src, idxRow, idxCol);
        }
    } else if constexpr (is_conv_tile_v<SrcTileData>) {
        TEXTRACT_CONVTILE_IMPL(dst, src, indexRow, indexCol);
    } else {
        TEXTRACT_TILE_IMPL(dst, src, indexRow, indexCol);
    }
}

// vector quant
template <typename FpTileData>
__tf__ PTO_INTERNAL void SetFPC(typename FpTileData::TileDType __in__ fp, uint16_t indexCol)
{
    __fbuf__ typename FpTileData::DType *dstAddrFp =
        (__fbuf__ typename FpTileData::DType *)__cce_get_tile_ptr(fp) + indexCol;
    uint64_t deqTensorAddr = ((uint64_t)dstAddrFp >> static_cast<uint64_t>(7)) << 8;
    set_fpc(deqTensorAddr);
}

// relu
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint16_t indexRow = 0, uint16_t indexCol = 0)
{
    static_assert((DstTileData::Loc == TileType::Mat), "Destination TileType only support Mat.");
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType>();
    static_assert((!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor),
                  "Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor).");
    constexpr QuantMode_t quantPre = GetCastPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    TExtractAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), dst.GetValidRow(),
                                                                   dst.GetValidCol(), indexRow, indexCol);
}

// scalar quant
template <typename DstTileData, typename SrcTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, uint64_t preQuantScalar, uint16_t indexRow = 0,
                                uint16_t indexCol = 0)
{
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    static_assert((DstTileData::Loc == TileType::Mat), "Destination TileType only support Mat.");
    static_assert((!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor),
                  "Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor).");
    constexpr QuantMode_t quantPre = GetScalarPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    set_quant_pre(preQuantScalar);
    TExtractAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), dst.GetValidRow(),
                                                                   dst.GetValidCol(), indexRow, indexCol);
}

template <typename DstTileData, typename SrcTileData, typename FpTileData, ReluPreMode reluMode = ReluPreMode::NoRelu>
PTO_INTERNAL void TEXTRACT_IMPL(DstTileData &dst, SrcTileData &src, FpTileData &fp, uint16_t indexRow = 0,
                                uint16_t indexCol = 0)
{
    CheckTMovAccValid<DstTileData, SrcTileData, typename DstTileData::DType, typename SrcTileData::DType, true>();
    static_assert((DstTileData::Loc == TileType::Mat), "Destination TileType only support Mat.");
    static_assert((!DstTileData::isRowMajor && DstTileData::SFractal == SLayout::RowMajor),
                  "Dst fractal format should be (BFractal: ColMajor, SFractal: RowMajor).");
    static_assert(FpTileData::Loc == TileType::Scaling, "Fp only support Scaling.");
    constexpr QuantMode_t quantPre = GetVectorPreQuantMode<typename SrcTileData::DType, typename DstTileData::DType>();
    SetFPC<FpTileData>(fp.data(), indexCol);
    TExtractAccToMat<DstTileData, SrcTileData, quantPre, reluMode>(dst.data(), src.data(), dst.GetValidRow(),
                                                                   dst.GetValidCol(), indexRow, indexCol);
}
} // namespace pto
#endif // TEXTRACT_HPP
