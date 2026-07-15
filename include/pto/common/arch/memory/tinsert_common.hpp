/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TINSERT_COMMON_MEMORY
#define TINSERT_COMMON_MEMORY
#include <pto/common/utils.hpp>

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TInsertVecToVecNDScalar(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint32_t indexRow,
    uint32_t indexCol)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    constexpr uint32_t dstRowStride = DstTileData::RowStride;
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    dstAddr[indexRow * dstRowStride + indexCol] = srcAddr[0];
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TInsertVecToVecNDAligned(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint32_t indexRow, uint32_t indexCol)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    constexpr uint32_t srcRowStride = SrcTileData::RowStride;
    constexpr uint32_t dstRowStride = DstTileData::RowStride;
    __ubuf__ T* dstStart = dstAddr + indexRow * dstRowStride + indexCol;
    uint32_t rowBytes = static_cast<uint32_t>(validCol) * sizeof(T);
    if (validCol == dstRowStride && validCol == srcRowStride) {
        uint32_t totalBytes = static_cast<uint32_t>(validRow) * rowBytes;
        uint16_t burstLen = static_cast<uint16_t>(totalBytes / BLOCK_BYTE_SIZE);
        pto_copy_ubuf_to_ubuf((__ubuf__ void*)dstStart, (__ubuf__ void*)srcAddr, 1, burstLen, 0, 0);
    } else {
        uint16_t rowBurst = static_cast<uint16_t>(rowBytes / BLOCK_BYTE_SIZE);
        uint16_t srcGap = static_cast<uint16_t>((srcRowStride - validCol) * sizeof(T) / BLOCK_BYTE_SIZE);
        uint16_t dstGap = static_cast<uint16_t>((dstRowStride - validCol) * sizeof(T) / BLOCK_BYTE_SIZE);
        pto_copy_ubuf_to_ubuf((__ubuf__ void*)dstStart, (__ubuf__ void*)srcAddr, validRow, rowBurst, srcGap, dstGap);
    }
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TInsertVecToVecNZScalar(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint32_t indexRow,
    uint32_t indexCol)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
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
__tf__ AICORE void TInsertVecToVecNZAligned(
    typename DstTileData::TileDType __out__ dst, typename SrcTileData::TileDType __in__ src, uint16_t validRow,
    uint16_t validCol, uint32_t indexRow, uint32_t indexCol)
{
    __ubuf__ T* dstAddr = (__ubuf__ T*)__cce_get_tile_ptr(dst);
    __ubuf__ T* srcAddr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    constexpr uint32_t typeSize = sizeof(T);
    constexpr uint32_t c0Size = BLOCK_BYTE_SIZE / typeSize;
    constexpr uint32_t srcRows = SrcTileData::Rows;
    constexpr uint32_t dstRows = DstTileData::Rows;
    uint16_t burstNum = static_cast<uint16_t>(validCol / c0Size);
    uint16_t burstLen = static_cast<uint16_t>((validRow * c0Size * typeSize) / BLOCK_BYTE_SIZE);
    uint32_t dstOffset = (indexCol / c0Size) * dstRows * c0Size + indexRow * c0Size;
    uint16_t srcGap = static_cast<uint16_t>(srcRows - validRow);
    uint16_t dstGap = static_cast<uint16_t>(dstRows - validRow);
    pto_copy_ubuf_to_ubuf(
        (__ubuf__ void*)(dstAddr + dstOffset), (__ubuf__ void*)srcAddr, burstNum, burstLen, srcGap, dstGap);
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void CheckTInsertVecToVecCommon()
{
    using DstT = typename DstTileData::DType;
    using SrcT = typename SrcTileData::DType;
    static_assert(std::is_same<DstT, SrcT>::value, "TINSERT Vec->Vec : Source and destination data types must match.");
    static_assert(
        std::is_same<DstT, int8_t>::value || std::is_same<DstT, uint8_t>::value || std::is_same<DstT, int16_t>::value ||
            std::is_same<DstT, uint16_t>::value || std::is_same<DstT, half>::value ||
            std::is_same<DstT, bfloat16_t>::value || std::is_same<DstT, float>::value ||
            std::is_same<DstT, int32_t>::value || std::is_same<DstT, uint32_t>::value,
        "TINSERT Vec->Vec : Unsupported data type for A3.");
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void CheckTInsertVecToVecND()
{
    using T = typename DstTileData::DType;
    static_assert(
        SrcTileData::Rows <= DstTileData::Rows, "TINSERT ND Vec->Vec : Source Rows must not exceed destination Rows.");
    static_assert(
        SrcTileData::Cols <= DstTileData::Cols, "TINSERT ND Vec->Vec : Source Cols must not exceed destination Cols.");
    static_assert(
        SrcTileData::RowStride * sizeof(T) % BLOCK_BYTE_SIZE == 0,
        "TINSERT ND Vec->Vec : SrcTile RowStride bytes must be 32-byte aligned.");
    static_assert(
        DstTileData::RowStride * sizeof(T) % BLOCK_BYTE_SIZE == 0,
        "TINSERT ND Vec->Vec : DstTile RowStride bytes must be 32-byte aligned.");
    if constexpr (!(SrcTileData::ValidRow == 1 && SrcTileData::ValidCol == 1)) {
        static_assert(
            (SrcTileData::ValidCol * sizeof(T)) % sizeof(uint16_t) == 0,
            "TINSERT ND Vec->Vec : SrcTile ValidCol bytes must be at least 2-byte aligned.");
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void CheckTInsertVecToVecNZ()
{
    using T = typename DstTileData::DType;
    constexpr uint32_t kC0Size = BLOCK_BYTE_SIZE / sizeof(T);
    static_assert(
        SrcTileData::Rows <= DstTileData::Rows, "TINSERT NZ Vec->Vec : Source Rows must not exceed destination Rows.");
    static_assert(
        SrcTileData::Cols <= DstTileData::Cols, "TINSERT NZ Vec->Vec : Source Cols must not exceed destination Cols.");
    static_assert(SrcTileData::Rows % FRACTAL_NZ_ROW == 0, "TINSERT NZ Vec->Vec : SrcTile Rows must be 16-aligned.");
    static_assert(DstTileData::Rows % FRACTAL_NZ_ROW == 0, "TINSERT NZ Vec->Vec : DstTile Rows must be 16-aligned.");
    static_assert(SrcTileData::Cols % kC0Size == 0, "TINSERT NZ Vec->Vec : SrcTile Cols must be c0Size-aligned.");
    static_assert(DstTileData::Cols % kC0Size == 0, "TINSERT NZ Vec->Vec : DstTile Cols must be c0Size-aligned.");
    if constexpr (!(SrcTileData::ValidRow == 1 && SrcTileData::ValidCol == 1)) {
        static_assert(
            (SrcTileData::ValidCol * sizeof(T)) % sizeof(uint16_t) == 0,
            "TINSERT NZ Vec->Vec : SrcTile ValidCol bytes must be at least 2-byte aligned.");
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TInsertVecToVecNDDispatch(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol)
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
        PTO_ASSERT(
            idxCol * sizeof(T) % BLOCK_BYTE_SIZE == 0,
            "TINSERT ND Vec->Vec : indexCol bytes must be 32-byte aligned (A3 limitation).");
        PTO_ASSERT(
            idxRow + SrcTileData::ValidRow <= DstTileData::Rows,
            "TINSERT ND Vec->Vec : indexRow + srcValidRow exceeds destination rows!");
        PTO_ASSERT(
            idxCol + SrcTileData::ValidCol <= DstTileData::Cols,
            "TINSERT ND Vec->Vec : indexCol + srcValidCol exceeds destination cols!");
        uint16_t validRow = static_cast<uint16_t>(src.GetValidRow());
        uint16_t validCol = static_cast<uint16_t>(src.GetValidCol());
        if constexpr ((SrcTileData::ValidCol * sizeof(T)) % BLOCK_BYTE_SIZE == 0) {
            TInsertVecToVecNDAligned<T, DstTileData, SrcTileData>(
                dst.data(), src.data(), validRow, validCol, idxRow, idxCol);
        } else {
            TInsertVecToVecNDUnaligned<T, DstTileData, SrcTileData>(
                dst.data(), src.data(), validRow, validCol, idxRow, idxCol);
        }
    }
}

template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TInsertVecToVecNZDispatch(DstTileData& dst, SrcTileData& src, uint16_t indexRow, uint16_t indexCol)
{
    using T = typename DstTileData::DType;
    constexpr uint32_t kC0Size = BLOCK_BYTE_SIZE / sizeof(T);

    uint32_t idxRow = static_cast<uint32_t>(indexRow);
    uint32_t idxCol = static_cast<uint32_t>(indexCol);
    if constexpr (SrcTileData::ValidRow == 1 && SrcTileData::ValidCol == 1) {
        PTO_ASSERT(idxRow < DstTileData::Rows, "TINSERT NZ Vec->Vec : indexRow exceeds dstRows!");
        PTO_ASSERT(idxCol < DstTileData::Cols, "TINSERT NZ Vec->Vec : indexCol exceeds dstCols!");
        TInsertVecToVecNZScalar<T, DstTileData, SrcTileData>(dst.data(), src.data(), idxRow, idxCol);
    } else {
        CheckTInsertVecToVecNZ<DstTileData, SrcTileData>();
        PTO_ASSERT(idxRow % FRACTAL_NZ_ROW == 0, "TINSERT NZ Vec->Vec : indexRow must be 16-aligned (A3 limitation).");
        PTO_ASSERT(idxCol % kC0Size == 0, "TINSERT NZ Vec->Vec : indexCol must be c0Size-aligned (A3 limitation).");
        uint16_t validRow = static_cast<uint16_t>(src.GetValidRow());
        uint16_t validCol = static_cast<uint16_t>(src.GetValidCol());
        PTO_ASSERT(
            idxRow + validRow <= DstTileData::Rows,
            "TINSERT NZ Vec->Vec : indexRow + validRow exceeds destination rows!");
        PTO_ASSERT(
            idxCol + validCol <= DstTileData::Cols,
            "TINSERT NZ Vec->Vec : indexCol + validCol exceeds destination cols!");
        if constexpr ((SrcTileData::ValidCol % kC0Size) == 0) {
            TInsertVecToVecNZAligned<T, DstTileData, SrcTileData>(
                dst.data(), src.data(), validRow, validCol, idxRow, idxCol);
        } else {
            TInsertVecToVecNZUnaligned<T, DstTileData, SrcTileData>(
                dst.data(), src.data(), validRow, validCol, idxRow, idxCol);
        }
    }
}

template <typename FpTileData>
__tf__ PTO_INTERNAL void SetFPCInsert(typename FpTileData::TileDType __in__ fp)
{
    using FpType = typename FpTileData::DType;
    __fbuf__ FpType* dstAddrFp = (__fbuf__ FpType*)__cce_get_tile_ptr(fp);
    uint64_t deqTensorAddr = ((uint64_t)dstAddrFp >> static_cast<uint64_t>(7))
                             << 8; // fpc[15:8] means Quant_PRE_ADDR, uint of 128(2^7)bytes
    set_fpc(deqTensorAddr);
}

#endif
