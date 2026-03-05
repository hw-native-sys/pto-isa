/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TINSERT_CUSTOM_HPP
#define TINSERT_CUSTOM_HPP

#ifndef TINSERT_MODE_DEFINED
#define TINSERT_MODE_DEFINED
enum class TInsertMode : uint8_t
{
    NZ = 0,
    NZ_PLUS_1 = 1,
    SPLIT2_NZ_PLUS_1 = 2,
    SPLIT4_NZ_PLUS_1 = 3,
};
#endif

namespace pto {

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
    // NZ layout: each column block (c0Size cols) has dstRow * c0Size elements
    // Column offset: (indexCol / c0Size) * dstRow * c0Size to reach the right column block
    // Row offset: indexRow * c0Size within that column block
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

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TInsertSplit2Impl(typename DstTileData::TileDType __out__ dst,
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
    uint16_t halfBurstNum = totalBurstNum >> 1;

    // NZ layout: each column block (c0Size cols) has DstTileData::Rows * c0Size elements
    // Column offset: (indexCol / c0Size) * DstTileData::Rows * c0Size to reach the right column block
    // Row offset: indexRow * c0Size within that column block
    uint32_t colBlockOffset = (indexCol / c0Size) * DstTileData::Rows * c0Size;
    uint32_t rowOffset = indexRow * c0Size + (indexCol % c0Size);
    uint32_t dstOffset = colBlockOffset + rowOffset;

    // srcGap=1 for NZ+1 source layout, dstGap=0 for plain NZ destination
    __cbuf__ T *dstAddr0 = dstAddr + dstOffset;
    copy_ubuf_to_cbuf(dstAddr0, srcAddr, 0, halfBurstNum, burstLen, 1, 0);

    // Source offset accounts for NZ+1 layout: (burstLen + 1) per column
    uint32_t srcBlockOffset = halfBurstNum * (burstLen + 1) * BLOCK_BYTE_SIZE / typeSize;
    // Destination offset for second half: halfBurstNum column blocks in NZ format
    uint32_t dstBlockOffset = halfBurstNum * DstTileData::Rows * c0Size;
    __ubuf__ T *srcAddr2 = srcAddr + srcBlockOffset;
    __cbuf__ T *dstAddr2 = dstAddr0 + dstBlockOffset;

    copy_ubuf_to_cbuf(dstAddr2, srcAddr2, 0, halfBurstNum, burstLen, 1, 0);
}

template <typename T, typename DstTileData, typename SrcTileData>
__tf__ AICORE void TInsertSplit4Impl(typename DstTileData::TileDType __out__ dst,
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
    uint16_t quarterBurstNum = totalBurstNum >> 2;
    // srcBlockSize accounts for NZ+1 layout: (burstLen + 1) per column
    uint32_t srcBlockSize = (burstLen + 1) * BLOCK_BYTE_SIZE / typeSize;
    // dstBlockSize for NZ layout: DstTileData::Rows * c0Size per column block
    uint32_t dstBlockSize = DstTileData::Rows * c0Size;

    // NZ layout: each column block (c0Size cols) has DstTileData::Rows * c0Size elements
    // Column offset: (indexCol / c0Size) * DstTileData::Rows * c0Size to reach the right column block
    // Row offset: indexRow * c0Size within that column block
    uint32_t colBlockOffset = (indexCol / c0Size) * DstTileData::Rows * c0Size;
    uint32_t rowOffset = indexRow * c0Size + (indexCol % c0Size);
    uint32_t dstOffset = colBlockOffset + rowOffset;

    // srcGap=1 for NZ+1 source layout, dstGap=0 for plain NZ destination
    __cbuf__ T *dstAddr0 = dstAddr + dstOffset;
    copy_ubuf_to_cbuf(dstAddr0, srcAddr, 0, quarterBurstNum, burstLen, 1, 0);

    __ubuf__ T *srcQ1 = srcAddr + quarterBurstNum * srcBlockSize;
    __cbuf__ T *dstQ1 = dstAddr0 + quarterBurstNum * dstBlockSize;
    copy_ubuf_to_cbuf(dstQ1, srcQ1, 0, quarterBurstNum, burstLen, 1, 0);

    __ubuf__ T *srcQ2 = srcAddr + 2 * quarterBurstNum * srcBlockSize;
    __cbuf__ T *dstQ2 = dstAddr0 + 2 * quarterBurstNum * dstBlockSize;
    copy_ubuf_to_cbuf(dstQ2, srcQ2, 0, quarterBurstNum, burstLen, 1, 0);

    __ubuf__ T *srcQ3 = srcAddr + 3 * quarterBurstNum * srcBlockSize;
    __cbuf__ T *dstQ3 = dstAddr0 + 3 * quarterBurstNum * dstBlockSize;
    copy_ubuf_to_cbuf(dstQ3, srcQ3, 0, quarterBurstNum, burstLen, 1, 0);
}

template <TInsertMode mode = TInsertMode::NZ, typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TINSERT_CUSTOM(DstTileData &dst, SrcTileData &src, uint32_t indexRow = 0, uint32_t indexCol = 0)
{
    using T = typename SrcTileData::DType;
    static_assert(DstTileData::Loc == TileType::Mat, "TINSERT CUSTOM : Destination must be Mat tile (L1/cbuf)");
    static_assert(SrcTileData::Loc == TileType::Vec, "TINSERT CUSTOM : Source must be Vec tile (UB/ubuf)");
    static_assert(std::is_same<typename DstTileData::DType, typename SrcTileData::DType>::value,
                  "TINSERT CUSTOM : Source and destination data types must match");
    static_assert(!SrcTileData::isRowMajor && (SrcTileData::SFractal == SLayout::RowMajor),
                  "TINSERT CUSTOM : Source must be NZ format (column-major, RowMajor fractal)");
    static_assert((std::is_same<T, half>::value) || (std::is_same<T, bfloat16_t>::value) ||
                      (std::is_same<T, float>::value) || (std::is_same<T, int32_t>::value) ||
                      (std::is_same<T, float8_e4m3_t>::value) || (std::is_same<T, float8_e5m2_t>::value) ||
                      (std::is_same<T, hifloat8_t>::value) || (std::is_same<T, int8_t>::value),
                  "TINSERT CUSTOM : Dst and src must be "
                  "float/int32_t/half/bfloat16_t/int8_t/float8_e4m3_t/float8_e5m2_t/hifloat8_t.");
    PTO_ASSERT(indexRow + SrcTileData::Rows <= DstTileData::Rows,
               "TINSERT CUSTOM : The sum of indexRow and srcRow should be less than dstRow!");
    PTO_ASSERT(indexCol + SrcTileData::Cols <= DstTileData::Cols,
               "TINSERT CUSTOM : The sum of indexCol and srcCol should be less than dstCol!");

    uint16_t validRow = static_cast<uint16_t>(src.GetValidRow());
    uint16_t validCol = static_cast<uint16_t>(src.GetValidCol());
    uint16_t dstRow = static_cast<uint16_t>(dst.GetValidRow());

    if constexpr (mode == TInsertMode::SPLIT2_NZ_PLUS_1) {
        TInsertSplit2Impl<T, DstTileData, SrcTileData>(dst.data(), src.data(), mode, validRow, validCol, indexRow,
                                                       indexCol);
    } else if constexpr (mode == TInsertMode::SPLIT4_NZ_PLUS_1) {
        TInsertSplit4Impl<T, DstTileData, SrcTileData>(dst.data(), src.data(), mode, validRow, validCol, indexRow,
                                                       indexCol);
    } else {
        TInsertImpl<T, DstTileData, SrcTileData>(dst.data(), src.data(), mode, validRow, validCol, dstRow, indexRow,
                                                 indexCol);
    }
}

} // namespace pto

#endif // TINSERT_CUSTOM_HPP
