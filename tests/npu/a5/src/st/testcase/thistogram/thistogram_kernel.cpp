/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Define missing constants needed by THistogram.hpp (they live locally in TTopK.hpp)
namespace pto {
constexpr unsigned ElemPerRepeatB8 = 256;  // REPEAT_BYTE / sizeof(uint8_t)
constexpr unsigned ElemPerRepeatB16 = 128; // REPEAT_BYTE / sizeof(uint16_t)
} // namespace pto

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

#define PTO_DIV_ROUNDUP(x, y) (((x) + (y)-1) / (y))
#define PTO_CEIL(x, y) (PTO_DIV_ROUNDUP(x, y) * (y))

// ---------------------------------------------------------------------------
// uint16 kernel
// ---------------------------------------------------------------------------
template <int validRows, int validCols, HistByte byte>
__global__ AICORE void runTHistogram(__gm__ uint16_t *src, __gm__ uint32_t __out__ *dst, __gm__ uint8_t *idx)
{
    constexpr uint16_t alignedSrcCol = PTO_CEIL(validCols, BLOCK_BYTE_SIZE / sizeof(uint16_t));
    constexpr uint16_t alignedIdxBytes = PTO_CEIL(validRows * sizeof(uint8_t), BLOCK_BYTE_SIZE); // Align idx to 32B
    constexpr bool isMSB = (byte == HistByte::BYTE_1);

    using GlobalDataSrc =
        GlobalTensor<uint16_t, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using GlobalDataDst = GlobalTensor<uint32_t, Shape<1, 1, 1, validRows, 256>, pto::Stride<1, 1, 1, 256, 1>>;
    using GlobalDataIdx = GlobalTensor<uint8_t, Shape<1, 1, 1, validRows, 1>, pto::Stride<1, 1, 1, 1, 1>, Layout::DN>;

    using TileDataSrc = Tile<TileType::Vec, uint16_t, validRows, alignedSrcCol, BLayout::RowMajor, -1, -1>;
    using TileDataDst = Tile<TileType::Vec, uint32_t, validRows, 256, BLayout::RowMajor, -1, -1>;
    using TileDataIdx = Tile<TileType::Vec, uint8_t, alignedIdxBytes, 1, BLayout::ColMajor, -1, -1>;

    TileDataSrc srcTile(validRows, validCols);
    TileDataDst dstTile(validRows, 256);
    TileDataIdx idxTile(validRows, 1);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);
    TASSIGN(idxTile, 0x30000);

    GlobalDataSrc srcGlobal(src);
    GlobalDataDst dstGlobal(dst);
    GlobalDataIdx idxGlobal(idx);

    TLOAD(srcTile, srcGlobal);
    if constexpr (!isMSB) {
        TLOAD(idxTile, idxGlobal);
    }

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    THISTOGRAM<byte>(dstTile, srcTile, idxTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);
}

template <int validRows, int validCols, HistByte byte>
void LaunchTHistogramU16(uint16_t *src, uint32_t *dst, void *stream, uint8_t *idx)
{
    runTHistogram<validRows, validCols, byte><<<1, nullptr, stream>>>(src, dst, idx);
}

template void LaunchTHistogramU16<2, 128, HistByte::BYTE_1>(uint16_t *src, uint32_t *dst, void *stream, uint8_t *idx);
template void LaunchTHistogramU16<4, 64, HistByte::BYTE_1>(uint16_t *src, uint32_t *dst, void *stream, uint8_t *idx);
template void LaunchTHistogramU16<8, 128, HistByte::BYTE_1>(uint16_t *src, uint32_t *dst, void *stream, uint8_t *idx);
template void LaunchTHistogramU16<1, 256, HistByte::BYTE_1>(uint16_t *src, uint32_t *dst, void *stream, uint8_t *idx);
template void LaunchTHistogramU16<4, 256, HistByte::BYTE_1>(uint16_t *src, uint32_t *dst, void *stream, uint8_t *idx);
template void LaunchTHistogramU16<2, 100, HistByte::BYTE_1>(uint16_t *src, uint32_t *dst, void *stream, uint8_t *idx);
template void LaunchTHistogramU16<2, 128, HistByte::BYTE_0>(uint16_t *src, uint32_t *dst, void *stream, uint8_t *idx);
template void LaunchTHistogramU16<4, 64, HistByte::BYTE_0>(uint16_t *src, uint32_t *dst, void *stream, uint8_t *idx);
template void LaunchTHistogramU16<8, 128, HistByte::BYTE_0>(uint16_t *src, uint32_t *dst, void *stream, uint8_t *idx);
template void LaunchTHistogramU16<1, 256, HistByte::BYTE_0>(uint16_t *src, uint32_t *dst, void *stream, uint8_t *idx);
template void LaunchTHistogramU16<4, 256, HistByte::BYTE_0>(uint16_t *src, uint32_t *dst, void *stream, uint8_t *idx);
template void LaunchTHistogramU16<2, 100, HistByte::BYTE_0>(uint16_t *src, uint32_t *dst, void *stream, uint8_t *idx);

// ---------------------------------------------------------------------------
// uint32 kernel
// ---------------------------------------------------------------------------
template <int validRows, int validCols, HistByte byte>
__global__ AICORE void runTHistogramU32(__gm__ uint32_t *src, __gm__ uint32_t __out__ *dst, __gm__ uint8_t *idx)
{
    // Use uint8-aligned columns for both src and idx so TileIdx::Cols == TileSrc::Cols
    constexpr uint16_t alignedCol = PTO_CEIL(validCols, BLOCK_BYTE_SIZE / sizeof(uint8_t));
    constexpr int byteVal = static_cast<int>(byte);
    // idx tile shape: (3 - byteVal, validCols) for byte < 3, else a dummy (1,1)
    constexpr int numIdxRows = 3 - byteVal;
    constexpr int idxRows = numIdxRows > 0 ? numIdxRows : 1;
    constexpr int idxCols = numIdxRows > 0 ? validCols : 1;
    constexpr uint16_t alignedIdxCol = numIdxRows > 0 ? alignedCol : PTO_CEIL(1, BLOCK_BYTE_SIZE / sizeof(uint8_t));

    using GlobalDataSrc =
        GlobalTensor<uint32_t, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using GlobalDataDst = GlobalTensor<uint32_t, Shape<1, 1, 1, validRows, 256>, pto::Stride<1, 1, 1, 256, 1>>;
    using GlobalDataIdx = GlobalTensor<uint8_t, Shape<1, 1, 1, idxRows, idxCols>, pto::Stride<1, 1, 1, idxCols, 1>>;

    using TileDataSrc = Tile<TileType::Vec, uint32_t, validRows, alignedCol, BLayout::RowMajor, -1, -1>;
    using TileDataDst = Tile<TileType::Vec, uint32_t, validRows, 256, BLayout::RowMajor, -1, -1>;
    using TileDataIdx = Tile<TileType::Vec, uint8_t, idxRows, alignedIdxCol, BLayout::RowMajor, -1, -1>;

    TileDataSrc srcTile(validRows, validCols);
    TileDataDst dstTile(validRows, 256);
    TileDataIdx idxTile(idxRows, idxCols);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);
    TASSIGN(idxTile, 0x30000);

    GlobalDataSrc srcGlobal(src);
    GlobalDataDst dstGlobal(dst);
    GlobalDataIdx idxGlobal(idx);

    TLOAD(srcTile, srcGlobal);
    if constexpr (byte != HistByte::BYTE_3) {
        TLOAD(idxTile, idxGlobal);
    }

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    THISTOGRAM<byte>(dstTile, srcTile, idxTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);
}

template <int validRows, int validCols, HistByte byte>
void LaunchTHistogramU32(uint32_t *src, uint32_t *dst, void *stream, uint8_t *idx)
{
    runTHistogramU32<validRows, validCols, byte><<<1, nullptr, stream>>>(src, dst, idx);
}

// BYTE_3: histogram of byte3 (MSB), no filtering
template void LaunchTHistogramU32<1, 128, HistByte::BYTE_3>(uint32_t *, uint32_t *, void *, uint8_t *);
template void LaunchTHistogramU32<1, 256, HistByte::BYTE_3>(uint32_t *, uint32_t *, void *, uint8_t *);
template void LaunchTHistogramU32<2, 128, HistByte::BYTE_3>(uint32_t *, uint32_t *, void *, uint8_t *);
template void LaunchTHistogramU32<2, 4096, HistByte::BYTE_3>(uint32_t *, uint32_t *, void *, uint8_t *);
template void LaunchTHistogramU32<4, 4096, HistByte::BYTE_3>(uint32_t *, uint32_t *, void *, uint8_t *);
template void LaunchTHistogramU32<2, 192, HistByte::BYTE_3>(uint32_t *, uint32_t *, void *, uint8_t *);
template void LaunchTHistogramU32<6, 912, HistByte::BYTE_3>(uint32_t *, uint32_t *, void *, uint8_t *);
// BYTE_2: histogram of byte2, filtered by byte3
template void LaunchTHistogramU32<1, 128, HistByte::BYTE_2>(uint32_t *, uint32_t *, void *, uint8_t *);
template void LaunchTHistogramU32<1, 256, HistByte::BYTE_2>(uint32_t *, uint32_t *, void *, uint8_t *);
template void LaunchTHistogramU32<2, 128, HistByte::BYTE_2>(uint32_t *, uint32_t *, void *, uint8_t *);
template void LaunchTHistogramU32<2, 4096, HistByte::BYTE_2>(uint32_t *, uint32_t *, void *, uint8_t *);
template void LaunchTHistogramU32<4, 4096, HistByte::BYTE_2>(uint32_t *, uint32_t *, void *, uint8_t *);
template void LaunchTHistogramU32<2, 192, HistByte::BYTE_2>(uint32_t *, uint32_t *, void *, uint8_t *);
template void LaunchTHistogramU32<6, 912, HistByte::BYTE_2>(uint32_t *, uint32_t *, void *, uint8_t *);
// BYTE_1: histogram of byte1, filtered by byte3 & byte2
template void LaunchTHistogramU32<1, 128, HistByte::BYTE_1>(uint32_t *, uint32_t *, void *, uint8_t *);
template void LaunchTHistogramU32<1, 256, HistByte::BYTE_1>(uint32_t *, uint32_t *, void *, uint8_t *);
template void LaunchTHistogramU32<2, 4096, HistByte::BYTE_1>(uint32_t *, uint32_t *, void *, uint8_t *);
template void LaunchTHistogramU32<2, 192, HistByte::BYTE_1>(uint32_t *, uint32_t *, void *, uint8_t *);
template void LaunchTHistogramU32<6, 912, HistByte::BYTE_1>(uint32_t *, uint32_t *, void *, uint8_t *);
// BYTE_0: histogram of byte0 (LSB), filtered by all upper bytes
template void LaunchTHistogramU32<1, 128, HistByte::BYTE_0>(uint32_t *, uint32_t *, void *, uint8_t *);
template void LaunchTHistogramU32<1, 256, HistByte::BYTE_0>(uint32_t *, uint32_t *, void *, uint8_t *);
template void LaunchTHistogramU32<2, 4096, HistByte::BYTE_0>(uint32_t *, uint32_t *, void *, uint8_t *);
template void LaunchTHistogramU32<2, 192, HistByte::BYTE_0>(uint32_t *, uint32_t *, void *, uint8_t *);
template void LaunchTHistogramU32<6, 912, HistByte::BYTE_0>(uint32_t *, uint32_t *, void *, uint8_t *);
