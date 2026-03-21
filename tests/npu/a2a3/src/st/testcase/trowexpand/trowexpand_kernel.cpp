/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>
#include <iostream>

using namespace std;
using namespace pto;

template <typename T, int rows, int src_col, int src_validCol, int dst_col, int dst_validCol>
__global__ AICORE void runTROWEXPAND(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, rows, -1>;
    using DynStridDim5 = Stride<1, 1, rows, -1, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    GlobalData srcGlobal(src, DynShapeDim5(src_validCol), DynStridDim5(src_col));
    GlobalData dstGlobal(out, DynShapeDim5(dst_validCol), DynStridDim5(dst_col));
    using TileDataSrc = Tile<TileType::Vec, T, rows, src_col, BLayout::RowMajor, -1, -1>;
    using TileDataDst = Tile<TileType::Vec, T, rows, dst_col, BLayout::RowMajor, -1, -1>;

    TileDataSrc srcTile(rows, src_validCol);
    TileDataDst dstTile(rows, dst_validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, rows * src_col * sizeof(T)); // UB最大到0x40000

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWEXPAND(dstTile, srcTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int rows, int src_col, int dst_col>
__global__ AICORE void runTROWBRCB(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    constexpr bool isRowMajor = (rows == 1);
    constexpr BLayout tileLayout = isRowMajor ? BLayout::RowMajor : BLayout::ColMajor;
    constexpr Layout tensorLayout = isRowMajor ? Layout::ND : Layout::DN;

    using SrcGlobal = GlobalTensor<T, Shape<1, 1, 1, rows, src_col>, Stride<1, 1, rows, src_col, 1>, tensorLayout>;
    using DstGlobal = GlobalTensor<T, Shape<1, 1, 1, rows, dst_col>, Stride<1, 1, rows, dst_col, 1>>;
    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(out);
    using SrcTile = Tile<TileType::Vec, T, rows, src_col, tileLayout>;
    using DstTile = Tile<TileType::Vec, T, rows * src_col, dst_col>;

    SrcTile srcTile;
    DstTile dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, rows * src_col * sizeof(T));

    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TROWEXPAND(dstTile, srcTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int rows, int src_col, int src_validCol, int dst_col, int dst_validCol>
void launchTROWEXPAND(T *out, T *src, void *stream)
{
    if constexpr (dst_col * sizeof(T) == BLOCK_BYTE_SIZE) {
        runTROWBRCB<T, rows, src_col, dst_col><<<1, nullptr, stream>>>(out, src);
    } else {
        runTROWEXPAND<T, rows, src_col, src_validCol, dst_col, dst_validCol><<<1, nullptr, stream>>>(out, src);
    }
}

template void launchTROWEXPAND<uint16_t, 16, 16, 1, 512, 512>(uint16_t *out, uint16_t *src, void *stream);
template void launchTROWEXPAND<uint8_t, 16, 32, 1, 256, 256>(uint8_t *out, uint8_t *src, void *stream);
template void launchTROWEXPAND<uint32_t, 16, 8, 1, 128, 128>(uint32_t *out, uint32_t *src, void *stream);
template void launchTROWEXPAND<float, 16, 32, 1, 512, 512>(float *out, float *src, void *stream);
template void launchTROWEXPAND<uint16_t, 16, 16, 1, 256, 255>(uint16_t *out, uint16_t *src, void *stream);
template void launchTROWEXPAND<uint8_t, 16, 32, 1, 512, 511>(uint8_t *out, uint8_t *src, void *stream);
template void launchTROWEXPAND<uint32_t, 16, 8, 1, 128, 127>(uint32_t *out, uint32_t *src, void *stream);
template void launchTROWEXPAND<uint16_t, 16, 16, 1, 128, 127>(uint16_t *out, uint16_t *src, void *stream);
template void launchTROWEXPAND<uint8_t, 2, 32, 1, 64, 63>(uint8_t *out, uint8_t *src, void *stream);
template void launchTROWEXPAND<uint16_t, 4080, 1, 1, 16, 16>(uint16_t *out, uint16_t *src, void *stream);
template void launchTROWEXPAND<uint16_t, 16, 1, 1, 16, 16>(uint16_t *out, uint16_t *src, void *stream);
template void launchTROWEXPAND<uint32_t, 4080, 1, 1, 8, 8>(uint32_t *out, uint32_t *src, void *stream);
template void launchTROWEXPAND<uint32_t, 16, 1, 1, 8, 8>(uint32_t *out, uint32_t *src, void *stream);
template void launchTROWEXPAND<float, 4080, 1, 1, 8, 8>(float *out, float *src, void *stream);
template void launchTROWEXPAND<float, 16, 1, 1, 8, 8>(float *out, float *src, void *stream);
template void launchTROWEXPAND<int16_t, 16, 16, 1, 512, 512>(int16_t *out, int16_t *src, void *stream);
template void launchTROWEXPAND<int8_t, 16, 32, 1, 256, 256>(int8_t *out, int8_t *src, void *stream);
template void launchTROWEXPAND<int32_t, 16, 8, 1, 128, 128>(int32_t *out, int32_t *src, void *stream);
template void launchTROWEXPAND<int16_t, 16, 16, 1, 256, 255>(int16_t *out, int16_t *src, void *stream);
template void launchTROWEXPAND<int8_t, 16, 32, 1, 512, 511>(int8_t *out, int8_t *src, void *stream);
template void launchTROWEXPAND<int32_t, 16, 8, 1, 128, 127>(int32_t *out, int32_t *src, void *stream);
template void launchTROWEXPAND<int16_t, 16, 16, 1, 128, 127>(int16_t *out, int16_t *src, void *stream);
template void launchTROWEXPAND<int8_t, 2, 32, 1, 64, 63>(int8_t *out, int8_t *src, void *stream);