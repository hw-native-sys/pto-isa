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
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

template <typename T, int validRow, int validCol, int Row, int Col, bool src0eqdst>
__global__ AICORE void runTRowExpandSub(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    constexpr uint16_t src1Row = ((validRow * sizeof(T) + 31) / 32) * (32 / sizeof(T));
    using GlobalDataDst = GlobalTensor<T, Shape<1, 1, 1, Row, Col>, Stride<1, 1, 1, Col, 1>>;
    using TileDataDst = Tile<TileType::Vec, T, Row, Col, BLayout::RowMajor, -1, -1>;
    using GlobalDataSrc1 = GlobalTensor<T, Shape<1, 1, 1, src1Row, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;
    using TileDataSrc1 = Tile<TileType::Vec, T, src1Row, 1, BLayout::ColMajor, -1, -1>;
    TileDataDst src0Tile(validRow, validCol);
    TileDataSrc1 src1Tile(validRow, 1);
    TileDataDst dstTile(validRow, validCol);
    size_t size = Row * Col * sizeof(T);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x0);
    TASSIGN(src1Tile, size);

    GlobalDataDst src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataDst dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    if constexpr (src0eqdst) {
        TROWEXPANDSUB(dstTile, src0Tile, src1Tile);
    } else {
        TROWEXPANDSUB(dstTile, src1Tile, src0Tile);
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    out = dstGlobal.data();
}

template <typename T, int validRow, int validCol, int Row, int Col, bool src0eqdst>
__global__ AICORE void runTRowExpandSub2(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    constexpr uint16_t src1Row = ((validRow * sizeof(T) + 31) / 32) * (32 / sizeof(T));
    constexpr uint16_t src1Col = 32 / sizeof(T);
    using GlobalDataDst = GlobalTensor<T, Shape<1, 1, 1, Row, Col>, Stride<1, 1, 1, Col, 1>>;
    using TileDataDst = Tile<TileType::Vec, T, Row, Col, BLayout::RowMajor, -1, -1>;
    using GlobalDataSrc1 = GlobalTensor<T, Shape<1, 1, 1, src1Row, src1Col>, Stride<1, 1, 1, src1Col, 1>>;
    using TileDataSrc1 = Tile<TileType::Vec, T, src1Row, src1Col, BLayout::RowMajor, -1, -1>;
    TileDataDst src0Tile(validRow, validCol);
    TileDataDst dstTile(validRow, validCol);
    TileDataSrc1 src1Tile(validRow, src1Col);
    size_t size = Row * Col * sizeof(T);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x0);
    TASSIGN(src1Tile, size);

    GlobalDataDst src0Global(src0);
    GlobalDataDst dstGlobal(out);
    GlobalDataSrc1 src1Global(src1);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    if constexpr (src0eqdst) {
        TROWEXPANDSUB(dstTile, src0Tile, src1Tile);
    } else {
        TROWEXPANDSUB(dstTile, src1Tile, src0Tile);
    }
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    out = dstGlobal.data();
}

template <typename T, int validRow, int validCol, int Row, int Col, bool src0eqdst>
__global__ AICORE void runTRowExpandSub3(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    constexpr uint16_t src1Row = ((validRow * sizeof(T) + 31) / 32) * (32 / sizeof(T));
    using GlobalDataDst = GlobalTensor<T, Shape<1, 1, 1, Row, Col>, Stride<1, 1, 1, Col, 1>>;
    using TileDataDst = Tile<TileType::Vec, T, Row, Col, BLayout::RowMajor, -1, -1>;
    using GlobalDataSrc1 = GlobalTensor<T, Shape<1, 1, 1, src1Row, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;
    using TileDataSrc1 = Tile<TileType::Vec, T, src1Row, 1, BLayout::ColMajor, -1, -1>;
    using GlobalDataTmp = GlobalTensor<T, Shape<1, 1, 1, src1Row, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;
    using TileDataTmp = Tile<TileType::Vec, T, src1Row, 1, BLayout::ColMajor, -1, -1>;
    TileDataDst src0Tile(validRow, validCol);
    TileDataSrc1 src1Tile(validRow, 1);
    TileDataDst dstTile(validRow, validCol);
    // tmp只取初始地址，其他值不生效，分配空间>=src1Row*32B或>=8KB即可，行数<256时可与src1共用空间
    TileDataTmp tmpTile(validRow, validCol);
    size_t size = Row * Col * sizeof(T);
    size_t size1 = src1Row * sizeof(T);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(dstTile, 0x0);
    TASSIGN(src1Tile, size);
    TASSIGN(tmpTile, size + size1);

    GlobalDataDst src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataDst dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if constexpr (src0eqdst) {
        TROWEXPANDSUB(dstTile, src0Tile, src1Tile, tmpTile);
    } else {
        TROWEXPANDSUB(dstTile, src1Tile, src0Tile, tmpTile);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    out = dstGlobal.data();
}

template <typename T, int validRow, int validCol, int Row, int Col, bool src0eqdst>
void launchTRowExpandSub(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTRowExpandSub<half, validRow, validCol, Row, Col, src0eqdst>
            <<<1, nullptr, stream>>>((half *)(out), (half *)(src0), (half *)(src1));
    else
        runTRowExpandSub<T, validRow, validCol, Row, Col, src0eqdst><<<1, nullptr, stream>>>(out, src0, src1);
}

template <typename T, int validRow, int validCol, int Row, int Col, bool src0eqdst>
void launchTRowExpandSub2(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTRowExpandSub2<half, validRow, validCol, Row, Col, src0eqdst>
            <<<1, nullptr, stream>>>((half *)(out), (half *)(src0), (half *)(src1));
    else
        runTRowExpandSub2<T, validRow, validCol, Row, Col, src0eqdst><<<1, nullptr, stream>>>(out, src0, src1);
}

template <typename T, int validRow, int validCol, int Row, int Col, bool src0eqdst>
void launchTRowExpandSub3(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTRowExpandSub3<half, validRow, validCol, Row, Col, src0eqdst>
            <<<1, nullptr, stream>>>((half *)(out), (half *)(src0), (half *)(src1));
    else
        runTRowExpandSub3<T, validRow, validCol, Row, Col, src0eqdst><<<1, nullptr, stream>>>(out, src0, src1);
}

template void launchTRowExpandSub<float, 16, 16, 16, 16, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandSub<float, 16, 16, 32, 32, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandSub<aclFloat16, 16, 16, 16, 16, true>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1,
                                                                    void *stream);
template void launchTRowExpandSub<aclFloat16, 16, 16, 32, 32, true>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1,
                                                                    void *stream);
template void launchTRowExpandSub<float, 1, 16384, 1, 16384, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandSub<float, 2048, 1, 2048, 8, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandSub2<float, 16, 16, 16, 16, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandSub2<float, 16, 16, 32, 32, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandSub2<aclFloat16, 16, 16, 16, 16, true>(aclFloat16 *out, aclFloat16 *src0,
                                                                     aclFloat16 *src1, void *stream);
template void launchTRowExpandSub2<aclFloat16, 16, 16, 32, 32, true>(aclFloat16 *out, aclFloat16 *src0,
                                                                     aclFloat16 *src1, void *stream);
template void launchTRowExpandSub2<float, 1, 16384, 1, 16384, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandSub2<float, 2048, 1, 2048, 8, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandSub<float, 16, 16, 16, 16, false>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandSub2<float, 16, 16, 16, 16, false>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandSub3<float, 16, 16, 32, 32, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandSub3<aclFloat16, 16, 16, 16, 16, true>(aclFloat16 *out, aclFloat16 *src0,
                                                                     aclFloat16 *src1, void *stream);
template void launchTRowExpandSub3<float, 1, 16384, 1, 16384, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandSub3<float, 2048, 1, 2048, 8, true>(float *out, float *src0, float *src1, void *stream);
template void launchTRowExpandSub<int32_t, 16, 16, 16, 16, true>(int32_t *out, int32_t *src0, int32_t *src1,
                                                                 void *stream);
template void launchTRowExpandSub2<int32_t, 16, 16, 16, 16, true>(int32_t *out, int32_t *src0, int32_t *src1,
                                                                  void *stream);
template void launchTRowExpandSub<int16_t, 16, 16, 16, 16, true>(int16_t *out, int16_t *src0, int16_t *src1,
                                                                 void *stream);
template void launchTRowExpandSub2<int16_t, 16, 16, 16, 16, true>(int16_t *out, int16_t *src0, int16_t *src1,
                                                                  void *stream);
