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

using namespace pto;

template <typename T0, typename T1, int kGRows, int kGCols, int kTRows, int kTCols, int validRow, int validCol>
__global__ AICORE void runTSort32(__gm__ T0 __out__ *out, __gm__ T0 __in__ *src, __gm__ T1 __in__ *idx)
{
    const int totalByte = 8;
    const int totalNum = totalByte / sizeof(T0);
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T0, DynDim2Shape, DynDim2Stride>;
    using GlobalDataIdx = GlobalTensor<T1, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(kGRows, kGCols), DynDim2Stride(kGRows, kGCols));
    GlobalDataIdx idxGlobal(idx, DynDim2Shape(kGRows, kGCols), DynDim2Stride(kGRows, kGCols));
    GlobalData dstGlobal(out, DynDim2Shape(kGRows, kGCols * totalNum), DynDim2Stride(kGRows, kGCols * totalNum));

    using TileDataSrc = Tile<TileType::Vec, T0, kTRows, kTCols, BLayout::RowMajor, -1, -1>;
    using TileDataIdx = Tile<TileType::Vec, T1, kTRows, kTCols, BLayout::RowMajor, -1, -1>;
    using TileDataDst = Tile<TileType::Vec, T0, kTRows, kTCols * totalNum, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<TileType::Vec, T0, 1, 32, BLayout::RowMajor>;
    TileDataSrc srcTile(validRow, validCol);
    TileDataIdx idxTile(validRow, validCol);
    TileDataDst dstTile(validRow, validCol * totalNum);
    TmpTileData tmpTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(idxTile, kTRows * kTCols * sizeof(T0));
    TASSIGN(dstTile, kTRows * kTCols * sizeof(T0) + kTRows * kTCols * sizeof(T1));
    TASSIGN(tmpTile, 2 * kTRows * kTCols * sizeof(T0) + kTRows * kTCols * sizeof(T1));

    TLOAD(srcTile, srcGlobal);
    TLOAD(idxTile, idxGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TSORT32(dstTile, srcTile, idxTile, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T0, typename T1, int kGRows, int kGCols, int kTRows, int kTCols, int validRow, int validCol>
void launchTSort32(T0 *out, T0 *src, T1 *idx, aclrtStream stream)
{
    if constexpr (std::is_same_v<T0, aclFloat16>) {
        runTSort32<half, uint32_t, kGRows, kGCols, kTRows, kTCols, validRow, validCol>((half *)(out), (half *)(src),
                                                                                       idx);
    } else {
        runTSort32<T0, T1, kGRows, kGCols, kTRows, kTCols, validRow, validCol>(out, src, idx);
    }
}

template void launchTSort32<aclFloat16, uint32_t, 16, 16, 16, 16, 16, 16>(aclFloat16 *out, aclFloat16 *src,
                                                                          uint32_t *idx, aclrtStream stream);
template void launchTSort32<float, uint32_t, 8, 32, 8, 32, 8, 32>(float *out, float *src, uint32_t *idx,
                                                                  aclrtStream stream);
template void launchTSort32<float, uint32_t, 7, 32, 7, 32, 7, 32>(float *out, float *src, uint32_t *idx,
                                                                  aclrtStream stream);
template void launchTSort32<aclFloat16, uint32_t, 32, 16, 32, 16, 32, 16>(aclFloat16 *out, aclFloat16 *src,
                                                                          uint32_t *idx, aclrtStream stream);
