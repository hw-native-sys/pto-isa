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
#include <acl/acl.h>
#include <iostream>

using namespace std;
using namespace pto;

template <typename T, int cols, int src_row, int src_validRow>
__global__ AICORE void runTCOLMIN(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(src_validRow, cols), DynDim2Stride(src_row, cols));
    GlobalData dstGlobal(out, DynDim2Shape(1, cols), DynDim2Stride(1, cols));

    using srcTileData = Tile<TileType::Vec, T, src_row, cols, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, 1, cols, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(src_validRow, cols);
    dstTileData dstTile(1, cols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x28000);

// causes issues in automode as the tile returned from the TLOAD tfcall appears unused and this tload may not finish
// before the second tload
#ifndef __PTO_AUTO__
    // 清除脏数据
    TLOAD(dstTile, dstGlobal);
#endif

    // 搬运数据
    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TCOLMIN(dstTile, srcTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int cols, int src_row, int src_validRow>
void launchTCOLMIN(T *out, T *src, void *stream)
{
    cout << "launchTCOLMIN start!" << endl;

    runTCOLMIN<T, cols, src_row, src_validRow><<<1, nullptr, stream>>>(out, src);

    cout << "launchTCOLMIN end!" << endl;
}

template void launchTCOLMIN<int16_t, 16, 16, 8>(int16_t *out, int16_t *src, void *stream);
template void launchTCOLMIN<int32_t, 16, 16, 8>(int32_t *out, int32_t *src, void *stream);
template void launchTCOLMIN<float, 16, 16, 8>(float *out, float *src, void *stream);
template void launchTCOLMIN<int16_t, 128, 16, 8>(int16_t *out, int16_t *src, void *stream);
template void launchTCOLMIN<int32_t, 64, 16, 8>(int32_t *out, int32_t *src, void *stream);
template void launchTCOLMIN<float, 64, 16, 8>(float *out, float *src, void *stream);
template void launchTCOLMIN<int16_t, 512, 16, 8>(int16_t *out, int16_t *src, void *stream);
template void launchTCOLMIN<int32_t, 256, 16, 8>(int32_t *out, int32_t *src, void *stream);
template void launchTCOLMIN<float, 256, 16, 8>(float *out, float *src, void *stream);
template void launchTCOLMIN<int16_t, 512, 16, 7>(int16_t *out, int16_t *src, void *stream);
template void launchTCOLMIN<int32_t, 256, 32, 31>(int32_t *out, int32_t *src, void *stream);
template void launchTCOLMIN<float, 256, 32, 31>(float *out, float *src, void *stream);
template void launchTCOLMIN<float, 256, 16, 1>(float *out, float *src, void *stream);