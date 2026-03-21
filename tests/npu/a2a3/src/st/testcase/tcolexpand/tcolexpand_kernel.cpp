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
#include <iostream>

using namespace std;
using namespace pto;

template <typename T, int src_row, int src_col, int src_validCol, int dst_row, int dst_col, int dst_validRow,
          int dst_validCol>
__global__ AICORE void runTCOLEXPAND(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynDim2Shape = Shape<1, 1, 1, -1, -1>;
    using DynDim2Stride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<T, DynDim2Shape, DynDim2Stride>;
    GlobalData srcGlobal(src, DynDim2Shape(1, src_validCol), DynDim2Stride(src_row, src_col));
    GlobalData dstGlobal(out, DynDim2Shape(dst_validRow, dst_validCol), DynDim2Stride(dst_row, dst_col));

    using srcTileData = Tile<TileType::Vec, T, src_row, src_col, BLayout::RowMajor, -1, -1>;
    using dstTileData = Tile<TileType::Vec, T, dst_row, dst_col, BLayout::RowMajor, -1, -1>;
    srcTileData srcTile(1, src_validCol);
    dstTileData dstTile(dst_validRow, dst_validCol);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, src_row * src_col * sizeof(T));

// causes issues in automode as the tile returned from the TLOAD tfcall appears unused and this tload may not finish
// before the second tload
#ifndef __PTO_AUTO__
    TLOAD(dstTile, dstGlobal);
#endif

    // 搬运数据
    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TCOLEXPAND(dstTile, srcTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int src_row, int src_col, int src_validCol, int dst_row, int dst_col, int dst_validRow,
          int dst_validCol>
void launchTCOLEXPAND(T *out, T *src, void *stream)
{
    cout << "launchTCOLEXPAND start!" << endl;

    runTCOLEXPAND<T, src_row, src_col, src_validCol, dst_row, dst_col, dst_validRow, dst_validCol>
        <<<1, nullptr, stream>>>(out, src);

    cout << "launchTCOLEXPAND end!" << endl;
}

template void launchTCOLEXPAND<int16_t, 32, 32, 8, 32, 32, 16, 8>(int16_t *out, int16_t *src, void *stream);
template void launchTCOLEXPAND<int32_t, 16, 16, 8, 24, 16, 16, 8>(int32_t *out, int32_t *src, void *stream);
template void launchTCOLEXPAND<float, 16, 16, 8, 24, 16, 16, 8>(float *out, float *src, void *stream);
template void launchTCOLEXPAND<int16_t, 8, 128, 127, 16, 128, 8, 127>(int16_t *out, int16_t *src, void *stream);
template void launchTCOLEXPAND<int32_t, 3, 64, 63, 16, 64, 15, 63>(int32_t *out, int32_t *src, void *stream);
template void launchTCOLEXPAND<float, 3, 64, 63, 16, 64, 15, 63>(float *out, float *src, void *stream);
template void launchTCOLEXPAND<int16_t, 16, 256, 256, 12, 256, 6, 256>(int16_t *out, int16_t *src, void *stream);
template void launchTCOLEXPAND<int32_t, 4, 256, 256, 16, 256, 15, 256>(int32_t *out, int32_t *src, void *stream);
template void launchTCOLEXPAND<float, 6, 64, 64, 16, 64, 15, 64>(float *out, float *src, void *stream);
template void launchTCOLEXPAND<int16_t, 16, 256, 255, 16, 256, 7, 255>(int16_t *out, int16_t *src, void *stream);
template void launchTCOLEXPAND<int32_t, 8, 256, 255, 32, 256, 31, 255>(int32_t *out, int32_t *src, void *stream);
template void launchTCOLEXPAND<float, 1, 64, 63, 1, 64, 1, 63>(float *out, float *src, void *stream);