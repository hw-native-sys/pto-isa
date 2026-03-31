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
#include "tci_common.h"
#include "acl/acl.h"

using namespace std;
using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int descending, int mode>
__global__ AICORE void runTCI(__gm__ T __out__ *out, T start)
{
    using DynShapeDim5_dst = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5_dst = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData_dst = GlobalTensor<T, DynShapeDim5_dst, DynStridDim5_dst>;

    constexpr int dst_row = kGRows_;
    constexpr int dst_col = kGCols_;

    using TileData_dst = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData_dst dstTile(dst_row, dst_col);

    // A3 ub size 192kB, 0x30000
    TASSIGN(dstTile, 0x0);

    GlobalData_dst dstGlobal(out);

    if (mode == 0) {
        TCI<TileData_dst, T, descending>(dstTile, start);
    } else {
        using TileData_tmp = Tile<TileType::Vec, float, 1, 512, BLayout::RowMajor, 1, 512>; // 2KB tmp tile
        TileData_tmp tmpTile;
        TASSIGN(tmpTile, 0x20000);
        TCI<TileData_dst, TileData_tmp, T, descending>(dstTile, start, tmpTile);
    }

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <uint32_t GROW, uint32_t GCOL, uint32_t TROW, uint32_t TCOL, uint32_t descending, uint32_t mode>
void launchTCI_demo_b32(int32_t *out, void *stream)
{
    runTCI<int32_t, GROW, GCOL, TROW, TCOL, descending, mode><<<1, nullptr, stream>>>(out, 0);
}

template void launchTCI_demo_b32<1, 128, 1, 128, 0, 0>(int32_t *out, void *stream);
template void launchTCI_demo_b32<1, 128, 1, 128, 1, 0>(int32_t *out, void *stream);
template void launchTCI_demo_b32<1, 600, 1, 600, 0, 0>(int32_t *out, void *stream);
template void launchTCI_demo_b32<1, 600, 1, 600, 1, 0>(int32_t *out, void *stream);
template void launchTCI_demo_b32<1, 32, 1, 32, 0, 0>(int32_t *out, void *stream);
template void launchTCI_demo_b32<1, 32, 1, 32, 1, 0>(int32_t *out, void *stream);
template void launchTCI_demo_b32<1, 2000, 1, 2000, 0, 0>(int32_t *out, void *stream);
template void launchTCI_demo_b32<1, 2000, 1, 2000, 1, 0>(int32_t *out, void *stream);
template void launchTCI_demo_b32<1, 128, 1, 128, 0, 1>(int32_t *out, void *stream);
template void launchTCI_demo_b32<1, 128, 1, 128, 1, 1>(int32_t *out, void *stream);
template void launchTCI_demo_b32<1, 32, 1, 32, 0, 1>(int32_t *out, void *stream);
template void launchTCI_demo_b32<1, 32, 1, 32, 1, 1>(int32_t *out, void *stream);

template <uint32_t GROW, uint32_t GCOL, uint32_t TROW, uint32_t TCOL, uint32_t descending, uint32_t mode>
void launchTCI_demo_b16(int16_t *out, void *stream)
{
    runTCI<int16_t, GROW, GCOL, TROW, TCOL, descending, mode><<<1, nullptr, stream>>>(out, 0);
}

template void launchTCI_demo_b16<1, 256, 1, 256, 0, 0>(int16_t *out, void *stream);
template void launchTCI_demo_b16<1, 256, 1, 256, 1, 0>(int16_t *out, void *stream);
template void launchTCI_demo_b16<1, 800, 1, 800, 0, 0>(int16_t *out, void *stream);
template void launchTCI_demo_b16<1, 800, 1, 800, 1, 0>(int16_t *out, void *stream);
template void launchTCI_demo_b16<1, 64, 1, 64, 0, 0>(int16_t *out, void *stream);
template void launchTCI_demo_b16<1, 64, 1, 64, 1, 0>(int16_t *out, void *stream);
template void launchTCI_demo_b16<1, 5120, 1, 5120, 0, 0>(int16_t *out, void *stream);
template void launchTCI_demo_b16<1, 5120, 1, 5120, 1, 0>(int16_t *out, void *stream);
template void launchTCI_demo_b16<1, 256, 1, 256, 0, 1>(int16_t *out, void *stream);
template void launchTCI_demo_b16<1, 256, 1, 256, 1, 1>(int16_t *out, void *stream);
template void launchTCI_demo_b16<1, 800, 1, 800, 0, 1>(int16_t *out, void *stream);
template void launchTCI_demo_b16<1, 800, 1, 800, 1, 1>(int16_t *out, void *stream);
template void launchTCI_demo_b16<1, 3328, 1, 3328, 0, 1>(int16_t *out, void *stream);
template void launchTCI_demo_b16<1, 3328, 1, 3328, 1, 1>(int16_t *out, void *stream);