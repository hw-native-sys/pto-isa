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
#include <iostream>

using namespace std;
using namespace pto;

template <typename T, uint32_t ROWS_, uint32_t COLS_, uint32_t VALID_R_, uint32_t VALID_C, uint32_t ALIGN_C>
AICORE void runTSORT32(__gm__ T *out, __gm__ T *src, __gm__ uint32_t *idx)
{
    constexpr uint32_t TYPE_COEF = sizeof(float) / sizeof(T);
    constexpr uint32_t VALID_C_32B_aligned = PTO_CEIL(VALID_C, 32 / sizeof(uint32_t));
    using SrcShapeDim5 = pto::Shape<1, 1, 1, ROWS_, COLS_>;
    using SrcStridDim5 = pto::Stride<1, 1, 1, COLS_, 1>;
    using SrcGlobalData = GlobalTensor<T, SrcShapeDim5, SrcStridDim5>;

    using IdxShapeDim5 = pto::Shape<1, 1, 1, ROWS_, COLS_>;
    using IdxStridDim5 = pto::Stride<1, 1, 1, COLS_, 1>;
    using IdxGlobalData = GlobalTensor<uint32_t, IdxShapeDim5, IdxStridDim5>;

    using OutShapeDim5 = pto::Shape<1, 1, 1, ROWS_, TYPE_COEF * 2 * COLS_>;
    using OutStridDim5 = pto::Stride<1, 1, 1, TYPE_COEF * 2 * COLS_, 1>;
    using OutGlobalData = GlobalTensor<T, OutShapeDim5, OutStridDim5>;

    using SrcTileData = Tile<TileType::Vec, T, ROWS_, ALIGN_C, BLayout::RowMajor, -1, -1>;
    using IdxTileData = Tile<TileType::Vec, uint32_t, ROWS_, ALIGN_C, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, T, ROWS_, TYPE_COEF * 2 * ALIGN_C, BLayout::RowMajor, -1, -1>;
    using TmpTileData = Tile<TileType::Vec, T, 1, VALID_C_32B_aligned, BLayout::RowMajor>;

    SrcTileData srcTile(VALID_R_, VALID_C);
    IdxTileData idxTile(VALID_R_, ALIGN_C);
    DstTileData dstTile(VALID_R_, VALID_C * TYPE_COEF * 2);
    TmpTileData tmpTile;
    TASSIGN(srcTile, 0x0000);
    TASSIGN(idxTile, 0xF800);
    TASSIGN(dstTile, 0x1F000);
    TASSIGN(tmpTile, 0x3E000); // limited to 8KB = 0x2000

    SrcGlobalData srcGlobal(src);
    IdxGlobalData idxGlobal(idx);
    OutGlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    TLOAD(idxTile, idxGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TSORT32(dstTile, srcTile, idxTile, tmpTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);

#ifndef __PTO_AUTO__
    pipe_barrier(PIPE_ALL);
#endif
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTSORT32_1(__gm__ uint64_t *out, __gm__ uint64_t *src, __gm__ uint32_t *idx)
{
    constexpr uint32_t ROWS = 2;
    constexpr uint32_t COLS = 32;
    constexpr uint32_t VALID_R = 2;
    constexpr uint32_t VALID_C = 32;
    constexpr uint32_t ALIGN_C = (VALID_C + 31 - 1) / 32 * 32;
    runTSORT32<float, ROWS, COLS, VALID_R, VALID_C, ALIGN_C>(reinterpret_cast<__gm__ float *>(out),
                                                             reinterpret_cast<__gm__ float *>(src), idx);
}

extern "C" __global__ AICORE void launchTSORT32_2(__gm__ uint64_t *out, __gm__ uint64_t *src, __gm__ uint32_t *idx)
{
    constexpr uint32_t ROWS = 4;
    constexpr uint32_t COLS = 64;
    constexpr uint32_t VALID_R = 4;
    constexpr uint32_t VALID_C = 64;
    constexpr uint32_t ALIGN_C = (VALID_C + 31 - 1) / 32 * 32;

    runTSORT32<half, ROWS, COLS, VALID_R, VALID_C, ALIGN_C>(reinterpret_cast<__gm__ half *>(out),
                                                            reinterpret_cast<__gm__ half *>(src), idx);
}

extern "C" __global__ AICORE void launchTSORT32_3(__gm__ uint64_t *out, __gm__ uint64_t *src, __gm__ uint32_t *idx)
{
    constexpr uint32_t ROWS = 1;
    constexpr uint32_t COLS = 256 * 32;
    constexpr uint32_t VALID_R = 1;
    constexpr uint32_t VALID_C = 256 * 32;
    constexpr uint32_t ALIGN_C = (VALID_C + 31 - 1) / 32 * 32;

    runTSORT32<float, ROWS, COLS, VALID_R, VALID_C, ALIGN_C>(reinterpret_cast<__gm__ float *>(out),
                                                             reinterpret_cast<__gm__ float *>(src), idx);
}

extern "C" __global__ AICORE void launchTSORT32_4(__gm__ uint64_t *out, __gm__ uint64_t *src, __gm__ uint32_t *idx)
{
    constexpr uint32_t ROWS = 2;
    constexpr uint32_t COLS = 13;
    constexpr uint32_t VALID_R = 2;
    constexpr uint32_t VALID_C = 13;
    constexpr uint32_t ALIGN_C = (VALID_C * sizeof(float) + 31 - 1) / 32 * 32 / sizeof(float);

    runTSORT32<float, ROWS, COLS, VALID_R, VALID_C, ALIGN_C>(reinterpret_cast<__gm__ float *>(out),
                                                             reinterpret_cast<__gm__ float *>(src), idx);
}

extern "C" __global__ AICORE void launchTSORT32_5(__gm__ uint64_t *out, __gm__ uint64_t *src, __gm__ uint32_t *idx)
{
    constexpr uint32_t ROWS = 1;
    constexpr uint32_t COLS = 4164;
    constexpr uint32_t VALID_R = 1;
    constexpr uint32_t VALID_C = 4164;
    constexpr uint32_t ALIGN_C = 8192;

    runTSORT32<float, ROWS, COLS, VALID_R, VALID_C, ALIGN_C>(reinterpret_cast<__gm__ float *>(out),
                                                             reinterpret_cast<__gm__ float *>(src), idx);
}

extern "C" __global__ AICORE void launchTSORT32_6(__gm__ uint64_t *out, __gm__ uint64_t *src, __gm__ uint32_t *idx)
{
    constexpr uint32_t ROWS = 2;
    constexpr uint32_t COLS = 2084;
    constexpr uint32_t VALID_R = 2;
    constexpr uint32_t VALID_C = 2084;
    constexpr uint32_t ALIGN_C = 3072;

    runTSORT32<float, ROWS, COLS, VALID_R, VALID_C, ALIGN_C>(reinterpret_cast<__gm__ float *>(out),
                                                             reinterpret_cast<__gm__ float *>(src), idx);
}

template <int32_t testKey>
void launchTSORT32(uint64_t *out, uint64_t *src, uint32_t *idx, void *stream)
{
    cout << "launchTSORT32 start!" << endl;
    if constexpr (testKey == 1) {
        launchTSORT32_1<<<1, nullptr, stream>>>(out, src, idx);
    } else if constexpr (testKey == 2) {
        launchTSORT32_2<<<1, nullptr, stream>>>(out, src, idx);
    } else if constexpr (testKey == 3) {
        launchTSORT32_3<<<1, nullptr, stream>>>(out, src, idx);
    } else if constexpr (testKey == 4) {
        launchTSORT32_4<<<1, nullptr, stream>>>(out, src, idx);
    } else if constexpr (testKey == 5) {
        launchTSORT32_5<<<1, nullptr, stream>>>(out, src, idx);
    } else if constexpr (testKey == 6) {
        launchTSORT32_6<<<1, nullptr, stream>>>(out, src, idx);
    }
    cout << "launchTSORT32 end!" << endl;
}

template void launchTSORT32<1>(uint64_t *out, uint64_t *src, uint32_t *idx, void *stream);
template void launchTSORT32<2>(uint64_t *out, uint64_t *src, uint32_t *idx, void *stream);
template void launchTSORT32<3>(uint64_t *out, uint64_t *src, uint32_t *idx, void *stream);
template void launchTSORT32<4>(uint64_t *out, uint64_t *src, uint32_t *idx, void *stream);
template void launchTSORT32<5>(uint64_t *out, uint64_t *src, uint32_t *idx, void *stream);
template void launchTSORT32<6>(uint64_t *out, uint64_t *src, uint32_t *idx, void *stream);
