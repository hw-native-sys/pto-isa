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
#include <climits>
#include "acl/acl.h"

using namespace pto;

#define PTO_DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))

template <typename T, int validRows, int validCols, int upperOrLower>
__global__ AICORE void runTTri(__gm__ T __out__* out, int diagonal)
{
    constexpr uint16_t alignedCol = PTO_DIV_ROUNDUP(validCols, BLOCK_BYTE_SIZE) * BLOCK_BYTE_SIZE;

    using GlobalDataDst = GlobalTensor<T, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using TileDataDst = Tile<TileType::Vec, T, validRows, alignedCol, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(validRows, validCols);
    GlobalDataDst dstGlobal(out);

    TASSIGN(dstTile, 0x0);
    TTRI<TileDataDst, upperOrLower>(dstTile, diagonal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int validRows, int validCols, int upperOrLower>
void LaunchTTri(T* out, int diagonal, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTTri<half, validRows, validCols, upperOrLower><<<1, nullptr, stream>>>((half*)(out), diagonal);
    } else {
        runTTri<T, validRows, validCols, upperOrLower><<<1, nullptr, stream>>>(out, diagonal);
    }
}

template void LaunchTTri<aclFloat16, 20, 32, 0>(aclFloat16* out, int diagonal, void* stream);
template void LaunchTTri<uint8_t, 20, 32, 0>(uint8_t* out, int diagonal, void* stream);
template void LaunchTTri<float, 32, 91, 0>(float* out, int diagonal, void* stream);
template void LaunchTTri<float, 128, 128, 0>(float* out, int diagonal, void* stream);
template void LaunchTTri<float, 32, 91, 1>(float* out, int diagonal, void* stream);
template void LaunchTTri<float, 128, 128, 1>(float* out, int diagonal, void* stream);
template void LaunchTTri<float, 763, 32, 0>(float* out, int diagonal, void* stream);
template void LaunchTTri<float, 763, 32, 1>(float* out, int diagonal, void* stream);
template void LaunchTTri<int64_t, 4, 15, 1>(int64_t* out, int diagonal, void* stream);
template void LaunchTTri<uint64_t, 4, 15, 0>(uint64_t* out, int diagonal, void* stream);
template void LaunchTTri<int64_t, 4, 64, 1>(int64_t* out, int diagonal, void* stream);
template void LaunchTTri<uint64_t, 4, 64, 0>(uint64_t* out, int diagonal, void* stream);

// --- Dynamic (static != valid) variants ---

template <typename T, int staticRows, int staticCols, int validRows, int validCols, int upperOrLower>
__global__ AICORE void runTTriDyn(__gm__ T __out__* out, int diagonal)
{
    constexpr uint16_t alignedCol = PTO_DIV_ROUNDUP(staticCols, BLOCK_BYTE_SIZE) * BLOCK_BYTE_SIZE;

    using GlobalDataDst = GlobalTensor<T, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using TileDataDst = Tile<TileType::Vec, T, staticRows, alignedCol, BLayout::RowMajor, -1, -1>;
    TileDataDst dstTile(validRows, validCols);
    GlobalDataDst dstGlobal(out);

    TASSIGN(dstTile, 0x0);
    TTRI<TileDataDst, upperOrLower>(dstTile, diagonal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int staticRows, int staticCols, int validRows, int validCols, int upperOrLower>
void LaunchTTriDyn(T* out, int diagonal, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTTriDyn<half, staticRows, staticCols, validRows, validCols, upperOrLower>
            <<<1, nullptr, stream>>>((half*)(out), diagonal);
    } else {
        runTTriDyn<T, staticRows, staticCols, validRows, validCols, upperOrLower>
            <<<1, nullptr, stream>>>(out, diagonal);
    }
}

template void LaunchTTriDyn<aclFloat16, 30, 208, 30, 208, 1>(aclFloat16* out, int diagonal, void* stream);
template void LaunchTTriDyn<aclFloat16, 30, 208, 30, 176, 1>(aclFloat16* out, int diagonal, void* stream);
template void LaunchTTriDyn<aclFloat16, 293, 16, 269, 16, 0>(aclFloat16* out, int diagonal, void* stream);
template void LaunchTTriDyn<aclFloat16, 293, 16, 293, 16, 0>(aclFloat16* out, int diagonal, void* stream);
template void LaunchTTriDyn<aclFloat16, 293, 16, 287, 16, 0>(aclFloat16* out, int diagonal, void* stream);
template void LaunchTTriDyn<int8_t, 32, 128, 32, 128, 0>(int8_t* out, int diagonal, void* stream);
template void LaunchTTriDyn<int8_t, 32, 128, 24, 112, 0>(int8_t* out, int diagonal, void* stream);
template void LaunchTTriDyn<aclFloat16, 293, 16, 1, 16, 0>(aclFloat16* out, int diagonal, void* stream);
template void LaunchTTriDyn<aclFloat16, 293, 16, 2, 16, 0>(aclFloat16* out, int diagonal, void* stream);

template <typename T, int rows, int cols, int validRows, int validCols, int upperOrLower>
__global__ AICORE void runTTriGuard(__gm__ T* out, int diagonal)
{
    constexpr int guardRows = PTO_DIV_ROUNDUP(512, cols * sizeof(T));
    constexpr int totalRows = rows + guardRows;
    using GlobalData = GlobalTensor<T, Shape<1, 1, 1, totalRows, cols>, pto::Stride<1, 1, 1, cols, 1>>;
    using TileData = Tile<TileType::Vec, T, totalRows, cols, BLayout::RowMajor, -1, -1>;
    GlobalData output(out);
    TileData dst(totalRows, cols);
    TASSIGN(dst, 0);
    TLOAD(dst, output);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    dst.SetValidShape(validRows, validCols);
    TTRI<TileData, upperOrLower>(dst, diagonal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    dst.SetValidShape(totalRows, cols);
    TSTORE(output, dst);
}

template <typename T, int kind, int rows, int cols, int validRows, int validCols, int upperOrLower, int diagonal>
void LaunchTTriGuard(T* out, void* stream)
{
    if constexpr (kind == 1) {
        runTTriGuard<half, rows, cols, validRows, validCols, upperOrLower>
            <<<1, nullptr, stream>>>(reinterpret_cast<half*>(out), diagonal);
    } else if constexpr (kind == 2) {
        runTTriGuard<bfloat16_t, rows, cols, validRows, validCols, upperOrLower>
            <<<1, nullptr, stream>>>(reinterpret_cast<bfloat16_t*>(out), diagonal);
    } else {
        runTTriGuard<T, rows, cols, validRows, validCols, upperOrLower><<<1, nullptr, stream>>>(out, diagonal);
    }
}

#define PTO_TTRI_GUARD_CASE(name, type, kind, rows, cols, validRows, validCols, upper, diagonal) \
    template void LaunchTTriGuard<type, kind, rows, cols, validRows, validCols, upper, diagonal>(type*, void*);
#include "ttri_guard_cases.inc"
#undef PTO_TTRI_GUARD_CASE
