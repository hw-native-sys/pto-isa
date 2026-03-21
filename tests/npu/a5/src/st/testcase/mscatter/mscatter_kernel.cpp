/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "mscatter_common.h"

using namespace std;
using namespace pto;

template <typename T, typename TIdx, int kSrcRows, int kSrcCols, int kOutSize>
inline AICORE void runMSCATTER(__gm__ T __out__ *out, __gm__ T __in__ *src, __gm__ TIdx __in__ *indices)
{
    using DynShape_src = pto::Shape<1, 1, 1, kSrcRows, kSrcCols>;
    using DynStrid_src = pto::Stride<1, 1, 1, kSrcCols, 1>;
    using GlobalData_src = GlobalTensor<T, DynShape_src, DynStrid_src>;

    using DynShape_idx = pto::Shape<1, 1, 1, kSrcRows, kSrcCols>;
    using DynStrid_idx = pto::Stride<1, 1, 1, kSrcCols, 1>;
    using GlobalData_idx = GlobalTensor<TIdx, DynShape_idx, DynStrid_idx>;

    using DynShape_out = pto::Shape<1, 1, 1, 1, kOutSize>;
    using DynStrid_out = pto::Stride<1, 1, 1, kOutSize, 1>;
    using GlobalData_out = GlobalTensor<T, DynShape_out, DynStrid_out>;

    using TileData_src = Tile<TileType::Vec, T, kSrcRows, kSrcCols, BLayout::RowMajor, -1, -1>;
    using TileData_idx = Tile<TileType::Vec, TIdx, kSrcRows, kSrcCols, BLayout::RowMajor, -1, -1>;

    TileData_src srcTile(kSrcRows, kSrcCols);
    TileData_idx idxTile(kSrcRows, kSrcCols);

    // Place indices at offset 0 (same layout as MGATHER which works)
    TASSIGN(idxTile, 0x0);
    // Place src after indices, aligned to 32 bytes
    constexpr int idxBytes = kSrcRows * kSrcCols * sizeof(TIdx);
    constexpr int srcOffset = ((idxBytes + 31) / 32) * 32;
    TASSIGN(srcTile, srcOffset);

    GlobalData_src srcGlobal(src);
    GlobalData_idx idxGlobal(indices);
    GlobalData_out outGlobal(out);

    // Load indices first (at offset 0), then src
    TLOAD(idxTile, idxGlobal);
    TLOAD(srcTile, srcGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    MSCATTER(outGlobal, srcTile, idxTile);
#ifndef __PTO_AUTO__
    // MSCATTER uses SIMT to write from UB to GM, ensure completion before kernel exit
    pipe_barrier(PIPE_ALL);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
}

extern "C" __global__ AICORE void runMSCATTER_half_8x32_1024(__gm__ half *out, __gm__ half *src,
                                                             __gm__ int32_t *indices)
{
    runMSCATTER<half, int32_t, 8, 32, 1024>(out, src, indices);
}

extern "C" __global__ AICORE void runMSCATTER_half_16x64_2048(__gm__ half *out, __gm__ half *src,
                                                              __gm__ int32_t *indices)
{
    runMSCATTER<half, int32_t, 16, 64, 2048>(out, src, indices);
}

extern "C" __global__ AICORE void runMSCATTER_float_8x32_512(__gm__ float *out, __gm__ float *src,
                                                             __gm__ int32_t *indices)
{
    runMSCATTER<float, int32_t, 8, 32, 512>(out, src, indices);
}

extern "C" __global__ AICORE void runMSCATTER_float_16x32_1024(__gm__ float *out, __gm__ float *src,
                                                               __gm__ int32_t *indices)
{
    runMSCATTER<float, int32_t, 16, 32, 1024>(out, src, indices);
}

extern "C" __global__ AICORE void runMSCATTER_float_16x64_2048(__gm__ float *out, __gm__ float *src,
                                                               __gm__ int32_t *indices)
{
    runMSCATTER<float, int32_t, 16, 64, 2048>(out, src, indices);
}

extern "C" __global__ AICORE void runMSCATTER_float_8x8_128(__gm__ float *out, __gm__ float *src,
                                                            __gm__ int32_t *indices)
{
    runMSCATTER<float, int32_t, 8, 8, 128>(out, src, indices);
}

extern "C" __global__ AICORE void runMSCATTER_int32_8x16_256(__gm__ int32_t *out, __gm__ int32_t *src,
                                                             __gm__ int32_t *indices)
{
    runMSCATTER<int32_t, int32_t, 8, 16, 256>(out, src, indices);
}

extern "C" __global__ AICORE void runMSCATTER_int32_16x32_1024(__gm__ int32_t *out, __gm__ int32_t *src,
                                                               __gm__ int32_t *indices)
{
    runMSCATTER<int32_t, int32_t, 16, 32, 1024>(out, src, indices);
}

extern "C" __global__ AICORE void runMSCATTER_int32_16x16_512(__gm__ int32_t *out, __gm__ int32_t *src,
                                                              __gm__ int32_t *indices)
{
    runMSCATTER<int32_t, int32_t, 16, 16, 512>(out, src, indices);
}

extern "C" __global__ AICORE void runMSCATTER_uint8_16x32_1024(__gm__ uint8_t *out, __gm__ uint8_t *src,
                                                               __gm__ int32_t *indices)
{
    runMSCATTER<uint8_t, int32_t, 16, 32, 1024>(out, src, indices);
}

extern "C" __global__ AICORE void runMSCATTER_uint8_16x64_2048(__gm__ uint8_t *out, __gm__ uint8_t *src,
                                                               __gm__ int32_t *indices)
{
    runMSCATTER<uint8_t, int32_t, 16, 64, 2048>(out, src, indices);
}

template <typename T, typename TIdx, int kSrcRows, int kSrcCols, int kOutSize>
void LaunchMSCATTER(T *out, T *src, TIdx *indices, void *stream);

template <>
void LaunchMSCATTER<float, int32_t, 8, 32, 512>(float *out, float *src, int32_t *indices, void *stream)
{
    runMSCATTER_float_8x32_512<<<1, nullptr, stream>>>(out, src, indices);
}

template <>
void LaunchMSCATTER<float, int32_t, 16, 32, 1024>(float *out, float *src, int32_t *indices, void *stream)
{
    runMSCATTER_float_16x32_1024<<<1, nullptr, stream>>>(out, src, indices);
}

template <>
void LaunchMSCATTER<float, int32_t, 16, 64, 2048>(float *out, float *src, int32_t *indices, void *stream)
{
    runMSCATTER_float_16x64_2048<<<1, nullptr, stream>>>(out, src, indices);
}

template <>
void LaunchMSCATTER<float, int32_t, 8, 8, 128>(float *out, float *src, int32_t *indices, void *stream)
{
    runMSCATTER_float_8x8_128<<<1, nullptr, stream>>>(out, src, indices);
}

template <>
void LaunchMSCATTER<int32_t, int32_t, 8, 16, 256>(int32_t *out, int32_t *src, int32_t *indices, void *stream)
{
    runMSCATTER_int32_8x16_256<<<1, nullptr, stream>>>(out, src, indices);
}

template <>
void LaunchMSCATTER<int32_t, int32_t, 16, 32, 1024>(int32_t *out, int32_t *src, int32_t *indices, void *stream)
{
    runMSCATTER_int32_16x32_1024<<<1, nullptr, stream>>>(out, src, indices);
}

template <>
void LaunchMSCATTER<int32_t, int32_t, 16, 16, 512>(int32_t *out, int32_t *src, int32_t *indices, void *stream)
{
    runMSCATTER_int32_16x16_512<<<1, nullptr, stream>>>(out, src, indices);
}

template <>
void LaunchMSCATTER<uint8_t, int32_t, 16, 32, 1024>(uint8_t *out, uint8_t *src, int32_t *indices, void *stream)
{
    runMSCATTER_uint8_16x32_1024<<<1, nullptr, stream>>>(out, src, indices);
}

template <>
void LaunchMSCATTER<uint8_t, int32_t, 16, 64, 2048>(uint8_t *out, uint8_t *src, int32_t *indices, void *stream)
{
    runMSCATTER_uint8_16x64_2048<<<1, nullptr, stream>>>(out, src, indices);
}

template <int kSrcRows, int kSrcCols, int kOutSize>
void LaunchMSCATTERHalf(aclFloat16 *out, aclFloat16 *src, int32_t *indices, void *stream);

template <>
void LaunchMSCATTERHalf<8, 32, 1024>(aclFloat16 *out, aclFloat16 *src, int32_t *indices, void *stream)
{
    runMSCATTER_half_8x32_1024<<<1, nullptr, stream>>>((half *)out, (half *)src, indices);
}

template <>
void LaunchMSCATTERHalf<16, 64, 2048>(aclFloat16 *out, aclFloat16 *src, int32_t *indices, void *stream)
{
    runMSCATTER_half_16x64_2048<<<1, nullptr, stream>>>((half *)out, (half *)src, indices);
}
