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
#include "mgather_common.h"

using namespace std;
using namespace pto;

template <typename T, typename TIdx, int kTableRows, int kTableCols, int kOutRows, int kOutCols>
inline AICORE void runMGATHER(__gm__ T __out__ *out, __gm__ T __in__ *table, __gm__ TIdx __in__ *indices)
{
    using DynShape_table = pto::Shape<1, 1, 1, kTableRows, kTableCols>;
    using DynStrid_table = pto::Stride<1, 1, 1, kTableCols, 1>;
    using GlobalData_table = GlobalTensor<T, DynShape_table, DynStrid_table>;

    using DynShape_idx = pto::Shape<1, 1, 1, kOutRows, kOutCols>;
    using DynStrid_idx = pto::Stride<1, 1, 1, kOutCols, 1>;
    using GlobalData_idx = GlobalTensor<TIdx, DynShape_idx, DynStrid_idx>;

    using DynShape_out = pto::Shape<1, 1, 1, kOutRows, kOutCols>;
    using DynStrid_out = pto::Stride<1, 1, 1, kOutCols, 1>;
    using GlobalData_out = GlobalTensor<T, DynShape_out, DynStrid_out>;

    using TileData_idx = Tile<TileType::Vec, TIdx, kOutRows, kOutCols, BLayout::RowMajor, -1, -1>;
    using TileData_out = Tile<TileType::Vec, T, kOutRows, kOutCols, BLayout::RowMajor, -1, -1>;

    TileData_idx idxTile(kOutRows, kOutCols);
    TileData_out outTile(kOutRows, kOutCols);

    TASSIGN(idxTile, 0x0);
    TASSIGN(outTile, kOutRows * kOutCols * sizeof(TIdx));

    GlobalData_table tableGlobal(table);
    GlobalData_idx idxGlobal(indices);
    GlobalData_out outGlobal(out);

    TLOAD(idxTile, idxGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    MGATHER(outTile, tableGlobal, idxTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(outGlobal, outTile);
}

extern "C" __global__ AICORE void runMGATHER_half_16x64_8x32(__gm__ half *out, __gm__ half *table,
                                                             __gm__ int32_t *indices)
{
    runMGATHER<half, int32_t, 16, 64, 8, 32>(out, table, indices);
}

extern "C" __global__ AICORE void runMGATHER_half_16x128_8x64(__gm__ half *out, __gm__ half *table,
                                                              __gm__ int32_t *indices)
{
    runMGATHER<half, int32_t, 16, 128, 8, 64>(out, table, indices);
}

extern "C" __global__ AICORE void runMGATHER_half_32x128_16x64(__gm__ half *out, __gm__ half *table,
                                                               __gm__ int32_t *indices)
{
    runMGATHER<half, int32_t, 32, 128, 16, 64>(out, table, indices);
}

extern "C" __global__ AICORE void runMGATHER_half_16x256_8x128(__gm__ half *out, __gm__ half *table,
                                                               __gm__ int32_t *indices)
{
    runMGATHER<half, int32_t, 16, 256, 8, 128>(out, table, indices);
}

extern "C" __global__ AICORE void runMGATHER_half_64x64_32x32(__gm__ half *out, __gm__ half *table,
                                                              __gm__ int32_t *indices)
{
    runMGATHER<half, int32_t, 64, 64, 32, 32>(out, table, indices);
}

extern "C" __global__ AICORE void runMGATHER_float_8x64_4x32(__gm__ float *out, __gm__ float *table,
                                                             __gm__ int32_t *indices)
{
    runMGATHER<float, int32_t, 8, 64, 4, 32>(out, table, indices);
}

extern "C" __global__ AICORE void runMGATHER_float_16x64_8x32(__gm__ float *out, __gm__ float *table,
                                                              __gm__ int32_t *indices)
{
    runMGATHER<float, int32_t, 16, 64, 8, 32>(out, table, indices);
}

extern "C" __global__ AICORE void runMGATHER_float_32x64_16x32(__gm__ float *out, __gm__ float *table,
                                                               __gm__ int32_t *indices)
{
    runMGATHER<float, int32_t, 32, 64, 16, 32>(out, table, indices);
}

extern "C" __global__ AICORE void runMGATHER_float_16x16_8x8(__gm__ float *out, __gm__ float *table,
                                                             __gm__ int32_t *indices)
{
    runMGATHER<float, int32_t, 16, 16, 8, 8>(out, table, indices);
}

extern "C" __global__ AICORE void runMGATHER_int32_8x32_4x16(__gm__ int32_t *out, __gm__ int32_t *table,
                                                             __gm__ int32_t *indices)
{
    runMGATHER<int32_t, int32_t, 8, 32, 4, 16>(out, table, indices);
}

extern "C" __global__ AICORE void runMGATHER_int32_16x64_8x32(__gm__ int32_t *out, __gm__ int32_t *table,
                                                              __gm__ int32_t *indices)
{
    runMGATHER<int32_t, int32_t, 16, 64, 8, 32>(out, table, indices);
}

extern "C" __global__ AICORE void runMGATHER_int32_32x32_16x16(__gm__ int32_t *out, __gm__ int32_t *table,
                                                               __gm__ int32_t *indices)
{
    runMGATHER<int32_t, int32_t, 32, 32, 16, 16>(out, table, indices);
}

extern "C" __global__ AICORE void runMGATHER_uint8_16x64_8x32(__gm__ uint8_t *out, __gm__ uint8_t *table,
                                                              __gm__ int32_t *indices)
{
    runMGATHER<uint8_t, int32_t, 16, 64, 8, 32>(out, table, indices);
}

extern "C" __global__ AICORE void runMGATHER_uint8_32x64_16x32(__gm__ uint8_t *out, __gm__ uint8_t *table,
                                                               __gm__ int32_t *indices)
{
    runMGATHER<uint8_t, int32_t, 32, 64, 16, 32>(out, table, indices);
}

template <typename T, typename TIdx, int kTableRows, int kTableCols, int kOutRows, int kOutCols>
void LaunchMGATHER(T *out, T *table, TIdx *indices, void *stream);

template <>
void LaunchMGATHER<float, int32_t, 8, 64, 4, 32>(float *out, float *table, int32_t *indices, void *stream)
{
    runMGATHER_float_8x64_4x32<<<1, nullptr, stream>>>(out, table, indices);
}

template <>
void LaunchMGATHER<float, int32_t, 16, 64, 8, 32>(float *out, float *table, int32_t *indices, void *stream)
{
    runMGATHER_float_16x64_8x32<<<1, nullptr, stream>>>(out, table, indices);
}

template <>
void LaunchMGATHER<float, int32_t, 32, 64, 16, 32>(float *out, float *table, int32_t *indices, void *stream)
{
    runMGATHER_float_32x64_16x32<<<1, nullptr, stream>>>(out, table, indices);
}

template <>
void LaunchMGATHER<float, int32_t, 16, 16, 8, 8>(float *out, float *table, int32_t *indices, void *stream)
{
    runMGATHER_float_16x16_8x8<<<1, nullptr, stream>>>(out, table, indices);
}

template <>
void LaunchMGATHER<int32_t, int32_t, 8, 32, 4, 16>(int32_t *out, int32_t *table, int32_t *indices, void *stream)
{
    runMGATHER_int32_8x32_4x16<<<1, nullptr, stream>>>(out, table, indices);
}

template <>
void LaunchMGATHER<int32_t, int32_t, 16, 64, 8, 32>(int32_t *out, int32_t *table, int32_t *indices, void *stream)
{
    runMGATHER_int32_16x64_8x32<<<1, nullptr, stream>>>(out, table, indices);
}

template <>
void LaunchMGATHER<int32_t, int32_t, 32, 32, 16, 16>(int32_t *out, int32_t *table, int32_t *indices, void *stream)
{
    runMGATHER_int32_32x32_16x16<<<1, nullptr, stream>>>(out, table, indices);
}

template <>
void LaunchMGATHER<uint8_t, int32_t, 16, 64, 8, 32>(uint8_t *out, uint8_t *table, int32_t *indices, void *stream)
{
    runMGATHER_uint8_16x64_8x32<<<1, nullptr, stream>>>(out, table, indices);
}

template <>
void LaunchMGATHER<uint8_t, int32_t, 32, 64, 16, 32>(uint8_t *out, uint8_t *table, int32_t *indices, void *stream)
{
    runMGATHER_uint8_32x64_16x32<<<1, nullptr, stream>>>(out, table, indices);
}

template <int kTableRows, int kTableCols, int kOutRows, int kOutCols>
void LaunchMGATHERHalf(aclFloat16 *out, aclFloat16 *table, int32_t *indices, void *stream);

template <>
void LaunchMGATHERHalf<16, 64, 8, 32>(aclFloat16 *out, aclFloat16 *table, int32_t *indices, void *stream)
{
    runMGATHER_half_16x64_8x32<<<1, nullptr, stream>>>((half *)out, (half *)table, indices);
}

template <>
void LaunchMGATHERHalf<16, 128, 8, 64>(aclFloat16 *out, aclFloat16 *table, int32_t *indices, void *stream)
{
    runMGATHER_half_16x128_8x64<<<1, nullptr, stream>>>((half *)out, (half *)table, indices);
}

template <>
void LaunchMGATHERHalf<32, 128, 16, 64>(aclFloat16 *out, aclFloat16 *table, int32_t *indices, void *stream)
{
    runMGATHER_half_32x128_16x64<<<1, nullptr, stream>>>((half *)out, (half *)table, indices);
}

template <>
void LaunchMGATHERHalf<16, 256, 8, 128>(aclFloat16 *out, aclFloat16 *table, int32_t *indices, void *stream)
{
    runMGATHER_half_16x256_8x128<<<1, nullptr, stream>>>((half *)out, (half *)table, indices);
}

template <>
void LaunchMGATHERHalf<64, 64, 32, 32>(aclFloat16 *out, aclFloat16 *table, int32_t *indices, void *stream)
{
    runMGATHER_half_64x64_32x32<<<1, nullptr, stream>>>((half *)out, (half *)table, indices);
}
