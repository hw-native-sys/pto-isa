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
#include <unordered_map>
#include "tgather_common.h"

using namespace pto;
using namespace std;
template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, MaskPattern maskPattern>
AICORE void runTGATHER(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, (kTRows_ + 5), (kTCols_ + 32), BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData srcTile(kTRows_, kTCols_);
    DstTileData dstTile(kTRows_, kTCols_);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + (kTRows_ + 5) * (kTCols_ + 32) * sizeof(T));

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TGATHER<DstTileData, TileData, maskPattern>(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, MaskPattern maskPattern>
AICORE void runTGATHER_COL(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, (kGRows_ + 5), (kGCols_ + 32), BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData srcTile(kGRows_, kGCols_);
    DstTileData dstTile(kTRows_, kTCols_);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + (kGRows_ + 5) * (kGCols_ + 32) * sizeof(T));

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TGATHER<DstTileData, TileData, maskPattern, GatherAxis::GATHER_COL>(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void launchTGATHER_21(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<uint16_t, HALF_P0101_ROW, HALF_P0101_COL, HALF_P0101_ROW, HALF_P0101_COL, MaskPattern::P0101>(
        reinterpret_cast<__gm__ uint16_t *>(out), reinterpret_cast<__gm__ uint16_t *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_22(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<uint16_t, HALF_P1010_ROW, HALF_P1010_COL, HALF_P1010_ROW, HALF_P1010_COL, MaskPattern::P1010>(
        reinterpret_cast<__gm__ uint16_t *>(out), reinterpret_cast<__gm__ uint16_t *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_23(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<int16_t, HALF_P0001_ROW, HALF_P0001_COL, HALF_P0001_ROW, HALF_P0001_COL, MaskPattern::P0001>(
        reinterpret_cast<__gm__ int16_t *>(out), reinterpret_cast<__gm__ int16_t *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_24(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<int16_t, HALF_P0010_ROW, HALF_P0010_COL, HALF_P0010_ROW, HALF_P0010_COL, MaskPattern::P0010>(
        reinterpret_cast<__gm__ int16_t *>(out), reinterpret_cast<__gm__ int16_t *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_25(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<uint32_t, FLOAT_P0100_ROW, FLOAT_P0100_COL, FLOAT_P0100_ROW, FLOAT_P0100_COL, MaskPattern::P0100>(
        reinterpret_cast<__gm__ uint32_t *>(out), reinterpret_cast<__gm__ uint32_t *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_26(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<int32_t, FLOAT_P1000_ROW, FLOAT_P1000_COL, FLOAT_P1000_ROW, FLOAT_P1000_COL, MaskPattern::P1000>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int32_t *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_27(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<int32_t, FLOAT_P1111_ROW, FLOAT_P1111_COL, FLOAT_P1111_ROW, FLOAT_P1111_COL, MaskPattern::P1111>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int32_t *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_11(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<half, HALF_P0101_ROW, HALF_P0101_COL, HALF_P0101_ROW, HALF_P0101_COL, MaskPattern::P0101>(
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_12(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<half, HALF_P1010_ROW, HALF_P1010_COL, HALF_P1010_ROW, HALF_P1010_COL, MaskPattern::P1010>(
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_13(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<half, HALF_P0001_ROW, HALF_P0001_COL, HALF_P0001_ROW, HALF_P0001_COL, MaskPattern::P0001>(
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_15(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<half, HALF_P0100_ROW, HALF_P0100_COL, HALF_P0100_ROW, HALF_P0100_COL, MaskPattern::P0100>(
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_16(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<half, HALF_P1000_ROW, HALF_P1000_COL, HALF_P1000_ROW, HALF_P1000_COL, MaskPattern::P1000>(
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_1(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<float, FLOAT_P0101_ROW, FLOAT_P0101_COL, FLOAT_P0101_ROW, FLOAT_P0101_COL, MaskPattern::P0101>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_2(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<float, FLOAT_P1010_ROW, FLOAT_P1010_COL, FLOAT_P1010_ROW, FLOAT_P1010_COL, MaskPattern::P1010>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_3(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<float, FLOAT_P0001_ROW, FLOAT_P0001_COL, FLOAT_P0001_ROW, FLOAT_P0001_COL, MaskPattern::P0001>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_4(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<float, FLOAT_P0010_ROW, FLOAT_P0010_COL, FLOAT_P0010_ROW, FLOAT_P0010_COL, MaskPattern::P0010>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_5(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<float, FLOAT_P0100_ROW, FLOAT_P0100_COL, FLOAT_P0100_ROW, FLOAT_P0100_COL, MaskPattern::P0100>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_6(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<float, FLOAT_P1000_ROW, FLOAT_P1000_COL, FLOAT_P1000_ROW, FLOAT_P1000_COL, MaskPattern::P1000>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_7(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER<float, FLOAT_P1111_ROW, FLOAT_P1111_COL, FLOAT_P1111_ROW, FLOAT_P1111_COL, MaskPattern::P1111>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_COL_101(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER_COL<float, COL_HALF_P0101_ROW, COL_HALF_P0101_COL, COL_HALF_P0101_ROW / 2, COL_HALF_P0101_COL,
                   MaskPattern::P0101>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_COL_102(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER_COL<float, COL_HALF_P1010_ROW, COL_HALF_P1010_COL, COL_HALF_P1010_ROW / 2, COL_HALF_P1010_COL,
                   MaskPattern::P1010>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_COL_103(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER_COL<float, COL_HALF_P0001_ROW, COL_HALF_P0001_COL, COL_HALF_P0001_ROW / 4, COL_HALF_P0001_COL,
                   MaskPattern::P0001>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_COL_104(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER_COL<float, COL_HALF_P0010_ROW, COL_HALF_P0010_COL, COL_HALF_P0010_ROW / 4, COL_HALF_P0010_COL,
                   MaskPattern::P0010>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_COL_105(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER_COL<float, COL_HALF_P0100_ROW, COL_HALF_P0100_COL, COL_HALF_P0100_ROW / 4, COL_HALF_P0100_COL,
                   MaskPattern::P0100>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_COL_106(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER_COL<float, COL_HALF_P1000_ROW, COL_HALF_P1000_COL, COL_HALF_P1000_ROW / 4, COL_HALF_P1000_COL,
                   MaskPattern::P1000>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_COL_107(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER_COL<float, COL_HALF_P1111_ROW, COL_HALF_P1111_COL, COL_HALF_P1111_ROW, COL_HALF_P1111_COL,
                   MaskPattern::P1111>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_COL_111(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER_COL<half, COL_HALF_P0101_ROW, COL_HALF_P0101_COL, COL_HALF_P0101_ROW / 2, COL_HALF_P0101_COL,
                   MaskPattern::P0101>(reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_COL_112(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER_COL<half, COL_HALF_P1010_ROW, COL_HALF_P1010_COL, COL_HALF_P1010_ROW / 2, COL_HALF_P1010_COL,
                   MaskPattern::P1010>(reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_COL_113(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER_COL<half, COL_HALF_P0001_ROW, COL_HALF_P0001_COL, COL_HALF_P0001_ROW / 4, COL_HALF_P0001_COL,
                   MaskPattern::P0001>(reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_COL_114(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER_COL<half, COL_HALF_P0010_ROW, COL_HALF_P0010_COL, COL_HALF_P0010_ROW / 4, COL_HALF_P0010_COL,
                   MaskPattern::P0010>(reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_COL_115(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER_COL<half, COL_HALF_P0100_ROW, COL_HALF_P0100_COL, COL_HALF_P0100_ROW / 4, COL_HALF_P0100_COL,
                   MaskPattern::P0100>(reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_COL_116(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER_COL<half, COL_HALF_P1000_ROW, COL_HALF_P1000_COL, COL_HALF_P1000_ROW / 4, COL_HALF_P1000_COL,
                   MaskPattern::P1000>(reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ AICORE void launchTGATHER_COL_117(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTGATHER_COL<half, COL_HALF_P1111_ROW, COL_HALF_P1111_COL, COL_HALF_P1111_ROW, COL_HALF_P1111_COL,
                   MaskPattern::P1111>(reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

using LaunchFunc = void (*)(__gm__ uint8_t *, __gm__ uint8_t *);

static LaunchFunc GetLaunchFunction(int32_t tilingKey)
{
    static const std::unordered_map<int32_t, LaunchFunc> launchMap = {
        {FP0101, launchTGATHER_1},           {FP1010, launchTGATHER_2},           {FP0001, launchTGATHER_3},
        {FP0010, launchTGATHER_4},           {FP0100, launchTGATHER_5},           {FP1000, launchTGATHER_6},
        {FP1111, launchTGATHER_7},

        {HP0101, launchTGATHER_11},          {HP1010, launchTGATHER_12},          {HP0001, launchTGATHER_13},
        {HP0100, launchTGATHER_15},          {HP1000, launchTGATHER_16},

        {U16P0101, launchTGATHER_21},        {U16P1010, launchTGATHER_22},        {I16P0001, launchTGATHER_23},
        {I16P0010, launchTGATHER_24},        {U32P0100, launchTGATHER_25},        {I32P1000, launchTGATHER_26},
        {I32P1111, launchTGATHER_27},

        {COL_FP0101, launchTGATHER_COL_101}, {COL_FP1010, launchTGATHER_COL_102}, {COL_FP0001, launchTGATHER_COL_103},
        {COL_FP0010, launchTGATHER_COL_104}, {COL_FP0100, launchTGATHER_COL_105}, {COL_FP1000, launchTGATHER_COL_106},
        {COL_FP1111, launchTGATHER_COL_107},

        {COL_HP0101, launchTGATHER_COL_111}, {COL_HP1010, launchTGATHER_COL_112}, {COL_HP0001, launchTGATHER_COL_113},
        {COL_HP0010, launchTGATHER_COL_114}, {COL_HP0100, launchTGATHER_COL_115}, {COL_HP1000, launchTGATHER_COL_116},
        {COL_HP1111, launchTGATHER_COL_117},
    };

    auto it = launchMap.find(tilingKey);
    if (it != launchMap.end()) {
        return it->second;
    }
    return nullptr;
}

template <int32_t tilingKey>
void launchTGATHER_demo(uint8_t *out, uint8_t *src, void *stream)
{
    LaunchFunc func = GetLaunchFunction(tilingKey);
    if (func) {
        func(out, src);
    }
}

template void launchTGATHER_demo<FP0101>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<FP1010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<FP0001>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<FP0010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<FP0100>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<FP1000>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<FP1111>(uint8_t *out, uint8_t *src, void *stream);

template void launchTGATHER_demo<HP0101>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<HP1010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<HP0001>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<HP0100>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<HP1000>(uint8_t *out, uint8_t *src, void *stream);

template void launchTGATHER_demo<U16P0101>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<U16P1010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<I16P0001>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<I16P0010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<U32P0100>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<I32P1000>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<I32P1111>(uint8_t *out, uint8_t *src, void *stream);

template void launchTGATHER_demo<COL_FP0101>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<COL_FP1010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<COL_FP0001>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<COL_FP0010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<COL_FP0100>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<COL_FP1000>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<COL_FP1111>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<COL_HP0101>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<COL_HP1010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<COL_HP0001>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<COL_HP0010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<COL_HP0100>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<COL_HP1000>(uint8_t *out, uint8_t *src, void *stream);
template void launchTGATHER_demo<COL_HP1111>(uint8_t *out, uint8_t *src, void *stream);

template <typename Tsrc0, typename Tsrc1, int kGRows0_, int kGCols0_, int kGRows1_, int kGCols1_, int kTRows_,
          int kTCols_>
inline AICORE void runTGather1D(__gm__ Tsrc0 __out__ *out, __gm__ Tsrc0 __in__ *src0, __gm__ Tsrc1 __in__ *src1)
{
    using DynShapeDim5_src0 = pto::Shape<1, 1, 1, kGRows0_, kGCols0_>;
    using DynStridDim5_src0 = pto::Stride<1, 1, 1, kGCols0_, 1>;
    using GlobalData_src0 = GlobalTensor<Tsrc0, DynShapeDim5_src0, DynStridDim5_src0>;
    using DynShapeDim5_src1 = pto::Shape<1, 1, 1, kGRows1_, kGCols1_>;
    using DynStridDim5_src1 = pto::Stride<1, 1, 1, kGCols1_, 1>;
    using GlobalData_src1 = GlobalTensor<Tsrc1, DynShapeDim5_src1, DynStridDim5_src1>;
    using DynShapeDim5_dst = pto::Shape<1, 1, 1, kGRows1_, kGCols1_>;
    using DynStridDim5_dst = pto::Stride<1, 1, 1, kGCols1_, 1>;
    using GlobalData_dst = GlobalTensor<Tsrc0, DynShapeDim5_dst, DynStridDim5_dst>;

    constexpr int src0_row = kGRows0_;
    constexpr int src0_col = kGCols0_;
    constexpr int src1_row = kGRows1_;
    constexpr int src1_col = kGCols1_;
    constexpr int dst_row = kGRows1_;
    constexpr int dst_col = kGCols1_;

    using TileData_src0 = Tile<TileType::Vec, Tsrc0, kGRows0_, kGCols0_, BLayout::RowMajor, -1, -1>;
    using TileData_src1 = Tile<TileType::Vec, Tsrc1, kGRows1_, kGCols1_, BLayout::RowMajor, -1, -1>;
    using TileData_dst = Tile<TileType::Vec, Tsrc0, kGRows1_, kGCols1_, BLayout::RowMajor, -1, -1>;
    using TileData_tmp = Tile<TileType::Vec, Tsrc1, kGRows1_, kGCols1_, BLayout::RowMajor, -1, -1>;
    TileData_src0 src0Tile(src0_row, src0_col);
    TileData_src1 src1Tile(src1_row, src1_col);
    TileData_dst dstTile(dst_row, dst_col);
    TileData_tmp tmpTile(src1_row, src1_col);

    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x0 + src0_row * src0_col * sizeof(Tsrc0));
    TASSIGN(dstTile, 0x0 + src0_row * src0_col * sizeof(Tsrc0) + src1_row * src1_col * sizeof(Tsrc1));
    TASSIGN(tmpTile, 0x0 + src0_row * src0_col * sizeof(Tsrc0) + src1_row * src1_col * (sizeof(Tsrc1) + sizeof(Tsrc0)));

    GlobalData_src0 src0Global(src0);
    GlobalData_src1 src1Global(src1);
    GlobalData_dst dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TGATHER(dstTile, src0Tile, src1Tile, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

extern "C" __global__ AICORE void test_tgather1D_float(__gm__ float *out, __gm__ float *src0, __gm__ int32_t *src1)
{
    runTGather1D<float, int32_t, 32, 1024, 16, 64, 32, 1024>(out, src0, src1);
}

extern "C" __global__ AICORE void test_tgather1D_int32(__gm__ int32_t *out, __gm__ int32_t *src0, __gm__ int32_t *src1)
{
    runTGather1D<int32_t, int32_t, 32, 512, 16, 256, 32, 512>(out, src0, src1);
}

extern "C" __global__ AICORE void test_tgather1D_half(__gm__ int16_t *out, __gm__ int16_t *src0, __gm__ int32_t *src1)
{
    runTGather1D<int16_t, int32_t, 16, 1024, 16, 128, 16, 1024>(out, src0, src1);
}

extern "C" __global__ AICORE void test_tgather1D_int16(__gm__ int16_t *out, __gm__ int16_t *src0, __gm__ int32_t *src1)
{
    runTGather1D<int16_t, int32_t, 32, 256, 32, 64, 32, 256>(out, src0, src1);
}

void launchTGATHER1D_demo_float(float *out, float *src0, int32_t *src1, aclrtStream stream)
{
    cout << "launch TGATHER float start!" << endl;
    test_tgather1D_float(out, src0, src1);
    cout << "launch TGATHER float end!" << endl;
}

void launchTGATHER1D_demo_int32(int32_t *out, int32_t *src0, int32_t *src1, aclrtStream stream)
{
    cout << "launch TGATHER int32 start!" << endl;
    test_tgather1D_int32(out, src0, src1);
    cout << "launch TGATHER int32 end!" << endl;
}

void launchTGATHER1D_demo_half(int16_t *out, int16_t *src0, int32_t *src1, aclrtStream stream)
{
    cout << "launch TGATHER half start!" << endl;
    test_tgather1D_half(out, src0, src1);
    cout << "launch TGATHER half end!" << endl;
}

void launchTGATHER1D_demo_int16(int16_t *out, int16_t *src0, int32_t *src1, aclrtStream stream)
{
    cout << "launch TGATHER int16 start!" << endl;
    test_tgather1D_int16(out, src0, src1);
    cout << "launch TGATHER int16 end!" << endl;
}