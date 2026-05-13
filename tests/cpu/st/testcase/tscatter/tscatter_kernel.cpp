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
#include "tscatter_common.h"

using namespace pto;

template <int kTRows_, int kTCols_>
AICORE void runTScatter(__gm__ float __out__ *out, __gm__ float __in__ *src, __gm__ uint16_t __in__ *idx)
{
    using TileT = Tile<TileType::Vec, float, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    using IdxT = Tile<TileType::Vec, uint16_t, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    using SrcShape = Shape<1, 1, 1, kTRows_, kTCols_>;
    using SrcStride = Stride<1, 1, 1, kTCols_, 1>;
    using GTf = GlobalTensor<float, SrcShape, SrcStride>;
    using GTi = GlobalTensor<uint16_t, SrcShape, SrcStride>;

    TileT srcTile(kTRows_, kTCols_);
    TileT dstTile(kTRows_, kTCols_);
    IdxT idxTile(kTRows_, kTCols_);

    GTf srcGlobal(src);
    GTf dstGlobal(out);
    GTi idxGlobal(idx);

    TASSIGN(srcTile, 0);
    TASSIGN(dstTile, kTRows_ * kTCols_ * sizeof(typename TileT::DType));
    TASSIGN(idxTile, 2 * kTRows_ * kTCols_ * sizeof(typename TileT::DType));

    TLOAD(srcTile, srcGlobal);
    TLOAD(idxTile, idxGlobal);
    TEXPANDS(dstTile, 0.0f);
    TSCATTER(dstTile, srcTile, idxTile);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <int kTRows_, int kTCols_>
void LaunchTScatter(float *out, float *src, uint16_t *idx, void *stream)
{
    runTScatter<kTRows_, kTCols_>(out, src, idx);
}

template void LaunchTScatter<16, 16>(float *out, float *src, uint16_t *idx, void *stream);

// --- Mask-pattern TSCATTER ---

template <typename T, int kSrcRows_, int kSrcCols_, int kDstRows_, int kDstCols_, MaskPattern maskPattern>
AICORE void runTScatterMasked(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using DynShapeSrc = Shape<1, 1, 1, kSrcRows_, kSrcCols_>;
    using DynStridSrc = Stride<1, 1, 1, kSrcCols_, 1>;
    using GlobalSrc = GlobalTensor<T, DynShapeSrc, DynStridSrc>;

    using DynShapeDst = Shape<1, 1, 1, kDstRows_, kDstCols_>;
    using DynStridDst = Stride<1, 1, 1, kDstCols_, 1>;
    using GlobalDst = GlobalTensor<T, DynShapeDst, DynStridDst>;

    using SrcTileData = Tile<TileType::Vec, T, kSrcRows_, kSrcCols_, BLayout::RowMajor, -1, -1>;
    using DstTileData = Tile<TileType::Vec, T, kDstRows_, kDstCols_, BLayout::RowMajor, -1, -1>;

    SrcTileData srcTile(kSrcRows_, kSrcCols_);
    DstTileData dstTile(kDstRows_, kDstCols_);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x0 + kSrcRows_ * kSrcCols_ * sizeof(T));

    GlobalSrc srcGlobal(src);
    GlobalDst dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TSCATTER<maskPattern>(dstTile, srcTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

// --- float launchers ---

extern "C" __global__ AICORE void launchTSCATTER_FP0101(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTScatterMasked<float, FLOAT_P0101_ROW, FLOAT_P0101_COL / 2, FLOAT_P0101_ROW, FLOAT_P0101_COL,
                      MaskPattern::P0101>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ AICORE void launchTSCATTER_FP1010(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTScatterMasked<float, FLOAT_P1010_ROW, FLOAT_P1010_COL / 2, FLOAT_P1010_ROW, FLOAT_P1010_COL,
                      MaskPattern::P1010>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ AICORE void launchTSCATTER_FP0001(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTScatterMasked<float, FLOAT_P0001_ROW, FLOAT_P0001_COL / 4, FLOAT_P0001_ROW, FLOAT_P0001_COL,
                      MaskPattern::P0001>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ AICORE void launchTSCATTER_FP0010(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTScatterMasked<float, FLOAT_P0010_ROW, FLOAT_P0010_COL / 4, FLOAT_P0010_ROW, FLOAT_P0010_COL,
                      MaskPattern::P0010>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ AICORE void launchTSCATTER_FP0100(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTScatterMasked<float, FLOAT_P0100_ROW, FLOAT_P0100_COL / 4, FLOAT_P0100_ROW, FLOAT_P0100_COL,
                      MaskPattern::P0100>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ AICORE void launchTSCATTER_FP1000(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTScatterMasked<float, FLOAT_P1000_ROW, FLOAT_P1000_COL / 4, FLOAT_P1000_ROW, FLOAT_P1000_COL,
                      MaskPattern::P1000>(reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

extern "C" __global__ AICORE void launchTSCATTER_FP1111(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTScatterMasked<float, FLOAT_P1111_ROW, FLOAT_P1111_COL, FLOAT_P1111_ROW, FLOAT_P1111_COL, MaskPattern::P1111>(
        reinterpret_cast<__gm__ float *>(out), reinterpret_cast<__gm__ float *>(src));
}

// --- half launchers ---

extern "C" __global__ AICORE void launchTSCATTER_HP0101(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTScatterMasked<half, HALF_P0101_ROW, HALF_P0101_COL / 2, HALF_P0101_ROW, HALF_P0101_COL, MaskPattern::P0101>(
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ AICORE void launchTSCATTER_HP1010(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTScatterMasked<half, HALF_P1010_ROW, HALF_P1010_COL / 2, HALF_P1010_ROW, HALF_P1010_COL, MaskPattern::P1010>(
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ AICORE void launchTSCATTER_HP0001(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTScatterMasked<half, HALF_P0001_ROW, HALF_P0001_COL / 4, HALF_P0001_ROW, HALF_P0001_COL, MaskPattern::P0001>(
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ AICORE void launchTSCATTER_HP0100(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTScatterMasked<half, HALF_P0100_ROW, HALF_P0100_COL / 4, HALF_P0100_ROW, HALF_P0100_COL, MaskPattern::P0100>(
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

extern "C" __global__ AICORE void launchTSCATTER_HP1000(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTScatterMasked<half, HALF_P1000_ROW, HALF_P1000_COL / 4, HALF_P1000_ROW, HALF_P1000_COL, MaskPattern::P1000>(
        reinterpret_cast<__gm__ half *>(out), reinterpret_cast<__gm__ half *>(src));
}

// --- uint16/int16/uint32/int32 launchers ---

extern "C" __global__ AICORE void launchTSCATTER_U16P0101(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTScatterMasked<uint16_t, HALF_P0101_ROW, HALF_P0101_COL / 2, HALF_P0101_ROW, HALF_P0101_COL, MaskPattern::P0101>(
        reinterpret_cast<__gm__ uint16_t *>(out), reinterpret_cast<__gm__ uint16_t *>(src));
}

extern "C" __global__ AICORE void launchTSCATTER_U16P1010(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTScatterMasked<uint16_t, HALF_P1010_ROW, HALF_P1010_COL / 2, HALF_P1010_ROW, HALF_P1010_COL, MaskPattern::P1010>(
        reinterpret_cast<__gm__ uint16_t *>(out), reinterpret_cast<__gm__ uint16_t *>(src));
}

extern "C" __global__ AICORE void launchTSCATTER_I16P0001(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTScatterMasked<int16_t, HALF_P0001_ROW, HALF_P0001_COL / 4, HALF_P0001_ROW, HALF_P0001_COL, MaskPattern::P0001>(
        reinterpret_cast<__gm__ int16_t *>(out), reinterpret_cast<__gm__ int16_t *>(src));
}

extern "C" __global__ AICORE void launchTSCATTER_I16P0010(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTScatterMasked<int16_t, HALF_P0010_ROW, HALF_P0010_COL / 4, HALF_P0010_ROW, HALF_P0010_COL, MaskPattern::P0010>(
        reinterpret_cast<__gm__ int16_t *>(out), reinterpret_cast<__gm__ int16_t *>(src));
}

extern "C" __global__ AICORE void launchTSCATTER_U32P0100(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTScatterMasked<uint32_t, FLOAT_P0100_ROW, FLOAT_P0100_COL / 4, FLOAT_P0100_ROW, FLOAT_P0100_COL,
                      MaskPattern::P0100>(reinterpret_cast<__gm__ uint32_t *>(out),
                                          reinterpret_cast<__gm__ uint32_t *>(src));
}

extern "C" __global__ AICORE void launchTSCATTER_I32P1000(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTScatterMasked<int32_t, FLOAT_P1000_ROW, FLOAT_P1000_COL / 4, FLOAT_P1000_ROW, FLOAT_P1000_COL,
                      MaskPattern::P1000>(reinterpret_cast<__gm__ int32_t *>(out),
                                          reinterpret_cast<__gm__ int32_t *>(src));
}

extern "C" __global__ AICORE void launchTSCATTER_I32P1111(__gm__ uint8_t *out, __gm__ uint8_t *src)
{
    runTScatterMasked<int32_t, FLOAT_P1111_ROW, FLOAT_P1111_COL, FLOAT_P1111_ROW, FLOAT_P1111_COL, MaskPattern::P1111>(
        reinterpret_cast<__gm__ int32_t *>(out), reinterpret_cast<__gm__ int32_t *>(src));
}

// --- dispatch ---

template <int32_t tilingKey>
void launchTSCATTER_masked(uint8_t *out, uint8_t *src, void *stream)
{
    if constexpr (tilingKey == FP0101) {
        launchTSCATTER_FP0101(out, src);
    } else if constexpr (tilingKey == FP1010) {
        launchTSCATTER_FP1010(out, src);
    } else if constexpr (tilingKey == FP0001) {
        launchTSCATTER_FP0001(out, src);
    } else if constexpr (tilingKey == FP0010) {
        launchTSCATTER_FP0010(out, src);
    } else if constexpr (tilingKey == FP0100) {
        launchTSCATTER_FP0100(out, src);
    } else if constexpr (tilingKey == FP1000) {
        launchTSCATTER_FP1000(out, src);
    } else if constexpr (tilingKey == FP1111) {
        launchTSCATTER_FP1111(out, src);
    } else if constexpr (tilingKey == HP0101) {
        launchTSCATTER_HP0101(out, src);
    } else if constexpr (tilingKey == HP1010) {
        launchTSCATTER_HP1010(out, src);
    } else if constexpr (tilingKey == HP0001) {
        launchTSCATTER_HP0001(out, src);
    } else if constexpr (tilingKey == HP0100) {
        launchTSCATTER_HP0100(out, src);
    } else if constexpr (tilingKey == HP1000) {
        launchTSCATTER_HP1000(out, src);
    } else if constexpr (tilingKey == U16P0101) {
        launchTSCATTER_U16P0101(out, src);
    } else if constexpr (tilingKey == U16P1010) {
        launchTSCATTER_U16P1010(out, src);
    } else if constexpr (tilingKey == I16P0001) {
        launchTSCATTER_I16P0001(out, src);
    } else if constexpr (tilingKey == I16P0010) {
        launchTSCATTER_I16P0010(out, src);
    } else if constexpr (tilingKey == U32P0100) {
        launchTSCATTER_U32P0100(out, src);
    } else if constexpr (tilingKey == I32P1000) {
        launchTSCATTER_I32P1000(out, src);
    } else if constexpr (tilingKey == I32P1111) {
        launchTSCATTER_I32P1111(out, src);
    }
}

template void launchTSCATTER_masked<FP0101>(uint8_t *out, uint8_t *src, void *stream);
template void launchTSCATTER_masked<FP1010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTSCATTER_masked<FP0001>(uint8_t *out, uint8_t *src, void *stream);
template void launchTSCATTER_masked<FP0010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTSCATTER_masked<FP0100>(uint8_t *out, uint8_t *src, void *stream);
template void launchTSCATTER_masked<FP1000>(uint8_t *out, uint8_t *src, void *stream);
template void launchTSCATTER_masked<FP1111>(uint8_t *out, uint8_t *src, void *stream);
template void launchTSCATTER_masked<HP0101>(uint8_t *out, uint8_t *src, void *stream);
template void launchTSCATTER_masked<HP1010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTSCATTER_masked<HP0001>(uint8_t *out, uint8_t *src, void *stream);
template void launchTSCATTER_masked<HP0100>(uint8_t *out, uint8_t *src, void *stream);
template void launchTSCATTER_masked<HP1000>(uint8_t *out, uint8_t *src, void *stream);
template void launchTSCATTER_masked<U16P0101>(uint8_t *out, uint8_t *src, void *stream);
template void launchTSCATTER_masked<U16P1010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTSCATTER_masked<I16P0001>(uint8_t *out, uint8_t *src, void *stream);
template void launchTSCATTER_masked<I16P0010>(uint8_t *out, uint8_t *src, void *stream);
template void launchTSCATTER_masked<U32P0100>(uint8_t *out, uint8_t *src, void *stream);
template void launchTSCATTER_masked<I32P1000>(uint8_t *out, uint8_t *src, void *stream);
template void launchTSCATTER_masked<I32P1111>(uint8_t *out, uint8_t *src, void *stream);
