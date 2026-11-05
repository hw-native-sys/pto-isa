/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdlib>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/common/debug.h>
#include <pto/common/pto_tile.hpp>

using namespace pto;

/*-----------------Static shapes------------------*/

template <typename T, int N, int D, int C1, int H, int W, int C0>
__global__ AICORE void runTStoreConv_NDC1HWC0(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr size_t kC0 = 32 / sizeof(T);
    static_assert(C0 == kC0);

    // Define the dimensions for readability
    constexpr int64_t W_dim = W;
    constexpr int64_t H_dim = H;
    constexpr int64_t C1_dim = C1;
    constexpr int64_t D_dim = D;

    using ShapeDim5 = Shape<N, D, C1, H, W>;
    using StrideDim5 = pto::Stride<
        D_dim * C1_dim * H_dim * W_dim * C0, // Stride for N (Total elements)
        C1_dim * H_dim * W_dim * C0,         // Stride for D
        H_dim * W_dim * C0,                  // Stride for C1
        W_dim * C0,                          // Stride for H (Crucial: W * C0)
        C0                                   // Stride for W (Jump one full C0 vector)
        >;
    using GlobalData = GlobalTensor<T, ShapeDim5, StrideDim5, Layout::NDC1HWC0>;

    constexpr size_t srcElemNum = N * D * C1 * H * W * C0;
    constexpr size_t srcBufferSize = srcElemNum * sizeof(T);

    using SrcConvTile = ConvTile<TileType::Mat, T, srcBufferSize, Layout::NDC1HWC0, ConvTileShape<N, D, C1, H, W, C0>>;
    SrcConvTile srcTile;
    static_assert(srcTile.totalDimCount == 6);

    TASSIGN(srcTile, 0x0);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    TSTORE(dstGlobal, srcTile);
}

template <typename T, int N, int C1, int H, int W, int C0>
__global__ AICORE void runTStoreConv_NC1HWC0(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr size_t kC0 = 32 / sizeof(T);
    static_assert(C0 == kC0);

    // Define the dimensions for readability
    constexpr int64_t W_dim = W;
    constexpr int64_t H_dim = H;
    constexpr int64_t C1_dim = C1;

    using ShapeDim5 = Shape<N, C1, H, W, C0>;
    using StrideDim5 = pto::Stride<
        C1_dim * H_dim * W_dim * C0, // Stride for N
        H_dim * W_dim * C0,          // Stride for C1
        W_dim * C0,                  // Stride for H
        C0,                          // Stride for W
        1                            // Stride for C0 (contiguous)
        >;
    using GlobalData = GlobalTensor<T, ShapeDim5, StrideDim5, Layout::NC1HWC0>;

    constexpr size_t srcElemNum = N * C1 * H * W * C0;
    constexpr size_t srcBufferSize = srcElemNum * sizeof(T);

    using SrcConvTile = ConvTile<TileType::Mat, T, srcBufferSize, Layout::NC1HWC0, ConvTileShape<N, C1, H, W>>;
    SrcConvTile srcTile;
    static_assert(srcTile.totalDimCount == 4);

    TASSIGN(srcTile, 0x0);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    TSTORE(dstGlobal, srcTile);
}

/*------------------Dynamic dims------------------*/
// One dynamic tile dimension (N).
template <typename T, int kWholeN, int D, int C1, int H, int W, int C0>
__global__ AICORE void runTStoreConvDynN(__gm__ T __out__* out, __gm__ T __in__* src, int64_t n)
{
    constexpr size_t kC0 = 32 / sizeof(T);
    static_assert(C0 == kC0);

    // Define the dimensions for readability
    constexpr int64_t W_dim = W;
    constexpr int64_t H_dim = H;
    constexpr int64_t C1_dim = C1;
    constexpr int64_t D_dim = D;

    // Whole (physical) global tensor.
    using ShapeDim5 = Shape<kWholeN, D, C1, H, W>;
    using StrideDim5 = pto::Stride<
        D_dim * C1_dim * H_dim * W_dim * C0, // Stride for N
        C1_dim * H_dim * W_dim * C0,         // Stride for D
        H_dim * W_dim * C0,                  // Stride for C1
        W_dim * C0,                          // Stride for H
        C0                                   // Stride for W
        >;
    using GlobalData = GlobalTensor<T, ShapeDim5, StrideDim5, Layout::NDC1HWC0>;

    constexpr size_t wholeElemNum = kWholeN * D * C1 * H * W * C0;
    constexpr size_t bufferSize = wholeElemNum * sizeof(T);

    // N (dim 0) is DYNAMIC, its runtime value comes in as a constructor argument.
    using SrcConvTile =
        ConvTile<TileType::Mat, T, bufferSize, Layout::NDC1HWC0, ConvTileShape<pto::DYNAMIC, D, C1, H, W, C0>>;
    SrcConvTile srcTile(n);
    static_assert(srcTile.totalDimCount == 6);
    static_assert(srcTile.dynamicDimCount == 1);

    TASSIGN(srcTile, 0x0);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    TSTORE(dstGlobal, srcTile);
}

// Two dynamic tile dimensions (N and D).
template <typename T, int kWholeN, int kWholeD, int C1, int H, int W, int C0>
__global__ AICORE void runTStoreConvDynND(__gm__ T __out__* out, __gm__ T __in__* src, int64_t n, int64_t d)
{
    constexpr size_t kC0 = 32 / sizeof(T);
    static_assert(C0 == kC0);

    // Define the dimensions for readability
    constexpr int64_t W_dim = W;
    constexpr int64_t H_dim = H;
    constexpr int64_t C1_dim = C1;

    using ShapeDim5 = Shape<kWholeN, kWholeD, C1, H, W>;
    using StrideDim5 = pto::Stride<
        kWholeD * C1_dim * H_dim * W_dim * C0, // Stride for N
        C1_dim * H_dim * W_dim * C0,           // Stride for D
        H_dim * W_dim * C0,                    // Stride for C1
        W_dim * C0,                            // Stride for H
        C0                                     // Stride for W
        >;
    using GlobalData = GlobalTensor<T, ShapeDim5, StrideDim5, Layout::NDC1HWC0>;

    constexpr size_t wholeElemNum = kWholeN * kWholeD * C1 * H * W * C0;
    constexpr size_t bufferSize = wholeElemNum * sizeof(T);

    // N (dim 0) and D (dim 1) are DYNAMIC: runtime values are passed in order of
    // appearance of the DYNAMIC dimensions.
    using SrcConvTile = ConvTile<
        TileType::Mat, T, bufferSize, Layout::NDC1HWC0, ConvTileShape<pto::DYNAMIC, pto::DYNAMIC, C1, H, W, C0>>;
    SrcConvTile srcTile(n, d);
    static_assert(srcTile.totalDimCount == 6);
    static_assert(srcTile.dynamicDimCount == 2);

    TASSIGN(srcTile, 0x0);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    TSTORE(dstGlobal, srcTile);
}

// Two dynamic tile dimensions (N and D) plus dynamic GlobalTensor shape and stride.
template <typename T, int kWholeN, int kWholeD, int C1, int H, int W, int C0>
__global__ AICORE void runTStoreConvDynNDGlobal(__gm__ T __out__* out, __gm__ T __in__* src, int64_t n, int64_t d)
{
    constexpr size_t kC0 = 32 / sizeof(T);
    static_assert(C0 == kC0);

    // Define the dimensions for readability
    constexpr int64_t W_dim = W;
    constexpr int64_t H_dim = H;
    constexpr int64_t C1_dim = C1;
    constexpr int64_t kWholeD_dim = kWholeD;

    // Global shape and stride are DYNAMIC: runtime values are passed to the GlobalTensor
    // constructor. Only the leading two dims vary at runtime, the rest stay static.
    using DynShape = Shape<pto::DYNAMIC, pto::DYNAMIC, C1, H, W>;
    using DynStride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, H_dim * W_dim * C0, W_dim * C0, C0>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride, Layout::NDC1HWC0>;

    constexpr size_t wholeElemNum = kWholeN * kWholeD * C1 * H * W * C0;
    constexpr size_t bufferSize = wholeElemNum * sizeof(T);

    using SrcConvTile = ConvTile<
        TileType::Mat, T, bufferSize, Layout::NDC1HWC0, ConvTileShape<pto::DYNAMIC, pto::DYNAMIC, C1, H, W, C0>>;
    SrcConvTile srcTile(n, d);
    static_assert(srcTile.totalDimCount == 6);
    static_assert(srcTile.dynamicDimCount == 2);

    TASSIGN(srcTile, 0x0);

    // Runtime shape is the actual (sub-block) size, runtime strides keep the whole-shape
    // layout, so the tile addresses a sub-block of the whole tensor.
    DynShape runtimeShape(n, d, 0, 0, 0);
    DynStride runtimeStride(kWholeD_dim * C1_dim * H_dim * W_dim * C0, C1_dim * H_dim * W_dim * C0, 0, 0, 0);

    GlobalData srcGlobal(src, runtimeShape, runtimeStride);
    GlobalData dstGlobal(out, runtimeShape, runtimeStride);

    TLOAD(srcTile, srcGlobal);
    TSTORE(dstGlobal, srcTile);
}

/*-----------------ValidShape cases------------------*/
// Only the valid box is written to GM; the host pre-fills the output with a sentinel and the golden
// keeps that sentinel outside the valid box.
template <typename T, int N, int C1, int H, int W, int vN, int vC1, int vH, int vW, bool Dynamic>
__global__ AICORE void runTStoreConvValid_NC1HWC0(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr size_t kC0 = 32 / sizeof(T);
    constexpr size_t bufferSize = size_t(N) * C1 * H * W * kC0 * sizeof(T);

    using ShapeDim5 = Shape<N, C1, H, W, kC0>;
    using StrideDim5 = pto::Stride<(int64_t)C1 * H * W * kC0, (int64_t)H * W * kC0, (int64_t)W * kC0, kC0, 1>;
    using GlobalData = GlobalTensor<T, ShapeDim5, StrideDim5, Layout::NC1HWC0>;

    using TShape = ConvTileShape<N, C1, H, W>;
    using VShape = std::conditional_t<
        Dynamic, ConvTileShape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>, ConvTileShape<vN, vC1, vH, vW>>;
    using SrcConvTile = ConvTile<TileType::Mat, T, bufferSize, Layout::NC1HWC0, TShape, VShape>;

    SrcConvTile srcTile;
    static_assert(srcTile.totalDimCount == 4);
    TASSIGN(srcTile, 0x0);
    if constexpr (Dynamic) {
        srcTile.SetValidShape(int64_t(vN), int64_t(vC1), int64_t(vH), int64_t(vW));
    }

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);
    TLOAD(srcTile, srcGlobal);
    TSTORE(dstGlobal, srcTile);
}

template <typename T, int N, int D, int C1, int H, int W, int vN, int vD, int vC1, int vH, int vW, bool Dynamic>
__global__ AICORE void runTStoreConvValid_NDC1HWC0(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr size_t kC0 = 32 / sizeof(T);
    constexpr size_t bufferSize = size_t(N) * D * C1 * H * W * kC0 * sizeof(T);

    using ShapeDim5 = Shape<N, D, C1, H, W>;
    using StrideDim5 = pto::Stride<
        (int64_t)D * C1 * H * W * kC0, (int64_t)C1 * H * W * kC0, (int64_t)H * W * kC0, (int64_t)W * kC0, kC0>;
    using GlobalData = GlobalTensor<T, ShapeDim5, StrideDim5, Layout::NDC1HWC0>;

    using TShape = ConvTileShape<N, D, C1, H, W, kC0>;
    using VShape = std::conditional_t<
        Dynamic, ConvTileShape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>,
        ConvTileShape<vN, vD, vC1, vH, vW, kC0>>;
    using SrcConvTile = ConvTile<TileType::Mat, T, bufferSize, Layout::NDC1HWC0, TShape, VShape>;

    SrcConvTile srcTile;
    static_assert(srcTile.totalDimCount == 6);
    TASSIGN(srcTile, 0x0);
    if constexpr (Dynamic) {
        srcTile.SetValidShape(int64_t(vN), int64_t(vD), int64_t(vC1), int64_t(vH), int64_t(vW), int64_t(kC0));
    }

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);
    TLOAD(srcTile, srcGlobal);
    TSTORE(dstGlobal, srcTile);
}

// Flat FRACTAL_Z target view [1, C1HW, N/16, 16, C0]; the tile is [C1HW, N/16, 16, C0].
template <typename T, int D0, int D1, int v0, int v1, int v2>
__global__ AICORE void runTStoreConvValid_FractalZ4D(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr int C0 = 32 / sizeof(T);
    constexpr int N0 = 16;
    constexpr size_t bufferSize = size_t(D0) * D1 * N0 * C0 * sizeof(T);

    using GShape = Shape<1, D0, D1, N0, C0>;
    using GStride = Stride<(int64_t)D0 * D1 * N0 * C0, (int64_t)D1 * N0 * C0, (int64_t)N0 * C0, C0, 1>;
    using GlobalData = GlobalTensor<T, GShape, GStride, Layout::FRACTAL_Z>;

    using TShape = ConvTileShape<D0, D1, N0, C0>;
    using VShape = ConvTileShape<v0, v1, v2, C0>;
    using SrcConvTile = ConvTile<TileType::Mat, T, bufferSize, Layout::FRACTAL_Z, TShape, VShape>;

    SrcConvTile srcTile;
    static_assert(srcTile.totalDimCount == 4);
    TASSIGN(srcTile, 0x0);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);
    TLOAD(srcTile, srcGlobal);
    TSTORE(dstGlobal, srcTile);
}

template <typename T, int C1, int H, int W, int N, int vC1, int vH, int vW, int vN, bool Dynamic>
__global__ AICORE void runTStoreConvValid_FractalZ5D(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr int C0 = 32 / sizeof(T);
    constexpr size_t bufferSize = size_t(C1) * H * W * N * C0 * sizeof(T);

    using GShape = Shape<C1, H, W, N, C0>;
    using GStride = Stride<(int64_t)H * W * N * C0, (int64_t)W * N * C0, (int64_t)N * C0, C0, 1>;
    using GlobalData = GlobalTensor<T, GShape, GStride, Layout::FRACTAL_Z>;

    using TShape = ConvTileShape<C1, H, W, N, C0>;
    using VShape = std::conditional_t<
        Dynamic, ConvTileShape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>,
        ConvTileShape<vC1, vH, vW, vN, C0>>;
    using SrcConvTile = ConvTile<TileType::Mat, T, bufferSize, Layout::FRACTAL_Z, TShape, VShape>;

    SrcConvTile srcTile;
    static_assert(srcTile.totalDimCount == 5);
    TASSIGN(srcTile, 0x0);
    if constexpr (Dynamic) {
        srcTile.SetValidShape(int64_t(vC1), int64_t(vH), int64_t(vW), int64_t(vN), int64_t(C0));
    }

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);
    TLOAD(srcTile, srcGlobal);
    TSTORE(dstGlobal, srcTile);
}

enum class TStoreConvCase {
    NDC1HWC0Static,
    NC1HWC0Static,
    NDC1HWC0DynN,
    NDC1HWC0DynND,
    NDC1HWC0DynNDGlobal,
    NC1HWC0ValidStatic,
    NC1HWC0ValidDynamic,
    NDC1HWC0ValidStatic,
    NDC1HWC0ValidDynamic,
    FRACTAL_Z4DValidStatic,
    FRACTAL_Z5DValidStatic,
    FRACTAL_Z5DValidDynamic,
};

template <typename T, TStoreConvCase caseKey, int s0, int s1, int s2, int s3, int s4>
void LaunchTStoreConv(T* out, T* src, int64_t n, int64_t d, void* stream)
{
    (void)stream;
    if constexpr (caseKey == TStoreConvCase::NDC1HWC0Static) {
        (void)n;
        (void)d;
        runTStoreConv_NDC1HWC0<T, s0, s1, s2, s3, s4, 32 / sizeof(T)>(out, src);
    } else if constexpr (caseKey == TStoreConvCase::NC1HWC0Static) {
        (void)n;
        (void)d;
        (void)s4;
        runTStoreConv_NC1HWC0<T, s0, s1, s2, s3, 32 / sizeof(T)>(out, src);
    } else if constexpr (caseKey == TStoreConvCase::NDC1HWC0DynN) {
        (void)d;
        runTStoreConvDynN<T, s0, s1, s2, s3, s4, 32 / sizeof(T)>(out, src, n);
    } else if constexpr (caseKey == TStoreConvCase::NDC1HWC0DynND) {
        runTStoreConvDynND<T, s0, s1, s2, s3, s4, 32 / sizeof(T)>(out, src, n, d);
    } else if constexpr (caseKey == TStoreConvCase::NDC1HWC0DynNDGlobal) {
        runTStoreConvDynNDGlobal<T, s0, s1, s2, s3, s4, 32 / sizeof(T)>(out, src, n, d);
    }
}

template <
    typename T, TStoreConvCase caseKey, int s0, int s1, int s2, int s3, int s4, int v0, int v1, int v2, int v3, int v4>
void LaunchTStoreConvValid(T* out, T* src, void* stream)
{
    (void)stream;
    if constexpr (caseKey == TStoreConvCase::NC1HWC0ValidStatic) {
        (void)v4;
        runTStoreConvValid_NC1HWC0<T, s0, s1, s2, s3, v0, v1, v2, v3, false>(out, src);
    } else if constexpr (caseKey == TStoreConvCase::NC1HWC0ValidDynamic) {
        (void)v4;
        runTStoreConvValid_NC1HWC0<T, s0, s1, s2, s3, v0, v1, v2, v3, true>(out, src);
    } else if constexpr (caseKey == TStoreConvCase::NDC1HWC0ValidStatic) {
        runTStoreConvValid_NDC1HWC0<T, s0, s1, s2, s3, s4, v0, v1, v2, v3, v4, false>(out, src);
    } else if constexpr (caseKey == TStoreConvCase::NDC1HWC0ValidDynamic) {
        runTStoreConvValid_NDC1HWC0<T, s0, s1, s2, s3, s4, v0, v1, v2, v3, v4, true>(out, src);
    } else if constexpr (caseKey == TStoreConvCase::FRACTAL_Z4DValidStatic) {
        (void)s2;
        (void)s3;
        (void)s4;
        (void)v3;
        (void)v4;
        runTStoreConvValid_FractalZ4D<T, s0, s1, v0, v1, v2>(out, src);
    } else if constexpr (caseKey == TStoreConvCase::FRACTAL_Z5DValidStatic) {
        (void)v4;
        runTStoreConvValid_FractalZ5D<T, s0, s1, s2, s3, v0, v1, v2, v3, false>(out, src);
    } else if constexpr (caseKey == TStoreConvCase::FRACTAL_Z5DValidDynamic) {
        (void)v4;
        runTStoreConvValid_FractalZ5D<T, s0, s1, s2, s3, v0, v1, v2, v3, true>(out, src);
    }
}

template void LaunchTStoreConv<float, TStoreConvCase::NDC1HWC0Static, 1, 1, 1, 2, 8>(
    float* out, float* src, int64_t n, int64_t d, void* stream);
template void LaunchTStoreConv<float, TStoreConvCase::NDC1HWC0Static, 2, 3, 4, 1, 7>(
    float* out, float* src, int64_t n, int64_t d, void* stream);
template void LaunchTStoreConv<int32_t, TStoreConvCase::NDC1HWC0Static, 1, 2, 4, 16, 8>(
    int32_t* out, int32_t* src, int64_t n, int64_t d, void* stream);
template void LaunchTStoreConv<float, TStoreConvCase::NDC1HWC0Static, 3, 2, 3, 3, 5>(
    float* out, float* src, int64_t n, int64_t d, void* stream);
template void LaunchTStoreConv<int32_t, TStoreConvCase::NDC1HWC0Static, 3, 3, 5, 3, 3>(
    int32_t* out, int32_t* src, int64_t n, int64_t d, void* stream);
template void LaunchTStoreConv<float, TStoreConvCase::NC1HWC0Static, 3, 3, 5, 3, 1>(
    float* out, float* src, int64_t n, int64_t d, void* stream);
template void LaunchTStoreConv<float, TStoreConvCase::NDC1HWC0DynN, 4, 1, 1, 2, 8>(
    float* out, float* src, int64_t n, int64_t d, void* stream);
template void LaunchTStoreConv<float, TStoreConvCase::NDC1HWC0DynND, 4, 3, 1, 7, 8>(
    float* out, float* src, int64_t n, int64_t d, void* stream);
template void LaunchTStoreConv<int32_t, TStoreConvCase::NDC1HWC0DynN, 3, 2, 4, 8, 8>(
    int32_t* out, int32_t* src, int64_t n, int64_t d, void* stream);
template void LaunchTStoreConv<float, TStoreConvCase::NDC1HWC0DynNDGlobal, 4, 3, 2, 4, 8>(
    float* out, float* src, int64_t n, int64_t d, void* stream);
template void LaunchTStoreConv<float, TStoreConvCase::NDC1HWC0DynN, 4, 3, 3, 3, 5>(
    float* out, float* src, int64_t n, int64_t d, void* stream);

// ValidShape instantiations
template void LaunchTStoreConvValid<float, TStoreConvCase::NC1HWC0ValidStatic, 2, 3, 4, 5, 1, 1, 2, 3, 4, 1>(
    float* out, float* src, void* stream);
template void LaunchTStoreConvValid<int32_t, TStoreConvCase::NC1HWC0ValidDynamic, 2, 3, 4, 5, 1, 1, 2, 3, 4, 1>(
    int32_t* out, int32_t* src, void* stream);
template void LaunchTStoreConvValid<float, TStoreConvCase::NDC1HWC0ValidStatic, 2, 3, 2, 4, 5, 1, 2, 1, 3, 4>(
    float* out, float* src, void* stream);
template void LaunchTStoreConvValid<int32_t, TStoreConvCase::NDC1HWC0ValidDynamic, 2, 3, 2, 4, 5, 1, 2, 1, 3, 4>(
    int32_t* out, int32_t* src, void* stream);
// FRACTAL_Z 4D: flat target [1, 8, 2, 16, 16], valid (6, 1, 10, C0)
template void LaunchTStoreConvValid<half, TStoreConvCase::FRACTAL_Z4DValidStatic, 8, 2, 16, 1, 1, 6, 1, 10, 1, 1>(
    half* out, half* src, void* stream);
// FRACTAL_Z 5D: target (4, 2, 6, 16, C0=32), valid (3, 1, 4, 10, C0)
template void LaunchTStoreConvValid<int8_t, TStoreConvCase::FRACTAL_Z5DValidStatic, 4, 2, 6, 16, 1, 3, 1, 4, 10, 1>(
    int8_t* out, int8_t* src, void* stream);
// FRACTAL_Z 5D dynamic valid: target (2, 3, 4, 8, C0=32), valid (1, 2, 3, 6, C0)
template void LaunchTStoreConvValid<int8_t, TStoreConvCase::FRACTAL_Z5DValidDynamic, 2, 3, 4, 8, 1, 1, 2, 3, 6, 1>(
    int8_t* out, int8_t* src, void* stream);
