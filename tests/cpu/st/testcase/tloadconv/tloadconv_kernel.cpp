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
#include <limits>
#include <algorithm>

using namespace std;
using namespace pto;

using namespace pto;

// Unified TLOAD runner: Static shapes only
template <typename T, int N, int C1, int H, int W, Layout L>
void runTloadDynamic(__gm__ T* out, __gm__ T* src)
{
    constexpr int C0 = 32 / sizeof(T);
    constexpr uint32_t totalElements = N * C1 * H * W * C0;
    constexpr uint32_t bufferSize = totalElements * sizeof(T);

    // 1. Static Strides
    using StrideNC1HWC0 = Stride<(int64_t)C1 * H * W * C0, (int64_t)H * W * C0, (int64_t)W * C0, (int64_t)C0, 1>;
    using StrideFractalZ = Stride<(int64_t)H * W * N * C0, (int64_t)W * N * C0, (int64_t)N * C0, (int64_t)C0, 1>;
    using SelectedStride = std::conditional_t<L == Layout::NC1HWC0, StrideNC1HWC0, StrideFractalZ>;

    // 2. Global Shape
    using GShape = std::conditional_t<L == Layout::NC1HWC0, Shape<N, C1, H, W, C0>, Shape<C1, H, W, N, C0>>;

    // 4. Initialize Global Tensor via CONSTRUCTOR (since SetAddr is private)
    GlobalTensor<T, GShape, SelectedStride, L> srcGlobal(src);

    // 3. Dynamic Tile Shape (Keeping required statics [16, C0])
    using TShape = std::conditional_t<
        L == Layout::NC1HWC0, ConvTileShape<N, C1, H, W>, ConvTileShape<(C1 * H * W), (N / 16), 16, C0>>;

    // 5. Initialize ConvTile
    using MyTile = ConvTile<TileType::Mat, T, bufferSize, L, TShape>;
    MyTile convTile;

    // 5. Allocate and Manually Assign Dummy L1/UB Memory
    // We use a vector for automatic cleanup to prevent memory leaks in the sim
    std::vector<T> localBuffer(totalElements);

    // Explicitly cast the RAM address to the expected TileDType
    convTile.data() = reinterpret_cast<typename MyTile::TileDType>(localBuffer.data());

    // 6. Execute
    TLOAD(convTile, srcGlobal);

    // 7. Verification Copy
    for (uint32_t i = 0; i < totalElements; i++) {
        out[i] = convTile.data()[i];
    }
}

// Specialized runner for 5D Fractal Z Tiles
template <typename T, int C1, int H, int W, int N>
void runTloadFractalZ5D(__gm__ T* out, __gm__ T* src)
{
    // 1. Calculate C0 based on data type (32 bytes / size of T)
    constexpr int C0 = 32 / sizeof(T);

    // 2. Hardware and buffer constants
    constexpr uint32_t totalElements = C1 * H * W * N * C0;
    constexpr uint32_t bufferSize = totalElements * sizeof(T);
    constexpr Layout L = Layout::FRACTAL_Z;

    // 3. Static Strides for Global Memory: [C1, H, W, N, C0]
    using StrideFractalZ = Stride<
        (int64_t)H * W * N * C0, // S0: Jump between C1 groups
        (int64_t)W * N * C0,     // S1: Jump between H rows
        (int64_t)N * C0,         // S2: Jump between W columns
        (int64_t)C0,             // S3: Jump between N fractals
        1                        // S4: Contiguous C0
        >;

    // 4. Global Shape (5D)
    using GShape = Shape<C1, H, W, N, C0>;

    // 5. Initialize Global Tensor
    GlobalTensor<T, GShape, StrideFractalZ, L> srcGlobal(src);

    // 6. 5D Tile Shape for Fractal Z: [C1, H, W, N, C0]
    // Matches your TLoad logic: dstShape0(C1), dstShape1(H), dstShape2(W), dstShape3(N)
    using TShape = ConvTileShape<C1, H, W, N, C0>;

    // 7. Initialize ConvTile
    using MyTile = ConvTile<TileType::Mat, T, bufferSize, L, TShape>;
    MyTile convTile;

    // 8. Allocate and Manually Assign Dummy L1/UB Memory (CPU Simulation)
    std::vector<T> localBuffer(totalElements);
    convTile.data() = reinterpret_cast<typename MyTile::TileDType>(localBuffer.data());

    // 9. Execute (Uses the nBurst / gmGap logic you provided)
    TLOAD_IMPL(convTile, srcGlobal);

    // 10. Verification Copy: Move from assigned memory to output GM
    for (uint32_t i = 0; i < totalElements; i++) {
        out[i] = convTile.data()[i];
    }
}

// Specialized runner for NC1HWC0 Tiles with one dynamic tile dimension (N).
template <typename T, int kWholeN, int C1, int H, int W>
void runTloadNC1HWC0DynN(__gm__ T* out, __gm__ T* src, int64_t n)
{
    // 1. Calculate C0 based on data type (32 bytes / size of T)
    constexpr int C0 = 32 / sizeof(T);

    // 2. Hardware and buffer constants
    constexpr uint32_t totalElements = kWholeN * C1 * H * W * C0;
    constexpr uint32_t bufferSize = totalElements * sizeof(T);

    // 3. Static Strides for Global Memory: [kWholeN, C1, H, W, C0]
    // Strides follow the whole shape, so the runtime-sized tile addresses a sub-block
    using StrideNC1HWC0 = Stride<(int64_t)C1 * H * W * C0, (int64_t)H * W * C0, (int64_t)W * C0, (int64_t)C0, 1>;

    // 4. Global Shape (5D, whole)
    using GShape = Shape<kWholeN, C1, H, W, C0>;

    // 5. Initialize Global Tensor
    GlobalTensor<T, GShape, StrideNC1HWC0, Layout::NC1HWC0> srcGlobal(src);

    // 6. Tile Shape: N (dim 0) is DYNAMIC, its runtime value comes in as a constructor argument
    using TShape = ConvTileShape<pto::DYNAMIC, C1, H, W>;

    // 7. Initialize ConvTile
    using MyTile = ConvTile<TileType::Mat, T, bufferSize, Layout::NC1HWC0, TShape>;
    MyTile convTile(n);

    // 8. Allocate and Manually Assign Dummy L1/UB Memory (CPU Simulation)
    std::vector<T> localBuffer(totalElements);
    convTile.data() = reinterpret_cast<typename MyTile::TileDType>(localBuffer.data());

    // 9. Execute
    TLOAD(convTile, srcGlobal);

    // 10. Verification Copy: Move from assigned memory to output GM
    for (uint32_t i = 0; i < totalElements; i++) {
        out[i] = convTile.data()[i];
    }
}

// Specialized runner for NC1HWC0 Tiles with one dynamic tile dimension (N) plus dynamic GlobalTensor shape and stride.
template <typename T, int kWholeN, int C1, int H, int W>
void runTloadNC1HWC0DynNGlobal(__gm__ T* out, __gm__ T* src, int64_t n)
{
    // 1. Calculate C0 based on data type (32 bytes / size of T)
    constexpr int C0 = 32 / sizeof(T);

    // 2. Hardware and buffer constants
    constexpr uint32_t totalElements = kWholeN * C1 * H * W * C0;
    constexpr uint32_t bufferSize = totalElements * sizeof(T);

    // 3. Dynamic GlobalTensor types: N (dim 0) of both shape and stride is DYNAMIC
    using DynShape = Shape<pto::DYNAMIC, C1, H, W, C0>;
    using DynStride = Stride<pto::DYNAMIC, (int64_t)H * W * C0, (int64_t)W * C0, (int64_t)C0, 1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride, Layout::NC1HWC0>;

    // 4. Tile Shape: N (dim 0) is DYNAMIC, its runtime value comes in as a constructor argument
    using TShape = ConvTileShape<pto::DYNAMIC, C1, H, W>;

    // 5. Initialize ConvTile
    using MyTile = ConvTile<TileType::Mat, T, bufferSize, Layout::NC1HWC0, TShape>;
    MyTile convTile(n);

    // 6. Allocate and Manually Assign Dummy L1/UB Memory (CPU Simulation)
    std::vector<T> localBuffer(totalElements);
    convTile.data() = reinterpret_cast<typename MyTile::TileDType>(localBuffer.data());

    // 7. Initialize Global Tensor with the runtime shape and the whole-based stride
    DynShape runtimeShape(n, 0, 0, 0, 0);
    DynStride runtimeStride((int64_t)C1 * H * W * C0, 0, 0, 0, 0);
    GlobalData srcGlobal(src, runtimeShape, runtimeStride);

    // 8. Execute
    TLOAD(convTile, srcGlobal);

    // 9. Verification Copy: Move from assigned memory to output GM
    for (uint32_t i = 0; i < totalElements; i++) {
        out[i] = convTile.data()[i];
    }
}

// Specialized runner for static NDC1HWC0 Tiles.
template <typename T, int N, int D, int C1, int H, int W>
void runTloadNDC1HWC0(__gm__ T* out, __gm__ T* src)
{
    // 1. Calculate C0 based on data type (32 bytes / size of T)
    constexpr int C0 = 32 / sizeof(T);

    // 2. Hardware and buffer constants
    constexpr uint32_t totalElements = N * D * C1 * H * W * C0;
    constexpr uint32_t bufferSize = totalElements * sizeof(T);

    // 3. Static Strides for Global Memory: [N, D, C1, H, W], C0 folded into the strides
    using GShape = Shape<N, D, C1, H, W>;
    using GStride = Stride<
        (int64_t)D * C1 * H * W * C0, // Stride for N
        (int64_t)C1 * H * W * C0,     // Stride for D
        (int64_t)H * W * C0,          // Stride for C1
        (int64_t)W * C0,              // Stride for H
        (int64_t)C0                   // Stride for W
        >;

    // 4. Initialize Global Tensor
    GlobalTensor<T, GShape, GStride, Layout::NDC1HWC0> srcGlobal(src);

    // 5. 6D Tile Shape for NDC1HWC0: [N, D, C1, H, W, C0]
    using TShape = ConvTileShape<N, D, C1, H, W, C0>;

    // 6. Initialize ConvTile
    using MyTile = ConvTile<TileType::Mat, T, bufferSize, Layout::NDC1HWC0, TShape>;
    MyTile convTile;

    // 7. Allocate and Manually Assign Dummy L1/UB Memory (CPU Simulation)
    std::vector<T> localBuffer(totalElements);
    convTile.data() = reinterpret_cast<typename MyTile::TileDType>(localBuffer.data());

    // 8. Execute
    TLOAD(convTile, srcGlobal);

    // 9. Verification Copy: Move from assigned memory to output GM
    for (uint32_t i = 0; i < totalElements; i++) {
        out[i] = convTile.data()[i];
    }
}

// Unified Dispatcher for GTest
enum class TLoadConvCase {
    NC1HWC0Static,
    NC1HWC0DynN,
    NC1HWC0DynNGlobal,
    NDC1HWC0Static,
    FRACTAL_Z4DStatic,
    FRACTAL_Z5DStatic,
};

template <typename T, TLoadConvCase caseKey, int s0, int s1, int s2, int s3, int s4>
void LaunchTLoadConv(uint8_t* out, uint8_t* src, int64_t n, uint64_t* gLog, void* stream)
{
    (void)gLog;
    (void)stream;
    if constexpr (caseKey == TLoadConvCase::NC1HWC0Static) {
        (void)n;
        (void)s4;
        runTloadDynamic<T, s0, s1, s2, s3, Layout::NC1HWC0>((__gm__ T*)out, (__gm__ T*)src);
    } else if constexpr (caseKey == TLoadConvCase::NC1HWC0DynN) {
        (void)s4;
        runTloadNC1HWC0DynN<T, s0, s1, s2, s3>((__gm__ T*)out, (__gm__ T*)src, n);
    } else if constexpr (caseKey == TLoadConvCase::NC1HWC0DynNGlobal) {
        (void)s4;
        runTloadNC1HWC0DynNGlobal<T, s0, s1, s2, s3>((__gm__ T*)out, (__gm__ T*)src, n);
    } else if constexpr (caseKey == TLoadConvCase::NDC1HWC0Static) {
        (void)n;
        runTloadNDC1HWC0<T, s0, s1, s2, s3, s4>((__gm__ T*)out, (__gm__ T*)src);
    } else if constexpr (caseKey == TLoadConvCase::FRACTAL_Z4DStatic) {
        (void)n;
        (void)s4;
        runTloadDynamic<T, s0, s1, s2, s3, Layout::FRACTAL_Z>((__gm__ T*)out, (__gm__ T*)src);
    } else if constexpr (caseKey == TLoadConvCase::FRACTAL_Z5DStatic) {
        (void)n;
        (void)s4;
        runTloadFractalZ5D<T, s0, s1, s2, s3>((__gm__ T*)out, (__gm__ T*)src);
    }
}

// Template instantiations
template void LaunchTLoadConv<half, TLoadConvCase::NC1HWC0Static, 1, 2, 4, 4, 1>(
    uint8_t*, uint8_t*, int64_t, uint64_t*, void*);
template void LaunchTLoadConv<float, TLoadConvCase::NC1HWC0Static, 1, 4, 10, 10, 1>(
    uint8_t*, uint8_t*, int64_t, uint64_t*, void*);
template void LaunchTLoadConv<half, TLoadConvCase::FRACTAL_Z4DStatic, 16, 2, 1, 18, 1>(
    uint8_t*, uint8_t*, int64_t, uint64_t*, void*);
template void LaunchTLoadConv<int8_t, TLoadConvCase::FRACTAL_Z5DStatic, 4, 2, 6, 16, 1>(
    uint8_t*, uint8_t*, int64_t, uint64_t*, void*);
template void LaunchTLoadConv<half, TLoadConvCase::NC1HWC0Static, 3, 3, 5, 3, 1>(
    uint8_t*, uint8_t*, int64_t, uint64_t*, void*);
template void LaunchTLoadConv<int8_t, TLoadConvCase::NC1HWC0Static, 3, 5, 3, 3, 1>(
    uint8_t*, uint8_t*, int64_t, uint64_t*, void*);
template void LaunchTLoadConv<half, TLoadConvCase::NDC1HWC0Static, 2, 3, 2, 4, 5>(
    uint8_t*, uint8_t*, int64_t, uint64_t*, void*);
template void LaunchTLoadConv<half, TLoadConvCase::NC1HWC0DynN, 4, 2, 4, 4, 1>(
    uint8_t*, uint8_t*, int64_t, uint64_t*, void*);
template void LaunchTLoadConv<float, TLoadConvCase::NC1HWC0DynNGlobal, 4, 2, 3, 5, 1>(
    uint8_t*, uint8_t*, int64_t, uint64_t*, void*);
#ifdef CPU_SIM_BFLOAT_ENABLED
template void LaunchTLoadConv<bfloat16_t, TLoadConvCase::NC1HWC0Static, 1, 2, 4, 4, 1>(
    uint8_t*, uint8_t*, int64_t, uint64_t*, void*);
#endif
