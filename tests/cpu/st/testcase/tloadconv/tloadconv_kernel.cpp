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
void runTloadDynamic(__gm__ T *out, __gm__ T *src)
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
    using TShape = std::conditional_t<L == Layout::NC1HWC0, ConvTileShape<N, C1, H, W>,
                                      ConvTileShape<(C1 * H * W), (N / 16), 16, C0>>;

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
void runTloadFractalZ5D(__gm__ T *out, __gm__ T *src)
{
    // 1. Calculate C0 based on data type (32 bytes / size of T)
    constexpr int C0 = 32 / sizeof(T);

    // 2. Hardware and buffer constants
    constexpr uint32_t totalElements = C1 * H * W * N * C0;
    constexpr uint32_t bufferSize = totalElements * sizeof(T);
    constexpr Layout L = Layout::FRACTAL_Z;

    // 3. Static Strides for Global Memory: [C1, H, W, N, C0]
    using StrideFractalZ = Stride<(int64_t)H * W * N * C0, // S0: Jump between C1 groups
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

extern "C" __global__ AICORE void launch_1(__gm__ uint8_t *o, __gm__ uint8_t *s)
{
    runTloadDynamic<half, 1, 2, 4, 4, Layout::NC1HWC0>((__gm__ half *)o, (__gm__ half *)s);
}

extern "C" __global__ AICORE void launch_2(__gm__ uint8_t *o, __gm__ uint8_t *s)
{
    runTloadDynamic<float, 1, 4, 10, 10, Layout::NC1HWC0>((__gm__ float *)o, (__gm__ float *)s);
}

extern "C" __global__ AICORE void launch_3(__gm__ uint8_t *o, __gm__ uint8_t *s)
{
    runTloadDynamic<half, 16, 2, 1, 18, Layout::FRACTAL_Z>((__gm__ half *)o, (__gm__ half *)s);
}

extern "C" __global__ AICORE void launch_4(__gm__ uint8_t *o, __gm__ uint8_t *s)
{
    runTloadFractalZ5D<int8_t, 4, 2, 6, 16>((__gm__ int8_t *)o, (__gm__ int8_t *)s);
}

#ifdef CPU_SIM_BFLOAT_ENABLED
extern "C" __global__ AICORE void launch_5(__gm__ uint8_t *o, __gm__ uint8_t *s)
{
    runTloadDynamic<bfloat16_t, 1, 2, 4, 4, Layout::NC1HWC0>((__gm__ bfloat16_t *)o, (__gm__ bfloat16_t *)s);
}
#endif

// Unified Dispatcher for GTest
template <int32_t testKey>
void launchTLOAD(uint8_t *out, uint8_t *src, uint64_t *gLog, void *stream)
{
    if constexpr (testKey == 1)
        launch_1(out, src);
    else if constexpr (testKey == 2)
        launch_2(out, src);
    else if constexpr (testKey == 3)
        launch_3(out, src);
    else if constexpr (testKey == 4)
        launch_4(out, src);
#ifdef CPU_SIM_BFLOAT_ENABLED
    else if constexpr (testKey == 5)
        launch_5(out, src);
#endif
}

// Template instantiations
template void launchTLOAD<1>(uint8_t *, uint8_t *, uint64_t *, void *);
template void launchTLOAD<2>(uint8_t *, uint8_t *, uint64_t *, void *);
template void launchTLOAD<3>(uint8_t *, uint8_t *, uint64_t *, void *);
template void launchTLOAD<4>(uint8_t *, uint8_t *, uint64_t *, void *);
#ifdef CPU_SIM_BFLOAT_ENABLED
template void launchTLOAD<5>(uint8_t *, uint8_t *, uint64_t *, void *);
#endif
