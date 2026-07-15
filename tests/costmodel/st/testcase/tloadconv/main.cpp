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
#include <gtest/gtest.h>
#include <vector>
#include <type_traits>

#include "cost_check.hpp"

using namespace pto;

namespace {

template <typename T, int N, int C1, int H, int W, Layout L, float profiling, float accuracy>
void runTloadConv()
{
    constexpr int C0 = 32 / sizeof(T);
    constexpr uint32_t totalElements = N * C1 * H * W * C0;
    constexpr uint32_t bufferSize = totalElements * sizeof(T);

    using StrideNC1HWC0 = Stride<(int64_t)C1 * H * W * C0, (int64_t)H * W * C0, (int64_t)W * C0, (int64_t)C0, 1>;
    using StrideFractalZ = Stride<(int64_t)H * W * N * C0, (int64_t)W * N * C0, (int64_t)N * C0, (int64_t)C0, 1>;
    using SelectedStride = std::conditional_t<L == Layout::NC1HWC0, StrideNC1HWC0, StrideFractalZ>;

    using GShape = std::conditional_t<L == Layout::NC1HWC0, Shape<N, C1, H, W, C0>, Shape<C1, H, W, N, C0>>;

    GlobalTensor<T, GShape, SelectedStride, L> srcGlobal(reinterpret_cast<T*>(0x10000));

    using TShape = std::conditional_t<
        L == Layout::NC1HWC0, ConvTileShape<N, C1, H, W>, ConvTileShape<(C1 * H * W), (N / 16), 16, C0>>;

    using MyTile = ConvTile<TileType::Mat, T, bufferSize, L, TShape>;
    MyTile convTile;

    std::vector<T> localBuffer(totalElements);
    convTile.data() = reinterpret_cast<typename MyTile::TileDType>(localBuffer.data());

    TLOAD(convTile, srcGlobal);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

template <typename T, int C1, int H, int W, int N, float profiling, float accuracy>
void runTloadConvFractalZ5D()
{
    constexpr int C0 = 32 / sizeof(T);
    constexpr uint32_t totalElements = C1 * H * W * N * C0;
    constexpr uint32_t bufferSize = totalElements * sizeof(T);
    constexpr Layout L = Layout::FRACTAL_Z;

    using StrideFractalZ = Stride<(int64_t)H * W * N * C0, (int64_t)W * N * C0, (int64_t)N * C0, (int64_t)C0, 1>;
    using GShape = Shape<C1, H, W, N, C0>;

    GlobalTensor<T, GShape, StrideFractalZ, L> srcGlobal(reinterpret_cast<T*>(0x10000));

    using TShape = ConvTileShape<C1, H, W, N, C0>;
    using MyTile = ConvTile<TileType::Mat, T, bufferSize, L, TShape>;
    MyTile convTile;

    std::vector<T> localBuffer(totalElements);
    convTile.data() = reinterpret_cast<typename MyTile::TileDType>(localBuffer.data());

    TLOAD_IMPL(convTile, srcGlobal);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TLoadConv, c01_nc1hwc0_half_1_2_4_4) { runTloadConv<half, 1, 2, 4, 4, Layout::NC1HWC0, 28.0f, 0.464285f>(); }

TEST(TLoadConv, c02_nc1hwc0_float_1_4_10_10)
{
    runTloadConv<float, 1, 4, 10, 10, Layout::NC1HWC0, 116.0f, 0.594827f>();
}

TEST(TLoadConv, c03_fractalz_half_16_2_1_18) { runTloadConv<half, 16, 2, 1, 18, Layout::FRACTAL_Z, 96.0f, 0.0f>(); }

TEST(TLoadConv, c04_fractalz5d_int8_4_2_6_16) { runTloadConvFractalZ5D<int8_t, 4, 2, 6, 16, 96.0f, 0.0f>(); }
