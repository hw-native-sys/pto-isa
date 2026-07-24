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

#include "cost_check.hpp"

using namespace pto;

namespace {

// case shape is static, but testing would do dynamic or static test
template <int shape0, int shape1, int shape2, int shape3, int shape4>
inline auto getOptDynShape(int gShape0, int gShape1, int gShape2, int gShape3, int gShape4)
{
    if constexpr (shape0 == 1) {
        using DynShapeDim5 = Shape<1, -1, -1, -1, -1>;
        DynShapeDim5 dynShape(gShape1, gShape2, gShape3, gShape4);
        return dynShape;
    } else if constexpr (shape0 == 1 && shape1 == 1) {
        using DynShapeDim5 = Shape<1, 1, -1, -1, -1>;
        DynShapeDim5 dynShape(gShape2, gShape3, gShape4);
        return dynShape;
    } else if constexpr (shape0 == 1 && shape1 == 1 && shape2 == 1) {
        using DynShapeDim5 = Shape<1, 1, 1, -1, -1>;
        DynShapeDim5 dynShape(gShape3, gShape4);
        return dynShape;
    } else if constexpr (shape0 == 1 && shape1 == 1 && shape2 == 1 && shape3 == 1) {
        using DynShapeDim5 = Shape<1, 1, 1, 1, -1>;
        DynShapeDim5 dynShape(gShape4);
        return dynShape;
    } else {
        using DynShapeDim5 = Shape<-1, -1, -1, -1, -1>;
        DynShapeDim5 dynShape(gShape0, gShape1, gShape2, gShape3, gShape4);
        return dynShape;
    }
}

template <
    typename T, int shape0, int shape1, int shape2, int shape3, int shape4, int tRows, int tCols, BLayout major,
    int dyn, Layout Layout_ = Layout::ND>
inline auto getGlobalTensor(T* addr, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4)
{
    if constexpr (dyn) {
        int stride0 = gShape1 * gShape2 * shape3 * shape4;
        int stride1 = gShape2 * shape3 * shape4;
        int stride2 = shape3 * shape4;

        using DynStrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
        auto dynShape =
            getOptDynShape<shape0, shape1, shape2, shape3, shape4>(gShape0, gShape1, gShape2, gShape3, gShape4);
        using GlobalData = GlobalTensor<T, decltype(dynShape), DynStrideDim5, Layout_>;

        if constexpr (major == BLayout::RowMajor) {
            GlobalData srcGlobal(addr, dynShape, DynStrideDim5(stride0, stride1, stride2, shape4, 1));
            return srcGlobal;
        } else {
            GlobalData srcGlobal(addr, dynShape, DynStrideDim5(stride0, stride1, stride2, 1, shape4));
            return srcGlobal;
        }
    } else {
        constexpr int stride0 = shape1 * shape2 * shape3 * shape4;
        constexpr int stride1 = shape2 * shape3 * shape4;
        constexpr int stride2 = shape3 * shape4;
        using StaticShapeDim5 = Shape<shape0, shape1, shape2, tRows, tCols>;

        if constexpr (major == BLayout::RowMajor) {
            using StaticStrideDim5 = pto::Stride<stride0, stride1, stride2, shape4, 1>;
            using GlobalData = GlobalTensor<T, StaticShapeDim5, StaticStrideDim5, Layout_>;
            GlobalData srcGlobal(addr);
            return srcGlobal;
        } else {
            using StaticStrideDim5 = pto::Stride<stride0, stride1, stride2, 1, shape4>;
            using GlobalData = GlobalTensor<T, StaticShapeDim5, StaticStrideDim5, Layout_>;
            GlobalData srcGlobal(addr);
            return srcGlobal;
        }
    }
}

template <
    typename T, int shape0, int shape1, int shape2, int shape3, int shape4, int kTRows, int kTCols, int dyn,
    PadValue PadVal, int gShape0, int gShape1, int gShape2, int gCols, float profiling, float accuracy>
void runTLoadND()
{
    using TileData = Tile<TileType::Vec, T, kTRows, kTCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadVal>;
    TileData vecTile(kTRows, gCols);
    TASSIGN(vecTile, 0x0);

    constexpr int kGTRows = kTRows / shape0 / shape1 / shape2;
    auto srcGlobal =
        getGlobalTensor<T, shape0, shape1, shape2, kGTRows, shape4, kGTRows, shape4, BLayout::RowMajor, dyn>(
            reinterpret_cast<T*>(0x10000), gShape0, gShape1, gShape2, kGTRows, shape4);

    TLOAD(vecTile, srcGlobal);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

template <
    typename T, int shape0, int shape1, int shape2, int shape3, int shape4, int kTRows, int kTCols, int dyn,
    PadValue PadVal, int gShape0, int gShape1, int gShape2, int gRows, int gCols, float profiling, float accuracy>
void runTLoadDN()
{
    using TileData = Tile<TileType::Vec, T, kTRows, kTCols, BLayout::ColMajor, -1, -1, SLayout::NoneBox, 512, PadVal>;
    TileData vecTile(gRows, gCols);
    TASSIGN(vecTile, 0x0);

    constexpr int kGTCols = kTCols / shape0 / shape1 / shape2;
    auto srcGlobal = getGlobalTensor<
        T, shape0, shape1, shape2, shape3, kGTCols, shape3, kGTCols, BLayout::ColMajor, dyn, Layout::DN>(
        reinterpret_cast<T*>(0x10000), gShape0, gShape1, gShape2, shape3, kGTCols);

    TLOAD(vecTile, srcGlobal);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TLoad, c01_nd_float_128x128)
{
    runTLoadND<float, 1, 1, 1, 128, 128, 128, 128, 1, PadValue::Null, 1, 1, 1, 128, 776.0f, 0.557989f>();
}

TEST(TLoad, c02_nd_float_256x64)
{
    runTLoadND<float, 2, 2, 2, 256, 64, 256, 64, 1, PadValue::Null, 2, 2, 2, 64, 2624.0f, 0.423780f>();
}

TEST(TLoad, c03_nd_float_128x127_padmax)
{
    runTLoadND<float, 1, 1, 1, 128, 127, 128, 128, 1, PadValue::Max, 1, 1, 1, 127, 773.0f, 0.562742f>();
}

TEST(TLoad, c04_nd_int16_128x127_padmax)
{
    runTLoadND<int16_t, 1, 1, 1, 128, 127, 128, 128, 1, PadValue::Max, 1, 1, 1, 127, 519.0f, 0.928709f>();
}

TEST(TLoad, c05_nd_uint8_128x127_padmin)
{
    runTLoadND<uint8_t, 1, 1, 1, 128, 127, 128, 128, 1, PadValue::Min, 1, 1, 1, 127, 392.0f, 0.709183f>();
}

TEST(TLoad, c06_nd_int16_64x128_dyn)
{
    runTLoadND<int16_t, 1, 1, 32, 64, 128, 64, 128, 1, PadValue::Null, 1, 1, 32, 128, 8576.0f, 0.029850f>();
}

TEST(TLoad, c07_nd_int16_64x128_static)
{
    runTLoadND<int16_t, 1, 1, 32, 64, 128, 64, 128, 0, PadValue::Null, 1, 1, 32, 128, 8576.0f, 0.029850f>();
}

TEST(TLoad, c08_nd_float_256x60_padmax)
{
    runTLoadND<float, 2, 2, 2, 256, 60, 256, 64, 1, PadValue::Max, 2, 2, 2, 60, 2497.0f, 0.420104f>();
}

TEST(TLoad, c09_dn_float_64x128)
{
    runTLoadDN<float, 1, 1, 32, 64, 128, 64, 128, 1, PadValue::Null, 1, 1, 32, 64, 128, 8704.0f, 0.062500f>();
}

TEST(TLoad, c10_dn_float_256x60)
{
    runTLoadDN<float, 2, 2, 2, 255, 60, 256, 64, 1, PadValue::Null, 2, 2, 2, 255, 60, 8352.0f, 0.133141f>();
}
