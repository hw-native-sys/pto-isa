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

#include "binary_vec_test_context.hpp"
#include "cost_check.hpp"

using namespace pto;

namespace {

template <typename T, int row, int validRow, int col, int validCol, PadValue padValue, float profiling, float accuracy>
void runTMax()
{
    pto::test::BinaryVecTestContext<T, row, validRow, col, validCol, padValue> ctx;

    TMAX(ctx.dstTile, ctx.src0Tile, ctx.src1Tile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TMax, case_float_64x64_64x64_64x64_PAD_VALUE_NULL)
{
    runTMax<float, 64, 64, 64, 64, PadValue::Null, 0.0f, 0.0f>();
}

TEST(TMax, case_int32_64x64_64x64_64x64_PAD_VALUE_NULL)
{
    runTMax<int32_t, 64, 64, 64, 64, PadValue::Null, 0.0f, 0.0f>();
}

TEST(TMax, case_half_64x64_64x64_64x64_PAD_VALUE_NULL)
{
    runTMax<half, 64, 64, 64, 64, PadValue::Null, 0.0f, 0.0f>();
}

TEST(TMax, case_int16_64x64_64x64_64x64_PAD_VALUE_NULL)
{
    runTMax<int16_t, 64, 64, 64, 64, PadValue::Null, 0.0f, 0.0f>();
}

TEST(TMax, case_float_60x60_64x64_60x60_PAD_VALUE_MAX)
{
    runTMax<float, 64, 60, 64, 60, PadValue::Max, 0.0f, 0.0f>();
}

TEST(TMax, case_int32_60x60_64x64_60x60_PAD_VALUE_MAX)
{
    runTMax<int32_t, 64, 60, 64, 60, PadValue::Max, 0.0f, 0.0f>();
}

TEST(TMax, case_half_1x3600_2x4096_1x3600_PAD_VALUE_MAX)
{
    runTMax<half, 2, 1, 4096, 3600, PadValue::Max, 0.0f, 0.0f>();
}

TEST(TMax, case_int16_16x200_20x512_16x200_PAD_VALUE_MAX)
{
    runTMax<int16_t, 20, 16, 512, 200, PadValue::Max, 0.0f, 0.0f>();
}
