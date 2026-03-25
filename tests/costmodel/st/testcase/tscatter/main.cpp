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
#include "test_common.h"
#include <gtest/gtest.h>
#include <cmath>

using namespace std;
using namespace pto;
using namespace PtoTestCommon;

class TSCATTERTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

template <typename ST, typename DT, typename IT, int rows, int cols, int validRow, int validCol, float profiling,
          float accuracy>
void testScatter()
{
    Tile<TileType::Vec, DT, rows, cols, BLayout::RowMajor, validRow, validCol, SLayout::NoneBox> dst;
    Tile<TileType::Vec, ST, rows, cols, BLayout::RowMajor, validRow, validCol, SLayout::NoneBox> src;
    Tile<TileType::Vec, IT, rows, cols, BLayout::RowMajor, validRow, validCol, SLayout::NoneBox> idx;

    std::fill(dst.data(), dst.data() + rows * cols, static_cast<DT>(0));
    std::fill(src.data(), src.data() + rows * cols, static_cast<ST>(1));
    std::fill(idx.data(), idx.data() + rows * cols, static_cast<IT>(0));

    TSCATTER(dst, src, idx);

    float costResult = dst.GetCycle();
    float precision = 1 - fabs(profiling - costResult) / profiling;
    bool ret = precision >= accuracy;
    EXPECT_TRUE(ret);
}

// float 16x16 full: 14 + 256 * 2 = 526
TEST_F(TSCATTERTest, case_float_16x16_full)
{
    testScatter<float, float, uint32_t, 16, 16, 16, 16, 526.0f, 1.0f>();
}

// float 32x32 full: 14 + 1024 * 2 = 2062
TEST_F(TSCATTERTest, case_float_32x32_full)
{
    testScatter<float, float, uint32_t, 32, 32, 32, 32, 2062.0f, 1.0f>();
}

// half 16x16 full: 14 + 256 * 1 = 270
TEST_F(TSCATTERTest, case_half_16x16_full)
{
    testScatter<half, half, uint16_t, 16, 16, 16, 16, 270.0f, 1.0f>();
}

// int32_t 16x16 full: 14 + 256 * 2 = 526
TEST_F(TSCATTERTest, case_int32_16x16_full)
{
    testScatter<int32_t, int32_t, uint32_t, 16, 16, 16, 16, 526.0f, 1.0f>();
}

// int16_t 16x16 full: 14 + 256 * 1 = 270
TEST_F(TSCATTERTest, case_int16_16x16_full)
{
    testScatter<int16_t, int16_t, uint16_t, 16, 16, 16, 16, 270.0f, 1.0f>();
}

// float 16x16 partial validRow=12 validCol=10: 14 + 120 * 2 = 254
TEST_F(TSCATTERTest, case_float_16x16_partial_12x10)
{
    testScatter<float, float, uint32_t, 16, 16, 12, 10, 254.0f, 1.0f>();
}
