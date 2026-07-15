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
#include <cstdint>
#include <vector>

#include "cost_check.hpp"

using namespace pto;

namespace {

template <typename T, int cols, int srcRow, int srcValidRow, float profiling, float accuracy>
void runTColMin()
{
    using SrcTile = Tile<TileType::Vec, T, srcRow, cols, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, T, 1, cols, BLayout::RowMajor, -1, -1>;
    SrcTile srcTile(srcValidRow, cols);
    DstTile dstTile(1, cols);

    std::vector<T> srcBuf(srcRow * cols, T{1});
    std::vector<T> dstBuf(cols, T{0});
    TASSIGN(srcTile, reinterpret_cast<std::uintptr_t>(srcBuf.data()));
    TASSIGN(dstTile, reinterpret_cast<std::uintptr_t>(dstBuf.data()));

    TCOLMIN(dstTile, srcTile);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TColMin, case1) { runTColMin<int16_t, 16, 16, 8, 0.0f, 0.0f>(); }
TEST(TColMin, case2) { runTColMin<int32_t, 16, 16, 8, 0.0f, 0.0f>(); }
TEST(TColMin, case3) { runTColMin<float, 16, 16, 8, 0.0f, 0.0f>(); }
TEST(TColMin, case4) { runTColMin<int16_t, 128, 16, 8, 0.0f, 0.0f>(); }
TEST(TColMin, case5) { runTColMin<int32_t, 64, 16, 8, 0.0f, 0.0f>(); }
TEST(TColMin, case6) { runTColMin<float, 64, 16, 8, 0.0f, 0.0f>(); }
TEST(TColMin, case7) { runTColMin<int16_t, 512, 16, 8, 0.0f, 0.0f>(); }
TEST(TColMin, case8) { runTColMin<int32_t, 256, 16, 8, 0.0f, 0.0f>(); }
TEST(TColMin, case9) { runTColMin<float, 256, 16, 8, 0.0f, 0.0f>(); }
TEST(TColMin, case10) { runTColMin<int16_t, 512, 16, 7, 0.0f, 0.0f>(); }
TEST(TColMin, case11) { runTColMin<int32_t, 256, 32, 31, 0.0f, 0.0f>(); }
TEST(TColMin, case12) { runTColMin<float, 256, 32, 31, 0.0f, 0.0f>(); }
TEST(TColMin, case13) { runTColMin<float, 256, 16, 1, 0.0f, 0.0f>(); }
