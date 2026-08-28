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
#include "cpu_tile_test_utils.h"

#include <gtest/gtest.h>

using namespace pto;
using namespace CpuTileTestUtils;

namespace {

TEST(TColProdTest, ProducesProductPerColumn)
{
    using SrcTile = Tile<TileType::Vec, int32_t, 3, 8, BLayout::RowMajor, 3, 4>;
    using DstTile = Tile<TileType::Vec, int32_t, 1, 8, BLayout::RowMajor, 1, 4>;

    SrcTile src;
    DstTile dst;
    std::size_t addr = 0;
    AssignTileStorage(addr, src, dst);

    const int values[3][4] = {{2, 3, 4, 5}, {1, -2, 3, -4}, {7, 1, 2, 0}};
    for (int r = 0; r < src.GetValidRow(); ++r) {
        for (int c = 0; c < src.GetValidCol(); ++c) {
            SetValue(src, r, c, values[r][c]);
        }
    }

    TCOLPROD(dst, src);

    const int expected[4] = {14, -6, 24, 0};
    for (int c = 0; c < dst.GetValidCol(); ++c) {
        ExpectValueEquals(GetValue(dst, 0, c), expected[c]);
    }
}

} // namespace
