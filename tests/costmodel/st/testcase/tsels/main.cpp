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

template <
    typename T, typename TMask, int dstTileH, int dstTileW, int maskTileH, int maskTileW, int srcTileH, int srcTileW,
    int vRows, int vCols, float profiling, float accuracy>
void runTSels(T scalar)
{
    using DstTile = Tile<TileType::Vec, T, dstTileH, dstTileW, BLayout::RowMajor, -1, -1>;
    using MaskTile = Tile<TileType::Vec, TMask, maskTileH, maskTileW, BLayout::RowMajor, -1, -1>;
    using SrcTile = Tile<TileType::Vec, T, srcTileH, srcTileW, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;
    DstTile dstTile(vRows, vCols);
    MaskTile maskTile(vRows, maskTileW);
    SrcTile srcTile(vRows, vCols);
    TmpTile tmpTile(1, 32);

    std::vector<T> dstBuf(dstTileH * dstTileW, T{});
    std::vector<TMask> maskBuf(maskTileH * maskTileW, TMask{0});
    std::vector<T> srcBuf(srcTileH * srcTileW, T{});
    std::vector<uint8_t> tmpBuf(32, uint8_t{0});
    TASSIGN(dstTile, reinterpret_cast<std::uintptr_t>(dstBuf.data()));
    TASSIGN(maskTile, reinterpret_cast<std::uintptr_t>(maskBuf.data()));
    TASSIGN(srcTile, reinterpret_cast<std::uintptr_t>(srcBuf.data()));
    TASSIGN(tmpTile, reinterpret_cast<std::uintptr_t>(tmpBuf.data()));

    TSELS(dstTile, maskTile, srcTile, tmpTile, scalar);

    EXPECT_CYCLE_NEAR(profiling, accuracy);
}

} // namespace

TEST(TSels, case_uint16_uint8_2x16_2x32_2x16_2x16)
{
    runTSels<uint16_t, uint8_t, 2, 16, 2, 32, 2, 16, 2, 16, 0.0f, 0.0f>(1);
}

TEST(TSels, case_uint16_uint16_2x16_2x16_2x16_2x16)
{
    runTSels<uint16_t, uint16_t, 2, 16, 2, 16, 2, 16, 2, 16, 0.0f, 0.0f>(1);
}

TEST(TSels, case_uint16_uint32_2x16_2x8_2x16_2x16)
{
    runTSels<uint16_t, uint32_t, 2, 16, 2, 8, 2, 16, 2, 16, 0.0f, 0.0f>(1);
}

TEST(TSels, case_uint32_uint8_2x8_2x32_2x8_2x8) { runTSels<uint32_t, uint8_t, 2, 8, 2, 32, 2, 8, 2, 8, 0.0f, 0.0f>(1); }

TEST(TSels, case_uint32_uint16_2x8_2x16_2x8_2x8)
{
    runTSels<uint32_t, uint16_t, 2, 8, 2, 16, 2, 8, 2, 8, 0.0f, 0.0f>(1);
}

TEST(TSels, case_uint32_uint32_2x8_2x8_2x8_2x8) { runTSels<uint32_t, uint32_t, 2, 8, 2, 8, 2, 8, 2, 8, 0.0f, 0.0f>(1); }

TEST(TSels, case_half_uint8_2x16_2x32_2x16_2x16)
{
    runTSels<half, uint8_t, 2, 16, 2, 32, 2, 16, 2, 16, 0.0f, 0.0f>((half)1.0f);
}

TEST(TSels, case_half_uint16_2x16_2x16_2x16_2x16)
{
    runTSels<half, uint16_t, 2, 16, 2, 16, 2, 16, 2, 16, 0.0f, 0.0f>((half)1.0f);
}

TEST(TSels, case_half_uint32_2x16_2x8_2x16_2x16)
{
    runTSels<half, uint32_t, 2, 16, 2, 8, 2, 16, 2, 16, 0.0f, 0.0f>((half)1.0f);
}

TEST(TSels, case_float_uint8_2x8_2x32_2x8_2x8) { runTSels<float, uint8_t, 2, 8, 2, 32, 2, 8, 2, 8, 0.0f, 0.0f>(1.0f); }

TEST(TSels, case_float_uint16_2x8_2x16_2x8_2x8)
{
    runTSels<float, uint16_t, 2, 8, 2, 16, 2, 8, 2, 8, 0.0f, 0.0f>(1.0f);
}

TEST(TSels, case_float_uint32_2x8_2x8_2x8_2x8) { runTSels<float, uint32_t, 2, 8, 2, 8, 2, 8, 2, 8, 0.0f, 0.0f>(1.0f); }

TEST(TSels, case_uint16_uint8_2x32_2x64_2x128_2x31)
{
    runTSels<uint16_t, uint8_t, 2, 32, 2, 64, 2, 128, 2, 31, 0.0f, 0.0f>(1);
}

TEST(TSels, case_float_uint8_2x32_2x64_2x128_2x31)
{
    runTSels<float, uint8_t, 2, 32, 2, 64, 2, 128, 2, 31, 0.0f, 0.0f>(1.0f);
}

TEST(TSels, case_half_uint8_32x672_32x96_32x672_32x666)
{
    runTSels<half, uint8_t, 32, 672, 32, 96, 32, 672, 32, 666, 0.0f, 0.0f>((half)1.0f);
}

TEST(TSels, case_float_uint8_32x672_32x96_32x672_32x666)
{
    runTSels<float, uint8_t, 32, 672, 32, 96, 32, 672, 32, 666, 0.0f, 0.0f>(1.0f);
}

TEST(TSels, case_float_uint8_1x8192_1x4096_1x8192_1x8192)
{
    runTSels<float, uint8_t, 1, 8192, 1, 4096, 1, 8192, 1, 8192, 0.0f, 0.0f>(1.0f);
}
