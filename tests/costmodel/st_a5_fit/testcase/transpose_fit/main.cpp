/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>
#include <pto/costmodel/lightweight_costmodel.hpp>

namespace {
using namespace pto::mocker::lightweight;

CostModelInput Input(DType dtype, int r, int c)
{
    CostModelInput input{.op = PtoOpcode::TTRANS, .dtype = dtype, .rows = c, .cols = r, .arch = CostModelArch::A5};
    input.transpose = {true, r, c, r, c, c, c, r, c, r, r};
    return input;
}

TEST(A5TransposeFit, MeasuredScatterCurves)
{
    struct Sample {
        DType dtype;
        int rows;
        int cols;
        double cycles;
    };
    const Sample samples[] = {
        {DType::Float, 16, 32, 323},    {DType::Float, 16, 64, 563},     {DType::Float, 16, 128, 1102},
        {DType::Float, 32, 64, 2145.1}, {DType::Float, 32, 128, 4267.1}, {DType::Float, 32, 192, 6413.1},
        {DType::Half, 16, 32, 704},     {DType::Half, 16, 64, 1344},     {DType::Half, 16, 128, 2608},
        {DType::Half, 32, 64, 5225.1},  {DType::Half, 32, 128, 10312.1}, {DType::Half, 32, 192, 15554.1}};
    for (auto s : samples) {
        SCOPED_TRACE(std::to_string(s.rows) + "x" + std::to_string(s.cols));
        auto result = EstimateCycles(Input(s.dtype, s.rows, s.cols));
        EXPECT_GT(result.cycles, 0);
        EXPECT_LE(std::abs(result.cycles - s.cycles) / s.cycles, 0.03);
    }
}

TEST(A5TransposeFit, SourcePhysicalShapeControlsDispatch)
{
    auto input = Input(DType::Float, 64, 32);
    // Valid source 16x32 is wide, but physical source 64x32 selects gather.
    input.transpose.src_valid_rows = 16;
    input.transpose.dst_valid_cols = 16;
    a5::TransposeFeatures features;
    ASSERT_TRUE(a5::BuildTransposeFeatures(input, features));
    EXPECT_FALSE(features.scatter);
    EXPECT_EQ(features.outer_iterations, 32);
    EXPECT_EQ(features.tail_elements, 16);
    EXPECT_EQ(features.indexed_stride_bytes, 128);
    EXPECT_EQ(EstimateCycles(input).cycles, 0);
    input = Input(DType::Float, 16, 128);
    ASSERT_TRUE(a5::BuildTransposeFeatures(input, features));
    EXPECT_TRUE(features.scatter);
    EXPECT_EQ(features.repeats_per_iteration, 2);
    EXPECT_EQ(features.full_chunks_per_iteration, 1);
    EXPECT_EQ(features.tail_elements, 64); // Last full chunk still uses tail block.
    EXPECT_EQ(features.indexed_stride_bytes, 64);
}

TEST(A5TransposeFit, RejectMissingOrUncalibratedGeometry)
{
    auto input = Input(DType::Float, 16, 64);
    input.transpose = {};
    EXPECT_EQ(EstimateCycles(input).cycles, 0);
    input = Input(DType::Float, 48, 96);
    EXPECT_EQ(EstimateCycles(input).cycles, 0);
    input = Input(DType::Half, 64, 128);
    EXPECT_EQ(EstimateCycles(input).cycles, 0);
    input = Input(DType::Float, 16, 256);
    EXPECT_EQ(EstimateCycles(input).cycles, 0);
    input = Input(DType::Float, 16, 64);
    input.transpose.dst_row_stride = 32;
    EXPECT_EQ(EstimateCycles(input).cycles, 0);
    input = Input(DType::Float, 16, 64);
    input.transpose.ordinary_row_major = false;
    EXPECT_EQ(EstimateCycles(input).cycles, 0);
}

TEST(A5TransposeFit, PublicApiPropagatesSourceAndDestination)
{
    using namespace pto;
    auto& ctx = perf_sim::GetCostModelCtx();
    const auto oldArch = ctx.npu_arch;
    ctx.npu_arch = perf_sim::kAscend950Arch;
    using S = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1>;
    using D = Tile<TileType::Vec, float, 128, 16, BLayout::RowMajor, -1, -1>;
    S src(16, 128), tmp(16, 128);
    D dst(128, 16);
    TASSIGN(src, 0);
    TASSIGN(dst, 32768);
    TASSIGN(tmp, 65536);
    const auto geometry = mocker::TransposeFitGeometry(dst, src);
    EXPECT_EQ(geometry.src_rows, 16);
    EXPECT_EQ(geometry.src_row_stride, 128);
    EXPECT_EQ(geometry.dst_row_stride, 16);
    TTRANS(dst, src, tmp);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), EstimateCycles(Input(DType::Float, 16, 128)).cycles);
    EXPECT_GT(mocker::GetLastPtoInstrCycles(), 0);
    ctx.npu_arch = oldArch;
}
} // namespace
