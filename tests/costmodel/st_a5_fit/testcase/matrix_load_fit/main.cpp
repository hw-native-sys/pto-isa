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

#include "a5_cost_check.hpp"

namespace {

using namespace ::pto::mocker::lightweight;

void ExpectCycleNear(const char* name, const CostModelInput& input, double profilingCycles)
{
    constexpr double kRequiredPrecision = 0.90;
    pto::test::ExpectA5CycleNear(name, input, profilingCycles, kRequiredPrecision);
}

TEST(A5MatrixLoadFit, MatmulFp16Fp32)
{
    ExpectCycleNear(
        "TMATMUL",
        CostModelInput{
            .op = PtoOpcode::TMATMUL,
            .dtype = DType::Half,
            .rows = 128,
            .cols = 128,
            .arch = CostModelArch::A5,
            .dtype2 = DType::Half,
            .k = 128,
            .dst_dtype = DType::Float},
        512.4);
    ExpectCycleNear(
        "TMATMUL",
        CostModelInput{
            .op = PtoOpcode::TMATMUL,
            .dtype = DType::Half,
            .rows = 16,
            .cols = 16,
            .arch = CostModelArch::A5,
            .dtype2 = DType::Half,
            .k = 128,
            .dst_dtype = DType::Float},
        24.4);
    ExpectCycleNear(
        "TMATMUL",
        CostModelInput{
            .op = PtoOpcode::TMATMUL,
            .dtype = DType::Half,
            .rows = 1,
            .cols = 2,
            .arch = CostModelArch::A5,
            .dtype2 = DType::Half,
            .k = 64,
            .dst_dtype = DType::Float},
        12.4);
}

TEST(A5MatrixLoadFit, MatmulAdapterPreservesInnerDimension)
{
    constexpr uint64_t kExpectedCycles = 24;
    auto& runtimeContext = ::pto::perf_sim::GetCostModelCtx();
    const uint32_t originalArch = runtimeContext.npu_arch;
    runtimeContext.npu_arch = ::pto::perf_sim::kAscend950Arch;
    EXPECT_EQ(::pto::perf_sim::EstimateLightweightCycles("TMATMUL", 16, 16, "fp16", "fp32", 128), kExpectedCycles);
    runtimeContext.npu_arch = originalArch;
}

TEST(A5MatrixLoadFit, AdapterPreservesA5PhysicalAndValidShapePath)
{
    auto& runtimeContext = ::pto::perf_sim::GetCostModelCtx();
    const uint32_t originalArch = runtimeContext.npu_arch;
    runtimeContext.npu_arch = ::pto::perf_sim::kAscend950Arch;

    const ::pto::perf_sim::LightweightEstimateOptions options{
        .physical_rows = 16,
        .physical_cols = 256,
        .valid_rows = 16,
        .valid_cols = 256,
        .a5_shape_path = A5ShapePathHint::Path2D,
    };
    EXPECT_EQ(
        ::pto::perf_sim::EstimateLightweightCycles(
            "TABS", 16, 256, "fp16", "unknown", 0, TransferTileType::Unknown, options),
        130U);

    runtimeContext.npu_arch = originalArch;
}

TEST(A5MatrixLoadFit, AdapterKeepsValidMatrixAndLegacyDimensions)
{
    auto& context = ::pto::perf_sim::GetCostModelCtx();
    const uint32_t originalArch = context.npu_arch;
    const ::pto::perf_sim::LightweightEstimateOptions options{
        .physical_rows = 128, .physical_cols = 128, .valid_rows = 16, .valid_cols = 16};
    context.npu_arch = ::pto::perf_sim::kAscend950Arch;
    EXPECT_EQ(
        ::pto::perf_sim::EstimateLightweightCycles(
            "TMATMUL", 16, 16, "fp16", "fp32", 128, TransferTileType::Unknown, options),
        24U);
    context.npu_arch = 2201;
    auto legacyOptions = options;
    legacyOptions.valid_cols = 32;
    const auto expected = ::pto::perf_sim::EstimateLightweightCycles("TADDS", 16, 32, "fp16");
    EXPECT_GT(expected, 0U);
    EXPECT_EQ(
        ::pto::perf_sim::EstimateLightweightCycles(
            "TADDS", 16, 32, "fp16", "unknown", 0, TransferTileType::Unknown, legacyOptions),
        expected);
    context.npu_arch = originalArch;
}

TEST(A5MatrixLoadFit, MissingA5FitDoesNotUseGenericFallback)
{
    auto& context = ::pto::perf_sim::GetCostModelCtx();
    const uint32_t originalArch = context.npu_arch;
    const ::pto::perf_sim::LightweightEstimateOptions indexOptions{.dtype2 = DType::Uint16, .has_dtype2 = true};
    for (const auto arch : {::pto::perf_sim::kAscend950Arch, ::pto::perf_sim::kAscend950UrmaArch}) {
        context.npu_arch = arch;
        EXPECT_EQ(
            ::pto::perf_sim::EstimateInstrCycles(
                "TGATHER", 16, 128, "fp16", "fp16", 0, TransferTileType::Unknown, indexOptions),
            0U);
        EXPECT_EQ(
            ::pto::perf_sim::EstimateInstrCycles("TLOAD", 12, 128, "fp16", "unknown", 0, TransferTileType::MatTile),
            0U);
        EXPECT_EQ(::pto::perf_sim::EstimateInstrCycles("TMATMUL", 16, 16, "fp16", "fp32", 128), 24U);
    }
    context.npu_arch = 2201;
    EXPECT_GT(::pto::perf_sim::EstimateInstrCycles("UNSUPPORTED_TEST_OPCODE", 16, 128, "fp16"), 0U);
    context.npu_arch = originalArch;
}

TEST(A5MatrixLoadFit, LoadFp16NdLargeTransfer)
{
    ExpectCycleNear(
        "TLOAD",
        CostModelInput{
            .op = PtoOpcode::TLOAD,
            .dtype = DType::Half,
            .rows = 12,
            .cols = 128,
            .arch = CostModelArch::A5,
            .layout = MemLayout::ND,
            .tile_type = TransferTileType::VecTile,
            .data_size = 1536},
        30.8);
    ExpectCycleNear(
        "TLOAD",
        CostModelInput{
            .op = PtoOpcode::TLOAD,
            .dtype = DType::Half,
            .rows = 16,
            .cols = 256,
            .arch = CostModelArch::A5,
            .layout = MemLayout::ND,
            .tile_type = TransferTileType::VecTile,
            .data_size = 4096},
        82.0);
}

TEST(A5MatrixLoadFit, TransferAdapterPreservesTileType)
{
    constexpr uint64_t kExpectedLoadCycles = 31;
    constexpr uint64_t kExpectedStoreCycles = 50;
    auto& runtimeContext = ::pto::perf_sim::GetCostModelCtx();
    const uint32_t originalArch = runtimeContext.npu_arch;
    runtimeContext.npu_arch = ::pto::perf_sim::kAscend950Arch;

    EXPECT_EQ(
        ::pto::perf_sim::EstimateLightweightCycles("TLOAD", 12, 128, "fp16", "unknown", 0, TransferTileType::VecTile),
        kExpectedLoadCycles);
    EXPECT_EQ(
        ::pto::perf_sim::EstimateLightweightCycles("TSTORE", 16, 64, "fp16", "unknown", 0, TransferTileType::VecTile),
        kExpectedStoreCycles);
    EXPECT_EQ(
        ::pto::perf_sim::EstimateLightweightCycles("TLOAD", 12, 128, "fp16", "unknown", 0, TransferTileType::MatTile),
        0U);
    EXPECT_EQ(
        ::pto::perf_sim::EstimateLightweightCycles("TSTORE", 16, 64, "fp16", "unknown", 0, TransferTileType::AccTile),
        0U);

    runtimeContext.npu_arch = originalArch;
}

TEST(A5MatrixLoadFit, ResidentCubeAndTransferMeasuredProfiles)
{
    struct Case {
        PtoOpcode op;
        DType src;
        DType dst;
        int64_t m;
        int64_t k;
        int64_t n;
        double measured;
    };
    const Case cases[] = {
        {PtoOpcode::TLOAD, DType::Float, DType::Float, 16, 0, 128, 82.1},
        {PtoOpcode::TLOAD, DType::Float, DType::Float, 16, 0, 256, 164.0},
        {PtoOpcode::TLOAD, DType::Float, DType::Float, 16, 0, 64, 41.1},
        {PtoOpcode::TLOAD, DType::Float, DType::Float, 32, 0, 32, 41.1},
        {PtoOpcode::TLOAD, DType::Float, DType::Float, 4, 0, 128, 20.6},
        {PtoOpcode::TLOAD, DType::Float, DType::Float, 4, 0, 256, 41.1},
        {PtoOpcode::TLOAD, DType::Float, DType::Float, 64, 0, 128, 327.8},
        {PtoOpcode::TLOAD, DType::Float, DType::Float, 64, 0, 64, 164.0},
        {PtoOpcode::TLOAD, DType::Float, DType::Float, 7, 0, 96, 27.0},
        {PtoOpcode::TLOAD, DType::Half, DType::Half, 16, 0, 128, 41.1},
        {PtoOpcode::TLOAD, DType::Half, DType::Half, 16, 0, 256, 82.1},
        {PtoOpcode::TLOAD, DType::Half, DType::Half, 4, 0, 256, 20.6},
        {PtoOpcode::TLOAD, DType::Half, DType::Half, 7, 0, 96, 15.0},
        {PtoOpcode::TLOAD, DType::Int32, DType::Int32, 16, 0, 128, 82.0},
        {PtoOpcode::TLOAD, DType::Int32, DType::Int32, 16, 0, 256, 164.0},
        {PtoOpcode::TLOAD, DType::Int32, DType::Int32, 16, 0, 64, 41.1},
        {PtoOpcode::TLOAD, DType::Int32, DType::Int32, 32, 0, 32, 41.1},
        {PtoOpcode::TLOAD, DType::Int32, DType::Int32, 4, 0, 128, 20.6},
        {PtoOpcode::TLOAD, DType::Int32, DType::Int32, 4, 0, 256, 41.1},
        {PtoOpcode::TLOAD, DType::Int32, DType::Int32, 64, 0, 128, 327.8},
        {PtoOpcode::TLOAD, DType::Int32, DType::Int32, 64, 0, 64, 164.0},
        {PtoOpcode::TLOAD, DType::Int32, DType::Int32, 7, 0, 96, 27.0},
        {PtoOpcode::TMATMUL, DType::BFloat16, DType::Float, 128, 64, 64, 128.0},
        {PtoOpcode::TMATMUL, DType::BFloat16, DType::Float, 16, 128, 16, 24.0},
        {PtoOpcode::TMATMUL, DType::BFloat16, DType::Float, 16, 16, 128, 8.0},
        {PtoOpcode::TMATMUL, DType::BFloat16, DType::Float, 16, 16, 16, 4.0},
        {PtoOpcode::TMATMUL, DType::BFloat16, DType::Float, 16, 64, 64, 16.0},
        {PtoOpcode::TMATMUL, DType::BFloat16, DType::Float, 1, 64, 2, 12.0},
        {PtoOpcode::TMATMUL, DType::BFloat16, DType::Float, 1, 64, 256, 64.0},
        {PtoOpcode::TMATMUL, DType::BFloat16, DType::Float, 32, 64, 64, 32.0},
        {PtoOpcode::TMATMUL, DType::BFloat16, DType::Float, 64, 128, 64, 128.0},
        {PtoOpcode::TMATMUL, DType::BFloat16, DType::Float, 64, 32, 64, 32.0},
        {PtoOpcode::TMATMUL, DType::BFloat16, DType::Float, 64, 64, 128, 128.0},
        {PtoOpcode::TMATMUL, DType::Float, DType::Float, 128, 64, 128, 4096.0},
        {PtoOpcode::TMATMUL, DType::Float, DType::Float, 128, 64, 64, 2048.0},
        {PtoOpcode::TMATMUL, DType::Float, DType::Float, 16, 128, 16, 384.0},
        {PtoOpcode::TMATMUL, DType::Float, DType::Float, 16, 16, 128, 128.0},
        {PtoOpcode::TMATMUL, DType::Float, DType::Float, 16, 16, 16, 48.0},
        {PtoOpcode::TMATMUL, DType::Float, DType::Float, 16, 64, 64, 256.0},
        {PtoOpcode::TMATMUL, DType::Float, DType::Float, 1, 64, 2, 192.0},
        {PtoOpcode::TMATMUL, DType::Float, DType::Float, 32, 64, 64, 512.0},
        {PtoOpcode::TMATMUL, DType::Float, DType::Float, 64, 128, 64, 2048.0},
        {PtoOpcode::TMATMUL, DType::Float, DType::Float, 64, 32, 64, 512.0},
        {PtoOpcode::TMATMUL, DType::Float, DType::Float, 64, 64, 128, 2048.0},
        {PtoOpcode::TMATMUL, DType::Float, DType::Float, 64, 64, 64, 1024.0},
        {PtoOpcode::TMATMUL, DType::Int8, DType::Int32, 128, 128, 128, 256.0},
        {PtoOpcode::TMATMUL, DType::Int8, DType::Int32, 128, 64, 128, 128.0},
        {PtoOpcode::TMATMUL, DType::Int8, DType::Int32, 128, 64, 64, 64.0},
        {PtoOpcode::TMATMUL, DType::Int8, DType::Int32, 16, 128, 16, 12.0},
        {PtoOpcode::TMATMUL, DType::Int8, DType::Int32, 16, 32, 16, 4.0},
        {PtoOpcode::TMATMUL, DType::Int8, DType::Int32, 16, 32, 256, 16.0},
        {PtoOpcode::TMATMUL, DType::Int8, DType::Int32, 16, 64, 64, 8.0},
        {PtoOpcode::TMATMUL, DType::Int8, DType::Int32, 1, 64, 2, 6.0},
        {PtoOpcode::TMATMUL, DType::Int8, DType::Int32, 1, 64, 256, 32.0},
        {PtoOpcode::TMATMUL, DType::Int8, DType::Int32, 32, 64, 64, 16.0},
        {PtoOpcode::TMATMUL, DType::Int8, DType::Int32, 64, 128, 64, 64.0},
        {PtoOpcode::TMATMUL, DType::Int8, DType::Int32, 64, 32, 64, 16.0},
        {PtoOpcode::TMATMUL, DType::Int8, DType::Int32, 64, 64, 128, 64.0},
        {PtoOpcode::TMATMUL, DType::Int8, DType::Int32, 64, 64, 64, 32.0},
        {PtoOpcode::TSTORE, DType::Float8E4M3, DType::Float8E4M3, 16, 0, 128, 50.1},
        {PtoOpcode::TSTORE, DType::Float8E4M3, DType::Float8E4M3, 16, 0, 256, 66.1},
        {PtoOpcode::TSTORE, DType::Float8E4M3, DType::Float8E4M3, 1, 0, 128, 35.9},
        {PtoOpcode::TSTORE, DType::Float8E4M3, DType::Float8E4M3, 32, 0, 64, 50.1},
        {PtoOpcode::TSTORE, DType::Float8E4M3, DType::Float8E4M3, 4, 0, 128, 38.5},
        {PtoOpcode::TSTORE, DType::Float8E4M3, DType::Float8E4M3, 4, 0, 256, 42.6},
        {PtoOpcode::TSTORE, DType::Float8E5M2, DType::Float8E5M2, 16, 0, 128, 50.1},
        {PtoOpcode::TSTORE, DType::Float8E5M2, DType::Float8E5M2, 16, 0, 256, 66.1},
        {PtoOpcode::TSTORE, DType::Float8E5M2, DType::Float8E5M2, 1, 0, 128, 35.8},
        {PtoOpcode::TSTORE, DType::Float8E5M2, DType::Float8E5M2, 32, 0, 64, 50.1},
        {PtoOpcode::TSTORE, DType::Float8E5M2, DType::Float8E5M2, 4, 0, 128, 38.5},
        {PtoOpcode::TSTORE, DType::Float8E5M2, DType::Float8E5M2, 4, 0, 256, 42.6},
        {PtoOpcode::TSTORE, DType::HFloat8, DType::HFloat8, 16, 0, 128, 50.1},
        {PtoOpcode::TSTORE, DType::HFloat8, DType::HFloat8, 16, 0, 256, 66.1},
        {PtoOpcode::TSTORE, DType::HFloat8, DType::HFloat8, 1, 0, 128, 35.9},
        {PtoOpcode::TSTORE, DType::HFloat8, DType::HFloat8, 32, 0, 64, 50.1},
        {PtoOpcode::TSTORE, DType::HFloat8, DType::HFloat8, 4, 0, 128, 38.5},
        {PtoOpcode::TSTORE, DType::HFloat8, DType::HFloat8, 4, 0, 256, 42.6},
    };
    for (const auto& item : cases) {
        SCOPED_TRACE(static_cast<int>(item.op));
        SCOPED_TRACE(item.m);
        SCOPED_TRACE(item.n);
        CostModelInput input{};
        input.op = item.op;
        input.dtype = input.dtype2 = item.src;
        input.dst_dtype = item.dst;
        input.rows = item.m;
        input.k = item.k;
        input.cols = item.n;
        input.arch = CostModelArch::A5;
        input.tile_type = TransferTileType::VecTile;
        CostModelResult result{};
        ASSERT_TRUE(EstimateCycles(input, result));
        EXPECT_LE(std::abs(result.cycles - item.measured) / item.measured, 0.10);
    }
}
TEST(A5MatrixFit, OutputAreaUtilizationProfiles)
{
    struct Profile {
        DType src;
        DType dst;
        int64_t m;
        int64_t k;
        int64_t n;
        double cycles;
    };
    const Profile profiles[] = {
        {DType::BFloat16, DType::Float, 16, 32, 16, 6.0},  {DType::BFloat16, DType::Float, 16, 32, 32, 6.0},
        {DType::BFloat16, DType::Float, 16, 32, 48, 6.0},  {DType::BFloat16, DType::Float, 16, 32, 96, 12.0},
        {DType::BFloat16, DType::Float, 16, 64, 16, 12.0}, {DType::Float, DType::Float, 16, 32, 96, 192.0},
        {DType::Float, DType::Float, 16, 64, 96, 384.0},   {DType::Float, DType::Float, 48, 32, 96, 576.0},
        {DType::Float, DType::Float, 48, 64, 96, 1152.0},  {DType::Int8, DType::Int32, 16, 64, 16, 6.0},
        {DType::BFloat16, DType::Float, 16, 64, 96, 24.0}, {DType::BFloat16, DType::Float, 48, 32, 32, 12.0},
        {DType::BFloat16, DType::Float, 48, 32, 96, 36.0}, {DType::BFloat16, DType::Float, 48, 64, 96, 72.0},
        {DType::Float, DType::Float, 16, 32, 32, 96.0},    {DType::Float, DType::Float, 16, 64, 16, 192.0},
        {DType::Float, DType::Float, 48, 32, 16, 96.0},    {DType::Float, DType::Float, 48, 64, 16, 192.0},
        {DType::Int8, DType::Int32, 16, 32, 32, 4.0},      {DType::Int8, DType::Int32, 16, 64, 32, 6.0},
        {DType::BFloat16, DType::Float, 16, 64, 32, 12.0}, {DType::Float, DType::Float, 16, 64, 32, 192.0},
        {DType::Float, DType::Float, 48, 32, 32, 192.0},   {DType::Float, DType::Float, 48, 64, 32, 384.0},
        {DType::Int8, DType::Int32, 16, 32, 48, 4.0},      {DType::Int8, DType::Int32, 16, 64, 48, 6.0},
        {DType::Int8, DType::Int32, 16, 64, 96, 12.0},     {DType::Int8, DType::Int32, 48, 32, 16, 4.0},
        {DType::Int8, DType::Int32, 48, 32, 32, 6.0},      {DType::Int8, DType::Int32, 48, 32, 48, 9.0},
        {DType::Int8, DType::Int32, 48, 32, 96, 18.0},     {DType::Int8, DType::Int32, 48, 64, 16, 6.0},
        {DType::Int8, DType::Int32, 48, 64, 32, 12.0},     {DType::Int8, DType::Int32, 48, 64, 48, 18.0},
        {DType::Int8, DType::Int32, 48, 64, 96, 36.0},     {DType::BFloat16, DType::Float, 16, 64, 48, 12.0},
        {DType::BFloat16, DType::Float, 48, 32, 16, 6.0},  {DType::BFloat16, DType::Float, 48, 32, 48, 18.0},
        {DType::BFloat16, DType::Float, 48, 64, 16, 12.0}, {DType::BFloat16, DType::Float, 48, 64, 48, 36.0},
        {DType::Float, DType::Float, 16, 32, 16, 96.0},    {DType::Float, DType::Float, 16, 32, 48, 96.0},
        {DType::Float, DType::Float, 16, 64, 48, 192.0},   {DType::Float, DType::Float, 48, 32, 48, 288.0},
        {DType::Float, DType::Float, 48, 64, 48, 576.0},   {DType::Int8, DType::Int32, 16, 32, 96, 6.0},
    };
    for (const auto& profile : profiles) {
        CostModelInput input{};
        input.op = PtoOpcode::TMATMUL;
        input.dtype = input.dtype2 = profile.src;
        input.dst_dtype = profile.dst;
        input.rows = profile.m;
        input.k = profile.k;
        input.cols = profile.n;
        input.arch = CostModelArch::A5;
        CostModelResult result{};
        ASSERT_TRUE(EstimateCycles(input, result));
        EXPECT_NEAR(result.cycles, profile.cycles, profile.cycles * 0.10);
    }
}

} // namespace
