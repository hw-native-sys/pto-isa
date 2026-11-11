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

} // namespace
