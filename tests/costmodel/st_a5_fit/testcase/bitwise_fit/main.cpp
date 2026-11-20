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

#include <pto/costmodel/lightweight_costmodel.hpp>

#include "a5_cost_check.hpp"

namespace {

using namespace ::pto::mocker::lightweight;

struct BitwiseFitCase {
    PtoOpcode op;
    const char* name;
    DType dtype;
    int64_t rows;
    int64_t cols;
    double profilingCycles;
};

constexpr BitwiseFitCase kCases[] = {
    {PtoOpcode::TAND, "TAND", DType::Int16, 16, 16, 114.0},
    {PtoOpcode::TAND, "TAND", DType::Int16, 16, 256, 130.0},
    {PtoOpcode::TAND, "TAND", DType::Int32, 64, 128, 466.0},
    {PtoOpcode::TOR, "TOR", DType::Int16, 128, 128, 786.0},
    {PtoOpcode::TOR, "TOR", DType::Int32, 16, 256, 162.0},
    {PtoOpcode::TNOT, "TNOT", DType::Int16, 32, 32, 210.0},
    {PtoOpcode::TNOT, "TNOT", DType::Int16, 64, 128, 402.0},
    {PtoOpcode::TANDS, "TANDS", DType::Int16, 1, 1024, 43.0},
    {PtoOpcode::TANDS, "TANDS", DType::Int16, 24, 256, 186.0},
    {PtoOpcode::TORS, "TORS", DType::Int16, 1, 16, 36.0},
    {PtoOpcode::TORS, "TORS", DType::Int16, 48, 64, 306.0},
    {PtoOpcode::TXOR, "TXOR", DType::Int16, 2, 64, 39.0},
    {PtoOpcode::TXOR, "TXOR", DType::Int16, 64, 128, 402.0},
    {PtoOpcode::TXORS, "TXORS", DType::Int16, 16, 128, 114.0},
    {PtoOpcode::TXORS, "TXORS", DType::Int16, 24, 256, 186.0},
};

TEST(A5BitwiseFit, HardwareProfiles)
{
    constexpr double kRequiredPrecision = 0.90;
    for (const auto& testCase : kCases) {
        CostModelInput input{
            .op = testCase.op,
            .dtype = testCase.dtype,
            .rows = testCase.rows,
            .cols = testCase.cols,
            .arch = CostModelArch::A5,
            .valid_rows = testCase.rows,
            .valid_cols = testCase.cols,
        };
        pto::test::ExpectA5CycleNear(testCase.name, input, testCase.profilingCycles, kRequiredPrecision);
    }
}

TEST(A5BitwiseFit, Int16AndMissingShapeKeys)
{
    constexpr BitwiseFitCase kAdditionalCases[] = {
        {PtoOpcode::TAND, "TAND/2D", DType::Int16, 4, 128, 42.0},
        {PtoOpcode::TAND, "TAND/2D", DType::Int16, 4, 256, 46.0},
        {PtoOpcode::TAND, "TAND/2D", DType::Int16, 16, 128, 114.0},
        {PtoOpcode::TAND, "TAND/2D", DType::Int16, 16, 256, 130.0},
        {PtoOpcode::TAND, "TAND/2D/tail", DType::Int16, 7, 96, 60.0},
        {PtoOpcode::TAND, "TAND/2D", DType::Uint16, 16, 128, 114.0},
        {PtoOpcode::TAND, "TAND/2D/tail", DType::Uint16, 16, 16, 114.0},
        {PtoOpcode::TAND, "TAND/2D", DType::Uint16, 16, 256, 130.0},
        {PtoOpcode::TAND, "TAND/2D", DType::Uint16, 4, 128, 42.0},
        {PtoOpcode::TAND, "TAND/2D", DType::Uint16, 4, 256, 46.0},
        {PtoOpcode::TAND, "TAND/2D/tail", DType::Uint16, 64, 64, 402.0},
        {PtoOpcode::TAND, "TAND/2D/tail", DType::Uint16, 7, 96, 60.0},
    };
    for (const auto& testCase : kAdditionalCases) {
        const CostModelInput input{
            .op = testCase.op,
            .dtype = testCase.dtype,
            .rows = testCase.rows,
            .cols = testCase.cols,
            .arch = CostModelArch::A5,
            .a5_shape_path = testCase.rows == 1 ? pto::mocker::lightweight::A5ShapePathHint::Path1D :
                                                  pto::mocker::lightweight::A5ShapePathHint::Path2D,
        };
        CostModelResult result{};
        ASSERT_TRUE(EstimateCycles(input, result)) << testCase.name;
        EXPECT_NEAR(result.cycles, testCase.profilingCycles, testCase.profilingCycles * 0.10) << testCase.name;
    }
}

TEST(A5SupplementalProfiles, ExactTwoDimensionalHardwareProfiles)
{
    struct Profile {
        PtoOpcode op;
        DType dtype;
        int64_t rows;
        int64_t cols;
        double cycles;
    };
    constexpr Profile profiles[] = {
        {PtoOpcode::TANDS, DType::Uint16, 16, 128, 114.0}, {PtoOpcode::TANDS, DType::Uint16, 16, 16, 114.0},
        {PtoOpcode::TANDS, DType::Uint16, 16, 256, 130.0}, {PtoOpcode::TANDS, DType::Uint16, 4, 128, 42.0},
        {PtoOpcode::TANDS, DType::Uint16, 4, 256, 46.0},   {PtoOpcode::TANDS, DType::Uint16, 64, 64, 402.0},
        {PtoOpcode::TANDS, DType::Uint16, 7, 96, 60.0},    {PtoOpcode::TNOT, DType::Int16, 16, 128, 114.0},
        {PtoOpcode::TNOT, DType::Int16, 16, 256, 130.0},   {PtoOpcode::TNOT, DType::Int16, 4, 128, 42.0},
        {PtoOpcode::TNOT, DType::Int16, 4, 256, 46.0},     {PtoOpcode::TNOT, DType::Int16, 7, 96, 60.0},
        {PtoOpcode::TNOT, DType::Uint16, 16, 128, 114.0},  {PtoOpcode::TNOT, DType::Uint16, 16, 16, 114.0},
        {PtoOpcode::TNOT, DType::Uint16, 16, 256, 130.0},  {PtoOpcode::TNOT, DType::Uint16, 4, 128, 42.0},
        {PtoOpcode::TNOT, DType::Uint16, 4, 256, 46.0},    {PtoOpcode::TNOT, DType::Uint16, 64, 64, 402.0},
        {PtoOpcode::TNOT, DType::Uint16, 7, 96, 60.0},     {PtoOpcode::TOR, DType::Int16, 16, 128, 114.0},
        {PtoOpcode::TOR, DType::Int16, 16, 256, 130.0},    {PtoOpcode::TOR, DType::Int16, 4, 128, 42.0},
        {PtoOpcode::TOR, DType::Int16, 4, 256, 46.0},      {PtoOpcode::TOR, DType::Int16, 7, 96, 60.0},
        {PtoOpcode::TOR, DType::Uint16, 16, 128, 114.0},   {PtoOpcode::TOR, DType::Uint16, 16, 16, 114.0},
        {PtoOpcode::TOR, DType::Uint16, 16, 256, 130.0},   {PtoOpcode::TOR, DType::Uint16, 4, 128, 42.0},
        {PtoOpcode::TOR, DType::Uint16, 4, 256, 46.0},     {PtoOpcode::TOR, DType::Uint16, 64, 64, 402.0},
        {PtoOpcode::TOR, DType::Uint16, 7, 96, 60.0},      {PtoOpcode::TORS, DType::Uint16, 16, 128, 114.0},
        {PtoOpcode::TORS, DType::Uint16, 16, 16, 114.0},   {PtoOpcode::TORS, DType::Uint16, 16, 256, 130.0},
        {PtoOpcode::TORS, DType::Uint16, 4, 128, 42.0},    {PtoOpcode::TORS, DType::Uint16, 4, 256, 46.0},
        {PtoOpcode::TORS, DType::Uint16, 64, 64, 402.0},   {PtoOpcode::TORS, DType::Uint16, 7, 96, 60.0},
        {PtoOpcode::TXOR, DType::Uint16, 16, 128, 114.0},  {PtoOpcode::TXOR, DType::Uint16, 16, 16, 114.0},
        {PtoOpcode::TXOR, DType::Uint16, 16, 256, 130.0},  {PtoOpcode::TXOR, DType::Uint16, 4, 128, 42.0},
        {PtoOpcode::TXOR, DType::Uint16, 4, 256, 46.0},    {PtoOpcode::TXOR, DType::Uint16, 64, 64, 402.0},
        {PtoOpcode::TXOR, DType::Uint16, 7, 96, 60.0},     {PtoOpcode::TXORS, DType::Uint16, 16, 128, 114.0},
        {PtoOpcode::TXORS, DType::Uint16, 16, 16, 114.0},  {PtoOpcode::TXORS, DType::Uint16, 16, 256, 130.0},
        {PtoOpcode::TXORS, DType::Uint16, 4, 128, 42.0},   {PtoOpcode::TXORS, DType::Uint16, 4, 256, 46.0},
        {PtoOpcode::TXORS, DType::Uint16, 64, 64, 402.0},  {PtoOpcode::TXORS, DType::Uint16, 7, 96, 60.0},
    };
    for (const auto& profile : profiles) {
        const CostModelInput input{
            .op = profile.op,
            .dtype = profile.dtype,
            .rows = profile.rows,
            .cols = profile.cols,
            .arch = CostModelArch::A5,
            .a5_shape_path = A5ShapePathHint::Path2D,
        };
        CostModelResult result{};
        ASSERT_TRUE(EstimateCycles(input, result));
        EXPECT_NEAR(result.cycles, profile.cycles, profile.cycles * 0.10)
            << static_cast<int>(profile.op) << " " << profile.rows << "x" << profile.cols;
    }
}

TEST(A5BitwiseFit, SingleRowSingleRepeatMeasuredProfiles)
{
    struct Case {
        PtoOpcode op;
        DType dtype;
        int64_t cols;
        double measured;
    };
    const Case cases[] = {
        {PtoOpcode::TAND, DType::Int16, 16, 37.0},    {PtoOpcode::TAND, DType::Int16, 128, 37.0},
        {PtoOpcode::TAND, DType::Uint16, 128, 37.0},  {PtoOpcode::TAND, DType::Uint16, 16, 37.0},
        {PtoOpcode::TNOT, DType::Int16, 16, 36.0},    {PtoOpcode::TNOT, DType::Int16, 128, 36.0},
        {PtoOpcode::TNOT, DType::Uint16, 128, 36.0},  {PtoOpcode::TNOT, DType::Uint16, 16, 36.0},
        {PtoOpcode::TOR, DType::Int16, 16, 37.0},     {PtoOpcode::TOR, DType::Int16, 128, 37.0},
        {PtoOpcode::TOR, DType::Uint16, 128, 37.0},   {PtoOpcode::TOR, DType::Uint16, 16, 37.0},
        {PtoOpcode::TANDS, DType::Uint16, 128, 36.0}, {PtoOpcode::TANDS, DType::Uint16, 16, 36.0},
        {PtoOpcode::TORS, DType::Uint16, 128, 36.0},  {PtoOpcode::TORS, DType::Uint16, 16, 36.0},
        {PtoOpcode::TXOR, DType::Uint16, 128, 37.0},  {PtoOpcode::TXOR, DType::Uint16, 16, 37.0},
        {PtoOpcode::TXORS, DType::Uint16, 128, 36.0}, {PtoOpcode::TXORS, DType::Uint16, 16, 36.0},
    };
    for (const auto& item : cases) {
        CostModelInput input{
            .op = item.op,
            .dtype = item.dtype,
            .rows = 1,
            .cols = item.cols,
            .arch = CostModelArch::A5,
            .a5_shape_path = A5ShapePathHint::Path1D};
        CostModelResult result{};
        ASSERT_TRUE(EstimateCycles(input, result));
        EXPECT_LE(std::abs(result.cycles - item.measured) / item.measured, 0.10);
        pto::mocker::lightweight::a5::fit::VfCurveKey key{};
        uint64_t loop, outer, inner, tail, rowInner;
        input.cols = 256;
        ASSERT_TRUE(
            pto::mocker::lightweight::a5::BuildVfCurveKeyAndLoopCount(input, key, loop, outer, inner, tail, rowInner));
        EXPECT_NE(key.op_params, "single_row_single_repeat");
    }
}
} // namespace
