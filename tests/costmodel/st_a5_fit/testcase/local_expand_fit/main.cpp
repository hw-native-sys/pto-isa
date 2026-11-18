/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>

#include <gtest/gtest.h>

#include <pto/costmodel/lightweight_costmodel.hpp>

#include "a5_cost_check.hpp"

namespace {

using namespace ::pto::mocker::lightweight;

struct FitCase {
    PtoOpcode op;
    const char* name;
    DType dtype;
    int64_t rows;
    int64_t cols;
    double profilingCycles;
};

constexpr FitCase kCases[] = {
    {PtoOpcode::TPARTADD, "TPARTADD", DType::Half, 1, 16, 38.0},
    {PtoOpcode::TPARTADD, "TPARTADD", DType::Half, 24, 256, 186.0},
    {PtoOpcode::TPARTMUL, "TPARTMUL", DType::Half, 1, 1024, 46.0},
    {PtoOpcode::TPARTMUL, "TPARTMUL", DType::Half, 48, 64, 306.0},
    {PtoOpcode::TPARTMAX, "TPARTMAX", DType::Half, 2, 64, 39.0},
    {PtoOpcode::TPARTMAX, "TPARTMAX", DType::Half, 64, 128, 402.0},
    {PtoOpcode::TPARTMIN, "TPARTMIN", DType::Half, 1, 512, 41.0},
    {PtoOpcode::TPARTMIN, "TPARTMIN", DType::Half, 16, 256, 130.0},
    {PtoOpcode::TROWEXPAND, "TROWEXPAND", DType::Half, 8, 128, 66.0},
    {PtoOpcode::TROWEXPANDADD, "TROWEXPANDADD", DType::Half, 16, 128, 114.0},
    {PtoOpcode::TROWEXPANDSUB, "TROWEXPANDSUB", DType::Half, 8, 128, 66.0},
    {PtoOpcode::TROWEXPANDMUL, "TROWEXPANDMUL", DType::Half, 16, 128, 114.0},
    {PtoOpcode::TROWEXPANDDIV, "TROWEXPANDDIV", DType::Half, 8, 128, 78.0},
    {PtoOpcode::TROWEXPANDMAX, "TROWEXPANDMAX", DType::Half, 16, 128, 114.0},
    {PtoOpcode::TROWEXPANDMIN, "TROWEXPANDMIN", DType::Half, 8, 128, 66.0},
    {PtoOpcode::TROWEXPANDEXPDIF, "TROWEXPANDEXPDIF", DType::Half, 8, 128, 81.0},
    {PtoOpcode::TCOLEXPAND, "TCOLEXPAND", DType::Half, 16, 128, 49.0},
    {PtoOpcode::TCOLEXPANDADD, "TCOLEXPANDADD", DType::Half, 8, 128, 66.0},
    {PtoOpcode::TCOLEXPANDSUB, "TCOLEXPANDSUB", DType::Half, 16, 128, 114.0},
    {PtoOpcode::TCOLEXPANDMUL, "TCOLEXPANDMUL", DType::Half, 8, 128, 66.0},
    {PtoOpcode::TCOLEXPANDDIV, "TCOLEXPANDDIV", DType::Half, 8, 128, 78.0},
    {PtoOpcode::TCOLEXPANDMAX, "TCOLEXPANDMAX", DType::Half, 16, 128, 114.0},
    {PtoOpcode::TCOLEXPANDMIN, "TCOLEXPANDMIN", DType::Half, 8, 128, 66.0},
    {PtoOpcode::TCOLEXPANDEXPDIF, "TCOLEXPANDEXPDIF", DType::Half, 8, 128, 81.0},
    {PtoOpcode::TEXPANDS, "TEXPANDS", DType::Half, 16, 128, 114.0},
};

TEST(A5LocalExpandFit, HardwareProfiles)
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
        {PtoOpcode::TCOLEXPANDADD, DType::Float, 16, 128, 130.0},
        {PtoOpcode::TCOLEXPANDADD, DType::Float, 16, 16, 114.0},
        {PtoOpcode::TCOLEXPANDADD, DType::Float, 16, 256, 162.0},
        {PtoOpcode::TCOLEXPANDADD, DType::Float, 16, 64, 114.0},
        {PtoOpcode::TCOLEXPANDADD, DType::Float, 32, 32, 210.0},
        {PtoOpcode::TCOLEXPANDADD, DType::Float, 4, 128, 46.0},
        {PtoOpcode::TCOLEXPANDADD, DType::Float, 4, 256, 54.0},
        {PtoOpcode::TCOLEXPANDADD, DType::Float, 64, 128, 466.0},
        {PtoOpcode::TCOLEXPANDADD, DType::Float, 64, 64, 402.0},
        {PtoOpcode::TCOLEXPANDADD, DType::Float, 7, 96, 67.0},
        {PtoOpcode::TCOLEXPANDMAX, DType::Float, 16, 128, 130.0},
        {PtoOpcode::TCOLEXPANDMAX, DType::Float, 16, 16, 114.0},
        {PtoOpcode::TCOLEXPANDMAX, DType::Float, 16, 256, 162.0},
        {PtoOpcode::TCOLEXPANDMAX, DType::Float, 16, 64, 114.0},
        {PtoOpcode::TCOLEXPANDMAX, DType::Float, 32, 32, 210.0},
        {PtoOpcode::TCOLEXPANDMAX, DType::Float, 4, 128, 46.0},
        {PtoOpcode::TCOLEXPANDMAX, DType::Float, 4, 256, 54.0},
        {PtoOpcode::TCOLEXPANDMAX, DType::Float, 64, 128, 466.0},
        {PtoOpcode::TCOLEXPANDMAX, DType::Float, 64, 64, 402.0},
        {PtoOpcode::TCOLEXPANDMAX, DType::Float, 7, 96, 67.0},
        {PtoOpcode::TCOLEXPANDMIN, DType::Float, 16, 128, 130.0},
        {PtoOpcode::TCOLEXPANDMIN, DType::Float, 16, 16, 114.0},
        {PtoOpcode::TCOLEXPANDMIN, DType::Float, 16, 256, 162.0},
        {PtoOpcode::TCOLEXPANDMIN, DType::Float, 16, 64, 114.0},
        {PtoOpcode::TCOLEXPANDMIN, DType::Float, 32, 32, 210.0},
        {PtoOpcode::TCOLEXPANDMIN, DType::Float, 4, 128, 46.0},
        {PtoOpcode::TCOLEXPANDMIN, DType::Float, 4, 256, 54.0},
        {PtoOpcode::TCOLEXPANDMIN, DType::Float, 64, 128, 466.0},
        {PtoOpcode::TCOLEXPANDMIN, DType::Float, 64, 64, 402.0},
        {PtoOpcode::TCOLEXPANDMIN, DType::Float, 7, 96, 67.0},
        {PtoOpcode::TCOLEXPANDMUL, DType::Float, 16, 128, 130.0},
        {PtoOpcode::TCOLEXPANDMUL, DType::Float, 16, 16, 114.0},
        {PtoOpcode::TCOLEXPANDMUL, DType::Float, 16, 256, 162.0},
        {PtoOpcode::TCOLEXPANDMUL, DType::Float, 16, 64, 114.0},
        {PtoOpcode::TCOLEXPANDMUL, DType::Float, 32, 32, 210.0},
        {PtoOpcode::TCOLEXPANDMUL, DType::Float, 4, 128, 46.0},
        {PtoOpcode::TCOLEXPANDMUL, DType::Float, 4, 256, 54.0},
        {PtoOpcode::TCOLEXPANDMUL, DType::Float, 64, 128, 466.0},
        {PtoOpcode::TCOLEXPANDMUL, DType::Float, 64, 64, 402.0},
        {PtoOpcode::TCOLEXPANDMUL, DType::Float, 7, 96, 67.0},
        {PtoOpcode::TCOLEXPANDSUB, DType::Float, 16, 128, 130.0},
        {PtoOpcode::TCOLEXPANDSUB, DType::Float, 16, 16, 114.0},
        {PtoOpcode::TCOLEXPANDSUB, DType::Float, 16, 256, 162.0},
        {PtoOpcode::TCOLEXPANDSUB, DType::Float, 16, 64, 114.0},
        {PtoOpcode::TCOLEXPANDSUB, DType::Float, 32, 32, 210.0},
        {PtoOpcode::TCOLEXPANDSUB, DType::Float, 4, 128, 46.0},
        {PtoOpcode::TCOLEXPANDSUB, DType::Float, 4, 256, 54.0},
        {PtoOpcode::TCOLEXPANDSUB, DType::Float, 64, 128, 466.0},
        {PtoOpcode::TCOLEXPANDSUB, DType::Float, 64, 64, 402.0},
        {PtoOpcode::TCOLEXPANDSUB, DType::Float, 7, 96, 67.0},
        {PtoOpcode::TEXPANDS, DType::Int16, 16, 128, 114.0},
        {PtoOpcode::TEXPANDS, DType::Int16, 16, 16, 114.0},
        {PtoOpcode::TEXPANDS, DType::Int16, 16, 256, 130.0},
        {PtoOpcode::TEXPANDS, DType::Int16, 4, 128, 42.0},
        {PtoOpcode::TEXPANDS, DType::Int16, 4, 256, 46.0},
        {PtoOpcode::TEXPANDS, DType::Int16, 64, 64, 402.0},
        {PtoOpcode::TEXPANDS, DType::Int16, 7, 96, 60.0},
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

TEST(A5ValidatedProfiles, IdentifiableCrossValidatedHardwareProfiles)
{
    struct Profile {
        PtoOpcode op;
        DType dtype;
        int64_t rows;
        int64_t cols;
        double cycles;
    };
    constexpr Profile profiles[] = {
        {PtoOpcode::TCOLEXPAND, DType::Float, 16, 128, 65.0},
        {PtoOpcode::TCOLEXPAND, DType::Float, 16, 16, 49.0},
        {PtoOpcode::TCOLEXPAND, DType::Float, 16, 256, 97.0},
        {PtoOpcode::TCOLEXPAND, DType::Float, 16, 64, 49.0},
        {PtoOpcode::TCOLEXPAND, DType::Float, 1, 128, 35.0},
        {PtoOpcode::TCOLEXPAND, DType::Float, 32, 32, 65.0},
        {PtoOpcode::TCOLEXPAND, DType::Float, 4, 128, 41.0},
        {PtoOpcode::TCOLEXPAND, DType::Float, 4, 256, 50.0},
        {PtoOpcode::TCOLEXPAND, DType::Float, 64, 128, 161.0},
        {PtoOpcode::TCOLEXPAND, DType::Float, 64, 64, 97.0},
        {PtoOpcode::TCOLEXPAND, DType::Float, 7, 96, 47.0},
        {PtoOpcode::TCOLEXPANDDIV, DType::Float, 16, 128, 130.0},
        {PtoOpcode::TCOLEXPANDDIV, DType::Float, 16, 16, 114.0},
        {PtoOpcode::TCOLEXPANDDIV, DType::Float, 16, 256, 173.0},
        {PtoOpcode::TCOLEXPANDDIV, DType::Float, 16, 64, 114.0},
        {PtoOpcode::TCOLEXPANDDIV, DType::Float, 1, 128, 49.0},
        {PtoOpcode::TCOLEXPANDDIV, DType::Float, 32, 32, 210.0},
        {PtoOpcode::TCOLEXPANDDIV, DType::Float, 4, 128, 61.0},
        {PtoOpcode::TCOLEXPANDDIV, DType::Float, 4, 256, 77.0},
        {PtoOpcode::TCOLEXPANDDIV, DType::Float, 64, 128, 466.0},
        {PtoOpcode::TCOLEXPANDDIV, DType::Float, 64, 64, 402.0},
        {PtoOpcode::TCOLEXPANDDIV, DType::Float, 7, 96, 73.0},
        {PtoOpcode::TCOLEXPANDEXPDIF, DType::Float, 16, 128, 130.0},
        {PtoOpcode::TCOLEXPANDEXPDIF, DType::Float, 16, 16, 114.0},
        {PtoOpcode::TCOLEXPANDEXPDIF, DType::Float, 16, 256, 174.0},
        {PtoOpcode::TCOLEXPANDEXPDIF, DType::Float, 16, 64, 114.0},
        {PtoOpcode::TCOLEXPANDEXPDIF, DType::Float, 1, 128, 50.0},
        {PtoOpcode::TCOLEXPANDEXPDIF, DType::Float, 32, 32, 210.0},
        {PtoOpcode::TCOLEXPANDEXPDIF, DType::Float, 4, 128, 62.0},
        {PtoOpcode::TCOLEXPANDEXPDIF, DType::Float, 4, 256, 78.0},
        {PtoOpcode::TCOLEXPANDEXPDIF, DType::Float, 64, 128, 466.0},
        {PtoOpcode::TCOLEXPANDEXPDIF, DType::Float, 64, 64, 402.0},
        {PtoOpcode::TCOLEXPANDEXPDIF, DType::Float, 7, 96, 74.0},
        {PtoOpcode::TROWEXPANDDIV, DType::Float, 16, 128, 130.0},
        {PtoOpcode::TROWEXPANDDIV, DType::Float, 16, 16, 114.0},
        {PtoOpcode::TROWEXPANDDIV, DType::Float, 16, 256, 173.0},
        {PtoOpcode::TROWEXPANDDIV, DType::Float, 16, 64, 114.0},
        {PtoOpcode::TROWEXPANDDIV, DType::Float, 1, 128, 49.0},
        {PtoOpcode::TROWEXPANDDIV, DType::Float, 32, 32, 210.0},
        {PtoOpcode::TROWEXPANDDIV, DType::Float, 4, 128, 61.0},
        {PtoOpcode::TROWEXPANDDIV, DType::Float, 4, 256, 77.0},
        {PtoOpcode::TROWEXPANDDIV, DType::Float, 64, 128, 466.0},
        {PtoOpcode::TROWEXPANDDIV, DType::Float, 64, 64, 402.0},
        {PtoOpcode::TROWEXPANDDIV, DType::Float, 7, 96, 73.0},
        {PtoOpcode::TROWEXPANDEXPDIF, DType::Float, 16, 128, 130.0},
        {PtoOpcode::TROWEXPANDEXPDIF, DType::Float, 16, 16, 114.0},
        {PtoOpcode::TROWEXPANDEXPDIF, DType::Float, 16, 256, 174.0},
        {PtoOpcode::TROWEXPANDEXPDIF, DType::Float, 16, 64, 114.0},
        {PtoOpcode::TROWEXPANDEXPDIF, DType::Float, 1, 128, 50.0},
        {PtoOpcode::TROWEXPANDEXPDIF, DType::Float, 32, 32, 210.0},
        {PtoOpcode::TROWEXPANDEXPDIF, DType::Float, 4, 128, 62.0},
        {PtoOpcode::TROWEXPANDEXPDIF, DType::Float, 4, 256, 78.0},
        {PtoOpcode::TROWEXPANDEXPDIF, DType::Float, 64, 128, 466.0},
        {PtoOpcode::TROWEXPANDEXPDIF, DType::Float, 64, 64, 402.0},
        {PtoOpcode::TROWEXPANDEXPDIF, DType::Float, 7, 96, 74.0},
        {PtoOpcode::TEXPANDS, DType::Float, 16, 128, 130.0},
        {PtoOpcode::TEXPANDS, DType::Float, 16, 16, 114.0},
        {PtoOpcode::TEXPANDS, DType::Float, 16, 256, 146.0},
        {PtoOpcode::TEXPANDS, DType::Float, 16, 64, 114.0},
        {PtoOpcode::TEXPANDS, DType::Float, 32, 32, 210.0},
        {PtoOpcode::TEXPANDS, DType::Float, 4, 128, 46.0},
        {PtoOpcode::TEXPANDS, DType::Float, 4, 256, 50.0},
        {PtoOpcode::TEXPANDS, DType::Float, 64, 128, 466.0},
        {PtoOpcode::TEXPANDS, DType::Float, 64, 64, 402.0},
        {PtoOpcode::TEXPANDS, DType::Float, 7, 96, 67.0},
        {PtoOpcode::TEXPANDS, DType::Int32, 16, 128, 130.0},
        {PtoOpcode::TEXPANDS, DType::Int32, 16, 16, 114.0},
        {PtoOpcode::TEXPANDS, DType::Int32, 16, 256, 146.0},
        {PtoOpcode::TEXPANDS, DType::Int32, 4, 128, 46.0},
        {PtoOpcode::TEXPANDS, DType::Int32, 4, 256, 50.0},
        {PtoOpcode::TEXPANDS, DType::Int32, 64, 64, 402.0},
        {PtoOpcode::TEXPANDS, DType::Int32, 7, 96, 67.0},
        {PtoOpcode::TROWEXPAND, DType::Float, 16, 128, 147.0},
        {PtoOpcode::TROWEXPAND, DType::Float, 16, 16, 129.0},
        {PtoOpcode::TROWEXPAND, DType::Float, 16, 256, 179.0},
        {PtoOpcode::TROWEXPAND, DType::Float, 16, 64, 131.0},
        {PtoOpcode::TROWEXPAND, DType::Float, 32, 32, 241.0},
        {PtoOpcode::TROWEXPAND, DType::Float, 4, 128, 51.0},
        {PtoOpcode::TROWEXPAND, DType::Float, 4, 256, 59.0},
        {PtoOpcode::TROWEXPAND, DType::Float, 64, 128, 531.0},
        {PtoOpcode::TROWEXPAND, DType::Float, 64, 64, 467.0},
        {PtoOpcode::TROWEXPAND, DType::Float, 7, 96, 75.0},
        {PtoOpcode::TROWEXPANDADD, DType::Float, 16, 128, 130.0},
        {PtoOpcode::TROWEXPANDADD, DType::Float, 16, 16, 114.0},
        {PtoOpcode::TROWEXPANDADD, DType::Float, 16, 256, 146.0},
        {PtoOpcode::TROWEXPANDADD, DType::Float, 16, 64, 114.0},
        {PtoOpcode::TROWEXPANDADD, DType::Float, 32, 32, 210.0},
        {PtoOpcode::TROWEXPANDADD, DType::Float, 4, 128, 46.0},
        {PtoOpcode::TROWEXPANDADD, DType::Float, 4, 256, 53.0},
        {PtoOpcode::TROWEXPANDADD, DType::Float, 64, 128, 466.0},
        {PtoOpcode::TROWEXPANDADD, DType::Float, 64, 64, 402.0},
        {PtoOpcode::TROWEXPANDADD, DType::Float, 7, 96, 67.0},
        {PtoOpcode::TROWEXPANDMAX, DType::Float, 16, 128, 130.0},
        {PtoOpcode::TROWEXPANDMAX, DType::Float, 16, 16, 114.0},
        {PtoOpcode::TROWEXPANDMAX, DType::Float, 16, 256, 146.0},
        {PtoOpcode::TROWEXPANDMAX, DType::Float, 16, 64, 114.0},
        {PtoOpcode::TROWEXPANDMAX, DType::Float, 32, 32, 210.0},
        {PtoOpcode::TROWEXPANDMAX, DType::Float, 4, 128, 46.0},
        {PtoOpcode::TROWEXPANDMAX, DType::Float, 4, 256, 52.0},
        {PtoOpcode::TROWEXPANDMAX, DType::Float, 64, 128, 466.0},
        {PtoOpcode::TROWEXPANDMAX, DType::Float, 64, 64, 402.0},
        {PtoOpcode::TROWEXPANDMAX, DType::Float, 7, 96, 67.0},
        {PtoOpcode::TROWEXPANDMIN, DType::Float, 16, 128, 130.0},
        {PtoOpcode::TROWEXPANDMIN, DType::Float, 16, 16, 114.0},
        {PtoOpcode::TROWEXPANDMIN, DType::Float, 16, 256, 146.0},
        {PtoOpcode::TROWEXPANDMIN, DType::Float, 16, 64, 114.0},
        {PtoOpcode::TROWEXPANDMIN, DType::Float, 32, 32, 210.0},
        {PtoOpcode::TROWEXPANDMIN, DType::Float, 4, 128, 46.0},
        {PtoOpcode::TROWEXPANDMIN, DType::Float, 4, 256, 52.0},
        {PtoOpcode::TROWEXPANDMIN, DType::Float, 64, 128, 466.0},
        {PtoOpcode::TROWEXPANDMIN, DType::Float, 64, 64, 402.0},
        {PtoOpcode::TROWEXPANDMIN, DType::Float, 7, 96, 67.0},
        {PtoOpcode::TROWEXPANDSUB, DType::Float, 16, 128, 130.0},
        {PtoOpcode::TROWEXPANDSUB, DType::Float, 16, 16, 114.0},
        {PtoOpcode::TROWEXPANDSUB, DType::Float, 16, 256, 146.0},
        {PtoOpcode::TROWEXPANDSUB, DType::Float, 16, 64, 114.0},
        {PtoOpcode::TROWEXPANDSUB, DType::Float, 32, 32, 210.0},
        {PtoOpcode::TROWEXPANDSUB, DType::Float, 4, 128, 46.0},
        {PtoOpcode::TROWEXPANDSUB, DType::Float, 4, 256, 53.0},
        {PtoOpcode::TROWEXPANDSUB, DType::Float, 64, 128, 466.0},
        {PtoOpcode::TROWEXPANDSUB, DType::Float, 64, 64, 402.0},
        {PtoOpcode::TROWEXPANDSUB, DType::Float, 7, 96, 67.0},
    };
    for (const auto& profile : profiles) {
        CostModelInput input{};
        input.op = profile.op;
        input.dtype = profile.dtype;
        input.rows = profile.rows;
        input.cols = profile.cols;
        input.arch = CostModelArch::A5;
        input.a5_shape_path = profile.rows == 1 ? A5ShapePathHint::Path1D : A5ShapePathHint::Path2D;
        CostModelResult result{};
        ASSERT_TRUE(EstimateCycles(input, result));
        EXPECT_NEAR(result.cycles, profile.cycles, profile.cycles * 0.10)
            << static_cast<int>(profile.op) << " " << profile.rows << "x" << profile.cols;
    }
}

TEST(A5LocalExpandFit, RowMultiplyIndependentHardwareProfiles)
{
    struct Case {
        int64_t rows;
        int64_t cols;
        double measured;
    };
    const Case cases[] = {{16, 128, 130}, {16, 16, 114}, {16, 256, 146}, {16, 64, 114}, {32, 32, 210},
                          {4, 128, 46},   {4, 256, 54},  {64, 128, 466}, {64, 64, 402}, {7, 96, 67}};
    for (const auto& item : cases) {
        CostModelInput input{
            .op = PtoOpcode::TROWEXPANDMUL,
            .dtype = DType::Float,
            .rows = item.rows,
            .cols = item.cols,
            .arch = CostModelArch::A5,
            .a5_shape_path = A5ShapePathHint::Path2D};
        CostModelResult result{};
        ASSERT_TRUE(EstimateCycles(input, result));
        EXPECT_LE(std::abs(result.cycles - item.measured) / item.measured, 0.10);
    }
}

} // namespace
