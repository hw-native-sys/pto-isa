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

struct TabsFitCase {
    DType dtype;
    int64_t rows;
    int64_t cols;
    int64_t validRows;
    int64_t validCols;
    double profilingCycles;
};

constexpr TabsFitCase kTabsCases[] = {
    {DType::Float, 1, 64, 1, 64, 36.002},        {DType::Float, 1, 640, 1, 640, 45.002},
    {DType::Float, 1, 1280, 1, 1280, 56.001},    {DType::Float, 1, 63, 1, 63, 36.002},
    {DType::Float, 1, 1279, 1, 1279, 56.001},    {DType::Float, 2, 96, 2, 64, 37.003},
    {DType::Float, 10, 352, 10, 320, 108.032},   {DType::Float, 20, 1312, 20, 1280, 566.033},
    {DType::Float, 3, 224, 3, 192, 44.033},      {DType::Float, 3, 672, 3, 640, 66.004},
    {DType::Float, 2, 96, 2, 63, 37.003},        {DType::Float, 10, 352, 10, 319, 108.005},
    {DType::Float, 20, 1312, 20, 1279, 566.084}, {DType::Float, 3, 224, 3, 191, 44.128},
    {DType::Float, 3, 672, 3, 639, 66.074},      {DType::Half, 1, 128, 1, 128, 36.002},
    {DType::Half, 1, 1280, 1, 1280, 45.002},     {DType::Half, 1, 2560, 1, 2560, 56.001},
    {DType::Half, 1, 127, 1, 127, 36.002},       {DType::Half, 1, 2559, 1, 2559, 56.001},
    {DType::Half, 2, 144, 2, 128, 37.003},       {DType::Half, 10, 656, 10, 640, 108.005},
    {DType::Half, 20, 2576, 20, 2560, 692.031},  {DType::Half, 3, 400, 3, 384, 44.037},
    {DType::Half, 3, 1296, 3, 1280, 67.004},     {DType::Half, 2, 144, 2, 127, 38.003},
    {DType::Half, 10, 656, 10, 639, 133.005},    {DType::Half, 20, 2576, 20, 2559, 751.002},
    {DType::Half, 3, 400, 3, 383, 49.076},       {DType::Half, 3, 1296, 3, 1279, 75.005},
    {DType::Half, 4, 144, 4, 127, 42.005},       {DType::Half, 20, 144, 20, 127, 138.005},
    {DType::Half, 4, 272, 4, 255, 48.005},       {DType::Half, 17, 272, 17, 255, 137.006},
    {DType::Half, 7, 528, 7, 511, 86.005},       {DType::Half, 8, 528, 8, 511, 96.005},
    {DType::Half, 20, 528, 20, 511, 208.006},    {DType::Half, 13, 912, 13, 895, 218.005},
    {DType::Half, 17, 1552, 17, 1535, 398.005},  {DType::Half, 4, 2064, 4, 2047, 132.003},
};

TEST(A5ElementwiseFit, AbsHardwareProfiles)
{
    constexpr double kRequiredPrecisionWithHelperTolerance = 0.95;
    for (const auto& testCase : kTabsCases) {
        CostModelInput input{
            .op = PtoOpcode::TABS,
            .dtype = testCase.dtype,
            .rows = testCase.rows,
            .cols = testCase.cols,
            .arch = CostModelArch::A5,
            .valid_rows = testCase.validRows,
            .valid_cols = testCase.validCols,
        };
        pto::test::ExpectA5CycleNear("TABS", input, testCase.profilingCycles, kRequiredPrecisionWithHelperTolerance);
    }
}

struct ElementwiseFitCase {
    PtoOpcode op;
    const char* name;
    DType dtype;
    int64_t rows;
    int64_t cols;
    double profilingCycles;
};

constexpr ElementwiseFitCase kElementwiseCases[] = {
    {PtoOpcode::TDIV, "TDIV", DType::Half, 1, 16, 53.0},
    {PtoOpcode::TDIV, "TDIV", DType::Half, 1, 1024, 78.0},
    {PtoOpcode::TDIV, "TDIV", DType::Half, 24, 256, 238.0},
    {PtoOpcode::TMAX, "TMAX", DType::Half, 1, 16, 37.0},
    {PtoOpcode::TMAX, "TMAX", DType::Half, 24, 256, 186.0},
    {PtoOpcode::TMIN, "TMIN", DType::Half, 1, 1024, 44.0},
    {PtoOpcode::TMIN, "TMIN", DType::Half, 48, 64, 306.0},
    {PtoOpcode::TADDS, "TADDS", DType::Half, 1, 16, 37.0},
    {PtoOpcode::TADDS, "TADDS", DType::Half, 16, 128, 114.0},
    {PtoOpcode::TSUBS, "TSUBS", DType::Half, 1, 1024, 44.0},
    {PtoOpcode::TSUBS, "TSUBS", DType::Half, 24, 256, 186.0},
    {PtoOpcode::TMULS, "TMULS", DType::Half, 1, 16, 38.0},
    {PtoOpcode::TMULS, "TMULS", DType::Half, 48, 64, 306.0},
    {PtoOpcode::TMAXS, "TMAXS", DType::Half, 1, 1024, 43.0},
    {PtoOpcode::TMAXS, "TMAXS", DType::Half, 16, 128, 114.0},
    {PtoOpcode::TRELU, "TRELU", DType::Half, 1, 16, 36.0},
    {PtoOpcode::TRELU, "TRELU", DType::Half, 24, 256, 186.0},
    {PtoOpcode::TLRELU, "TLRELU", DType::Half, 1, 1024, 45.0},
    {PtoOpcode::TLRELU, "TLRELU", DType::Half, 48, 64, 306.9},
    {PtoOpcode::TLOG, "TLOG", DType::Half, 1, 16, 53.0},
    {PtoOpcode::TLOG, "TLOG", DType::Half, 24, 256, 238.0},
    {PtoOpcode::TFMOD, "TFMOD", DType::Float, 1, 1024, 110.0},
    {PtoOpcode::TFMOD, "TFMOD", DType::Float, 24, 256, 379.0},
    {PtoOpcode::TFMODS, "TFMODS", DType::Float, 1, 16, 60.0},
    {PtoOpcode::TFMODS, "TFMODS", DType::Float, 64, 128, 606.0},
    {PtoOpcode::TREM, "TREM", DType::Float, 1, 64, 71.0},
    {PtoOpcode::TREM, "TREM", DType::Float, 16, 256, 467.0},
    {PtoOpcode::TREMS, "TREMS", DType::Float, 1, 1024, 111.0},
    {PtoOpcode::TREMS, "TREMS", DType::Float, 48, 64, 354.0},
    {PtoOpcode::TPOW, "TPOW", DType::Float, 1, 16, 90.0},
    {PtoOpcode::TPOW, "TPOW", DType::Float, 16, 256, 1877.0},
    {PtoOpcode::TPOWS, "TPOWS", DType::Float, 16, 16, 166.5},
    {PtoOpcode::TPOWS, "TPOWS", DType::Float, 64, 128, 3912.5},
    {PtoOpcode::TMADD, "TMADD", DType::Half, 1, 1024, 57.0},
    {PtoOpcode::TMADD, "TMADD", DType::Half, 24, 256, 234.0},
    {PtoOpcode::TMULA, "TMULA", DType::Half, 1, 16, 42.0},
    {PtoOpcode::TMULA, "TMULA", DType::Half, 48, 64, 354.0},
    {PtoOpcode::TPRELU, "TPRELU", DType::Half, 1, 512, 42.0},
    {PtoOpcode::TPRELU, "TPRELU", DType::Half, 128, 128, 786.0},
    {PtoOpcode::TAXPY, "TAXPY", DType::Half, 16, 128, 114.0},
};

TEST(A5ElementwiseFit, HardwareProfiles)
{
    constexpr double kRequiredPrecision = 0.90;
    for (const auto& testCase : kElementwiseCases) {
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

TEST(A5ElementwiseFit, Fp16SingleRepeatUsesMeasuredStartupRegion)
{
    struct SmallRepeatCase {
        PtoOpcode op;
        double cycles;
    };
    constexpr SmallRepeatCase kCases[] = {
        {PtoOpcode::TADD, 38.0}, {PtoOpcode::TSUB, 38.0},  {PtoOpcode::TMUL, 39.0},
        {PtoOpcode::TNEG, 38.0}, {PtoOpcode::TMINS, 36.0},
    };
    for (const auto& testCase : kCases) {
        CostModelResult result{};
        ASSERT_TRUE(EstimateCycles(
            CostModelInput{
                .op = testCase.op,
                .dtype = DType::Half,
                .rows = 1,
                .cols = 16,
                .arch = CostModelArch::A5,
                .valid_rows = 1,
                .valid_cols = 16,
            },
            result));
        EXPECT_EQ(result.cycles, testCase.cycles);
    }
}

TEST(A5ElementwiseFit, Fp16LongSingleRowAndTwoDimensionalPaths)
{
    struct RegressionCase {
        PtoOpcode op;
        const char* name;
        int64_t rows;
        int64_t cols;
        double profilingCycles;
    };
    constexpr RegressionCase kCases[] = {
        {PtoOpcode::TADD, "TADD/long-row", 1, 43648, 737.0}, {PtoOpcode::TMUL, "TMUL/long-row", 1, 43648, 737.0},
        {PtoOpcode::TMAX, "TMAX/long-row", 1, 43648, 736.0}, {PtoOpcode::TMIN, "TMIN/long-row", 1, 43648, 736.0},
        {PtoOpcode::TABS, "TABS/long-row", 1, 65472, 548.0}, {PtoOpcode::TABS, "TABS/2D", 128, 128, 786.0},
        {PtoOpcode::TNEG, "TNEG/2D", 16, 256, 130.0},        {PtoOpcode::TNEG, "TNEG/narrow-2D", 4092, 16, 24570.0},
    };
    constexpr double kRequiredPrecision = 0.95;
    for (const auto& testCase : kCases) {
        const CostModelInput input{
            .op = testCase.op,
            .dtype = DType::Half,
            .rows = testCase.rows,
            .cols = testCase.cols,
            .arch = CostModelArch::A5,
            .valid_rows = testCase.rows,
            .valid_cols = testCase.cols,
            .a5_shape_path = testCase.rows == 1 ? A5ShapePathHint::Path1D : A5ShapePathHint::Path2D,
        };
        pto::test::ExpectA5CycleNear(testCase.name, input, testCase.profilingCycles, kRequiredPrecision);
    }
}

} // namespace
