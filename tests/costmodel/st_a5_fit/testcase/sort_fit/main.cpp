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
    {PtoOpcode::TMRGSORT, "TMRGSORT", DType::Half, 1, 8192, 1123.0},
    {PtoOpcode::TSORT32, "TSORT32", DType::Half, 128, 32, 1290.1},
};

TEST(A5SortFit, HardwareProfiles)
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

TEST(A5SortFit, SortWithoutTemporaryTileShortRows)
{
    struct SortCase {
        DType dtype;
        const char* name;
        int64_t rows;
        int64_t cols;
        double profilingCycles;
    };
    constexpr SortCase kSortCases[] = {
        {DType::Half, "TSORT32/no_tmp/fp16", 16, 16, 138.2},
        {DType::Float, "TSORT32/no_tmp/fp32", 16, 16, 137.2},
        {DType::Half, "TSORT32/no_tmp/fp16", 64, 16, 522.2},
    };
    constexpr double kRequiredPrecision = 0.95;
    for (const auto& testCase : kSortCases) {
        const CostModelInput input{
            .op = PtoOpcode::TSORT32,
            .dtype = testCase.dtype,
            .rows = testCase.rows,
            .cols = testCase.cols,
            .arch = CostModelArch::A5,
            .valid_rows = testCase.rows,
            .valid_cols = testCase.cols,
            .a5_shape_path = A5ShapePathHint::Path2D,
            .a5_op_params = "no_tmp_subrepeat",
        };
        pto::test::ExpectA5CycleNear(testCase.name, input, testCase.profilingCycles, kRequiredPrecision);
    }
}

} // namespace
