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
#include <string_view>

#include <gtest/gtest.h>

#include <pto/common/type.hpp>
#include <pto/costmodel/lightweight_costmodel.hpp>

#include "a5_cost_check.hpp"

namespace {

using ::pto::VFImplKind;
using ::pto::mocker::lightweight::CostModelArch;
using ::pto::mocker::lightweight::CostModelInput;
using ::pto::mocker::lightweight::DType;
using ::pto::mocker::lightweight::PtoOpcode;

struct A5VfFitCase {
    DType dtype;
    int64_t rows;
    int64_t cols;
    int64_t valid_rows;
    int64_t valid_cols;
    double profiling_cycles;
    double precision;
    VFImplKind vf_impl_kind;
    std::string_view op_params;
};

void RunCase(PtoOpcode op, const char* name, const A5VfFitCase& c)
{
    CostModelInput input{
        .op = op,
        .dtype = c.dtype,
        .rows = c.rows,
        .cols = c.cols,
        .arch = CostModelArch::A5,
        .valid_rows = c.valid_rows,
        .valid_cols = c.valid_cols,
        .vf_impl_kind = c.vf_impl_kind,
        .a5_op_params = c.op_params,
    };
    pto::test::ExpectA5CycleNear(name, input, c.profiling_cycles, c.precision);
}

constexpr A5VfFitCase kCases0[] = {
    // fp16 1D POST_UPDATE full
    {DType::Half, 1, 128, 1, 128, 69.0, 0.9855, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Half, 1, 640, 1, 640, 86.0, 0.9883, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Half, 1, 1280, 1, 1280, 103.0, 0.9902, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
};

TEST(A5TexpFit, fp16_1D_POST_UPDATE_full_none)
{
    for (const auto& c : kCases0) {
        RunCase(PtoOpcode::TEXP, "TEXP", c);
    }
}

constexpr A5VfFitCase kCases1[] = {
    // fp16 2D NO_POST_UPDATE full
    {DType::Half, 3, 192, 3, 128, 77.0, 0.9870, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Half, 3, 448, 3, 384, 102.0, 0.9803, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Half, 3, 1344, 3, 1280, 183.0, 0.9945, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Half, 10, 192, 10, 128, 102.0, 0.9901, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Half, 10, 448, 10, 384, 183.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Half, 10, 1344, 10, 1280, 463.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
};

TEST(A5TexpFit, fp16_2D_NO_POST_UPDATE_full_none)
{
    for (const auto& c : kCases1) {
        RunCase(PtoOpcode::TEXP, "TEXP", c);
    }
}

constexpr A5VfFitCase kCases2[] = {
    // fp32 1D NO_POST_UPDATE full
    {DType::Float, 1, 640, 1, 640, 81.0, 0.9876, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
};

TEST(A5TexpFit, fp32_1D_NO_POST_UPDATE_full_none)
{
    for (const auto& c : kCases2) {
        RunCase(PtoOpcode::TEXP, "TEXP", c);
    }
}

constexpr A5VfFitCase kCases3[] = {
    // fp32 1D POST_UPDATE full
    {DType::Float, 1, 64, 1, 64, 64.0, 0.9843, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 192, 1, 192, 69.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 320, 1, 320, 73.0, 0.9863, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 640, 1, 640, 82.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 1280, 1, 1280, 102.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
};

TEST(A5TexpFit, fp32_1D_POST_UPDATE_full_none)
{
    for (const auto& c : kCases3) {
        RunCase(PtoOpcode::TEXP, "TEXP", c);
    }
}

constexpr A5VfFitCase kCases4[] = {
    // fp32 1D POST_UPDATE tail
    {DType::Float, 1, 64, 1, 1, 64.0, 0.9843, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 192, 1, 129, 69.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 320, 1, 257, 73.0, 0.9863, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 640, 1, 577, 82.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 1280, 1, 1217, 102.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
};

TEST(A5TexpFit, fp32_1D_POST_UPDATE_tail_none)
{
    for (const auto& c : kCases4) {
        RunCase(PtoOpcode::TEXP, "TEXP", c);
    }
}

constexpr A5VfFitCase kCases5[] = {
    // fp32 2D NO_POST_UPDATE full
    {DType::Float, 2, 160, 2, 128, 73.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 224, 2, 192, 76.0, 0.9868, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 672, 2, 640, 101.0, 0.9801, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 1312, 2, 1280, 141.0, 0.9929, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 160, 3, 128, 80.0, 0.9875, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 224, 3, 192, 85.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 672, 3, 640, 122.0, 0.9836, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 1312, 3, 1280, 182.0, 0.9890, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 160, 10, 128, 129.0, 0.9534, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 224, 10, 192, 141.0, 0.9929, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 672, 10, 640, 262.0, 0.9618, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 1312, 10, 1280, 462.0, 0.9891, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 160, 20, 128, 199.0, 0.9346, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 224, 20, 192, 221.0, 0.9909, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 672, 20, 640, 462.0, 0.9567, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 1312, 20, 1280, 862.0, 0.9895, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
};

TEST(A5TexpFit, fp32_2D_NO_POST_UPDATE_full_none)
{
    for (const auto& c : kCases5) {
        RunCase(PtoOpcode::TEXP, "TEXP", c);
    }
}

constexpr A5VfFitCase kCases6[] = {
    // fp32 2D NO_POST_UPDATE tail
    {DType::Float, 2, 160, 2, 65, 73.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 224, 2, 129, 76.0, 0.9868, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 672, 2, 577, 101.0, 0.9801, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 1312, 2, 1217, 141.0, 0.9929, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 160, 3, 65, 80.0, 0.9875, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 224, 3, 129, 85.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 672, 3, 577, 122.0, 0.9836, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 1312, 3, 1217, 182.0, 0.9890, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 160, 10, 65, 129.0, 0.9534, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 224, 10, 129, 141.0, 0.9929, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 672, 10, 577, 262.0, 0.9618, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 1312, 10, 1217, 462.0, 0.9891, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 160, 20, 65, 199.0, 0.9346, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 224, 20, 129, 221.0, 0.9909, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 672, 20, 577, 462.0, 0.9567, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 1312, 20, 1217, 862.0, 0.9895, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
};

TEST(A5TexpFit, fp32_2D_NO_POST_UPDATE_tail_none)
{
    for (const auto& c : kCases6) {
        RunCase(PtoOpcode::TEXP, "TEXP", c);
    }
}

} // namespace
