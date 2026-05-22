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

void RunCase(PtoOpcode op, const char *name, const A5VfFitCase &c)
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
    {DType::Half, 1, 1280, 1, 1280, 64.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
};

TEST(A5TminsFit, fp16_1D_POST_UPDATE_full_none)
{
    for (const auto &c : kCases0) {
        RunCase(PtoOpcode::TMINS, "TMINS", c);
    }
}

constexpr A5VfFitCase kCases1[] = {
    // fp32 1D NO_POST_UPDATE full
    {DType::Float, 1, 640, 1, 640, 63.0, 0.9841, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
};

TEST(A5TminsFit, fp32_1D_NO_POST_UPDATE_full_none)
{
    for (const auto &c : kCases1) {
        RunCase(PtoOpcode::TMINS, "TMINS", c);
    }
}

constexpr A5VfFitCase kCases2[] = {
    // fp32 1D POST_UPDATE full
    {DType::Float, 1, 64, 1, 64, 53.0, 0.9811, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 192, 1, 192, 57.0, 0.9824, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 320, 1, 320, 59.0, 0.9830, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 640, 1, 640, 64.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 1280, 1, 1280, 74.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
};

TEST(A5TminsFit, fp32_1D_POST_UPDATE_full_none)
{
    for (const auto &c : kCases2) {
        RunCase(PtoOpcode::TMINS, "TMINS", c);
    }
}

constexpr A5VfFitCase kCases3[] = {
    // fp32 1D POST_UPDATE tail
    {DType::Float, 1, 64, 1, 1, 53.0, 0.9811, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 192, 1, 129, 57.0, 0.9824, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 320, 1, 257, 59.0, 0.9830, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 640, 1, 577, 64.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 1280, 1, 1217, 74.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
};

TEST(A5TminsFit, fp32_1D_POST_UPDATE_tail_none)
{
    for (const auto &c : kCases3) {
        RunCase(PtoOpcode::TMINS, "TMINS", c);
    }
}

constexpr A5VfFitCase kCases4[] = {
    // fp32 2D NO_POST_UPDATE full
    {DType::Float, 2, 160, 2, 128, 63.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 224, 2, 192, 64.0, 0.9843, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 672, 2, 640, 77.0, 0.9480, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 1312, 2, 1280, 103.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 160, 3, 128, 70.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 224, 3, 192, 73.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 672, 3, 640, 94.0, 0.9787, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 1312, 3, 1280, 134.0, 0.9626, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 160, 10, 128, 119.0, 0.9831, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 224, 10, 192, 129.0, 0.9922, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 672, 10, 640, 199.0, 0.9698, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 1312, 10, 1280, 323.0, 0.9752, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 160, 20, 128, 189.0, 0.9735, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 224, 20, 192, 209.0, 0.9856, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 672, 20, 640, 349.0, 0.9656, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 1312, 20, 1280, 583.0, 0.9965, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
};

TEST(A5TminsFit, fp32_2D_NO_POST_UPDATE_full_none)
{
    for (const auto &c : kCases4) {
        RunCase(PtoOpcode::TMINS, "TMINS", c);
    }
}

constexpr A5VfFitCase kCases5[] = {
    // fp32 2D NO_POST_UPDATE tail
    {DType::Float, 2, 160, 2, 65, 63.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 224, 2, 129, 64.0, 0.9843, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 672, 2, 577, 77.0, 0.9480, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 1312, 2, 1217, 103.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 160, 3, 65, 70.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 224, 3, 129, 73.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 672, 3, 577, 94.0, 0.9787, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 1312, 3, 1217, 134.0, 0.9626, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 160, 10, 65, 119.0, 0.9831, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 224, 10, 129, 129.0, 0.9922, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 672, 10, 577, 199.0, 0.9698, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 1312, 10, 1217, 323.0, 0.9752, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 160, 20, 65, 189.0, 0.9735, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 224, 20, 129, 209.0, 0.9856, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 672, 20, 577, 349.0, 0.9656, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 1312, 20, 1217, 583.0, 0.9965, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
};

TEST(A5TminsFit, fp32_2D_NO_POST_UPDATE_tail_none)
{
    for (const auto &c : kCases5) {
        RunCase(PtoOpcode::TMINS, "TMINS", c);
    }
}

} // namespace
