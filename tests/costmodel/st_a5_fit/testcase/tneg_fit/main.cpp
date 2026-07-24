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
    {DType::Half, 1, 1280, 1, 1280, 66.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
};

TEST(A5TnegFit, fp16_1D_POST_UPDATE_full_none)
{
    for (const auto& c : kCases0) {
        RunCase(PtoOpcode::TNEG, "TNEG", c);
    }
}

constexpr A5VfFitCase kCases1[] = {
    // fp32 1D NO_POST_UPDATE full
    {DType::Float, 1, 640, 1, 640, 65.0, 0.9846, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
};

TEST(A5TnegFit, fp32_1D_NO_POST_UPDATE_full_none)
{
    for (const auto& c : kCases1) {
        RunCase(PtoOpcode::TNEG, "TNEG", c);
    }
}

constexpr A5VfFitCase kCases2[] = {
    // fp32 1D POST_UPDATE full
    {DType::Float, 1, 64, 1, 64, 55.0, 0.9818, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 192, 1, 192, 59.0, 0.9830, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 320, 1, 320, 61.0, 0.9836, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 640, 1, 640, 66.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 1280, 1, 1280, 76.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
};

TEST(A5TnegFit, fp32_1D_POST_UPDATE_full_none)
{
    for (const auto& c : kCases2) {
        RunCase(PtoOpcode::TNEG, "TNEG", c);
    }
}

constexpr A5VfFitCase kCases3[] = {
    // fp32 1D POST_UPDATE tail
    {DType::Float, 1, 64, 1, 1, 55.0, 0.9818, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 192, 1, 129, 59.0, 0.9830, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 320, 1, 257, 61.0, 0.9836, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 640, 1, 577, 66.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 1280, 1, 1217, 76.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
};

TEST(A5TnegFit, fp32_1D_POST_UPDATE_tail_none)
{
    for (const auto& c : kCases3) {
        RunCase(PtoOpcode::TNEG, "TNEG", c);
    }
}

constexpr A5VfFitCase kCases4[] = {
    // fp32 2D NO_POST_UPDATE full
    {DType::Float, 2, 160, 2, 128, 65.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 224, 2, 192, 66.0, 0.9848, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 672, 2, 640, 78.0, 0.9358, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 1312, 2, 1280, 105.0, 0.9904, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 160, 3, 128, 72.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 224, 3, 192, 75.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 672, 3, 640, 96.0, 0.9791, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 1312, 3, 1280, 136.0, 0.9632, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 160, 10, 128, 121.0, 0.9834, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 224, 10, 192, 131.0, 0.9923, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 672, 10, 640, 201.0, 0.9701, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 1312, 10, 1280, 323.0, 0.9783, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 160, 20, 128, 191.0, 0.9790, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 224, 20, 192, 211.0, 0.9905, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 672, 20, 640, 351.0, 0.9686, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 1312, 20, 1280, 583.0, 0.9965, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
};

TEST(A5TnegFit, fp32_2D_NO_POST_UPDATE_full_none)
{
    for (const auto& c : kCases4) {
        RunCase(PtoOpcode::TNEG, "TNEG", c);
    }
}

constexpr A5VfFitCase kCases5[] = {
    // fp32 2D NO_POST_UPDATE tail
    {DType::Float, 2, 160, 2, 65, 65.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 224, 2, 129, 66.0, 0.9848, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 672, 2, 577, 78.0, 0.9358, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 1312, 2, 1217, 105.0, 0.9904, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 160, 3, 65, 72.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 224, 3, 129, 75.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 672, 3, 577, 96.0, 0.9791, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 1312, 3, 1217, 136.0, 0.9632, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 160, 10, 65, 121.0, 0.9834, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 224, 10, 129, 131.0, 0.9923, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 672, 10, 577, 201.0, 0.9701, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 1312, 10, 1217, 323.0, 0.9783, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 160, 20, 65, 191.0, 0.9790, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 224, 20, 129, 211.0, 0.9905, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 672, 20, 577, 351.0, 0.9686, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 1312, 20, 1217, 583.0, 0.9965, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
};

TEST(A5TnegFit, fp32_2D_NO_POST_UPDATE_tail_none)
{
    for (const auto& c : kCases5) {
        RunCase(PtoOpcode::TNEG, "TNEG", c);
    }
}

} // namespace
