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
    {DType::Half, 1, 128, 1, 128, 72.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Half, 1, 640, 1, 640, 87.0, 0.9885, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Half, 1, 1280, 1, 1280, 104.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
};

TEST(A5TdivsFit, fp16_1D_POST_UPDATE_full_none)
{
    for (const auto& c : kCases0) {
        RunCase(PtoOpcode::TDIVS, "TDIVS", c);
    }
}

constexpr A5VfFitCase kCases1[] = {
    // fp16 2D NO_POST_UPDATE full
    {DType::Half, 1, 192, 1, 128, 68.0, 0.9852, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Half, 1, 448, 1, 384, 78.0, 0.9871, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Half, 1, 1344, 1, 1280, 103.0, 0.9805, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Half, 3, 192, 3, 128, 78.0, 0.9871, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Half, 3, 448, 3, 384, 103.0, 0.9805, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Half, 3, 1344, 3, 1280, 184.0, 0.9945, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Half, 10, 192, 10, 128, 103.0, 0.9902, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Half, 10, 448, 10, 384, 184.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Half, 10, 1344, 10, 1280, 464.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
};

TEST(A5TdivsFit, fp16_2D_NO_POST_UPDATE_full_none)
{
    for (const auto& c : kCases1) {
        RunCase(PtoOpcode::TDIVS, "TDIVS", c);
    }
}

constexpr A5VfFitCase kCases2[] = {
    // fp32 1D NO_POST_UPDATE full
    {DType::Float, 1, 640, 1, 640, 82.0, 0.9756, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
};

TEST(A5TdivsFit, fp32_1D_NO_POST_UPDATE_full_none)
{
    for (const auto& c : kCases2) {
        RunCase(PtoOpcode::TDIVS, "TDIVS", c);
    }
}

constexpr A5VfFitCase kCases3[] = {
    // fp32 1D POST_UPDATE full
    {DType::Float, 1, 64, 1, 64, 67.0, 0.9850, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 192, 1, 192, 70.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 320, 1, 320, 74.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 640, 1, 640, 83.0, 0.9879, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 1280, 1, 1280, 103.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
};

TEST(A5TdivsFit, fp32_1D_POST_UPDATE_full_none)
{
    for (const auto& c : kCases3) {
        RunCase(PtoOpcode::TDIVS, "TDIVS", c);
    }
}

constexpr A5VfFitCase kCases4[] = {
    // fp32 1D POST_UPDATE tail
    {DType::Float, 1, 64, 1, 1, 67.0, 0.9850, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 192, 1, 129, 70.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 320, 1, 257, 74.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 640, 1, 577, 83.0, 0.9879, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
    {DType::Float, 1, 1280, 1, 1217, 103.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE, ""},
};

TEST(A5TdivsFit, fp32_1D_POST_UPDATE_tail_none)
{
    for (const auto& c : kCases4) {
        RunCase(PtoOpcode::TDIVS, "TDIVS", c);
    }
}

constexpr A5VfFitCase kCases5[] = {
    // fp32 2D NO_POST_UPDATE full
    {DType::Float, 2, 160, 2, 128, 74.0, 0.9864, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 224, 2, 192, 78.0, 0.9871, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 672, 2, 640, 102.0, 0.9705, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 1312, 2, 1280, 146.0, 0.9794, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 160, 3, 128, 82.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 224, 3, 192, 87.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 672, 3, 640, 123.0, 0.9756, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 1312, 3, 1280, 187.0, 0.9786, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 160, 10, 128, 130.0, 0.9615, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 224, 10, 192, 143.0, 0.9930, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 672, 10, 640, 267.0, 0.9700, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 1312, 10, 1280, 467.0, 0.9871, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 160, 20, 128, 201.0, 0.9303, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 224, 20, 192, 222.0, 0.9864, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 672, 20, 640, 467.0, 0.9593, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 1312, 20, 1280, 867.0, 0.9919, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
};

TEST(A5TdivsFit, fp32_2D_NO_POST_UPDATE_full_none)
{
    for (const auto& c : kCases5) {
        RunCase(PtoOpcode::TDIVS, "TDIVS", c);
    }
}

constexpr A5VfFitCase kCases6[] = {
    // fp32 2D NO_POST_UPDATE tail
    {DType::Float, 2, 160, 2, 65, 74.0, 0.9864, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 224, 2, 129, 78.0, 0.9871, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 672, 2, 577, 102.0, 0.9705, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 2, 1312, 2, 1217, 146.0, 0.9794, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 160, 3, 65, 82.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 224, 3, 129, 87.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 672, 3, 577, 123.0, 0.9756, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 3, 1312, 3, 1217, 187.0, 0.9786, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 160, 10, 65, 130.0, 0.9615, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 224, 10, 129, 143.0, 0.9930, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 672, 10, 577, 267.0, 0.9700, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 10, 1312, 10, 1217, 467.0, 0.9871, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 160, 20, 65, 201.0, 0.9303, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 224, 20, 129, 222.0, 0.9864, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 672, 20, 577, 467.0, 0.9593, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
    {DType::Float, 20, 1312, 20, 1217, 867.0, 0.9919, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, ""},
};

TEST(A5TdivsFit, fp32_2D_NO_POST_UPDATE_tail_none)
{
    for (const auto& c : kCases6) {
        RunCase(PtoOpcode::TDIVS, "TDIVS", c);
    }
}

} // namespace
