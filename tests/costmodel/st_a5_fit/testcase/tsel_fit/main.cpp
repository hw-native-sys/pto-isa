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
    // int8 1D NO_POST_UPDATE full TSel_b16_8
    {DType::Int8, 1, 256, 1, 256, 54.0, 0.9814, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, "TSel_b16_8"},
    {DType::Int8, 1, 1280, 1, 1280, 64.0, 0.9843, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, "TSel_b16_8"},
    {DType::Int8, 1, 2560, 1, 2560, 74.0, 0.9864, VFImplKind::VFIMPL_1D_NO_POST_UPDATE, "TSel_b16_8"},
};

TEST(A5TselFit, int8_1D_NO_POST_UPDATE_full_TSel_b16_8)
{
    for (const auto& c : kCases0) {
        RunCase(PtoOpcode::TSEL, "TSEL", c);
    }
}

constexpr A5VfFitCase kCases1[] = {
    // int8 2D DEFAULT full TSel_b16_8
    {DType::Int8, 2, 640, 2, 512, 66.0, 0.9242, VFImplKind::VFIMPL_DEFAULT, "TSel_b16_8"},
    {DType::Int8, 2, 896, 2, 768, 67.0, 0.9850, VFImplKind::VFIMPL_DEFAULT, "TSel_b16_8"},
    {DType::Int8, 2, 2688, 2, 2560, 95.0, 0.9368, VFImplKind::VFIMPL_DEFAULT, "TSel_b16_8"},
    {DType::Int8, 3, 640, 3, 512, 77.0, 0.8831, VFImplKind::VFIMPL_DEFAULT, "TSel_b16_8"},
};

TEST(A5TselFit, int8_2D_DEFAULT_full_TSel_b16_8)
{
    for (const auto& c : kCases1) {
        RunCase(PtoOpcode::TSEL, "TSEL", c);
    }
}

constexpr A5VfFitCase kCases2[] = {
    // fp32 1D default op params, backend selects TSel_b32 odd/even
    {DType::Float, 1, 64, 1, 64, 60.0, 0.8666, VFImplKind::VFIMPL_DEFAULT, ""},
    {DType::Float, 1, 320, 1, 320, 91.0, 0.9120, VFImplKind::VFIMPL_DEFAULT, ""},
    {DType::Float, 1, 1280, 1, 1280, 97.0, 0.9896, VFImplKind::VFIMPL_DEFAULT, ""},
};

TEST(A5TselFit, fp32_1D_default_TSel_b32_parity)
{
    for (const auto& c : kCases2) {
        RunCase(PtoOpcode::TSEL, "TSEL", c);
    }
}

constexpr A5VfFitCase kCases3[] = {
    // fp32 2D default op params TSel_b32
    {DType::Float, 2, 160, 2, 128, 73.0, 0.5342, VFImplKind::VFIMPL_DEFAULT, ""},
    {DType::Float, 3, 224, 3, 192, 206.0, 0.7718, VFImplKind::VFIMPL_DEFAULT, ""},
    {DType::Float, 10, 672, 10, 640, 791.0, 0.8748, VFImplKind::VFIMPL_DEFAULT, ""},
};

TEST(A5TselFit, fp32_2D_default_TSel_b32)
{
    for (const auto& c : kCases3) {
        RunCase(PtoOpcode::TSEL, "TSEL", c);
    }
}

TEST(A5TselFit, DefaultKeyUsesNpuOperandWidth)
{
    namespace a5 = pto::mocker::lightweight::a5;
    for (DType dtype : {DType::Float, DType::Int32, DType::Uint32}) {
        EXPECT_EQ(a5::TselOpParams(dtype, a5::fit::ShapePath::Path1D, 1), "TSel_b32_odd");
        EXPECT_EQ(a5::TselOpParams(dtype, a5::fit::ShapePath::Path1D, 2), "TSel_b32_even");
        EXPECT_EQ(a5::TselOpParams(dtype, a5::fit::ShapePath::Path2D, 1), "TSel_b32");
    }
    for (DType dtype : {DType::Half, DType::BFloat16, DType::Int16, DType::Uint16, DType::Int8, DType::Uint8}) {
        EXPECT_EQ(a5::TselOpParams(dtype, a5::fit::ShapePath::Path1D, 1), "TSel_b16_8");
        EXPECT_EQ(a5::TselOpParams(dtype, a5::fit::ShapePath::Path2D, 2), "TSel_b16_8");
    }
}

} // namespace
