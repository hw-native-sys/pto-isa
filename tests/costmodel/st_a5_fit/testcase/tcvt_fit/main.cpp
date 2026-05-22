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
using ::pto::mocker::lightweight::RoundMode;
using ::pto::mocker::lightweight::SaturationMode;

struct A5TcvtFitCase {
    DType src_dtype;
    DType dst_dtype;
    int64_t rows;
    int64_t cols;
    int64_t valid_rows;
    int64_t valid_cols;
    double profiling_cycles;
    double precision;
    VFImplKind vf_impl_kind;
};

void RunTcvt(const A5TcvtFitCase &c)
{
    CostModelInput input{
        .op = PtoOpcode::TCVT,
        .dtype = c.src_dtype,
        .rows = c.rows,
        .cols = c.cols,
        .arch = CostModelArch::A5,
        .dst_dtype = c.dst_dtype,
        .round_mode = RoundMode::CAST_RINT,
        .valid_rows = c.valid_rows,
        .valid_cols = c.valid_cols,
        .vf_impl_kind = c.vf_impl_kind,
        .saturation_mode = SaturationMode::ON,
    };
    pto::test::ExpectA5CycleNear("TCVT", input, c.profiling_cycles, c.precision);
}

constexpr A5TcvtFitCase kCases0[] = {
    // bf16_to_fp16 1D NO_POST_UPDATE full
    {DType::BFloat16, DType::Half, 1, 128, 1, 128, 54.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::BFloat16, DType::Half, 1, 640, 1, 640, 60.0, 0.9833, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::BFloat16, DType::Half, 1, 1280, 1, 1280, 65.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
};

TEST(A5TcvtFit, bf16_to_fp16_1D_NO_POST_UPDATE_full)
{
    for (const auto &c : kCases0) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases1[] = {
    // bf16_to_fp16 2D POST_UPDATE full
    {DType::BFloat16, DType::Half, 3, 192, 3, 128, 58.0, 0.9482, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::BFloat16, DType::Half, 3, 448, 3, 384, 75.0, 0.9200, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::BFloat16, DType::Half, 3, 1344, 3, 1280, 95.0, 0.9894, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::BFloat16, DType::Half, 10, 192, 10, 128, 65.0, 0.7230, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::BFloat16, DType::Half, 10, 448, 10, 384, 131.0, 0.8320, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::BFloat16, DType::Half, 10, 1344, 10, 1280, 193.0, 0.9740, VFImplKind::VFIMPL_1D_POST_UPDATE},
};

TEST(A5TcvtFit, bf16_to_fp16_2D_POST_UPDATE_full)
{
    for (const auto &c : kCases1) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases2[] = {
    // bf16_to_fp32 1D NO_POST_UPDATE full
    {DType::BFloat16, DType::Float, 1, 64, 1, 64, 81.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::BFloat16, DType::Float, 1, 320, 1, 320, 73.0, 0.9863, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::BFloat16, DType::Float, 1, 640, 1, 640, 65.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
};

TEST(A5TcvtFit, bf16_to_fp32_1D_NO_POST_UPDATE_full)
{
    for (const auto &c : kCases2) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases3[] = {
    // bf16_to_fp32 2D POST_UPDATE full
    {DType::BFloat16, DType::Float, 3, 128, 3, 64, 59.0, 0.9661, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::BFloat16, DType::Float, 3, 256, 3, 192, 75.0, 0.9200, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::BFloat16, DType::Float, 3, 704, 3, 640, 95.0, 0.9894, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::BFloat16, DType::Float, 10, 128, 10, 64, 66.0, 0.7272, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::BFloat16, DType::Float, 10, 256, 10, 192, 131.0, 0.8320, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::BFloat16, DType::Float, 10, 704, 10, 640, 193.0, 0.9740, VFImplKind::VFIMPL_1D_POST_UPDATE},
};

TEST(A5TcvtFit, bf16_to_fp32_2D_POST_UPDATE_full)
{
    for (const auto &c : kCases3) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases4[] = {
    // bf16_to_fp4_e1m2 1D NO_POST_UPDATE full
    {DType::BFloat16, DType::Float4E1M2, 1, 128, 1, 128, 67.0, 0.9850, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::BFloat16, DType::Float4E1M2, 1, 640, 1, 640, 119.0, 0.9915, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::BFloat16, DType::Float4E1M2, 1, 1280, 1, 1280, 179.0, 0.9944, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
};

TEST(A5TcvtFit, bf16_to_fp4_e1m2_1D_NO_POST_UPDATE_full)
{
    for (const auto &c : kCases4) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases5[] = {
    // bf16_to_fp4_e1m2 2D POST_UPDATE full
    {DType::BFloat16, DType::Float4E1M2, 3, 192, 3, 128, 96.0, 0.1145, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::BFloat16, DType::Float4E1M2, 3, 448, 3, 384, 893.0, 0.7917, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::BFloat16, DType::Float4E1M2, 3, 1344, 3, 1280, 2552.0, 0.9992, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::BFloat16, DType::Float4E1M2, 10, 192, 10, 128, 180.0, 0.0000, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::BFloat16, DType::Float4E1M2, 10, 448, 10, 384, 2839.0, 0.8245, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::BFloat16, DType::Float4E1M2, 10, 1344, 10, 1280, 8369.0, 0.9863, VFImplKind::VFIMPL_1D_POST_UPDATE},
};

TEST(A5TcvtFit, bf16_to_fp4_e1m2_2D_POST_UPDATE_full)
{
    for (const auto &c : kCases5) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases6[] = {
    // bf16_to_fp4_e2m1 1D NO_POST_UPDATE full
    {DType::BFloat16, DType::Float4E2M1, 1, 128, 1, 128, 67.0, 0.9850, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::BFloat16, DType::Float4E2M1, 1, 640, 1, 640, 119.0, 0.9915, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::BFloat16, DType::Float4E2M1, 1, 1280, 1, 1280, 179.0, 0.9944, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
};

TEST(A5TcvtFit, bf16_to_fp4_e2m1_1D_NO_POST_UPDATE_full)
{
    for (const auto &c : kCases6) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases7[] = {
    // bf16_to_fp4_e2m1 2D POST_UPDATE full
    {DType::BFloat16, DType::Float4E2M1, 3, 192, 3, 128, 96.0, 0.1145, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::BFloat16, DType::Float4E2M1, 3, 448, 3, 384, 893.0, 0.7917, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::BFloat16, DType::Float4E2M1, 3, 1344, 3, 1280, 2552.0, 0.9992, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::BFloat16, DType::Float4E2M1, 10, 192, 10, 128, 180.0, 0.0000, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::BFloat16, DType::Float4E2M1, 10, 448, 10, 384, 2839.0, 0.8245, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::BFloat16, DType::Float4E2M1, 10, 1344, 10, 1280, 8369.0, 0.9863, VFImplKind::VFIMPL_1D_POST_UPDATE},
};

TEST(A5TcvtFit, bf16_to_fp4_e2m1_2D_POST_UPDATE_full)
{
    for (const auto &c : kCases7) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases8[] = {
    // fp16_to_fp32 1D NO_POST_UPDATE full
    {DType::Half, DType::Float, 1, 64, 1, 64, 81.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Half, DType::Float, 1, 320, 1, 320, 73.0, 0.9863, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Half, DType::Float, 1, 640, 1, 640, 65.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
};

TEST(A5TcvtFit, fp16_to_fp32_1D_NO_POST_UPDATE_full)
{
    for (const auto &c : kCases8) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases9[] = {
    // fp16_to_fp32 2D POST_UPDATE full
    {DType::Half, DType::Float, 3, 128, 3, 64, 59.0, 0.9661, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Half, DType::Float, 3, 256, 3, 192, 75.0, 0.9200, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Half, DType::Float, 3, 704, 3, 640, 95.0, 0.9894, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Half, DType::Float, 10, 128, 10, 64, 66.0, 0.7272, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Half, DType::Float, 10, 256, 10, 192, 131.0, 0.8320, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Half, DType::Float, 10, 704, 10, 640, 193.0, 0.9740, VFImplKind::VFIMPL_1D_POST_UPDATE},
};

TEST(A5TcvtFit, fp16_to_fp32_2D_POST_UPDATE_full)
{
    for (const auto &c : kCases9) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases10[] = {
    // fp16_to_hif8 1D NO_POST_UPDATE full
    {DType::Half, DType::HFloat8, 1, 128, 1, 128, 54.0, 0.9814, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Half, DType::HFloat8, 1, 640, 1, 640, 61.0, 0.9836, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Half, DType::HFloat8, 1, 1280, 1, 1280, 66.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
};

TEST(A5TcvtFit, fp16_to_hif8_1D_NO_POST_UPDATE_full)
{
    for (const auto &c : kCases10) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases11[] = {
    // fp16_to_hif8 2D POST_UPDATE full
    {DType::Half, DType::HFloat8, 3, 192, 3, 128, 58.0, 0.9310, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Half, DType::HFloat8, 3, 448, 3, 384, 76.0, 0.9078, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Half, DType::HFloat8, 3, 1344, 3, 1280, 96.0, 0.9895, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Half, DType::HFloat8, 10, 192, 10, 128, 65.0, 0.7230, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Half, DType::HFloat8, 10, 448, 10, 384, 132.0, 0.8257, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Half, DType::HFloat8, 10, 1344, 10, 1280, 194.0, 0.9742, VFImplKind::VFIMPL_1D_POST_UPDATE},
};

TEST(A5TcvtFit, fp16_to_hif8_2D_POST_UPDATE_full)
{
    for (const auto &c : kCases11) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases12[] = {
    // fp32_to_bf16 1D NO_POST_UPDATE full
    {DType::Float, DType::BFloat16, 1, 64, 1, 64, 53.0, 0.9811, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Float, DType::BFloat16, 1, 320, 1, 320, 60.0, 0.9833, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Float, DType::BFloat16, 1, 640, 1, 640, 65.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
};

TEST(A5TcvtFit, fp32_to_bf16_1D_NO_POST_UPDATE_full)
{
    for (const auto &c : kCases12) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases13[] = {
    // fp32_to_bf16 2D POST_UPDATE full
    {DType::Float, DType::BFloat16, 3, 192, 3, 128, 64.0, 0.9843, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::BFloat16, 3, 448, 3, 384, 83.0, 0.9518, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::BFloat16, 3, 1344, 3, 1280, 127.0, 0.9921, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::BFloat16, 10, 192, 10, 128, 78.0, 0.8717, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::BFloat16, 10, 448, 10, 384, 146.0, 0.9109, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::BFloat16, 10, 1344, 10, 1280, 288.0, 0.9895, VFImplKind::VFIMPL_1D_POST_UPDATE},
};

TEST(A5TcvtFit, fp32_to_bf16_2D_POST_UPDATE_full)
{
    for (const auto &c : kCases13) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases14[] = {
    // fp32_to_fp16 1D NO_POST_UPDATE full
    {DType::Float, DType::Half, 1, 64, 1, 64, 53.0, 0.9811, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Float, DType::Half, 1, 320, 1, 320, 60.0, 0.9833, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Float, DType::Half, 1, 640, 1, 640, 65.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
};

TEST(A5TcvtFit, fp32_to_fp16_1D_NO_POST_UPDATE_full)
{
    for (const auto &c : kCases14) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases15[] = {
    // fp32_to_fp16 2D POST_UPDATE full
    {DType::Float, DType::Half, 3, 192, 3, 128, 64.0, 0.9843, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::Half, 3, 448, 3, 384, 83.0, 0.9518, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::Half, 3, 1344, 3, 1280, 127.0, 0.9921, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::Half, 10, 192, 10, 128, 78.0, 0.8717, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::Half, 10, 448, 10, 384, 146.0, 0.9109, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::Half, 10, 1344, 10, 1280, 288.0, 0.9895, VFImplKind::VFIMPL_1D_POST_UPDATE},
};

TEST(A5TcvtFit, fp32_to_fp16_2D_POST_UPDATE_full)
{
    for (const auto &c : kCases15) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases16[] = {
    // fp32_to_fp8_e4m3 1D NO_POST_UPDATE full
    {DType::Float, DType::Float8E4M3, 1, 64, 1, 64, 67.0, 0.9850, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Float, DType::Float8E4M3, 1, 320, 1, 320, 118.0, 0.9915, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Float, DType::Float8E4M3, 1, 640, 1, 640, 178.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
};

TEST(A5TcvtFit, fp32_to_fp8_e4m3_1D_NO_POST_UPDATE_full)
{
    for (const auto &c : kCases16) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases17[] = {
    // fp32_to_fp8_e4m3 2D POST_UPDATE full
    {DType::Float, DType::Float8E4M3, 3, 192, 3, 128, 97.0, 0.1443, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::Float8E4M3, 3, 448, 3, 384, 890.0, 0.7932, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::Float8E4M3, 3, 1344, 3, 1280, 2549.0, 0.9992, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::Float8E4M3, 10, 192, 10, 128, 181.0, 0.0000, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::Float8E4M3, 10, 448, 10, 384, 2829.0, 0.8260, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::Float8E4M3, 10, 1344, 10, 1280, 8359.0, 0.9866, VFImplKind::VFIMPL_1D_POST_UPDATE},
};

TEST(A5TcvtFit, fp32_to_fp8_e4m3_2D_POST_UPDATE_full)
{
    for (const auto &c : kCases17) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases18[] = {
    // fp32_to_fp8_e5m2 1D NO_POST_UPDATE full
    {DType::Float, DType::Float8E5M2, 1, 64, 1, 64, 67.0, 0.9850, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Float, DType::Float8E5M2, 1, 320, 1, 320, 118.0, 0.9915, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Float, DType::Float8E5M2, 1, 640, 1, 640, 178.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
};

TEST(A5TcvtFit, fp32_to_fp8_e5m2_1D_NO_POST_UPDATE_full)
{
    for (const auto &c : kCases18) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases19[] = {
    // fp32_to_fp8_e5m2 2D POST_UPDATE full
    {DType::Float, DType::Float8E5M2, 3, 192, 3, 128, 97.0, 0.1443, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::Float8E5M2, 3, 448, 3, 384, 890.0, 0.7932, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::Float8E5M2, 3, 1344, 3, 1280, 2549.0, 0.9992, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::Float8E5M2, 10, 192, 10, 128, 181.0, 0.0000, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::Float8E5M2, 10, 448, 10, 384, 2829.0, 0.8260, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::Float8E5M2, 10, 1344, 10, 1280, 8359.0, 0.9866, VFImplKind::VFIMPL_1D_POST_UPDATE},
};

TEST(A5TcvtFit, fp32_to_fp8_e5m2_2D_POST_UPDATE_full)
{
    for (const auto &c : kCases19) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases20[] = {
    // fp32_to_hif8 1D NO_POST_UPDATE full
    {DType::Float, DType::HFloat8, 1, 64, 1, 64, 67.0, 0.9850, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Float, DType::HFloat8, 1, 320, 1, 320, 118.0, 0.9915, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Float, DType::HFloat8, 1, 640, 1, 640, 178.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
};

TEST(A5TcvtFit, fp32_to_hif8_1D_NO_POST_UPDATE_full)
{
    for (const auto &c : kCases20) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases21[] = {
    // fp32_to_hif8 2D POST_UPDATE full
    {DType::Float, DType::HFloat8, 3, 192, 3, 128, 97.0, 0.1443, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::HFloat8, 3, 448, 3, 384, 890.0, 0.7932, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::HFloat8, 3, 1344, 3, 1280, 2549.0, 0.9992, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::HFloat8, 10, 192, 10, 128, 181.0, 0.0000, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::HFloat8, 10, 448, 10, 384, 2829.0, 0.8260, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float, DType::HFloat8, 10, 1344, 10, 1280, 8359.0, 0.9866, VFImplKind::VFIMPL_1D_POST_UPDATE},
};

TEST(A5TcvtFit, fp32_to_hif8_2D_POST_UPDATE_full)
{
    for (const auto &c : kCases21) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases22[] = {
    // fp4_e1m2_to_bf16 1D NO_POST_UPDATE full
    {DType::Float4E1M2, DType::BFloat16, 1, 256, 1, 256, 91.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Float4E1M2, DType::BFloat16, 1, 1280, 1, 1280, 94.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Float4E1M2, DType::BFloat16, 1, 2560, 1, 2560, 99.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
};

TEST(A5TcvtFit, fp4_e1m2_to_bf16_1D_NO_POST_UPDATE_full)
{
    for (const auto &c : kCases22) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases23[] = {
    // fp4_e1m2_to_bf16 2D POST_UPDATE full
    {DType::Float4E1M2, DType::BFloat16, 3, 320, 3, 256, 75.0, 0.9733, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float4E1M2, DType::BFloat16, 3, 832, 3, 768, 102.0, 0.9509, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float4E1M2, DType::BFloat16, 3, 2624, 3, 2560, 166.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float4E1M2, DType::BFloat16, 10, 320, 10, 256, 103.0, 0.8932, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float4E1M2, DType::BFloat16, 10, 832, 10, 768, 194.0, 0.9278, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float4E1M2, DType::BFloat16, 10, 2624, 10, 2560, 407.0, 0.9901, VFImplKind::VFIMPL_1D_POST_UPDATE},
};

TEST(A5TcvtFit, fp4_e1m2_to_bf16_2D_POST_UPDATE_full)
{
    for (const auto &c : kCases23) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases24[] = {
    // fp4_e2m1_to_bf16 1D NO_POST_UPDATE full
    {DType::Float4E2M1, DType::BFloat16, 1, 256, 1, 256, 91.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Float4E2M1, DType::BFloat16, 1, 1280, 1, 1280, 94.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Float4E2M1, DType::BFloat16, 1, 2560, 1, 2560, 99.0, 1.0000, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
};

TEST(A5TcvtFit, fp4_e2m1_to_bf16_1D_NO_POST_UPDATE_full)
{
    for (const auto &c : kCases24) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases25[] = {
    // fp4_e2m1_to_bf16 2D POST_UPDATE full
    {DType::Float4E2M1, DType::BFloat16, 3, 320, 3, 256, 75.0, 0.9733, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float4E2M1, DType::BFloat16, 3, 832, 3, 768, 102.0, 0.9509, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float4E2M1, DType::BFloat16, 3, 2624, 3, 2560, 166.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float4E2M1, DType::BFloat16, 10, 320, 10, 256, 103.0, 0.8932, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float4E2M1, DType::BFloat16, 10, 832, 10, 768, 194.0, 0.9278, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float4E2M1, DType::BFloat16, 10, 2624, 10, 2560, 407.0, 0.9901, VFImplKind::VFIMPL_1D_POST_UPDATE},
};

TEST(A5TcvtFit, fp4_e2m1_to_bf16_2D_POST_UPDATE_full)
{
    for (const auto &c : kCases25) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases26[] = {
    // fp8_e4m3_to_fp32 1D NO_POST_UPDATE full
    {DType::Float8E4M3, DType::Float, 1, 64, 1, 64, 92.0, 0.9239, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Float8E4M3, DType::Float, 1, 320, 1, 320, 88.0, 0.8522, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Float8E4M3, DType::Float, 1, 640, 1, 640, 126.0, 0.9523, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
};

TEST(A5TcvtFit, fp8_e4m3_to_fp32_1D_NO_POST_UPDATE_full)
{
    for (const auto &c : kCases26) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases27[] = {
    // fp8_e4m3_to_fp32 2D POST_UPDATE full
    {DType::Float8E4M3, DType::Float, 3, 128, 3, 64, 74.0, 0.9324, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float8E4M3, DType::Float, 3, 256, 3, 192, 110.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float8E4M3, DType::Float, 3, 704, 3, 640, 217.0, 0.9907, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float8E4M3, DType::Float, 10, 128, 10, 64, 96.0, 0.9270, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float8E4M3, DType::Float, 10, 256, 10, 192, 216.0, 0.9537, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float8E4M3, DType::Float, 10, 704, 10, 640, 567.0, 0.9982, VFImplKind::VFIMPL_1D_POST_UPDATE},
};

TEST(A5TcvtFit, fp8_e4m3_to_fp32_2D_POST_UPDATE_full)
{
    for (const auto &c : kCases27) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases28[] = {
    // fp8_e5m2_to_fp32 1D NO_POST_UPDATE full
    {DType::Float8E5M2, DType::Float, 1, 64, 1, 64, 92.0, 0.9239, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Float8E5M2, DType::Float, 1, 320, 1, 320, 88.0, 0.8522, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::Float8E5M2, DType::Float, 1, 640, 1, 640, 126.0, 0.9523, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
};

TEST(A5TcvtFit, fp8_e5m2_to_fp32_1D_NO_POST_UPDATE_full)
{
    for (const auto &c : kCases28) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases29[] = {
    // fp8_e5m2_to_fp32 2D POST_UPDATE full
    {DType::Float8E5M2, DType::Float, 3, 128, 3, 64, 74.0, 0.9324, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float8E5M2, DType::Float, 3, 256, 3, 192, 110.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float8E5M2, DType::Float, 3, 704, 3, 640, 217.0, 0.9907, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float8E5M2, DType::Float, 10, 128, 10, 64, 96.0, 0.9270, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float8E5M2, DType::Float, 10, 256, 10, 192, 216.0, 0.9537, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::Float8E5M2, DType::Float, 10, 704, 10, 640, 567.0, 0.9982, VFImplKind::VFIMPL_1D_POST_UPDATE},
};

TEST(A5TcvtFit, fp8_e5m2_to_fp32_2D_POST_UPDATE_full)
{
    for (const auto &c : kCases29) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases30[] = {
    // hif8_to_fp32 1D NO_POST_UPDATE full
    {DType::HFloat8, DType::Float, 1, 64, 1, 64, 92.0, 0.9239, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::HFloat8, DType::Float, 1, 320, 1, 320, 88.0, 0.8522, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
    {DType::HFloat8, DType::Float, 1, 640, 1, 640, 126.0, 0.9523, VFImplKind::VFIMPL_1D_NO_POST_UPDATE},
};

TEST(A5TcvtFit, hif8_to_fp32_1D_NO_POST_UPDATE_full)
{
    for (const auto &c : kCases30) {
        RunTcvt(c);
    }
}

constexpr A5TcvtFitCase kCases31[] = {
    // hif8_to_fp32 2D POST_UPDATE full
    {DType::HFloat8, DType::Float, 3, 128, 3, 64, 74.0, 0.9324, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::HFloat8, DType::Float, 3, 256, 3, 192, 110.0, 1.0000, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::HFloat8, DType::Float, 3, 704, 3, 640, 217.0, 0.9907, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::HFloat8, DType::Float, 10, 128, 10, 64, 96.0, 0.9270, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::HFloat8, DType::Float, 10, 256, 10, 192, 216.0, 0.9537, VFImplKind::VFIMPL_1D_POST_UPDATE},
    {DType::HFloat8, DType::Float, 10, 704, 10, 640, 567.0, 0.9982, VFImplKind::VFIMPL_1D_POST_UPDATE},
};

TEST(A5TcvtFit, hif8_to_fp32_2D_POST_UPDATE_full)
{
    for (const auto &c : kCases31) {
        RunTcvt(c);
    }
}

} // namespace
