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
#include <pto/pto-inst.hpp>
#include <pto/costmodel/lightweight_costmodel.hpp>

namespace {
using namespace pto::mocker::lightweight;

TEST(A5SequenceFit, PublicApis)
{
    using namespace pto;
    auto& ctx = perf_sim::GetCostModelCtx();
    const auto arch = ctx.npu_arch;
    ctx.npu_arch = perf_sim::kAscend950Arch;
    using I = Tile<TileType::Vec, int32_t, 4, 128, BLayout::RowMajor, -1, -1>;
    using F = Tile<TileType::Vec, float, 4, 128, BLayout::RowMajor, -1, -1>;
    I i0(4, 128), i1(4, 128), id(4, 128);
    F f0(4, 128), f1(4, 128), fd(4, 128), fd1(4, 128);
    TASSIGN(i0, 0);
    TASSIGN(i1, 32768);
    TASSIGN(id, 65536);
    TASSIGN(f0, 0);
    TASSIGN(f1, 32768);
    TASSIGN(fd, 65536);
    TASSIGN(fd1, 98304);
    TSHL(id, i0, i1);
    EXPECT_GT(mocker::GetLastPtoInstrCycles(), 0);
    TSHR(id, i0, i1);
    EXPECT_GT(mocker::GetLastPtoInstrCycles(), 0);
    TSHLS(id, i0, 2);
    EXPECT_GT(mocker::GetLastPtoInstrCycles(), 0);
    TSHRS(id, i0, 2);
    EXPECT_GT(mocker::GetLastPtoInstrCycles(), 0);
    TPARTARGMAX(fd, f0, f1, id, i0, i1);
    EXPECT_GT(mocker::GetLastPtoInstrCycles(), 0);
    TPARTARGMIN(fd, f0, f1, id, i0, i1);
    EXPECT_GT(mocker::GetLastPtoInstrCycles(), 0);
    TINTERLEAVE(fd1, fd, f1, f0);
    EXPECT_GT(mocker::GetLastPtoInstrCycles(), 0);
    TDEINTERLEAVE(fd1, fd, f1, f0);
    EXPECT_GT(mocker::GetLastPtoInstrCycles(), 0);
    using C = Tile<TileType::Vec, int16_t, 1, 128, BLayout::RowMajor, -1, -1>;
    C c(1, 128), tmp(1, 128);
    TASSIGN(c, 0);
    TASSIGN(tmp, 32768);
    TCI<C, int16_t, 0>(c, 1024);
    const auto scalar = mocker::GetLastPtoInstrCycles();
    EXPECT_GT(scalar, 300);
    TCI<C, C, int16_t, 1>(c, 1024, tmp);
    EXPECT_LT(mocker::GetLastPtoInstrCycles(), scalar);
    EXPECT_GT(mocker::GetLastPtoInstrCycles(), 0);
    ctx.npu_arch = arch;
}

TEST(A5SequenceFit, RejectUncalibratedInputs)
{
    CostModelInput input{
        .op = PtoOpcode::TSHL,
        .dtype = DType::Int32,
        .rows = 4,
        .cols = 128,
        .arch = CostModelArch::A5,
        .a5_shape_path = A5ShapePathHint::Path2D,
        .a5_op_params = "dense"};
    EXPECT_GT(EstimateCycles(input).cycles, 0);
    input.dtype = DType::Uint32;
    EXPECT_EQ(EstimateCycles(input).cycles, 0);
    input.dtype = DType::Int32;
    input.valid_cols = 64;
    EXPECT_EQ(EstimateCycles(input).cycles, 0);
    input.valid_cols = 128;
    input.a5_shape_path = A5ShapePathHint::Path1D;
    EXPECT_EQ(EstimateCycles(input).cycles, 0);
    input.op = PtoOpcode::TDEINTERLEAVE;
    input.dtype = DType::Half;
    input.cols = 48;
    input.valid_cols = 48;
    EXPECT_EQ(EstimateCycles(input).cycles, 0);
    input.op = PtoOpcode::TPARTARGMAX;
    input.dtype = DType::Float;
    input.rows = 2;
    input.cols = 64;
    input.valid_cols = 64;
    input.a5_op_params = "dense_index32";
    EXPECT_EQ(EstimateCycles(input).cycles, 0);
    input.op = PtoOpcode::TTRANS;
    EXPECT_EQ(EstimateCycles(input).cycles, 0);
}

struct Sample {
    PtoOpcode op;
    DType dtype;
    int rows;
    int cols;
    const char* params;
    double actual;
};
#include "hardware_samples.hpp"
TEST(A5SequenceFit, MeasuredCycles)
{
    for (const auto& s : kSamples) {
        SCOPED_TRACE(
            std::string(PtoOpcodeToString(s.op)) + "/" + DTypeToString(s.dtype) + "/" + std::to_string(s.rows) + "x" +
            std::to_string(s.cols));
        CostModelInput input{
            .op = s.op,
            .dtype = s.dtype,
            .rows = s.rows,
            .cols = s.cols,
            .arch = CostModelArch::A5,
            .a5_shape_path = s.rows == 1 ? A5ShapePathHint::Path1D : A5ShapePathHint::Path2D,
            .a5_op_params = s.params};
        const auto predicted = EstimateCycles(input).cycles;
        EXPECT_GT(predicted, 0);
        EXPECT_LE(std::abs(predicted - s.actual) / s.actual, 0.10);
    }
}
} // namespace
