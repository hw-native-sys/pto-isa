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

TEST(A5MatrixCompletion, CompletionFormulaAndBoundaries)
{
    CostModelInput input{
        .op = PtoOpcode::TMATMUL_MX, .dtype = DType::Float8E4M3, .rows = 32, .cols = 32, .arch = CostModelArch::A5};
    input.k = 128;
    EXPECT_EQ(EstimateCycles(input).cycles, 49);
    input.k = 65;
    EXPECT_EQ(EstimateCycles(input).cycles, 49);
    input.k = 64;
    EXPECT_EQ(EstimateCycles(input).cycles, 37);
    input.dtype = DType::Float4E2M1;
    input.k = 65;
    EXPECT_EQ(EstimateCycles(input).cycles, 37);
    input.rows = input.cols = 16;
    input.k = 1;
    EXPECT_EQ(EstimateCycles(input).cycles, 28);
    input.op = PtoOpcode::TMATMUL_ACC;
    input.dtype = DType::Float;
    input.k = 64;
    EXPECT_EQ(EstimateCycles(input).cycles, 217);
    input.dtype = DType::Half;
    EXPECT_EQ(EstimateCycles(input).cycles, 37);
    input.dtype = DType::Uint8;
    EXPECT_EQ(EstimateCycles(input).cycles, 0);
    input.dtype = DType::Half;
    input.k = 0;
    EXPECT_EQ(EstimateCycles(input).cycles, 0);
    input.k = 4096;
    EXPECT_EQ(EstimateCycles(input).cycles, 0);
}

TEST(A5MatrixCompletion, AllNewOpcodeRoutesAndMixedPrecision)
{
    for (auto op :
         {PtoOpcode::TGEMV, PtoOpcode::TGEMV_ACC, PtoOpcode::TGEMV_BIAS, PtoOpcode::TGEMV_MX, PtoOpcode::TMATMUL_ACC,
          PtoOpcode::TMATMUL_BIAS, PtoOpcode::TMATMUL_MX}) {
        CostModelInput input{.op = op, .dtype = DType::Float8E4M3, .rows = 32, .cols = 32, .arch = CostModelArch::A5};
        input.k = 128;
        input.dtype2 = DType::Float8E5M2;
        input.has_dtype2 = true;
        const bool gemv = op == PtoOpcode::TGEMV || op == PtoOpcode::TGEMV_ACC || op == PtoOpcode::TGEMV_BIAS ||
                          op == PtoOpcode::TGEMV_MX;
        EXPECT_EQ(EstimateCycles(input).cycles, gemv ? 37 : 49) << PtoOpcodeToString(op);
        input.dtype2 = DType::Half;
        EXPECT_EQ(EstimateCycles(input).cycles, 0);
    }
}

TEST(A5MatrixCompletion, PublicInterfacesPreserveOperands)
{
    using namespace pto;
    auto& ctx = perf_sim::GetCostModelCtx();
    const auto oldArch = ctx.npu_arch;
    ctx.npu_arch = perf_sim::kAscend950Arch;
    using A = Tile<TileType::Left, float8_e4m3_t, 32, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using B = Tile<TileType::Right, float8_e5m2_t, 128, 32, BLayout::RowMajor, -1, -1, SLayout::ColMajor>;
    using C = Tile<TileType::Acc, float, 32, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using V = Tile<TileType::Acc, float, 16, 32, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;
    using S = Tile<TileType::Scaling, uint8_t, 32, 32>;
    using Bias = Tile<TileType::Bias, float, 1, 32>;
    A a(32, 128);
    B b(128, 32);
    C c(32, 32);
    V v(1, 32);
    S sa, sb;
    Bias bias;
    TASSIGN(a, 0);
    TASSIGN(b, 0);
    TASSIGN(c, 0);
    TASSIGN(sa, 0);
    TASSIGN(sb, 0);
    TASSIGN(bias, 0);
    TMATMUL_ACC(c, c, a, b);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 49);
    TMATMUL_ACC(c, a, b);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 49);
    TMATMUL_BIAS(c, a, b, bias);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 49);
    TGEMV(v, a, b);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 37);
    TGEMV_ACC(v, v, a, b);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 37);
    TGEMV_BIAS(v, a, b, bias);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 37);
    TMATMUL_MX(c, a, sa, b, sb);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 49);
    TMATMUL_MX(c, c, a, sa, b, sb);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 49);
    TMATMUL_MX(c, a, sa, b, sb, bias);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 49);
    TGEMV_MX(v, a, sa, b, sb);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 37);
    TGEMV_MX(v, v, a, sa, b, sb);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 37);
    TGEMV_MX(v, a, sa, b, sb, bias);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 37);
    TMATMUL_ACC<AccPhase::Partial>(c, c, a, b);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 49);
    TMATMUL_ACC<AccPhase::Final>(c, a, b);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 49);
    TMATMUL_BIAS<AccPhase::Final>(c, a, b, bias);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 49);
    TGEMV<AccPhase::Final>(v, a, b);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 37);
    TGEMV_ACC<AccPhase::Partial>(v, v, a, b);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 37);
    TGEMV_BIAS<AccPhase::Final>(v, a, b, bias);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 37);
    TMATMUL_MX<AccPhase::Final>(c, a, sa, b, sb);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 49);
    TMATMUL_MX<AccPhase::Partial>(c, c, a, sa, b, sb);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 49);
    TMATMUL_MX<AccPhase::Final>(c, a, sa, b, sb, bias);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 49);
    TGEMV_MX<AccPhase::Final>(v, a, sa, b, sb);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 37);
    TGEMV_MX<AccPhase::Partial>(v, v, a, sa, b, sb);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 37);
    TGEMV_MX<AccPhase::Final>(v, a, sa, b, sb, bias);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 37);
    // MX GEMV reads A's valid K, unlike ordinary GEMV. Do not infer it from C or B.
    A shortA(32, 64);
    EXPECT_EQ(mocker::MatrixFitShape(v, shortA, b, true, true).k, 64);
    EXPECT_EQ(mocker::MatrixFitShape(v, shortA, b, true, false).k, 128);
    ctx.npu_arch = oldArch;
}
} // namespace
