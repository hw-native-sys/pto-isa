/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <bit>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include <pto/pto-inst.hpp>
#include <pto/costmodel/lightweight_costmodel.hpp>

#include "a5_cost_check.hpp"

namespace {

using namespace ::pto::mocker::lightweight;

template <int Rows, int Cols>
void CheckDenseHalfFloatPublic(double measuredCycles)
{
    using namespace pto;
    auto& context = perf_sim::GetCostModelCtx();
    const auto saved = context;
    context.npu_arch = perf_sim::kAscend950Arch;
    using S = Tile<TileType::Vec, half, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using D = Tile<TileType::Vec, float, Rows, Cols, BLayout::RowMajor, -1, -1>;
    S source(Rows, Cols);
    D destination(Rows, Cols);
    TCVT(destination, source, RoundMode::CAST_NONE);
    const double cycles = mocker::GetLastPtoInstrCycles();
    EXPECT_GT(cycles, 0);
    EXPECT_NEAR(cycles, measuredCycles, measuredCycles * 0.1);
    context = saved;
}

TEST(A5ComparisonLayoutFit, DenseHalfFloatOriginalEvaluationAccuracy)
{
    // These four reported cases are excluded from the independent training set.
    CheckDenseHalfFloatPublic<128, 128>(914.0004);
    CheckDenseHalfFloatPublic<16, 256>(147.0002);
    CheckDenseHalfFloatPublic<64, 128>(466.0001);
    CheckDenseHalfFloatPublic<64, 64>(402.0001);
}

TEST(A5ComparisonLayoutFit, DenseHalfFloatIndependentShapeAccuracy)
{
    // Frozen after 20 disjoint training shapes; these eight shapes were not
    // inspected until the three-parameter model was selected and frozen.
    // Pure-AIV vector counter hot9 medians, (1024 - 256) iteration difference.
    CheckDenseHalfFloatPublic<40, 48>(258.0006513146004);
    CheckDenseHalfFloatPublic<40, 96>(298.00057111019584);
    CheckDenseHalfFloatPublic<40, 160>(337.9999514747415);
    CheckDenseHalfFloatPublic<40, 224>(338.9988528056696);
    CheckDenseHalfFloatPublic<80, 48>(497.99900061200043);
    CheckDenseHalfFloatPublic<80, 96>(577.9994273693932);
    CheckDenseHalfFloatPublic<80, 160>(658.0000284721427);
    CheckDenseHalfFloatPublic<80, 224>(659.0000626508038);
}

TEST(A5ComparisonLayoutFit, DenseHalfFloatCurveDomain)
{
    using namespace pto;
    CostModelInput input{
        .op = PtoOpcode::TCVT,
        .dtype = DType::Half,
        .rows = 64,
        .cols = 128,
        .arch = CostModelArch::A5,
        .dst_dtype = DType::Float,
        .dst_rows = 64,
        .dst_cols = 128,
        .valid_rows = 64,
        .valid_cols = 128,
        .a5_shape_path = A5ShapePathHint::Path2D};
    auto usesDenseCurve = [](const CostModelInput& value) {
        a5::fit::VfCurveKey key{};
        uint64_t loops = 0, outer = 0, inner = 0, tail = 0, rowInner = 0;
        return a5::BuildVfCurveKeyAndLoopCount(value, key, loops, outer, inner, tail, rowInner) &&
               key.op_params == "CAST_NONE_dense_fp16_fp32";
    };
    EXPECT_TRUE(usesDenseCurve(input));
    auto negative = input;
    negative.dst_cols = 256;
    EXPECT_FALSE(usesDenseCurve(negative));
    negative = input;
    negative.dst_rows = negative.dst_cols = 0;
    EXPECT_FALSE(usesDenseCurve(negative));
    negative = input;
    negative.valid_rows = 32;
    EXPECT_FALSE(usesDenseCurve(negative));
    negative = input;
    negative.a5_shape_path = A5ShapePathHint::Path1D;
    EXPECT_FALSE(usesDenseCurve(negative));
    negative = input;
    negative.round_mode = RoundMode::CAST_RINT;
    EXPECT_FALSE(usesDenseCurve(negative));
    negative = input;
    negative.vf_impl_kind = VFImplKind::VFIMPL_2D_NO_POST_UPDATE;
    EXPECT_FALSE(usesDenseCurve(negative));
    negative = input;
    negative.rows = negative.dst_rows = negative.valid_rows = 256;
    EXPECT_FALSE(usesDenseCurve(negative));
}

template <typename T, int Rows, uint64_t Expected>
void CheckA5OnlyCmpsPublic(double measuredCycles)
{
    using namespace pto;
    auto& context = perf_sim::GetCostModelCtx();
    const auto saved = context;
    context.npu_arch = perf_sim::kAscend950Arch;
    using Src = Tile<TileType::Vec, T, Rows, 32, BLayout::RowMajor, -1, -1>;
    using Dst = Tile<TileType::Vec, uint8_t, Rows, 32, BLayout::RowMajor, -1, -1>;
    Src src(Rows, 32);
    Dst dst(Rows, 32);
    TCMPS(dst, src, static_cast<T>(2), CmpMode::EQ);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), Expected);
    EXPECT_NEAR(static_cast<double>(mocker::GetLastPtoInstrCycles()), measuredCycles, 0.1 * measuredCycles);
    TCMPS(dst, src, static_cast<T>(2), CmpMode::NE);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    TCMPS(dst, src, static_cast<T>(3), CmpMode::EQ);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    Dst partial(Rows - 1, 32);
    TCMPS(partial, src, static_cast<T>(2), CmpMode::EQ);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    context.npu_arch = 2201;
    TCMPS(dst, src, static_cast<T>(2), CmpMode::EQ);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    context = saved;
}

TEST(A5ComparisonLayoutFit, A5OnlyCmpsTypesReachCalibratedBackend)
{
    // 2026-11-18 pure-AIV vector counter differences, 256/1024 iterations,
    // ten full-correctness launches per point. No fitting used these cases.
    CheckA5OnlyCmpsPublic<int8_t, 4092, 24570>(24569.999265377905);
    CheckA5OnlyCmpsPublic<uint8_t, 4092, 24570>(24569.99940546449);
    CheckA5OnlyCmpsPublic<uint32_t, 1636, 11281>(11470.000775407212);
}

TEST(A5ComparisonLayoutFit, StaticAndDynamicFullTilesHaveDifferentBinaryPaths)
{
    using namespace pto;
    auto& context = perf_sim::GetCostModelCtx();
    const auto saved = context;
    context.npu_arch = perf_sim::kAscend950Arch;
    using Static = Tile<TileType::Vec, half, 2728, 16, BLayout::RowMajor, 2728, 16>;
    using Dynamic = Tile<TileType::Vec, half, 2728, 16, BLayout::RowMajor, -1, -1>;
    Static staticDst, staticSrc0, staticSrc1;
    Dynamic dynamicDst(2728, 16), dynamicSrc0(2728, 16), dynamicSrc1(2728, 16);
    mocker::CostModelTileInfo staticInfo, dynamicInfo;
    EXPECT_TRUE(mocker::TryCostModelTileInfo(staticInfo, staticDst));
    EXPECT_TRUE(mocker::TryCostModelTileInfo(dynamicInfo, dynamicDst));
    EXPECT_EQ(staticInfo.a5_shape_path, A5ShapePathHint::Path1D);
    EXPECT_EQ(dynamicInfo.a5_shape_path, A5ShapePathHint::Path2D);

    // TBinOp.hpp dispatches on compile-time ValidCol, not runtime fullness.
    // These are formula/dispatch regressions, not new hardware measurements.
    TADD(staticDst, staticSrc0, staticSrc1);
    const auto staticCycles = mocker::GetLastPtoInstrCycles();
    TADD(dynamicDst, dynamicSrc0, dynamicSrc1);
    const auto dynamicCycles = mocker::GetLastPtoInstrCycles();
    EXPECT_EQ(staticCycles, 480);
    EXPECT_EQ(dynamicCycles, 16386);

    CostModelInput input{
        .op = PtoOpcode::TADD,
        .dtype = DType::Half,
        .rows = 2728,
        .cols = 16,
        .arch = CostModelArch::A5,
        .valid_rows = 2728,
        .valid_cols = 16,
    };
    CostModelResult result{};
    EXPECT_TRUE(EstimateCycles(input, result));
    EXPECT_EQ(result.cycles, staticCycles);
    input.a5_shape_path = dynamicInfo.a5_shape_path;
    EXPECT_TRUE(EstimateCycles(input, result));
    EXPECT_EQ(result.cycles, dynamicCycles);
    context = saved;
}

TEST(A5ComparisonLayoutFit, QuantVectorIndependentAddresses)
{
    using namespace pto;
    auto& context = perf_sim::GetCostModelCtx();
    const auto saved = context;
    context.npu_arch = perf_sim::kAscend950Arch;
    using Scaling = Tile<TileType::Scaling, uint64_t, 1, 128, BLayout::RowMajor, 1, 128>;
    Scaling fp;
    // Frozen independent addresses, not training addresses.
    for (auto address : {1536, 2560}) {
        TASSIGN(fp, address);
        SET_QUANT_VECTOR(fp);
        EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 4);
        EXPECT_NEAR(mocker::GetLastPtoInstrCycles(), 3.9997340927985747, 0.0003);
    }
    for (auto address : {512, 1025, 1537, 3073, 3200, 4096}) {
        TASSIGN(fp, address);
        SET_QUANT_VECTOR(fp);
        EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    }
    Scaling unassigned;
    SET_QUANT_VECTOR(unassigned);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    TASSIGN(fp, 0);
    SET_QUANT_VECTOR(fp);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    Tile<TileType::Scaling, int64_t, 1, 128, BLayout::RowMajor, 1, 128> wrongDtype;
    TASSIGN(wrongDtype, 1536);
    SET_QUANT_VECTOR(wrongDtype);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    Tile<TileType::Vec, uint64_t, 1, 128, BLayout::RowMajor, 1, 128> wrongLocation;
    TASSIGN(wrongLocation, 1536);
    SET_QUANT_VECTOR(wrongLocation);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    Tile<TileType::Scaling, uint64_t, 1, 64, BLayout::RowMajor, 1, 64> wrongShape;
    TASSIGN(wrongShape, 1536);
    SET_QUANT_VECTOR(wrongShape);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    Tile<TileType::Scaling, uint64_t, 1, 128, BLayout::RowMajor, -1, -1> partial(1, 64);
    TASSIGN(partial, 1536);
    SET_QUANT_VECTOR(partial);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    context = saved;
}

TEST(A5ComparisonLayoutFit, QuantScalarIndependentInputs)
{
    auto& context = pto::perf_sim::GetCostModelCtx();
    const auto saved = context;
    context.npu_arch = pto::perf_sim::kAscend950Arch;
    for (float scale : {1.5f, 2.5f, 4.0f, 5.0f}) {
        pto::SET_QUANT_SCALAR<half>(scale);
        EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 4);
        EXPECT_NEAR(pto::mocker::GetLastPtoInstrCycles(), 4.000191978430365, 0.0004);
    }
    for (float scale :
         {0.0f, -1.0f, 0.49f, 7.01f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        pto::SET_QUANT_SCALAR<half>(scale);
        EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 0);
    }
    pto::SET_QUANT_SCALAR<int8_t>(1.0f);
    EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 0);
    pto::SET_QUANT_SCALAR<float>(1.0f);
    EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 0);
    context = saved;
}

template <pto::SetFmatrixMode Mode>
void CheckImageConfigPublic()
{
    using namespace pto;
    using Source = ConvTile<TileType::Mat, half, 512, Layout::NC1HWC0, ConvTileShape<1, 2, 1, 16, 16>>;
    Source src;
    src.SetFmapH(1);
    src.SetFmapW(16);
    src.SetChannelSize(32);
    src.SetDstStride(1);
    // Frozen independent configuration values; both members of each value pair.
    for (int value : {24, 25, 48, 49}) {
        src.SetFmapW(value);
        SETFMATRIX<Source, Mode>(src);
        EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 2);
    }
    src.SetFmapW(66);
    SETFMATRIX<Source, Mode>(src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    src.SetFmapW(16);
    for (int value : {3, 4, 7, 8}) {
        src.SetDstStride(value);
        SET_IMG2COL_RPT<Source, Mode>(src);
        EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 2);
    }
    src.SetDstStride(11);
    SET_IMG2COL_RPT<Source, Mode>(src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    src.SetDstStride(1);
    for (int value : {2, 3, 6, 7}) {
        src.SetPadValue(static_cast<half>(value));
        SET_IMG2COL_PADDING<Source, Mode>(src);
        EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 2);
    }
    src.SetPadValue(static_cast<half>(0.5f));
    SET_IMG2COL_PADDING<Source, Mode>(src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    src.SetPadValue(static_cast<half>(0));
    src.SetFilterW(2);
    SETFMATRIX<Source, Mode>(src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    src.SetFilterW(1);
    SETFMATRIX<Source, SetFmatrixMode::FMATRIX_B_AUTO>(src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    SET_IMG2COL_RPT<Source, SetFmatrixMode::FMATRIX_B_AUTO>(src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    SET_IMG2COL_PADDING<Source, SetFmatrixMode::FMATRIX_B_AUTO>(src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
}

TEST(A5ComparisonLayoutFit, ImageConfigIndependentInputs)
{
    auto& context = pto::perf_sim::GetCostModelCtx();
    const auto saved = context;
    context.npu_arch = pto::perf_sim::kAscend950Arch;
    CheckImageConfigPublic<pto::SetFmatrixMode::FMATRIX_A_MANUAL>();
    CheckImageConfigPublic<pto::SetFmatrixMode::FMATRIX_B_MANUAL>();
    context = saved;
}

TEST(A5ComparisonLayoutFit, ImageConfigPreservesA2A3IntrinsicCalls)
{
    using namespace pto;
    auto& context = perf_sim::GetCostModelCtx();
    const auto saved = context;
    context.npu_arch = 2201;
    using Source = ConvTile<TileType::Mat, half, 1024, Layout::NC1HWC0, ConvTileShape<1, 2, 1, 16, 16>>;
    Source src;
    src.SetRepeatStride(7);
    src.SetRepeatTime(3);
    src.SetRepeatMode(1);
    src.SetPadValue(static_cast<half>(2));

    SET_IMG2COL_RPT<Source, SetFmatrixMode::FMATRIX_B_MANUAL>(src);
    auto calls = mocker::GetTrace().executed_pto.back().cce_calls;
    ASSERT_EQ(calls.size(), 1U);
    EXPECT_EQ(calls[0].name, "set_l3d_rpt");
    ASSERT_EQ(calls[0].args.size(), 1U);
    EXPECT_EQ(calls[0].args[0], 7U | (3U << 16) | (1U << 24));

    SET_IMG2COL_PADDING<Source, SetFmatrixMode::FMATRIX_B_MANUAL>(src);
    calls = mocker::GetTrace().executed_pto.back().cce_calls;
    ASSERT_EQ(calls.size(), 1U);
    EXPECT_EQ(calls[0].name, "set_padding");
    ASSERT_EQ(calls[0].args.size(), 1U);
    EXPECT_EQ(calls[0].args[0], std::bit_cast<uint16_t>(static_cast<half>(2)));

    SET_IMG2COL_RPT<Source, SetFmatrixMode::FMATRIX_B_AUTO>(src);
    EXPECT_TRUE(mocker::GetTrace().executed_pto.back().cce_calls.empty());
    SET_IMG2COL_PADDING<Source, SetFmatrixMode::FMATRIX_B_AUTO>(src);
    EXPECT_TRUE(mocker::GetTrace().executed_pto.back().cce_calls.empty());
    context = saved;
}

TEST(A5ComparisonLayoutFit, FpSlicesPreserveA2A3IntrinsicCalls)
{
    using namespace pto;
    auto& context = perf_sim::GetCostModelCtx();
    const auto saved = context;
    context.npu_arch = 2201;
    using Source = Tile<TileType::Acc, int32_t, 16, 32, BLayout::ColMajor, 16, 32, SLayout::RowMajor, 1024>;
    using Destination = Tile<TileType::Mat, half, 16, 32, BLayout::ColMajor, 16, 32, SLayout::RowMajor>;
    using Scaling = Tile<TileType::Scaling, uint64_t, 1, 32, BLayout::RowMajor, 1, 32>;
    Source src;
    Destination dst;
    Scaling scale;
    auto checkCalls = [] {
        const auto& calls = mocker::GetTrace().executed_pto.back().cce_calls;
        ASSERT_EQ(calls.size(), 2U);
        EXPECT_EQ(calls[0].name, "set_fpc");
        EXPECT_EQ(calls[1].name, "copy_matrix_cc_to_cbuf");
        EXPECT_GT(mocker::GetLastPtoInstrCycles(), 0);
    };
    TEXTRACT_FP(dst, src, scale, 0, 0);
    checkCalls();
    TINSERT_FP(dst, src, scale, 0, 0);
    checkCalls();
    // These Mat inputs have no A5 fit and must not execute A2/A3 intrinsics
    // merely because the formula test harness compiles against A2/A3 stubs.
    for (auto arch : {perf_sim::kAscend950Arch, perf_sim::kAscend950UrmaArch}) {
        context.npu_arch = arch;
        TEXTRACT_FP(dst, src, scale, 0, 0);
        EXPECT_TRUE(mocker::GetTrace().executed_pto.back().cce_calls.empty());
        EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
        TINSERT_FP(dst, src, scale, 0, 0);
        EXPECT_TRUE(mocker::GetTrace().executed_pto.back().cce_calls.empty());
        EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    }
    context = saved;
}

template <int Rows, int Cols>
void CheckImagePublicPath(double measured)
{
    using namespace pto;
    using Source =
        ConvTile<TileType::Mat, half, Rows * Cols * 2, Layout::NC1HWC0, ConvTileShape<1, Cols / 16, 1, Rows, 16>>;
    using Destination = Tile<TileType::Left, half, Rows, Cols, BLayout::ColMajor, Rows, Cols, SLayout::RowMajor>;
    Source src;
    Destination dst;
    std::vector<half> source(Rows * Cols), destination(Rows * Cols);
    TASSIGN(src, reinterpret_cast<uintptr_t>(source.data()));
    TASSIGN(dst, reinterpret_cast<uintptr_t>(destination.data()));
    src.SetFmapH(1);
    src.SetFmapW(Rows);
    src.SetChannelSize(Cols);
    src.SetDstStride(Rows / 16);
    mocker::ImageManualFitState() = {};
    TIMG2COL<Destination, Source, SetFmatrixMode::FMATRIX_B_MANUAL>(dst, src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    SETFMATRIX<Source, SetFmatrixMode::FMATRIX_B_MANUAL>(src);
    SET_IMG2COL_RPT<Source, SetFmatrixMode::FMATRIX_B_MANUAL>(src);
    SET_IMG2COL_PADDING<Source, SetFmatrixMode::FMATRIX_B_MANUAL>(src);
    TIMG2COL<Destination, Source, SetFmatrixMode::FMATRIX_B_MANUAL>(dst, src);
    EXPECT_LE(std::abs(mocker::GetLastPtoInstrCycles() - measured) / measured, 0.10);
    TIMG2COL<Destination, Source, SetFmatrixMode::FMATRIX_A_MANUAL>(dst, src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    TIMG2COL<Destination, Source, SetFmatrixMode::FMATRIX_B_MANUAL>(dst, src, 1, 0);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    src.SetFilterW(2);
    TIMG2COL<Destination, Source, SetFmatrixMode::FMATRIX_B_MANUAL>(dst, src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    src.SetFilterW(1);
    src.SetPadList(0, 1);
    SETFMATRIX<Source, SetFmatrixMode::FMATRIX_B_MANUAL>(src);
    TIMG2COL<Destination, Source, SetFmatrixMode::FMATRIX_B_MANUAL>(dst, src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    src.SetPadList(0, 0);
    SETFMATRIX<Source, SetFmatrixMode::FMATRIX_B_MANUAL>(src);
    TIMG2COL<Destination, Source, SetFmatrixMode::FMATRIX_B_AUTO>(dst, src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    TIMG2COL<Destination, Source, SetFmatrixMode::FMATRIX_B_MANUAL>(dst, src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    const auto saved = perf_sim::GetCostModelCtx();
    perf_sim::GetCostModelCtx() = {};
    EXPECT_EQ(perf_sim::GetCostModelCtx().imageManual.fmapH, -1);
    perf_sim::GetCostModelCtx() = saved;
    mocker::ImageManualFitState() = {};
}

TEST(A5ComparisonLayoutFit, ImagePublicIndependentShapes)
{
    auto& context = pto::perf_sim::GetCostModelCtx();
    const auto oldArch = context.npu_arch;
    context.npu_arch = pto::perf_sim::kAscend950Arch;
    CheckImagePublicPath<48, 32>(53.999518);
    CheckImagePublicPath<48, 96>(78.000235);
    CheckImagePublicPath<80, 32>(62.000584);
    CheckImagePublicPath<80, 96>(101.999979);
    context.npu_arch = oldArch;
}

TEST(A5ComparisonLayoutFit, ImageInternalIndependentShapes)
{
    const std::array<std::array<double, 3>, 4> points{
        {{48, 32, 53.999518}, {48, 96, 78.000235}, {80, 32, 62.000584}, {80, 96, 101.999979}}};
    for (const auto& point : points) {
        CostModelInput input{};
        input.arch = CostModelArch::A5;
        input.op = PtoOpcode::TIMG2COL;
        input.dtype = DType::Half;
        input.rows = input.valid_rows = point[0];
        input.cols = input.valid_cols = point[1];
        input.a5_op_params = "image_left_manual_1x1";
        CostModelResult result{};
        ASSERT_TRUE(EstimateCycles(input, result));
        EXPECT_NEAR(result.cycles, point[2], point[2] * .1);
        input.a5_op_params = {};
        EXPECT_FALSE(EstimateCycles(input, result));
        input.a5_op_params = "image_left_manual_1x1";
        input.dtype = DType::Float;
        EXPECT_FALSE(EstimateCycles(input, result));
    }
}

template <int Rows, int Cols, bool Insert>
void CheckFpPublicPath(double measured)
{
    using namespace pto;
    constexpr int sourceCols = Cols + (Insert ? 0 : 64);
    using Source = Tile<
        TileType::Acc, float, (Rows + 15) / 16 * 16, sourceCols, BLayout::ColMajor, Rows, sourceCols, SLayout::RowMajor,
        1024>;
    using Destination = Tile<
        TileType::Vec, half, Rows + 1, Cols + 96, BLayout::RowMajor, Insert ? Rows + 1 : Rows,
        Insert ? Cols + 96 : Cols>;
    using Scaling = Tile<TileType::Scaling, uint64_t, 1, sourceCols, BLayout::RowMajor, 1, sourceCols>;
    Source src;
    Destination dst;
    Scaling scale;
    std::vector<float> srcBuffer(Source::Numel, 32);
    std::vector<half> dstBuffer(Destination::Numel);
    std::vector<uint64_t> scaleBuffer(Scaling::Numel, 0x3f800000ULL);
    TASSIGN(src, reinterpret_cast<uintptr_t>(srcBuffer.data()));
    TASSIGN(dst, reinterpret_cast<uintptr_t>(dstBuffer.data()));
    TASSIGN(scale, reinterpret_cast<uintptr_t>(scaleBuffer.data()));
    if constexpr (Insert) {
        TINSERT_FP<Destination, Source, Scaling, ReluPreMode::NormalRelu>(dst, src, scale, 0, 64);
    } else {
        TEXTRACT_FP<Destination, Source, Scaling, ReluPreMode::NormalRelu>(dst, src, scale, 0, 64);
    }
    const auto cycles = pto::mocker::GetLastPtoInstrCycles();
    EXPECT_GT(cycles, 0);
    EXPECT_LE(std::abs(static_cast<double>(cycles) - measured) / measured, 0.10);
    // Wrong offsets and unmeasured ReLU modes must not fall through to the Vec slice curve.
    if constexpr (Insert) {
        TINSERT_FP<Destination, Source, Scaling, ReluPreMode::NormalRelu>(dst, src, scale, 0, 0);
    } else {
        TEXTRACT_FP<Destination, Source, Scaling, ReluPreMode::NormalRelu>(dst, src, scale, 0, 0);
    }
    EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 0);
    if constexpr (Insert) {
        TINSERT_FP(dst, src, scale, 0, 64);
    } else {
        TEXTRACT_FP(dst, src, scale, 0, 64);
    }
    EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 0);
}

TEST(A5ComparisonLayoutFit, FpPublicIndependentShapes)
{
    auto& context = pto::perf_sim::GetCostModelCtx();
    const auto oldArch = context.npu_arch;
    context.npu_arch = pto::perf_sim::kAscend950Arch;
    CheckFpPublicPath<24, 96, false>(102.99945);
    CheckFpPublicPath<24, 192, false>(148.99932);
    CheckFpPublicPath<40, 160, false>(213.00005);
    CheckFpPublicPath<40, 224, false>(216.99921);
    CheckFpPublicPath<56, 96, false>(167.00012);
    CheckFpPublicPath<56, 224, false>(280.99959);
    CheckFpPublicPath<24, 96, true>(103.00037);
    CheckFpPublicPath<24, 192, true>(148.99949);
    CheckFpPublicPath<40, 160, true>(212.99984);
    CheckFpPublicPath<40, 224, true>(216.99993);
    CheckFpPublicPath<56, 96, true>(166.99954);
    CheckFpPublicPath<56, 224, true>(280.99962);
    context.npu_arch = oldArch;
}

TEST(A5ComparisonLayoutFit, HistogramPublicApiAndGeometry)
{
    using namespace pto;
    auto& context = pto::perf_sim::GetCostModelCtx();
    const auto oldArch = context.npu_arch;
    context.npu_arch = pto::perf_sim::kAscend950Arch;
    using Data = Tile<TileType::Vec, uint32_t, 3, 512, BLayout::RowMajor, -1, -1>;
    using Bins = Tile<TileType::Vec, uint32_t, 3, 256, BLayout::RowMajor, -1, -1>;
    using Index = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;
    Data src(3, 320);
    Bins dst(3, 256);
    Index idx(1, 32);
    std::vector<uint32_t> source(Data::Numel), destination(Bins::Numel);
    std::vector<uint8_t> indices(Index::Numel);
    TASSIGN(src, reinterpret_cast<uintptr_t>(source.data()));
    TASSIGN(dst, reinterpret_cast<uintptr_t>(destination.data()));
    TASSIGN(idx, reinterpret_cast<uintptr_t>(indices.data()));
    THISTOGRAM<HistByte::BYTE_3>(dst, src, idx);
    EXPECT_NEAR(pto::mocker::GetLastPtoInstrCycles(), 128, 12.8);
    THISTOGRAM<HistByte::BYTE_2>(dst, src, idx);
    EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 0);
    Bins incompleteBins(3, 128);
    TASSIGN(incompleteBins, reinterpret_cast<uintptr_t>(destination.data()));
    THISTOGRAM<HistByte::BYTE_3>(incompleteBins, src, idx);
    EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 0);
    context.npu_arch = oldArch;
}

TEST(A5ComparisonLayoutFit, HistogramIndependentHardwareShapes)
{
    // Constant and 256-bin ramp inputs produced the same counters at these shapes.
    const std::array<std::array<double, 3>, 8> points{
        {{3, 96, 90.5},
         {7, 192, 153},
         {9, 160, 185.5},
         {2, 224, 81},
         {3, 320, 128},
         {5, 576, 237},
         {7, 896, 392},
         {6, 448, 208}}};
    for (const auto& p : points) {
        CostModelInput input{};
        input.arch = CostModelArch::A5;
        input.op = PtoOpcode::THISTOGRAM;
        input.dtype = DType::Uint32;
        input.rows = p[0];
        input.cols = ((static_cast<int64_t>(p[1]) + 255) / 256) * 256;
        input.valid_rows = p[0];
        input.valid_cols = p[1];
        input.a5_op_params = "u32_byte3_padded256";
        CostModelResult result{};
        ASSERT_TRUE(EstimateCycles(input, result));
        EXPECT_NEAR(result.cycles, p[2], p[2] * .1);
        input.a5_op_params = {};
        EXPECT_FALSE(EstimateCycles(input, result));
    }
}

TEST(A5ComparisonLayoutFit, QuantPublicApi)
{
    using namespace pto;
    auto& context = pto::perf_sim::GetCostModelCtx();
    const auto oldArch = context.npu_arch;
    context.npu_arch = pto::perf_sim::kAscend950Arch;
    using Src = Tile<TileType::Vec, float, 3, 96, BLayout::RowMajor, -1, -1>;
    using Dst = Tile<TileType::Vec, int8_t, 3, 96, BLayout::RowMajor, -1, -1>;
    using UnsignedDst = Tile<TileType::Vec, uint8_t, 3, 96, BLayout::RowMajor, -1, -1>;
    using Para = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, -1, -1>;
    Src src(3, 96);
    Dst dst(3, 96);
    UnsignedDst unsignedDst(3, 96);
    Para scale(1, 8), offset(1, 8);
    std::vector<float> srcBuffer(Src::Numel), scaleBuffer(8), offsetBuffer(8);
    std::vector<int8_t> dstBuffer(Dst::Numel);
    std::vector<uint8_t> unsignedBuffer(UnsignedDst::Numel);
    TASSIGN(src, reinterpret_cast<uintptr_t>(srcBuffer.data()));
    TASSIGN(dst, reinterpret_cast<uintptr_t>(dstBuffer.data()));
    TASSIGN(unsignedDst, reinterpret_cast<uintptr_t>(unsignedBuffer.data()));
    TASSIGN(scale, reinterpret_cast<uintptr_t>(scaleBuffer.data()));
    TASSIGN(offset, reinterpret_cast<uintptr_t>(offsetBuffer.data()));
    TQUANT<QuantType::INT8_SYM>(dst, src, scale);
    EXPECT_NEAR(pto::mocker::GetLastPtoInstrCycles(), 70, 7);
    TQUANT<QuantType::INT8_ASYM>(unsignedDst, src, scale, &offset);
    EXPECT_NEAR(pto::mocker::GetLastPtoInstrCycles(), 72, 7.2);
    TQUANT<QuantType::INT8_ASYM>(unsignedDst, src, scale);
    EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 0);
    context.npu_arch = oldArch;
}

TEST(A5ComparisonLayoutFit, QuantIndependentHardwareShapes)
{
    const std::array<std::array<double, 4>, 8> points{
        {{3, 96, 70, 72},
         {7, 192, 129, 141},
         {9, 160, 153, 163},
         {2, 224, 77, 78},
         {3, 320, 105, 112},
         {5, 576, 225, 242},
         {7, 896, 437, 469},
         {6, 448, 213, 231}}};
    for (bool asymmetric : {false, true}) {
        for (const auto& p : points) {
            CostModelInput input{};
            input.arch = CostModelArch::A5;
            input.op = PtoOpcode::TQUANT;
            input.dtype = asymmetric ? DType::Uint8 : DType::Int8;
            input.rows = p[0];
            input.cols = p[1];
            input.a5_op_params = asymmetric ? "dense_fp32_int8_asym" : "dense_fp32_int8_sym";
            CostModelResult result{};
            ASSERT_TRUE(EstimateCycles(input, result));
            const auto measured = p[asymmetric ? 3 : 2];
            EXPECT_NEAR(result.cycles, measured, measured * .1);
            input.a5_op_params = {};
            EXPECT_FALSE(EstimateCycles(input, result));
        }
    }
}

TEST(A5ComparisonLayoutFit, SlicePublicApiGeometry)
{
    using namespace pto;
    auto& context = pto::perf_sim::GetCostModelCtx();
    const auto oldArch = context.npu_arch;
    context.npu_arch = pto::perf_sim::kAscend950Arch;
    using Small = Tile<TileType::Vec, float, 3, 96, BLayout::RowMajor, -1, -1>;
    using Large = Tile<TileType::Vec, float, 4, 104, BLayout::RowMajor, -1, -1>;
    Small small(3, 96);
    Large large(4, 104);
    std::vector<float> smallBuffer(Small::Numel), largeBuffer(Large::Numel);
    TASSIGN(small, reinterpret_cast<uintptr_t>(smallBuffer.data()));
    TASSIGN(large, reinterpret_cast<uintptr_t>(largeBuffer.data()));
    TEXTRACT(small, large, 1, 8);
    EXPECT_GT(pto::mocker::GetLastPtoInstrCycles(), 0);
    TINSERT(large, small, 1, 8);
    EXPECT_GT(pto::mocker::GetLastPtoInstrCycles(), 0);
    TEXTRACT(small, large, 0, 8);
    EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 0);
    TINSERT(large, small, 1, 0);
    EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 0);
    using StaticSmall = Tile<TileType::Vec, float, 3, 96, BLayout::RowMajor, 3, 96>;
    StaticSmall dmaSmall;
    TASSIGN(dmaSmall, reinterpret_cast<uintptr_t>(smallBuffer.data()));
    TEXTRACT(dmaSmall, large, 1, 8);
    EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 0);
    context.npu_arch = oldArch;
}

TEST(A5ComparisonLayoutFit, SliceIndependentHardwareShapes)
{
    // rows, cols, extract-fp16, extract-b32, insert-fp16, insert-b32.
    const std::array<std::array<double, 6>, 8> points{
        {{3, 96, 36, 39, 36, 39},
         {7, 192, 61, 68, 61, 68},
         {9, 160, 73, 82, 73, 82},
         {2, 224, 37, 41, 37, 41},
         {3, 320, 42, 48, 42, 48},
         {5, 576, 60, 90, 60, 88},
         {7, 896, 93, 161, 95.1, 168.1},
         {6, 448, 61, 85, 61, 84}}};
    for (const auto dtype : {DType::Half, DType::Float, DType::Int32}) {
        for (const auto op : {PtoOpcode::TEXTRACT, PtoOpcode::TINSERT}) {
            for (const auto& p : points) {
                const bool insert = op == PtoOpcode::TINSERT;
                CostModelInput input{};
                input.arch = CostModelArch::A5;
                input.op = op;
                input.dtype = dtype;
                input.rows = p[0] + (insert ? 1 : 0);
                input.cols = p[1] + (insert ? (dtype == DType::Half ? 16 : 8) : 0);
                input.a5_op_params = "dynamic_slice_offset1_32B";
                CostModelResult result{};
                ASSERT_TRUE(EstimateCycles(input, result));
                const double measured = p[(insert ? 4 : 2) + (dtype == DType::Half ? 0 : 1)];
                EXPECT_NEAR(result.cycles, measured, measured * .1);
                input.a5_op_params = {};
                EXPECT_FALSE(EstimateCycles(input, result));
            }
        }
    }
}

TEST(A5ComparisonLayoutFit, RandomPublicApiRoundsAndGeometry)
{
    using namespace pto;
    auto& context = pto::perf_sim::GetCostModelCtx();
    const auto oldArch = context.npu_arch;
    context.npu_arch = pto::perf_sim::kAscend950Arch;
    using Data = Tile<TileType::Vec, uint32_t, 3, 320, BLayout::RowMajor, -1, -1>;
    Data dst(3, 320);
    std::vector<uint32_t> storage(Data::Numel);
    TASSIGN(dst, reinterpret_cast<uintptr_t>(storage.data()));
    TRandomKey key{123, 456};
    TRandomCounter counter{1, 2, 3, 4};
    TRANDOM<7>(dst, key, counter);
    const auto seven = pto::mocker::GetLastPtoInstrCycles();
    EXPECT_GT(seven, 0);
    TRANDOM(dst, key, counter);
    EXPECT_GT(pto::mocker::GetLastPtoInstrCycles(), seven);
    Data padded(3, 288);
    TASSIGN(padded, reinterpret_cast<uintptr_t>(storage.data()));
    TRANDOM(padded, key, counter);
    EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 0);
    context.npu_arch = oldArch;
}

TEST(A5ComparisonLayoutFit, RandomIndependentShapes)
{
    const std::array<std::array<double, 4>, 8> profiles{
        {{3, 96, 282, 379},
         {7, 192, 587.5, 795},
         {9, 160, 736, 1003},
         {2, 224, 214.5, 272.5},
         {3, 320, 515.5, 685},
         {5, 576, 1186.5, 1622.5},
         {7, 896, 2155.5, 2966},
         {6, 448, 960, 1308}}};
    for (const auto dtype : {DType::Int32, DType::Uint32}) {
        for (const auto& point : profiles) {
            for (const bool seven : {true, false}) {
                CostModelInput input{};
                input.arch = CostModelArch::A5;
                input.op = PtoOpcode::TRANDOM;
                input.dtype = dtype;
                input.rows = point[0];
                input.cols = point[1];
                input.a5_op_params = seven ? "dense_rounds7" : "dense_rounds10";
                CostModelResult result{};
                ASSERT_TRUE(EstimateCycles(input, result));
                const auto measured = point[seven ? 2 : 3];
                EXPECT_NEAR(result.cycles, measured, measured * .1);
                input.a5_op_params = "dense_rounds8";
                EXPECT_FALSE(EstimateCycles(input, result));
            }
        }
    }
}

TEST(A5ComparisonLayoutFit, InplacePaddingPublicApiAndDomain)
{
    using namespace pto;
    auto& context = pto::perf_sim::GetCostModelCtx();
    const auto oldArch = context.npu_arch;
    context.npu_arch = pto::perf_sim::kAscend950Arch;
    using Data = Tile<TileType::Vec, float, 4, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    Data tile(3, 120);
    std::vector<float> storage(Data::Numel);
    TASSIGN(tile, reinterpret_cast<uintptr_t>(storage.data()));
    TFILLPAD_INPLACE(tile, tile);
    EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 40);
    TFILLPAD<TFillPadMode::InPlace>(tile, tile);
    EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 40);
    Data unsupported(3, 112);
    TASSIGN(unsupported, reinterpret_cast<uintptr_t>(storage.data()));
    TFILLPAD_INPLACE(unsupported, unsupported);
    EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 0);
    context.npu_arch = oldArch;
}

TEST(A5ComparisonLayoutFit, InplacePaddingIndependentHoldout)
{
    for (auto dtype : {DType::Half, DType::Float, DType::Int32}) {
        for (const auto& point :
             std::vector<std::array<double, 3>>{{3, 96, 37}, {7, 192, 45}, {9, 160, 49}, {2, 224, 35}}) {
            CostModelInput input{};
            input.arch = CostModelArch::A5;
            input.op = PtoOpcode::TFILLPAD_INPLACE;
            input.dtype = dtype;
            input.rows = point[0];
            input.cols = point[1];
            input.valid_rows = input.rows - 1;
            input.a5_op_params = "suffix32B_no_bottom";
            CostModelResult result{};
            ASSERT_TRUE(EstimateCycles(input, result));
            EXPECT_NEAR(result.cycles, point[2], point[2] * .1);
            input.a5_op_params = {};
            EXPECT_FALSE(EstimateCycles(input, result));
        }
    }
}

template <pto::MaskPattern Pattern, bool Column>
void CheckPatternPublicCalls()
{
    using namespace pto;
    constexpr int factor = Pattern == MaskPattern::P1111                                    ? 1 :
                           (Pattern == MaskPattern::P0101 || Pattern == MaskPattern::P1010) ? 2 :
                                                                                              4;
    using Small = Tile<TileType::Vec, float, 4, 128, BLayout::RowMajor, 4, 128>;
    using Large = Tile<
        TileType::Vec, float, Column ? 4 * factor : 4, Column ? 128 : 128 * factor, BLayout::RowMajor,
        Column ? 4 * factor : 4, Column ? 128 : 128 * factor>;
    Small small;
    Large large;
    std::vector<float> smallBuffer(Small::Numel);
    std::vector<float> largeBuffer(Large::Numel);
    TASSIGN(small, reinterpret_cast<uintptr_t>(smallBuffer.data()));
    TASSIGN(large, reinterpret_cast<uintptr_t>(largeBuffer.data()));
    constexpr auto scatterAxis = Column ? ScatterAxis::SCATTER_COL : ScatterAxis::SCATTER_ROW;
    constexpr auto gatherAxis = Column ? GatherAxis::GATHER_COL : GatherAxis::GATHER_ROW;
    TSCATTER<Pattern, scatterAxis>(large, small);
    if constexpr (
        Pattern == MaskPattern::P1111 ||
        (!Column && (Pattern == MaskPattern::P0001 || Pattern == MaskPattern::P0101))) {
        EXPECT_GT(pto::mocker::GetLastPtoInstrCycles(), 0);
    } else {
        // Other masks and the column axis still have no calibrated fp32 curve.
        EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 0);
    }
    TGATHER<Small, Large, Pattern, gatherAxis>(small, large);
    EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 0);
}

TEST(A5ComparisonLayoutFit, PatternPublicApiMatchesDeviceSignatures)
{
    auto& context = pto::perf_sim::GetCostModelCtx();
    const auto previousArch = context.npu_arch;
    context.npu_arch = pto::perf_sim::kAscend950Arch;
    CheckPatternPublicCalls<pto::MaskPattern::P0101, false>();
    CheckPatternPublicCalls<pto::MaskPattern::P0101, true>();
    CheckPatternPublicCalls<pto::MaskPattern::P1010, false>();
    CheckPatternPublicCalls<pto::MaskPattern::P1010, true>();
    CheckPatternPublicCalls<pto::MaskPattern::P0001, false>();
    CheckPatternPublicCalls<pto::MaskPattern::P0001, true>();
    CheckPatternPublicCalls<pto::MaskPattern::P0010, false>();
    CheckPatternPublicCalls<pto::MaskPattern::P0010, true>();
    CheckPatternPublicCalls<pto::MaskPattern::P0100, false>();
    CheckPatternPublicCalls<pto::MaskPattern::P0100, true>();
    CheckPatternPublicCalls<pto::MaskPattern::P1000, false>();
    CheckPatternPublicCalls<pto::MaskPattern::P1000, true>();
    CheckPatternPublicCalls<pto::MaskPattern::P1111, false>();
    CheckPatternPublicCalls<pto::MaskPattern::P1111, true>();
    using Tile = pto::Tile<pto::TileType::Vec, float, 4, 128, pto::BLayout::RowMajor, 4, 128>;
    Tile src, dst;
    std::vector<float> srcBuffer(Tile::Numel), dstBuffer(Tile::Numel);
    pto::TASSIGN(src, reinterpret_cast<uintptr_t>(srcBuffer.data()));
    pto::TASSIGN(dst, reinterpret_cast<uintptr_t>(dstBuffer.data()));
    pto::TSCATTER(dst, src);
    EXPECT_NEAR(pto::mocker::GetLastPtoInstrCycles(), 46.0, 4.6);
    pto::TGATHER<Tile, Tile>(dst, src);
    EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 0);
    context.npu_arch = previousArch;
}

TEST(A5ComparisonLayoutFit, MissingIndexedFitDoesNotReuseMockCycles)
{
    using namespace pto;
    using Data = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, 16, 128>;
    using Index = Tile<TileType::Vec, uint32_t, 16, 128, BLayout::RowMajor, 16, 128>;
    Data src, dst;
    Index index, tmp;
    std::vector<float> srcBuffer(Data::Numel), dstBuffer(Data::Numel);
    std::vector<uint32_t> indexBuffer(Index::Numel), tmpBuffer(Index::Numel);
    TASSIGN(src, reinterpret_cast<uintptr_t>(srcBuffer.data()));
    TASSIGN(dst, reinterpret_cast<uintptr_t>(dstBuffer.data()));
    TASSIGN(tmp, reinterpret_cast<uintptr_t>(tmpBuffer.data()));
    TASSIGN(index, reinterpret_cast<uintptr_t>(indexBuffer.data()));
    auto& context = perf_sim::GetCostModelCtx();
    const auto previousArch = context.npu_arch;
    context.npu_arch = perf_sim::kAscend950Arch;
    // This fp32 indexed gather has no calibrated A5 fit key.
    TGATHER(dst, src, index, tmp);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    context.npu_arch = previousArch;
}

TEST(A5ComparisonLayoutFit, PatternLookupPreservesMaskAndAxis)
{
    CostModelInput input{
        .op = PtoOpcode::TSCATTER, .dtype = DType::Half, .rows = 4, .cols = 128, .arch = CostModelArch::A5};
    input.has_mask_pattern = true;
    input.mask_pattern = pto::MaskPattern::P0001;
    input.scatter_axis = pto::ScatterAxis::SCATTER_COL;
    input.a5_op_params = "index_b32";
    a5::VfCurveContext context;
    ASSERT_TRUE(a5::TryInitVfCurveContext(input, context));
    ASSERT_TRUE(a5::TryResolveVfOpParams(input, context));
    EXPECT_EQ(context.op_params, "pattern_P0001_col");
    input.op = PtoOpcode::TGATHER;
    input.gather_axis = pto::GatherAxis::GATHER_ROW;
    ASSERT_TRUE(a5::TryResolveVfOpParams(input, context));
    EXPECT_EQ(context.op_params, "pattern_P0001_row");
    input.has_dtype2 = true;
    EXPECT_FALSE(a5::TryResolveVfOpParams(input, context));

    input.has_dtype2 = false;
    input.op = PtoOpcode::TSCATTER;
    input.mask_pattern = pto::MaskPattern::P1111;
    const auto scatterCycles = EstimateCycles(input).cycles;
    input.op = PtoOpcode::TMOV;
    input.has_mask_pattern = false;
    input.a5_op_params = {};
    EXPECT_GT(scatterCycles, 0);
    EXPECT_EQ(scatterCycles, EstimateCycles(input).cycles);
}

struct VfFitCase {
    PtoOpcode op;
    const char* name;
    int64_t rows;
    int64_t cols;
    double profilingCycles;
};

template <pto::MaskPattern Pattern, int Rows, int Cols, int ValidRows, int ValidCols>
void CheckHalfScatterPublic(double measured)
{
    using namespace pto;
    constexpr int factor = Pattern == MaskPattern::P0001 ? 4 : 2;
    using Src = Tile<TileType::Vec, half, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using Dst = Tile<TileType::Vec, half, Rows, Cols * factor, BLayout::RowMajor, -1, -1>;
    Src src(ValidRows, ValidCols);
    Dst dst(ValidRows, ValidCols * factor);
    std::vector<half> srcBuffer(Src::Numel), dstBuffer(Dst::Numel);
    TASSIGN(src, reinterpret_cast<uintptr_t>(srcBuffer.data()));
    TASSIGN(dst, reinterpret_cast<uintptr_t>(dstBuffer.data()));
    EXPECT_STREQ(pto::mocker::CostModelTileDTypeString<Src>(), "fp16");
    TSCATTER<Pattern>(dst, src);
    const double cycles = pto::mocker::GetLastPtoInstrCycles();
    EXPECT_GT(cycles, 0);
    EXPECT_LE(std::abs(cycles - measured) / measured, 0.10);
}

TEST(A5ComparisonLayoutFit, HalfScatterPatternPublicHardwareProfiles)
{
    auto& context = pto::perf_sim::GetCostModelCtx();
    const auto previousArch = context.npu_arch;
    context.npu_arch = pto::perf_sim::kAscend950Arch;
    CheckHalfScatterPublic<pto::MaskPattern::P0001, 16, 64, 16, 64>(198.0);
    CheckHalfScatterPublic<pto::MaskPattern::P0001, 1023, 16, 1023, 16>(8728.0);
    CheckHalfScatterPublic<pto::MaskPattern::P0001, 1, 16368, 1, 16368>(1343.0);
    CheckHalfScatterPublic<pto::MaskPattern::P0101, 16, 64, 16, 64>(130.0);
    CheckHalfScatterPublic<pto::MaskPattern::P0101, 1, 32736, 1, 32736>(1203.0);
    CheckHalfScatterPublic<pto::MaskPattern::P0001, 4, 80, 3, 63>(87.0);
    CheckHalfScatterPublic<pto::MaskPattern::P0101, 8, 160, 7, 129>(105.0);
    CheckHalfScatterPublic<pto::MaskPattern::P0001, 5, 80, 5, 80>(100.0);
    CheckHalfScatterPublic<pto::MaskPattern::P0001, 8, 112, 6, 95>(121.0);
    CheckHalfScatterPublic<pto::MaskPattern::P0101, 1, 3072, 1, 3072>(159.0);
    CheckHalfScatterPublic<pto::MaskPattern::P0101, 8, 112, 6, 95>(83.0);
    context.npu_arch = previousArch;
}

TEST(A5ComparisonLayoutFit, ScatterPatternRequiresValidDestinationAndCalibratedMode)
{
    CostModelInput input{
        .op = PtoOpcode::TSCATTER, .dtype = DType::Half, .rows = 16, .cols = 64, .arch = CostModelArch::A5};
    input.has_mask_pattern = true;
    input.mask_pattern = pto::MaskPattern::P0001;
    uint64_t cycles = 0;
    EXPECT_FALSE(a5::TryEstimateA5ScatterPatternCycles(input, cycles));
    input.dst_rows = 16;
    input.dst_cols = 256;
    ASSERT_TRUE(a5::TryEstimateA5ScatterPatternCycles(input, cycles));
    input.scatter_axis = pto::ScatterAxis::SCATTER_COL;
    EXPECT_FALSE(a5::TryEstimateA5ScatterPatternCycles(input, cycles));
    input.scatter_axis = pto::ScatterAxis::SCATTER_ROW;
    input.dst_cols = 128;
    EXPECT_FALSE(a5::TryEstimateA5ScatterPatternCycles(input, cycles));
    input.dst_cols = 256;
    input.has_dtype2 = true;
    EXPECT_FALSE(a5::TryEstimateA5ScatterPatternCycles(input, cycles));
}

constexpr VfFitCase kCases[] = {
    {PtoOpcode::TCMP, "TCMP", 8, 128, 66.0},          {PtoOpcode::TCMPS, "TCMPS", 16, 128, 51.0},
    {PtoOpcode::TSELS, "TSELS", 8, 128, 66.0},        {PtoOpcode::TCONCAT, "TCONCAT", 128, 128, 3864.0},
    {PtoOpcode::TGATHER, "TGATHER", 16, 128, 136.0},  {PtoOpcode::TGATHERB, "TGATHERB", 8, 128, 67.0},
    {PtoOpcode::TSCATTER, "TSCATTER", 16, 128, 49.1}, {PtoOpcode::TMOV, "TMOV", 8, 128, 41.0},
    {PtoOpcode::TFILLPAD, "TFILLPAD", 8, 128, 54.0},  {PtoOpcode::TFILLPAD_EXPAND, "TFILLPAD_EXPAND", 16, 128, 89.0},
    {PtoOpcode::TTRI, "TTRI", 8, 128, 141.0},
};

TEST(A5ComparisonLayoutFit, HardwareProfiles)
{
    constexpr double kRequiredPrecision = 0.90;
    for (const auto& testCase : kCases) {
        CostModelInput input{
            .op = testCase.op,
            .dtype = DType::Half,
            .rows = testCase.rows,
            .cols = testCase.cols,
            .arch = CostModelArch::A5,
            .valid_rows = testCase.rows,
            .valid_cols = testCase.cols,
        };
        pto::test::ExpectA5CycleNear(testCase.name, input, testCase.profilingCycles, kRequiredPrecision);
    }
}

TEST(A5ComparisonLayoutFit, TwoDimensionalPathCompatibility)
{
    struct PathCase {
        PtoOpcode op;
        const char* name;
        DType dtype;
        int64_t rows;
        int64_t cols;
        double profilingCycles;
    };
    constexpr PathCase kPathCases[] = {
        {PtoOpcode::TCMPS, "TCMPS/2D", DType::Half, 16, 256, 130.0},
        {PtoOpcode::TCOLARGMAX, "TCOLARGMAX/2D", DType::Float, 16, 16, 102.5},
        {PtoOpcode::TCOLARGMIN, "TCOLARGMIN/2D", DType::Half, 1024, 16, 4162.0},
        {PtoOpcode::TGATHERB, "TGATHERB/2D", DType::Half, 64, 128, 285.0},
        {PtoOpcode::TMOV, "TMOV/2D", DType::Half, 16, 256, 130.0},
        {PtoOpcode::TPARTMUL, "TPARTMUL/2D", DType::Half, 16, 256, 130.0},
        {PtoOpcode::TPRELU, "TPRELU/2D", DType::Half, 16, 256, 130.0},
        {PtoOpcode::TROWARGMAX, "TROWARGMAX/2D", DType::Float, 64, 64, 547.0},
        {PtoOpcode::TTRI, "TTRI/2D", DType::Half, 16, 256, 293.0},
    };
    constexpr double kRequiredPrecision = 0.95;
    for (const auto& testCase : kPathCases) {
        const CostModelInput input{
            .op = testCase.op,
            .dtype = testCase.dtype,
            .rows = testCase.rows,
            .cols = testCase.cols,
            .arch = CostModelArch::A5,
            .valid_rows = testCase.rows,
            .valid_cols = testCase.cols,
            .a5_shape_path = A5ShapePathHint::Path2D,
        };
        pto::test::ExpectA5CycleNear(testCase.name, input, testCase.profilingCycles, kRequiredPrecision);
    }
}

TEST(A5ComparisonLayoutFit, IndexedOperationsUseIndexDType)
{
    struct IndexedCase {
        PtoOpcode op;
        const char* name;
        DType indexDtype;
        int64_t rows;
        int64_t cols;
        double profilingCycles;
    };
    constexpr IndexedCase kIndexedCases[] = {
        {PtoOpcode::TGATHER, "TGATHER/index_b32", DType::Int32, 16, 16, 114.0},
        {PtoOpcode::TGATHER, "TGATHER/index_b32", DType::Uint32, 1, 16, 52.0},
        {PtoOpcode::TSCATTER, "TSCATTER/index_b16", DType::Int16, 1, 16, 71.0},
        {PtoOpcode::TSCATTER, "TSCATTER/index_b16", DType::Uint16, 1, 16, 71.0},
        {PtoOpcode::TSCATTER, "TSCATTER/index_b16/long", DType::Int16, 1, 43648, 4750.1},
        {PtoOpcode::TSCATTER, "TSCATTER/index_b16/long", DType::Uint16, 1, 43648, 4750.1},
        {PtoOpcode::TSCATTER, "TSCATTER/index_b16", DType::Int16, 16, 128, 225.0},
        {PtoOpcode::TSCATTER, "TSCATTER/index_b16", DType::Int16, 48, 96, 675.1},
    };
    constexpr double kRequiredPrecision = 0.95;
    for (const auto& testCase : kIndexedCases) {
        const CostModelInput input{
            .op = testCase.op,
            .dtype = DType::Half,
            .rows = testCase.rows,
            .cols = testCase.cols,
            .arch = CostModelArch::A5,
            .dtype2 = testCase.indexDtype,
            .has_dtype2 = true,
            .valid_rows = testCase.rows,
            .valid_cols = testCase.cols,
            .a5_shape_path = testCase.rows == 1 ? A5ShapePathHint::Path1D : A5ShapePathHint::Path2D,
        };
        pto::test::ExpectA5CycleNear(testCase.name, input, testCase.profilingCycles, kRequiredPrecision);
    }
}

TEST(A5ComparisonLayoutFit, LongSingleRowIndexAndColumnArgPaths)
{
    constexpr double kRequiredPrecision = 0.95;
    const CostModelInput gatherInput{
        .op = PtoOpcode::TGATHER,
        .dtype = DType::Half,
        .rows = 1,
        .cols = 21824,
        .arch = CostModelArch::A5,
        .dtype2 = DType::Int32,
        .has_dtype2 = true,
        .valid_rows = 1,
        .valid_cols = 21824,
        .a5_shape_path = A5ShapePathHint::Path1D,
    };
    pto::test::ExpectA5CycleNear("TGATHER/index_b32/row1", gatherInput, 1032.2, kRequiredPrecision);

    for (const auto op : {PtoOpcode::TCOLARGMAX, PtoOpcode::TCOLARGMIN}) {
        const CostModelInput colArgInput{
            .op = op,
            .dtype = DType::Int32,
            .rows = 1,
            .cols = 21824,
            .arch = CostModelArch::A5,
            .valid_rows = 1,
            .valid_cols = 21824,
            .a5_shape_path = A5ShapePathHint::Path1D,
        };
        pto::test::ExpectA5CycleNear("TCOLARG/row1", colArgInput, 1723.0, kRequiredPrecision);
    }
}

TEST(A5SupplementalProfiles, ExactTwoDimensionalHardwareProfiles)
{
    struct Profile {
        PtoOpcode op;
        DType dtype;
        int64_t rows;
        int64_t cols;
        double cycles;
    };
    constexpr Profile profiles[] = {
        {PtoOpcode::TCMP, DType::BFloat16, 16, 128, 114.0},  {PtoOpcode::TCMP, DType::BFloat16, 16, 256, 130.0},
        {PtoOpcode::TCMP, DType::BFloat16, 229, 32, 1392.0}, {PtoOpcode::TCMP, DType::BFloat16, 32, 32, 210.0},
        {PtoOpcode::TCMP, DType::BFloat16, 4, 128, 42.0},    {PtoOpcode::TCMP, DType::BFloat16, 4, 256, 46.0},
        {PtoOpcode::TCMP, DType::BFloat16, 7, 96, 60.0},     {PtoOpcode::TTRI, DType::Int16, 16, 128, 261.0},
        {PtoOpcode::TTRI, DType::Int16, 16, 16, 261.0},      {PtoOpcode::TTRI, DType::Int16, 16, 256, 293.0},
        {PtoOpcode::TTRI, DType::Int16, 362, 16, 5451.0},    {PtoOpcode::TTRI, DType::Int16, 4, 128, 81.0},
        {PtoOpcode::TTRI, DType::Int16, 4, 256, 89.0},       {PtoOpcode::TTRI, DType::Int16, 64, 64, 981.0},
        {PtoOpcode::TTRI, DType::Int16, 7, 96, 126.0},
    };
    for (const auto& profile : profiles) {
        const CostModelInput input{
            .op = profile.op,
            .dtype = profile.dtype,
            .rows = profile.rows,
            .cols = profile.cols,
            .arch = CostModelArch::A5,
            .a5_shape_path = A5ShapePathHint::Path2D,
        };
        CostModelResult result{};
        ASSERT_TRUE(EstimateCycles(input, result));
        EXPECT_NEAR(result.cycles, profile.cycles, profile.cycles * 0.10)
            << static_cast<int>(profile.op) << " " << profile.rows << "x" << profile.cols;
    }
}

TEST(A5ScatterPatternProfiles, HeldOutDtypesDenseDestinations)
{
    struct Profile {
        DType dtype;
        pto::MaskPattern pattern;
        int64_t rows;
        int64_t cols;
        double cycles;
    };
    constexpr Profile profiles[] = {
        {DType::BFloat16, pto::MaskPattern::P0001, 1023, 16, 8728.0},
        {DType::BFloat16, pto::MaskPattern::P0001, 4, 128, 97.0},
        {DType::BFloat16, pto::MaskPattern::P0001, 16, 128, 228.0},
        {DType::BFloat16, pto::MaskPattern::P0001, 1, 16368, 1343.0},
        {DType::BFloat16, pto::MaskPattern::P0001, 16, 64, 198.0},
        {DType::BFloat16, pto::MaskPattern::P0001, 16, 256, 398.0},
        {DType::BFloat16, pto::MaskPattern::P0001, 7, 96, 120.0},
        {DType::BFloat16, pto::MaskPattern::P0001, 1, 128, 73.0},
        {DType::BFloat16, pto::MaskPattern::P0001, 1, 16, 70.0},
        {DType::BFloat16, pto::MaskPattern::P0001, 1, 1024, 143.0},
        {DType::Float, pto::MaskPattern::P0001, 16, 128, 398.0},
        {DType::Float, pto::MaskPattern::P0001, 7, 96, 191.0},
        {DType::Float, pto::MaskPattern::P0001, 1, 8184, 1343.0},
        {DType::Float, pto::MaskPattern::P0001, 4, 128, 141.0},
        {DType::Float, pto::MaskPattern::P0001, 16, 256, 727.0},
        {DType::Float, pto::MaskPattern::P0001, 1, 128, 81.0},
        {DType::Float, pto::MaskPattern::P0001, 1, 8, 70.0},
        {DType::Float, pto::MaskPattern::P0001, 16, 64, 228.0},
        {DType::Float, pto::MaskPattern::P0001, 1023, 8, 8728.0},
        {DType::Float, pto::MaskPattern::P0001, 1, 1024, 223.0},
        {DType::Int16, pto::MaskPattern::P0001, 4, 128, 97.0},
        {DType::Int16, pto::MaskPattern::P0001, 16, 128, 228.0},
        {DType::Int16, pto::MaskPattern::P0001, 1023, 16, 8728.0},
        {DType::Int16, pto::MaskPattern::P0001, 1, 16, 70.0},
        {DType::Int16, pto::MaskPattern::P0001, 1, 1024, 143.0},
        {DType::Int16, pto::MaskPattern::P0001, 1, 16368, 1343.0},
        {DType::Int16, pto::MaskPattern::P0001, 1, 128, 73.0},
        {DType::Int16, pto::MaskPattern::P0001, 16, 64, 198.0},
        {DType::Int16, pto::MaskPattern::P0001, 7, 96, 120.0},
        {DType::Int16, pto::MaskPattern::P0001, 16, 256, 398.0},
        {DType::Int32, pto::MaskPattern::P0001, 16, 256, 727.0},
        {DType::Int32, pto::MaskPattern::P0001, 4, 128, 141.0},
        {DType::Int32, pto::MaskPattern::P0001, 7, 96, 191.0},
        {DType::Int32, pto::MaskPattern::P0001, 16, 64, 228.0},
        {DType::Int32, pto::MaskPattern::P0001, 1, 1024, 223.0},
        {DType::Int32, pto::MaskPattern::P0001, 1023, 8, 8728.0},
        {DType::Int32, pto::MaskPattern::P0001, 1, 8, 70.0},
        {DType::Int32, pto::MaskPattern::P0001, 1, 128, 81.0},
        {DType::Int32, pto::MaskPattern::P0001, 16, 128, 398.0},
        {DType::Int32, pto::MaskPattern::P0001, 1, 8184, 1343.0},
        {DType::Uint16, pto::MaskPattern::P0001, 16, 256, 398.0},
        {DType::Uint16, pto::MaskPattern::P0001, 1, 128, 73.0},
        {DType::Uint16, pto::MaskPattern::P0001, 1023, 16, 8728.0},
        {DType::Uint16, pto::MaskPattern::P0001, 16, 64, 198.0},
        {DType::Uint16, pto::MaskPattern::P0001, 1, 1024, 143.0},
        {DType::Uint16, pto::MaskPattern::P0001, 1, 16, 70.0},
        {DType::Uint16, pto::MaskPattern::P0001, 4, 128, 97.0},
        {DType::Uint16, pto::MaskPattern::P0001, 16, 128, 228.0},
        {DType::Uint16, pto::MaskPattern::P0001, 7, 96, 120.0},
        {DType::Uint16, pto::MaskPattern::P0001, 1, 16368, 1343.0},
        {DType::Uint32, pto::MaskPattern::P0001, 4, 128, 141.0},
        {DType::Uint32, pto::MaskPattern::P0001, 1, 8184, 1343.0},
        {DType::Uint32, pto::MaskPattern::P0001, 16, 128, 398.0},
        {DType::Uint32, pto::MaskPattern::P0001, 7, 96, 191.0},
        {DType::Uint32, pto::MaskPattern::P0001, 1, 1024, 223.0},
        {DType::Uint32, pto::MaskPattern::P0001, 1, 8, 70.0},
        {DType::Uint32, pto::MaskPattern::P0001, 1023, 8, 8728.0},
        {DType::Uint32, pto::MaskPattern::P0001, 16, 64, 228.0},
        {DType::Uint32, pto::MaskPattern::P0001, 1, 128, 81.0},
        {DType::Uint32, pto::MaskPattern::P0001, 16, 256, 727.0},
        {DType::BFloat16, pto::MaskPattern::P0101, 4, 128, 73.0},
        {DType::BFloat16, pto::MaskPattern::P0101, 1, 16, 60.0},
        {DType::BFloat16, pto::MaskPattern::P0101, 16, 128, 146.0},
        {DType::BFloat16, pto::MaskPattern::P0101, 16, 64, 130.0},
        {DType::BFloat16, pto::MaskPattern::P0101, 16, 256, 198.0},
        {DType::BFloat16, pto::MaskPattern::P0101, 7, 96, 82.0},
        {DType::BFloat16, pto::MaskPattern::P0101, 1, 32736, 1203.0},
        {DType::BFloat16, pto::MaskPattern::P0101, 1, 128, 61.0},
        {DType::BFloat16, pto::MaskPattern::P0101, 1, 1024, 89.0},
        {DType::BFloat16, pto::MaskPattern::P0101, 2046, 16, 12807.0},
        {DType::Float, pto::MaskPattern::P0101, 1, 16368, 1203.0},
        {DType::Float, pto::MaskPattern::P0101, 16, 128, 198.0},
        {DType::Float, pto::MaskPattern::P0101, 16, 64, 146.0},
        {DType::Float, pto::MaskPattern::P0101, 7, 96, 106.0},
        {DType::Float, pto::MaskPattern::P0101, 1, 8, 60.0},
        {DType::Float, pto::MaskPattern::P0101, 4, 128, 89.0},
        {DType::Float, pto::MaskPattern::P0101, 16, 256, 348.0},
        {DType::Float, pto::MaskPattern::P0101, 1, 128, 65.0},
        {DType::Float, pto::MaskPattern::P0101, 2046, 8, 12807.0},
        {DType::Float, pto::MaskPattern::P0101, 1, 1024, 121.0},
        {DType::Int16, pto::MaskPattern::P0101, 4, 128, 73.0},
        {DType::Int16, pto::MaskPattern::P0101, 16, 64, 130.0},
        {DType::Int16, pto::MaskPattern::P0101, 16, 128, 146.0},
        {DType::Int16, pto::MaskPattern::P0101, 1, 16, 60.0},
        {DType::Int16, pto::MaskPattern::P0101, 1, 1024, 89.0},
        {DType::Int16, pto::MaskPattern::P0101, 1, 32736, 1203.0},
        {DType::Int16, pto::MaskPattern::P0101, 2046, 16, 12807.0},
        {DType::Int16, pto::MaskPattern::P0101, 1, 128, 61.0},
        {DType::Int16, pto::MaskPattern::P0101, 7, 96, 82.0},
        {DType::Int16, pto::MaskPattern::P0101, 16, 256, 198.0},
        {DType::Int32, pto::MaskPattern::P0101, 16, 256, 348.0},
        {DType::Int32, pto::MaskPattern::P0101, 4, 128, 89.0},
        {DType::Int32, pto::MaskPattern::P0101, 2046, 8, 12807.0},
        {DType::Int32, pto::MaskPattern::P0101, 7, 96, 106.0},
        {DType::Int32, pto::MaskPattern::P0101, 1, 16368, 1203.0},
        {DType::Int32, pto::MaskPattern::P0101, 1, 1024, 121.0},
        {DType::Int32, pto::MaskPattern::P0101, 1, 8, 60.0},
        {DType::Int32, pto::MaskPattern::P0101, 16, 64, 146.0},
        {DType::Int32, pto::MaskPattern::P0101, 1, 128, 65.0},
        {DType::Int32, pto::MaskPattern::P0101, 16, 128, 198.0},
        {DType::Uint16, pto::MaskPattern::P0101, 16, 256, 198.0},
        {DType::Uint16, pto::MaskPattern::P0101, 1, 128, 61.0},
        {DType::Uint16, pto::MaskPattern::P0101, 1, 16, 60.0},
        {DType::Uint16, pto::MaskPattern::P0101, 1, 1024, 89.0},
        {DType::Uint16, pto::MaskPattern::P0101, 16, 64, 130.0},
        {DType::Uint16, pto::MaskPattern::P0101, 2046, 16, 12807.0},
        {DType::Uint16, pto::MaskPattern::P0101, 1, 32736, 1203.0},
        {DType::Uint16, pto::MaskPattern::P0101, 4, 128, 73.0},
        {DType::Uint16, pto::MaskPattern::P0101, 16, 128, 146.0},
        {DType::Uint16, pto::MaskPattern::P0101, 7, 96, 82.0},
        {DType::Uint32, pto::MaskPattern::P0101, 4, 128, 89.0},
        {DType::Uint32, pto::MaskPattern::P0101, 16, 128, 198.0},
        {DType::Uint32, pto::MaskPattern::P0101, 7, 96, 106.0},
        {DType::Uint32, pto::MaskPattern::P0101, 16, 64, 146.0},
        {DType::Uint32, pto::MaskPattern::P0101, 1, 8, 60.0},
        {DType::Uint32, pto::MaskPattern::P0101, 1, 16368, 1203.0},
        {DType::Uint32, pto::MaskPattern::P0101, 1, 1024, 121.0},
        {DType::Uint32, pto::MaskPattern::P0101, 2046, 8, 12807.0},
        {DType::Uint32, pto::MaskPattern::P0101, 1, 128, 65.0},
        {DType::Uint32, pto::MaskPattern::P0101, 16, 256, 348.0},
    };
    for (const auto& profile : profiles) {
        CostModelInput input{};
        input.op = PtoOpcode::TSCATTER;
        input.dtype = profile.dtype;
        input.arch = CostModelArch::A5;
        input.rows = profile.rows;
        input.cols = profile.cols;
        input.dst_rows = profile.rows;
        const int64_t expansion = profile.pattern == pto::MaskPattern::P0001 ? 4 : 2;
        input.dst_cols = profile.cols * expansion;
        input.has_mask_pattern = true;
        input.mask_pattern = profile.pattern;
        CostModelResult result{};
        ASSERT_TRUE(EstimateCycles(input, result));
        EXPECT_NEAR(result.cycles, profile.cycles, profile.cycles * 0.10)
            << static_cast<int>(profile.dtype) << " " << profile.rows << "x" << profile.cols;
        input.dst_rows += 1;
        EXPECT_FALSE(EstimateCycles(input, result));
    }
}

TEST(A5ValidatedProfiles, IdentifiableCrossValidatedHardwareProfiles)
{
    struct Profile {
        PtoOpcode op;
        DType dtype;
        int64_t rows;
        int64_t cols;
        double cycles;
    };
    constexpr Profile profiles[] = {
        {PtoOpcode::TCMP, DType::BFloat16, 1, 128, 37.0}, {PtoOpcode::TCMP, DType::BFloat16, 1, 224, 38.0},
        {PtoOpcode::TCMP, DType::BFloat16, 1, 32, 37.0},  {PtoOpcode::TTRI, DType::Float, 16, 128, 293.0},
        {PtoOpcode::TTRI, DType::Float, 16, 16, 261.0},   {PtoOpcode::TTRI, DType::Float, 16, 256, 341.0},
        {PtoOpcode::TTRI, DType::Float, 16, 64, 261.0},   {PtoOpcode::TTRI, DType::Float, 1, 128, 39.0},
        {PtoOpcode::TTRI, DType::Float, 1, 256, 41.0},    {PtoOpcode::TTRI, DType::Float, 1, 8, 37.0},
        {PtoOpcode::TTRI, DType::Float, 256, 8, 3861.0},  {PtoOpcode::TTRI, DType::Float, 32, 32, 501.0},
        {PtoOpcode::TTRI, DType::Float, 4, 128, 89.0},    {PtoOpcode::TTRI, DType::Float, 4, 256, 101.0},
        {PtoOpcode::TTRI, DType::Float, 64, 128, 1109.0}, {PtoOpcode::TTRI, DType::Float, 64, 64, 981.0},
        {PtoOpcode::TTRI, DType::Float, 7, 96, 140.0},    {PtoOpcode::TTRI, DType::Int32, 16, 128, 293.0},
        {PtoOpcode::TTRI, DType::Int32, 16, 16, 261.0},   {PtoOpcode::TTRI, DType::Int32, 16, 256, 341.0},
        {PtoOpcode::TTRI, DType::Int32, 1, 128, 39.0},    {PtoOpcode::TTRI, DType::Int32, 1, 256, 41.0},
        {PtoOpcode::TTRI, DType::Int32, 1, 8, 37.0},      {PtoOpcode::TTRI, DType::Int32, 256, 8, 3861.0},
        {PtoOpcode::TTRI, DType::Int32, 4, 128, 89.0},    {PtoOpcode::TTRI, DType::Int32, 4, 256, 101.0},
        {PtoOpcode::TTRI, DType::Int32, 64, 64, 981.0},   {PtoOpcode::TTRI, DType::Int32, 7, 96, 140.0},
        {PtoOpcode::TMOV, DType::Float, 16, 128, 130.0},  {PtoOpcode::TMOV, DType::Float, 16, 16, 114.0},
        {PtoOpcode::TMOV, DType::Float, 16, 256, 146.0},  {PtoOpcode::TMOV, DType::Float, 16, 64, 114.0},
        {PtoOpcode::TMOV, DType::Float, 32, 32, 210.0},   {PtoOpcode::TMOV, DType::Float, 4, 128, 46.0},
        {PtoOpcode::TMOV, DType::Float, 4, 256, 50.0},    {PtoOpcode::TMOV, DType::Float, 64, 128, 466.0},
        {PtoOpcode::TMOV, DType::Float, 64, 64, 402.0},   {PtoOpcode::TMOV, DType::Float, 7, 96, 67.0},
    };
    for (const auto& profile : profiles) {
        CostModelInput input{};
        input.op = profile.op;
        input.dtype = profile.dtype;
        input.rows = profile.rows;
        input.cols = profile.cols;
        input.arch = CostModelArch::A5;
        input.a5_shape_path = profile.rows == 1 ? A5ShapePathHint::Path1D : A5ShapePathHint::Path2D;
        CostModelResult result{};
        ASSERT_TRUE(EstimateCycles(input, result));
        EXPECT_NEAR(result.cycles, profile.cycles, profile.cycles * 0.10)
            << static_cast<int>(profile.op) << " " << profile.rows << "x" << profile.cols;
    }
}

template <int Rows, int SourceCols>
void CheckConcatPublicHardwareProfile(double measuredCycles)
{
    using Source = pto::Tile<pto::TileType::Vec, float, Rows, SourceCols, pto::BLayout::RowMajor, -1, -1>;
    using Destination = pto::Tile<pto::TileType::Vec, float, Rows, SourceCols * 2, pto::BLayout::RowMajor, -1, -1>;
    Source source0(Rows, SourceCols), source1(Rows, SourceCols);
    Destination destination(Rows, SourceCols * 2);
    std::vector<float> source0Data(Source::Numel), source1Data(Source::Numel);
    std::vector<float> destinationData(Destination::Numel);
    pto::TASSIGN(source0, reinterpret_cast<uintptr_t>(source0Data.data()));
    pto::TASSIGN(source1, reinterpret_cast<uintptr_t>(source1Data.data()));
    pto::TASSIGN(destination, reinterpret_cast<uintptr_t>(destinationData.data()));
    pto::TCONCAT(destination, source0, source1);
    EXPECT_NEAR(pto::mocker::GetLastPtoInstrCycles(), measuredCycles, measuredCycles * 0.10);
}

TEST(A5ComparisonLayoutFit, ConcatPublicSourceGeometryProfiles)
{
    auto& context = pto::perf_sim::GetCostModelCtx();
    const auto previousArch = context.npu_arch;
    context.npu_arch = pto::perf_sim::kAscend950Arch;
    CheckConcatPublicHardwareProfile<16, 128>(1317);
    CheckConcatPublicHardwareProfile<16, 256>(2149);
    CheckConcatPublicHardwareProfile<1, 128>(102);
    CheckConcatPublicHardwareProfile<4, 128>(345);
    CheckConcatPublicHardwareProfile<4, 256>(553);
    CheckConcatPublicHardwareProfile<64, 128>(5205);
    CheckConcatPublicHardwareProfile<64, 64>(3605);
    CheckConcatPublicHardwareProfile<7, 96>(588);
    context.npu_arch = previousArch;
}

TEST(A5ComparisonLayoutFit, ConcatRejectsUncalibratedGeometry)
{
    CostModelInput input{};
    input.op = PtoOpcode::TCONCAT;
    input.dtype = DType::Float;
    input.arch = CostModelArch::A5;
    input.rows = 4;
    input.cols = 256;
    input.concat_src0_cols = 128;
    input.concat_src1_cols = 128;
    input.concat_dense_sources = true;
    CostModelResult result{};
    ASSERT_TRUE(EstimateCycles(input, result));
    input.concat_src0_cols = 64;
    input.concat_src1_cols = 192;
    EXPECT_FALSE(EstimateCycles(input, result));
    input.concat_src0_cols = 128;
    input.concat_src1_cols = 128;
    input.concat_dense_sources = false;
    EXPECT_FALSE(EstimateCycles(input, result));
    input.concat_dense_sources = true;
    input.valid_cols = 128;
    EXPECT_FALSE(EstimateCycles(input, result));
}

TEST(A5ComparisonLayoutFit, ScalarCompareSinglePackedIteration)
{
    struct Case {
        DType dtype;
        int64_t rows;
        int64_t cols;
        double measured;
    };
    const Case cases[] = {
        {DType::BFloat16, 16, 128, 114.0}, {DType::BFloat16, 295, 32, 1788.0}, {DType::BFloat16, 32, 32, 210.0},
        {DType::BFloat16, 4, 128, 42.0},   {DType::BFloat16, 7, 96, 60.0},     {DType::Float, 16, 128, 130.0},
        {DType::Float, 16, 64, 130.0},     {DType::Float, 229, 32, 1621.0},    {DType::Float, 32, 32, 242.0},
        {DType::Float, 4, 128, 50.0},      {DType::Float, 64, 128, 466.0},     {DType::Float, 64, 64, 466.0},
        {DType::Float, 7, 96, 67.0},       {DType::Int32, 16, 128, 130.0},     {DType::Int32, 16, 64, 130.0},
        {DType::Int32, 229, 32, 1621.0},   {DType::Int32, 32, 32, 242.0},      {DType::Int32, 4, 128, 50.0},
        {DType::Int32, 64, 128, 466.0},    {DType::Int32, 64, 64, 466.0},      {DType::Int32, 7, 96, 67.0},
        {DType::Uint32, 16, 128, 130.0},   {DType::Uint32, 229, 32, 1621.0},   {DType::Uint32, 32, 32, 242.0},
        {DType::Uint32, 4, 128, 50.0},     {DType::Uint32, 7, 96, 67.0},       {DType::Uint16, 16, 128, 114.0},
        {DType::Uint16, 295, 32, 1788.0},  {DType::Uint16, 32, 32, 210.0},     {DType::Uint16, 4, 128, 42.0},
        {DType::Uint16, 7, 96, 60.0},      {DType::Int8, 16, 128, 114.0},      {DType::Int8, 16, 256, 114.0},
        {DType::Int8, 32, 32, 210.0},      {DType::Int8, 362, 32, 2190.0},     {DType::Int8, 4, 128, 42.0},
        {DType::Int8, 4, 256, 42.0},       {DType::Int8, 7, 96, 60.0},         {DType::Uint8, 16, 128, 114.0},
        {DType::Uint8, 16, 256, 114.0},    {DType::Uint8, 32, 32, 210.0},      {DType::Uint8, 362, 32, 2190.0},
        {DType::Uint8, 4, 128, 42.0},      {DType::Uint8, 4, 256, 42.0},       {DType::Uint8, 7, 96, 60.0},
    };
    for (const auto& item : cases) {
        CostModelInput input{
            .op = PtoOpcode::TCMPS,
            .dtype = item.dtype,
            .rows = item.rows,
            .cols = item.cols,
            .arch = CostModelArch::A5,
            .a5_shape_path = A5ShapePathHint::Path2D};
        CostModelResult result{};
        ASSERT_TRUE(EstimateCycles(input, result));
        EXPECT_LE(std::abs(result.cycles - item.measured) / item.measured, 0.10);
        input.cols = 512;
        EXPECT_FALSE(EstimateCycles(input, result));
        input.cols = item.cols;
        input.valid_cols = item.cols - 1;
        EXPECT_FALSE(EstimateCycles(input, result));
        input.valid_cols = item.cols;
        input.rows = 1;
        EXPECT_FALSE(EstimateCycles(input, result));
    }
}
TEST(A5ComparisonLayoutFit, ScalarComparePublicOverloadBoundary)
{
    using namespace pto;
    using Src = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1>;
    using Dst = Tile<TileType::Vec, uint8_t, 16, 32, BLayout::RowMajor, -1, -1>;
    using Scalar = Tile<TileType::Vec, float, 1, 8, BLayout::RowMajor, 1, 1>;
    Src src(16, 128);
    Dst dst(16, 16);
    Scalar scalar;
    std::vector<float> srcData(Src::Numel), scalarData(Scalar::Numel);
    std::vector<uint8_t> dstData(Dst::Numel);
    TASSIGN(src, reinterpret_cast<uintptr_t>(srcData.data()));
    TASSIGN(dst, reinterpret_cast<uintptr_t>(dstData.data()));
    TASSIGN(scalar, reinterpret_cast<uintptr_t>(scalarData.data()));
    auto& context = perf_sim::GetCostModelCtx();
    const auto previousArch = context.npu_arch;
    context.npu_arch = perf_sim::kAscend950Arch;
    TCMPS(dst, src, 1.0f, CmpMode::EQ);
    EXPECT_NEAR(mocker::GetLastPtoInstrCycles(), 130, 13);
    TCMPS(dst, src, scalar, CmpMode::EQ);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    context.npu_arch = previousArch;
}

TEST(A5ComparisonLayoutFit, DenseRowScalarComparePublicInterface)
{
    using namespace pto;
    using Src = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1>;
    using Dst = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;
    Src src(1, 192);
    Dst dst(1, 24);
    std::vector<float> srcData(Src::Numel);
    std::vector<uint8_t> dstData(Dst::Numel);
    TASSIGN(src, reinterpret_cast<uintptr_t>(srcData.data()));
    TASSIGN(dst, reinterpret_cast<uintptr_t>(dstData.data()));
    auto& context = perf_sim::GetCostModelCtx();
    const auto previousArch = context.npu_arch;
    context.npu_arch = perf_sim::kAscend950Arch;
    TCMPS(dst, src, 1.0f, CmpMode::EQ);
    EXPECT_NEAR(mocker::GetLastPtoInstrCycles(), 46.0, 4.6);
    context.npu_arch = previousArch;
}

TEST(A5ComparisonLayoutFit, PackedB32CompareHardwareProfiles)
{
    struct Profile {
        int64_t rows;
        int64_t cols;
        double cycles;
    };
    constexpr Profile profiles[] = {{16, 256, 182.0}, {16, 64, 130.0},  {171, 32, 1215.3}, {32, 32, 242.1},
                                    {64, 128, 466.1}, {64, 64, 466.1},  {909, 32, 6382.3}, {2, 32, 49.0},
                                    {4, 64, 56.0},    {8, 128, 74.0},   {16, 32, 130.0},   {16, 128, 130.0},
                                    {32, 64, 242.1},  {32, 128, 242.1}, {64, 32, 466.1},   {64, 256, 614.1},
                                    {3, 64, 51.0},    {7, 128, 70.0},   {17, 64, 137.0},   {33, 128, 249.1}};
    for (auto dtype : {DType::Float, DType::Int32}) {
        for (const auto& profile : profiles) {
            CostModelInput input{};
            input.arch = CostModelArch::A5;
            input.op = PtoOpcode::TCMP;
            input.dtype = dtype;
            input.rows = profile.rows;
            input.cols = profile.cols;
            input.a5_shape_path = A5ShapePathHint::Path2D;
            CostModelResult result{};
            ASSERT_TRUE(EstimateCycles(input, result));
            EXPECT_NEAR(result.cycles, profile.cycles, profile.cycles * 0.10);
        }
    }
}

TEST(A5ComparisonLayoutFit, Uint16SelectDenseHardwareProfiles)
{
    struct Profile {
        int64_t rows;
        int64_t cols;
        double cycles;
    };
    constexpr Profile profiles[] = {
        {167, 160, 1187.5}, {167, 32, 1020.5}, {32, 32, 210.1}, {64, 64, 402.2},  {876, 32, 5276.6}, {2, 32, 40.0},
        {4, 64, 44.0},      {8, 128, 66.0},    {16, 32, 114.1}, {16, 128, 114.1}, {32, 64, 210.1},   {32, 128, 210.1},
        {64, 32, 402.2},    {64, 256, 466.2},  {3, 64, 42.0},   {7, 128, 60.0},   {17, 64, 120.1},   {33, 128, 216.1}};
    for (const auto& profile : profiles) {
        CostModelInput input{};
        input.arch = CostModelArch::A5;
        input.op = PtoOpcode::TSEL;
        input.dtype = DType::Uint16;
        input.rows = profile.rows;
        input.cols = profile.cols;
        input.a5_shape_path = A5ShapePathHint::Path2D;
        CostModelResult result{};
        ASSERT_TRUE(EstimateCycles(input, result));
        EXPECT_NEAR(result.cycles, profile.cycles, profile.cycles * 0.10);
    }
}

TEST(A5ComparisonLayoutFit, SelectPairedLoopAndDenseRowHardwareProfiles)
{
    struct Profile {
        DType dtype;
        int64_t rows;
        int64_t cols;
        double cycles;
    };
    constexpr Profile profiles[] = {
        {DType::Uint16, 1, 32, 38},      {DType::Uint16, 1, 8192, 170},   {DType::Uint16, 1, 96, 37},
        {DType::Uint16, 1, 160, 39},     {DType::Uint16, 1, 320, 41},     {DType::Uint16, 1, 768, 47},
        {DType::Uint16, 1, 3072, 84},    {DType::Uint16, 1, 6144, 137},   {DType::Uint16, 1, 28032, 541.1},
        {DType::Uint32, 1, 32, 52},      {DType::Uint32, 1, 192, 57},     {DType::Uint32, 1, 96, 45},
        {DType::Uint32, 1, 160, 57},     {DType::Uint32, 1, 320, 61},     {DType::Uint32, 1, 768, 60.5},
        {DType::Uint32, 1, 3072, 134},   {DType::Uint32, 1, 6144, 241.6}, {DType::Uint32, 1, 15096, 553.6},
        {DType::Uint32, 2, 64, 57},      {DType::Uint32, 2, 128, 48},     {DType::Uint32, 64, 320, 888.3},
        {DType::Uint32, 64, 384, 926.3}, {DType::Uint32, 3, 96, 51},      {DType::Uint32, 3, 160, 71},
        {DType::Uint32, 3, 224, 62},     {DType::Uint32, 3, 288, 94},     {DType::Uint32, 3, 352, 72.5},
        {DType::Uint32, 17, 96, 148.1},  {DType::Uint32, 17, 160, 202.6}, {DType::Uint32, 17, 224, 216.1},
        {DType::Uint32, 17, 288, 277.1}, {DType::Uint32, 17, 352, 268.1}, {DType::Uint32, 33, 96, 260.1},
        {DType::Uint32, 33, 160, 346.6}, {DType::Uint32, 33, 224, 392.1}, {DType::Uint32, 33, 288, 485.1},
        {DType::Uint32, 33, 352, 492.1}};
    for (const auto& profile : profiles) {
        SCOPED_TRACE(::testing::Message() << profile.rows << "x" << profile.cols);
        CostModelInput input{};
        input.arch = CostModelArch::A5;
        input.op = PtoOpcode::TSEL;
        input.dtype = profile.dtype;
        input.rows = profile.rows;
        input.cols = profile.cols;
        input.a5_shape_path = profile.rows == 1 ? A5ShapePathHint::Path1D : A5ShapePathHint::Path2D;
        CostModelResult result{};
        ASSERT_TRUE(EstimateCycles(input, result));
        EXPECT_NEAR(result.cycles, profile.cycles, profile.cycles * 0.10);
    }
}

TEST(A5ComparisonLayoutFit, ExpandScalarWideDenseHardwareProfiles)
{
    for (DType dtype : {DType::Float, DType::Int16, DType::Int32}) {
        const int64_t width = dtype == DType::Int16 ? 128 : 64;
        const int64_t boundary = dtype == DType::Int16 ? 130928 : 65464;
        for (int64_t cols :
             {int64_t{32}, int64_t{128}, int64_t{512}, int64_t{2048}, int64_t{8192}, int64_t{32768}, int64_t{65536},
              int64_t{96}, int64_t{768}, int64_t{4096}, int64_t{16384}, int64_t{49152}, boundary}) {
            CostModelInput input{};
            input.arch = CostModelArch::A5;
            input.op = PtoOpcode::TEXPANDS;
            input.dtype = dtype;
            input.rows = 1;
            input.cols = cols;
            input.a5_shape_path = A5ShapePathHint::Path1D;
            CostModelResult result{};
            ASSERT_TRUE(EstimateCycles(input, result));
            // All 39 collected train/holdout/boundary points match this line.
            const double measured = (cols + width - 1) / width + 33;
            EXPECT_NEAR(result.cycles, measured, measured * 0.10);
        }
    }
}

TEST(A5ComparisonLayoutFit, DensePackedScalarCompareHardwareProfiles)
{
    struct Profile {
        DType dtype;
        int64_t rows;
        int64_t cols;
        double measured;
        A5ShapePathHint path;
    };
    // Frozen training, shape-disjoint holdout, and original issue validation.
    const Profile profiles[] = {
        {DType::BFloat16, 1, 32, 36, A5ShapePathHint::Path1D},
        {DType::BFloat16, 1, 128, 36, A5ShapePathHint::Path1D},
        {DType::BFloat16, 1, 512, 39, A5ShapePathHint::Path1D},
        {DType::BFloat16, 1, 2048, 51, A5ShapePathHint::Path1D},
        {DType::BFloat16, 1, 8192, 122, A5ShapePathHint::Path1D},
        {DType::BFloat16, 1, 32768, 404.1, A5ShapePathHint::Path1D},
        {DType::BFloat16, 1, 65536, 780.1, A5ShapePathHint::Path1D},
        {DType::BFloat16, 1, 96, 36, A5ShapePathHint::Path1D},
        {DType::BFloat16, 1, 768, 41, A5ShapePathHint::Path1D},
        {DType::BFloat16, 1, 4096, 74, A5ShapePathHint::Path1D},
        {DType::BFloat16, 1, 16384, 216, A5ShapePathHint::Path1D},
        {DType::BFloat16, 1, 49152, 592.1, A5ShapePathHint::Path1D},
        {DType::BFloat16, 1, 87296, 1030.2, A5ShapePathHint::Path1D},
        {DType::BFloat16, 2, 256, 39, A5ShapePathHint::Path2D},
        {DType::BFloat16, 2, 384, 41, A5ShapePathHint::Path2D},
        {DType::BFloat16, 8, 256, 74, A5ShapePathHint::Path2D},
        {DType::BFloat16, 8, 384, 82, A5ShapePathHint::Path2D},
        {DType::BFloat16, 32, 256, 242.1, A5ShapePathHint::Path2D},
        {DType::BFloat16, 32, 384, 274.1, A5ShapePathHint::Path2D},
        {DType::BFloat16, 64, 256, 466.1, A5ShapePathHint::Path2D},
        {DType::BFloat16, 64, 384, 530.1, A5ShapePathHint::Path2D},
        {DType::BFloat16, 3, 192, 41, A5ShapePathHint::Path2D},
        {DType::BFloat16, 3, 320, 44, A5ShapePathHint::Path2D},
        {DType::BFloat16, 17, 192, 137, A5ShapePathHint::Path2D},
        {DType::BFloat16, 17, 320, 154, A5ShapePathHint::Path2D},
        {DType::BFloat16, 33, 192, 249.1, A5ShapePathHint::Path2D},
        {DType::BFloat16, 33, 320, 282.1, A5ShapePathHint::Path2D},
        {DType::BFloat16, 295, 288, 2378.5, A5ShapePathHint::Path2D},
        {DType::Float, 1, 32, 44, A5ShapePathHint::Path1D},
        {DType::Float, 1, 128, 44, A5ShapePathHint::Path1D},
        {DType::Float, 1, 512, 50, A5ShapePathHint::Path1D},
        {DType::Float, 1, 2048, 109, A5ShapePathHint::Path1D},
        {DType::Float, 1, 8192, 321, A5ShapePathHint::Path1D},
        {DType::Float, 1, 32768, 1177.1, A5ShapePathHint::Path1D},
        {DType::Float, 1, 96, 44, A5ShapePathHint::Path1D},
        {DType::Float, 1, 768, 64, A5ShapePathHint::Path1D},
        {DType::Float, 1, 4096, 180, A5ShapePathHint::Path1D},
        {DType::Float, 1, 16384, 609.1, A5ShapePathHint::Path1D},
        {DType::Float, 1, 49152, 1748.2, A5ShapePathHint::Path1D},
        {DType::Float, 2, 256, 50, A5ShapePathHint::Path2D},
        {DType::Float, 2, 384, 64, A5ShapePathHint::Path2D},
        {DType::Float, 8, 256, 106, A5ShapePathHint::Path2D},
        {DType::Float, 8, 384, 146, A5ShapePathHint::Path2D},
        {DType::Float, 32, 256, 322.1, A5ShapePathHint::Path2D},
        {DType::Float, 32, 384, 458.1, A5ShapePathHint::Path2D},
        {DType::Float, 64, 256, 610.1, A5ShapePathHint::Path2D},
        {DType::Float, 64, 384, 874.1, A5ShapePathHint::Path2D},
        {DType::Float, 3, 192, 64, A5ShapePathHint::Path2D},
        {DType::Float, 3, 320, 80, A5ShapePathHint::Path2D},
        {DType::Float, 17, 192, 190, A5ShapePathHint::Path2D},
        {DType::Float, 17, 320, 260, A5ShapePathHint::Path2D},
        {DType::Float, 33, 192, 334.1, A5ShapePathHint::Path2D},
        {DType::Float, 33, 320, 468.1, A5ShapePathHint::Path2D},
        {DType::Float, 16, 256, 178, A5ShapePathHint::Path2D},
        {DType::Float, 229, 224, 2098.5, A5ShapePathHint::Path2D},
        {DType::Int32, 1, 32, 44, A5ShapePathHint::Path1D},
        {DType::Int32, 1, 128, 44, A5ShapePathHint::Path1D},
        {DType::Int32, 1, 512, 50, A5ShapePathHint::Path1D},
        {DType::Int32, 1, 2048, 109, A5ShapePathHint::Path1D},
        {DType::Int32, 1, 8192, 321, A5ShapePathHint::Path1D},
        {DType::Int32, 1, 32768, 1177.1, A5ShapePathHint::Path1D},
        {DType::Int32, 1, 96, 44, A5ShapePathHint::Path1D},
        {DType::Int32, 1, 768, 64, A5ShapePathHint::Path1D},
        {DType::Int32, 1, 4096, 180, A5ShapePathHint::Path1D},
        {DType::Int32, 1, 16384, 609.1, A5ShapePathHint::Path1D},
        {DType::Int32, 1, 49152, 1748.2, A5ShapePathHint::Path1D},
        {DType::Int32, 2, 256, 50, A5ShapePathHint::Path2D},
        {DType::Int32, 2, 384, 64, A5ShapePathHint::Path2D},
        {DType::Int32, 8, 256, 106, A5ShapePathHint::Path2D},
        {DType::Int32, 8, 384, 146, A5ShapePathHint::Path2D},
        {DType::Int32, 32, 256, 322.1, A5ShapePathHint::Path2D},
        {DType::Int32, 32, 384, 458.1, A5ShapePathHint::Path2D},
        {DType::Int32, 64, 256, 610.1, A5ShapePathHint::Path2D},
        {DType::Int32, 64, 384, 874.1, A5ShapePathHint::Path2D},
        {DType::Int32, 3, 192, 64, A5ShapePathHint::Path2D},
        {DType::Int32, 3, 320, 80, A5ShapePathHint::Path2D},
        {DType::Int32, 17, 192, 190, A5ShapePathHint::Path2D},
        {DType::Int32, 17, 320, 260, A5ShapePathHint::Path2D},
        {DType::Int32, 33, 192, 334.1, A5ShapePathHint::Path2D},
        {DType::Int32, 33, 320, 468.1, A5ShapePathHint::Path2D},
        {DType::Int32, 16, 256, 178, A5ShapePathHint::Path2D},
        {DType::Int32, 229, 224, 2098.5, A5ShapePathHint::Path2D},
        {DType::Int8, 1, 32, 36, A5ShapePathHint::Path1D},
        {DType::Int8, 1, 128, 36, A5ShapePathHint::Path1D},
        {DType::Int8, 1, 512, 37, A5ShapePathHint::Path1D},
        {DType::Int8, 1, 2048, 43, A5ShapePathHint::Path1D},
        {DType::Int8, 1, 8192, 74, A5ShapePathHint::Path1D},
        {DType::Int8, 1, 32768, 216, A5ShapePathHint::Path1D},
        {DType::Int8, 1, 65536, 408.1, A5ShapePathHint::Path1D},
        {DType::Int8, 1, 96, 36, A5ShapePathHint::Path1D},
        {DType::Int8, 1, 768, 38, A5ShapePathHint::Path1D},
        {DType::Int8, 1, 4096, 51, A5ShapePathHint::Path1D},
        {DType::Int8, 1, 16384, 121, A5ShapePathHint::Path1D},
        {DType::Int8, 1, 49152, 312.1, A5ShapePathHint::Path1D},
        {DType::Int8, 1, 130944, 792.1, A5ShapePathHint::Path1D},
        {DType::Int8, 2, 384, 39, A5ShapePathHint::Path2D},
        {DType::Int8, 8, 384, 74, A5ShapePathHint::Path2D},
        {DType::Int8, 32, 384, 242.1, A5ShapePathHint::Path2D},
        {DType::Int8, 64, 384, 466.1, A5ShapePathHint::Path2D},
        {DType::Int8, 3, 320, 41, A5ShapePathHint::Path2D},
        {DType::Int8, 17, 320, 137, A5ShapePathHint::Path2D},
        {DType::Int8, 33, 320, 249.1, A5ShapePathHint::Path2D},
        {DType::Int8, 362, 352, 2552.6, A5ShapePathHint::Path2D},
        {DType::Uint16, 1, 32, 36, A5ShapePathHint::Path1D},
        {DType::Uint16, 1, 128, 36, A5ShapePathHint::Path1D},
        {DType::Uint16, 1, 512, 39, A5ShapePathHint::Path1D},
        {DType::Uint16, 1, 2048, 51, A5ShapePathHint::Path1D},
        {DType::Uint16, 1, 8192, 122, A5ShapePathHint::Path1D},
        {DType::Uint16, 1, 32768, 404.1, A5ShapePathHint::Path1D},
        {DType::Uint16, 1, 65536, 780.1, A5ShapePathHint::Path1D},
        {DType::Uint16, 1, 96, 36, A5ShapePathHint::Path1D},
        {DType::Uint16, 1, 768, 41, A5ShapePathHint::Path1D},
        {DType::Uint16, 1, 4096, 74, A5ShapePathHint::Path1D},
        {DType::Uint16, 1, 16384, 216, A5ShapePathHint::Path1D},
        {DType::Uint16, 1, 49152, 592.1, A5ShapePathHint::Path1D},
        {DType::Uint16, 1, 87296, 1030.2, A5ShapePathHint::Path1D},
        {DType::Uint16, 2, 256, 39, A5ShapePathHint::Path2D},
        {DType::Uint16, 2, 384, 41, A5ShapePathHint::Path2D},
        {DType::Uint16, 8, 256, 74, A5ShapePathHint::Path2D},
        {DType::Uint16, 8, 384, 82, A5ShapePathHint::Path2D},
        {DType::Uint16, 32, 256, 242.1, A5ShapePathHint::Path2D},
        {DType::Uint16, 32, 384, 274.1, A5ShapePathHint::Path2D},
        {DType::Uint16, 64, 256, 466.1, A5ShapePathHint::Path2D},
        {DType::Uint16, 64, 384, 530.1, A5ShapePathHint::Path2D},
        {DType::Uint16, 3, 192, 41, A5ShapePathHint::Path2D},
        {DType::Uint16, 3, 320, 44, A5ShapePathHint::Path2D},
        {DType::Uint16, 17, 192, 137, A5ShapePathHint::Path2D},
        {DType::Uint16, 17, 320, 154, A5ShapePathHint::Path2D},
        {DType::Uint16, 33, 192, 249.1, A5ShapePathHint::Path2D},
        {DType::Uint16, 33, 320, 282.1, A5ShapePathHint::Path2D},
        {DType::Uint16, 295, 288, 2378.5, A5ShapePathHint::Path2D},
        {DType::Uint32, 1, 32, 44, A5ShapePathHint::Path1D},
        {DType::Uint32, 1, 128, 44, A5ShapePathHint::Path1D},
        {DType::Uint32, 1, 512, 50, A5ShapePathHint::Path1D},
        {DType::Uint32, 1, 2048, 109, A5ShapePathHint::Path1D},
        {DType::Uint32, 1, 8192, 321, A5ShapePathHint::Path1D},
        {DType::Uint32, 1, 32768, 1177.1, A5ShapePathHint::Path1D},
        {DType::Uint32, 1, 96, 44, A5ShapePathHint::Path1D},
        {DType::Uint32, 1, 768, 64, A5ShapePathHint::Path1D},
        {DType::Uint32, 1, 4096, 179, A5ShapePathHint::Path1D},
        {DType::Uint32, 1, 16384, 603.1, A5ShapePathHint::Path1D},
        {DType::Uint32, 1, 49152, 1748.2, A5ShapePathHint::Path1D},
        {DType::Uint32, 2, 256, 50, A5ShapePathHint::Path2D},
        {DType::Uint32, 2, 384, 64, A5ShapePathHint::Path2D},
        {DType::Uint32, 8, 256, 106, A5ShapePathHint::Path2D},
        {DType::Uint32, 8, 384, 146, A5ShapePathHint::Path2D},
        {DType::Uint32, 32, 256, 322.1, A5ShapePathHint::Path2D},
        {DType::Uint32, 32, 384, 458.1, A5ShapePathHint::Path2D},
        {DType::Uint32, 64, 256, 610.1, A5ShapePathHint::Path2D},
        {DType::Uint32, 64, 384, 874.1, A5ShapePathHint::Path2D},
        {DType::Uint32, 3, 192, 64, A5ShapePathHint::Path2D},
        {DType::Uint32, 3, 320, 80, A5ShapePathHint::Path2D},
        {DType::Uint32, 17, 192, 190, A5ShapePathHint::Path2D},
        {DType::Uint32, 17, 320, 260, A5ShapePathHint::Path2D},
        {DType::Uint32, 33, 192, 334.1, A5ShapePathHint::Path2D},
        {DType::Uint32, 33, 320, 468.1, A5ShapePathHint::Path2D},
        {DType::Uint32, 229, 224, 2098.5, A5ShapePathHint::Path2D},
        {DType::Uint8, 1, 32, 36, A5ShapePathHint::Path1D},
        {DType::Uint8, 1, 128, 36, A5ShapePathHint::Path1D},
        {DType::Uint8, 1, 512, 37, A5ShapePathHint::Path1D},
        {DType::Uint8, 1, 2048, 43, A5ShapePathHint::Path1D},
        {DType::Uint8, 1, 8192, 74, A5ShapePathHint::Path1D},
        {DType::Uint8, 1, 32768, 216, A5ShapePathHint::Path1D},
        {DType::Uint8, 1, 65536, 408.1, A5ShapePathHint::Path1D},
        {DType::Uint8, 1, 96, 36, A5ShapePathHint::Path1D},
        {DType::Uint8, 1, 768, 38, A5ShapePathHint::Path1D},
        {DType::Uint8, 1, 4096, 51, A5ShapePathHint::Path1D},
        {DType::Uint8, 1, 16384, 121, A5ShapePathHint::Path1D},
        {DType::Uint8, 1, 49152, 312.1, A5ShapePathHint::Path1D},
        {DType::Uint8, 1, 130944, 792.1, A5ShapePathHint::Path1D},
        {DType::Uint8, 2, 384, 39, A5ShapePathHint::Path2D},
        {DType::Uint8, 8, 384, 74, A5ShapePathHint::Path2D},
        {DType::Uint8, 32, 384, 242.1, A5ShapePathHint::Path2D},
        {DType::Uint8, 64, 384, 466.1, A5ShapePathHint::Path2D},
        {DType::Uint8, 3, 320, 41, A5ShapePathHint::Path2D},
        {DType::Uint8, 17, 320, 137, A5ShapePathHint::Path2D},
        {DType::Uint8, 33, 320, 249.1, A5ShapePathHint::Path2D},
        {DType::Uint8, 362, 352, 2552.6, A5ShapePathHint::Path2D},
    };
    for (const auto& profile : profiles) {
        CostModelInput input{};
        input.arch = CostModelArch::A5;
        input.op = PtoOpcode::TCMPS;
        input.dtype = profile.dtype;
        input.rows = profile.rows;
        input.cols = profile.cols;
        input.a5_shape_path = profile.path;
        CostModelResult result{};
        ASSERT_TRUE(EstimateCycles(input, result));
        EXPECT_NEAR(result.cycles, profile.measured, profile.measured * 0.10);
    }
}

TEST(A5ComparisonLayoutFit, DenseMoveAndIdentityScatterHardwareProfiles)
{
    struct Profile {
        DType dtype;
        int64_t rows;
        int64_t cols;
        double measured;
    };
    const Profile profiles[] = {
        {DType::BFloat16, 1, 128, 34},
        {DType::BFloat16, 1, 512, 37},
        {DType::BFloat16, 1, 2048, 49},
        {DType::BFloat16, 1, 8192, 99},
        {DType::BFloat16, 1, 16384, 163},
        {DType::BFloat16, 1, 32768, 291.1},
        {DType::BFloat16, 1, 768, 39},
        {DType::BFloat16, 1, 4096, 67},
        {DType::BFloat16, 1, 12288, 131},
        {DType::BFloat16, 1, 32, 34},
        {DType::BFloat16, 1, 96, 34},
        {DType::BFloat16, 1, 16, 34},
        {DType::BFloat16, 1, 65472, 1024.1},
        {DType::BFloat16, 1, 1952, 49},
        {DType::BFloat16, 1, 1984, 49},
        {DType::BFloat16, 1, 2016, 49},
        {DType::BFloat16, 1, 4000, 68},
        {DType::BFloat16, 1, 4032, 68},
        {DType::BFloat16, 1, 4064, 68},
        {DType::BFloat16, 1, 8096, 128},
        {DType::BFloat16, 1, 8128, 128},
        {DType::BFloat16, 1, 8160, 128},
        {DType::BFloat16, 1, 12192, 192},
        {DType::BFloat16, 1, 12224, 192},
        {DType::BFloat16, 1, 12256, 192},
        {DType::BFloat16, 1, 16288, 256},
        {DType::BFloat16, 1, 16320, 256},
        {DType::BFloat16, 1, 16352, 256},
        {DType::BFloat16, 1, 24480, 384.1},
        {DType::BFloat16, 1, 24512, 384.1},
        {DType::BFloat16, 1, 24544, 384.1},
        {DType::BFloat16, 1, 32672, 512.1},
        {DType::BFloat16, 1, 32704, 512.1},
        {DType::BFloat16, 1, 32736, 512.1},
        {DType::BFloat16, 2, 16, 35},
        {DType::BFloat16, 2, 64, 35},
        {DType::BFloat16, 8, 16, 66},
        {DType::BFloat16, 8, 64, 66},
        {DType::BFloat16, 32, 16, 210.1},
        {DType::BFloat16, 32, 64, 210.1},
        {DType::BFloat16, 128, 16, 786.2},
        {DType::BFloat16, 128, 64, 786.2},
        {DType::BFloat16, 1024, 16, 6163.5},
        {DType::BFloat16, 1024, 64, 6163.5},
        {DType::BFloat16, 3, 16, 36},
        {DType::BFloat16, 3, 32, 36},
        {DType::BFloat16, 17, 16, 120},
        {DType::BFloat16, 17, 32, 120},
        {DType::BFloat16, 64, 16, 402.1},
        {DType::BFloat16, 64, 32, 402.1},
        {DType::BFloat16, 2048, 16, 12309},
        {DType::BFloat16, 2048, 32, 12309},
        {DType::BFloat16, 16, 64, 114},
        {DType::BFloat16, 4092, 16, 24576},
        {DType::BFloat16, 4, 32, 42},
        {DType::BFloat16, 4, 128, 42},
        {DType::BFloat16, 4, 256, 46},
        {DType::BFloat16, 4, 512, 50},
        {DType::BFloat16, 5, 64, 48},
        {DType::BFloat16, 5, 384, 53},
        {DType::BFloat16, 6, 32, 54},
        {DType::BFloat16, 6, 128, 54},
        {DType::BFloat16, 6, 256, 60},
        {DType::BFloat16, 6, 512, 66},
        {DType::BFloat16, 12, 32, 90},
        {DType::BFloat16, 12, 128, 90},
        {DType::BFloat16, 12, 256, 102},
        {DType::BFloat16, 12, 512, 114},
        {DType::BFloat16, 24, 64, 162},
        {DType::BFloat16, 24, 384, 186},
        {DType::BFloat16, 48, 32, 306.1},
        {DType::BFloat16, 48, 128, 306.1},
        {DType::BFloat16, 48, 256, 354.1},
        {DType::BFloat16, 48, 512, 402.1},
        {DType::Float, 1, 128, 35},
        {DType::Float, 1, 512, 41},
        {DType::Float, 1, 2048, 67},
        {DType::Float, 1, 8192, 163},
        {DType::Float, 1, 16384, 291.1},
        {DType::Float, 1, 32768, 547.1},
        {DType::Float, 1, 768, 45},
        {DType::Float, 1, 4096, 99},
        {DType::Float, 1, 12288, 227.1},
        {DType::Float, 1, 32, 34},
        {DType::Float, 1, 96, 35},
        {DType::Float, 1, 32736, 1024.1},
        {DType::Float, 1, 976, 49},
        {DType::Float, 1, 992, 49},
        {DType::Float, 1, 1008, 49},
        {DType::Float, 1, 2000, 68},
        {DType::Float, 1, 2016, 68},
        {DType::Float, 1, 2032, 68},
        {DType::Float, 1, 4048, 128},
        {DType::Float, 1, 4064, 128},
        {DType::Float, 1, 4080, 128},
        {DType::Float, 1, 6096, 192},
        {DType::Float, 1, 6112, 192},
        {DType::Float, 1, 6128, 192},
        {DType::Float, 1, 8144, 256},
        {DType::Float, 1, 8160, 256},
        {DType::Float, 1, 8176, 256},
        {DType::Float, 1, 12240, 384.1},
        {DType::Float, 1, 12256, 384.1},
        {DType::Float, 1, 12272, 384.1},
        {DType::Float, 1, 16336, 512.1},
        {DType::Float, 1, 16352, 512.1},
        {DType::Float, 1, 16368, 512.1},
        {DType::Float, 2, 8, 35},
        {DType::Float, 2, 64, 35},
        {DType::Float, 8, 8, 66},
        {DType::Float, 8, 64, 66},
        {DType::Float, 32, 8, 210.1},
        {DType::Float, 32, 64, 210.1},
        {DType::Float, 128, 8, 786.2},
        {DType::Float, 128, 64, 786.2},
        {DType::Float, 1024, 8, 6163.5},
        {DType::Float, 3, 8, 36},
        {DType::Float, 3, 32, 36},
        {DType::Float, 17, 8, 120},
        {DType::Float, 17, 32, 120},
        {DType::Float, 64, 8, 402.1},
        {DType::Float, 64, 32, 402.1},
        {DType::Float, 2048, 8, 12309},
        {DType::Float, 4, 16, 42},
        {DType::Float, 4, 64, 42},
        {DType::Float, 4, 128, 46},
        {DType::Float, 4, 256, 50},
        {DType::Float, 5, 32, 48},
        {DType::Float, 5, 192, 53},
        {DType::Float, 6, 16, 54},
        {DType::Float, 6, 64, 54},
        {DType::Float, 6, 128, 60},
        {DType::Float, 6, 256, 66},
        {DType::Float, 12, 16, 90},
        {DType::Float, 12, 64, 90},
        {DType::Float, 12, 128, 102},
        {DType::Float, 12, 256, 114},
        {DType::Float, 24, 32, 162},
        {DType::Float, 24, 192, 186},
        {DType::Float, 48, 16, 306.1},
        {DType::Float, 48, 64, 306.1},
        {DType::Float, 48, 128, 354.1},
        {DType::Float, 48, 256, 402.1},
        {DType::Int16, 1, 128, 34},
        {DType::Int16, 1, 512, 37},
        {DType::Int16, 1, 2048, 49},
        {DType::Int16, 1, 8192, 99},
        {DType::Int16, 1, 16384, 163},
        {DType::Int16, 1, 32768, 291.1},
        {DType::Int16, 1, 768, 39},
        {DType::Int16, 1, 4096, 67},
        {DType::Int16, 1, 12288, 131},
        {DType::Int16, 1, 32, 34},
        {DType::Int16, 1, 96, 34},
        {DType::Int16, 1, 16, 34},
        {DType::Int16, 1, 65472, 1024.1},
        {DType::Int16, 1, 1952, 49},
        {DType::Int16, 1, 1984, 49},
        {DType::Int16, 1, 2016, 49},
        {DType::Int16, 1, 4000, 68},
        {DType::Int16, 1, 4032, 68},
        {DType::Int16, 1, 4064, 68},
        {DType::Int16, 1, 8096, 128},
        {DType::Int16, 1, 8128, 128},
        {DType::Int16, 1, 8160, 128},
        {DType::Int16, 1, 12192, 192},
        {DType::Int16, 1, 12224, 192},
        {DType::Int16, 1, 12256, 192},
        {DType::Int16, 1, 16288, 256},
        {DType::Int16, 1, 16320, 256},
        {DType::Int16, 1, 16352, 256},
        {DType::Int16, 1, 24480, 384.1},
        {DType::Int16, 1, 24512, 384.1},
        {DType::Int16, 1, 24544, 384.1},
        {DType::Int16, 1, 32672, 512.1},
        {DType::Int16, 1, 32704, 512.1},
        {DType::Int16, 1, 32736, 512.1},
        {DType::Int16, 2, 16, 35},
        {DType::Int16, 2, 64, 35},
        {DType::Int16, 8, 16, 66},
        {DType::Int16, 8, 64, 66},
        {DType::Int16, 32, 16, 210.1},
        {DType::Int16, 32, 64, 210.1},
        {DType::Int16, 128, 16, 786.2},
        {DType::Int16, 128, 64, 786.2},
        {DType::Int16, 1024, 16, 6163.5},
        {DType::Int16, 1024, 64, 6163.5},
        {DType::Int16, 3, 16, 36},
        {DType::Int16, 3, 32, 36},
        {DType::Int16, 17, 16, 120},
        {DType::Int16, 17, 32, 120},
        {DType::Int16, 64, 16, 402.1},
        {DType::Int16, 64, 32, 402.1},
        {DType::Int16, 2048, 16, 12309},
        {DType::Int16, 2048, 32, 12309.1},
        {DType::Int16, 16, 64, 114},
        {DType::Int16, 4092, 16, 24576},
        {DType::Int16, 4, 32, 42},
        {DType::Int16, 4, 128, 42},
        {DType::Int16, 4, 256, 46},
        {DType::Int16, 4, 512, 50},
        {DType::Int16, 5, 64, 48},
        {DType::Int16, 5, 384, 53},
        {DType::Int16, 6, 32, 54},
        {DType::Int16, 6, 128, 54},
        {DType::Int16, 6, 256, 60},
        {DType::Int16, 6, 512, 66},
        {DType::Int16, 12, 32, 90},
        {DType::Int16, 12, 128, 90},
        {DType::Int16, 12, 256, 102},
        {DType::Int16, 12, 512, 114},
        {DType::Int16, 24, 64, 162},
        {DType::Int16, 24, 384, 186},
        {DType::Int16, 48, 32, 306.1},
        {DType::Int16, 48, 128, 306.1},
        {DType::Int16, 48, 256, 354.1},
        {DType::Int16, 48, 512, 402.1},
        {DType::Int32, 1, 128, 35},
        {DType::Int32, 1, 512, 41},
        {DType::Int32, 1, 2048, 67},
        {DType::Int32, 1, 8192, 163},
        {DType::Int32, 1, 16384, 291.1},
        {DType::Int32, 1, 32768, 547.1},
        {DType::Int32, 1, 768, 45},
        {DType::Int32, 1, 4096, 99},
        {DType::Int32, 1, 12288, 227.1},
        {DType::Int32, 1, 32, 34},
        {DType::Int32, 1, 96, 35},
        {DType::Int32, 1, 32736, 1024.1},
        {DType::Int32, 1, 8, 34},
        {DType::Int32, 1, 976, 49},
        {DType::Int32, 1, 992, 49},
        {DType::Int32, 1, 1008, 49},
        {DType::Int32, 1, 2000, 68},
        {DType::Int32, 1, 2016, 68},
        {DType::Int32, 1, 2032, 68},
        {DType::Int32, 1, 4048, 128},
        {DType::Int32, 1, 4064, 128},
        {DType::Int32, 1, 4080, 128},
        {DType::Int32, 1, 6096, 192},
        {DType::Int32, 1, 6112, 192},
        {DType::Int32, 1, 6128, 192},
        {DType::Int32, 1, 8144, 256},
        {DType::Int32, 1, 8160, 256},
        {DType::Int32, 1, 8176, 256},
        {DType::Int32, 1, 12240, 384.1},
        {DType::Int32, 1, 12256, 384.1},
        {DType::Int32, 1, 12272, 384.1},
        {DType::Int32, 1, 16336, 512.1},
        {DType::Int32, 1, 16352, 512.1},
        {DType::Int32, 1, 16368, 512.1},
        {DType::Int32, 2, 8, 35},
        {DType::Int32, 2, 64, 35},
        {DType::Int32, 8, 8, 66},
        {DType::Int32, 8, 64, 66},
        {DType::Int32, 32, 8, 210.1},
        {DType::Int32, 32, 64, 210.1},
        {DType::Int32, 128, 8, 786.2},
        {DType::Int32, 128, 64, 786.2},
        {DType::Int32, 1024, 8, 6163.5},
        {DType::Int32, 3, 8, 36},
        {DType::Int32, 3, 32, 36},
        {DType::Int32, 17, 8, 120},
        {DType::Int32, 17, 32, 120},
        {DType::Int32, 64, 8, 402.1},
        {DType::Int32, 64, 32, 402.1},
        {DType::Int32, 2048, 8, 12309},
        {DType::Int32, 16, 64, 114},
        {DType::Int32, 4092, 8, 24576},
        {DType::Int32, 4, 16, 42},
        {DType::Int32, 4, 64, 42},
        {DType::Int32, 4, 128, 46},
        {DType::Int32, 4, 256, 50},
        {DType::Int32, 5, 32, 48},
        {DType::Int32, 5, 192, 53},
        {DType::Int32, 6, 16, 54},
        {DType::Int32, 6, 64, 54},
        {DType::Int32, 6, 128, 60},
        {DType::Int32, 6, 256, 66},
        {DType::Int32, 12, 16, 90},
        {DType::Int32, 12, 64, 90},
        {DType::Int32, 12, 128, 102},
        {DType::Int32, 12, 256, 114},
        {DType::Int32, 24, 32, 162},
        {DType::Int32, 24, 192, 186.1},
        {DType::Int32, 48, 16, 306.1},
        {DType::Int32, 48, 64, 306.1},
        {DType::Int32, 48, 128, 354.1},
        {DType::Int32, 48, 256, 402.1},
        {DType::Uint16, 1, 128, 34},
        {DType::Uint16, 1, 512, 37},
        {DType::Uint16, 1, 2048, 49},
        {DType::Uint16, 1, 8192, 99},
        {DType::Uint16, 1, 16384, 163},
        {DType::Uint16, 1, 32768, 291.1},
        {DType::Uint16, 1, 768, 39},
        {DType::Uint16, 1, 4096, 67},
        {DType::Uint16, 1, 12288, 131},
        {DType::Uint16, 1, 32, 34},
        {DType::Uint16, 1, 96, 34},
        {DType::Uint16, 1, 16, 34},
        {DType::Uint16, 1, 65472, 1024.1},
        {DType::Uint16, 1, 1952, 49},
        {DType::Uint16, 1, 1984, 49},
        {DType::Uint16, 1, 2016, 49},
        {DType::Uint16, 1, 4000, 68},
        {DType::Uint16, 1, 4032, 68},
        {DType::Uint16, 1, 4064, 68},
        {DType::Uint16, 1, 8096, 128},
        {DType::Uint16, 1, 8128, 128},
        {DType::Uint16, 1, 8160, 128},
        {DType::Uint16, 1, 12192, 192},
        {DType::Uint16, 1, 12224, 192},
        {DType::Uint16, 1, 12256, 192},
        {DType::Uint16, 1, 16288, 256},
        {DType::Uint16, 1, 16320, 256},
        {DType::Uint16, 1, 16352, 256},
        {DType::Uint16, 1, 24480, 384.1},
        {DType::Uint16, 1, 24512, 384.1},
        {DType::Uint16, 1, 24544, 384.1},
        {DType::Uint16, 1, 32672, 512.1},
        {DType::Uint16, 1, 32704, 512.1},
        {DType::Uint16, 1, 32736, 512.1},
        {DType::Uint16, 2, 16, 35},
        {DType::Uint16, 2, 64, 35},
        {DType::Uint16, 8, 16, 66},
        {DType::Uint16, 8, 64, 66},
        {DType::Uint16, 32, 16, 210.1},
        {DType::Uint16, 32, 64, 210.1},
        {DType::Uint16, 128, 16, 786.2},
        {DType::Uint16, 128, 64, 786.2},
        {DType::Uint16, 1024, 16, 6163.5},
        {DType::Uint16, 1024, 64, 6163.5},
        {DType::Uint16, 3, 16, 36},
        {DType::Uint16, 3, 32, 36},
        {DType::Uint16, 17, 16, 120},
        {DType::Uint16, 17, 32, 120},
        {DType::Uint16, 64, 16, 402.1},
        {DType::Uint16, 64, 32, 402.1},
        {DType::Uint16, 2048, 16, 12309},
        {DType::Uint16, 2048, 32, 12309},
        {DType::Uint16, 16, 64, 114},
        {DType::Uint16, 4092, 16, 24576},
        {DType::Uint16, 4, 32, 42},
        {DType::Uint16, 4, 128, 42},
        {DType::Uint16, 4, 256, 46},
        {DType::Uint16, 4, 512, 50},
        {DType::Uint16, 5, 64, 48},
        {DType::Uint16, 5, 384, 53},
        {DType::Uint16, 6, 32, 54},
        {DType::Uint16, 6, 128, 54},
        {DType::Uint16, 6, 256, 60},
        {DType::Uint16, 6, 512, 66},
        {DType::Uint16, 12, 32, 90},
        {DType::Uint16, 12, 128, 90},
        {DType::Uint16, 12, 256, 102},
        {DType::Uint16, 12, 512, 114},
        {DType::Uint16, 24, 64, 162},
        {DType::Uint16, 24, 384, 186},
        {DType::Uint16, 48, 32, 306.1},
        {DType::Uint16, 48, 128, 306.1},
        {DType::Uint16, 48, 256, 354.1},
        {DType::Uint16, 48, 512, 402.1},
        {DType::Uint32, 1, 128, 35},
        {DType::Uint32, 1, 512, 41},
        {DType::Uint32, 1, 2048, 67},
        {DType::Uint32, 1, 8192, 163},
        {DType::Uint32, 1, 16384, 291.1},
        {DType::Uint32, 1, 32768, 547.1},
        {DType::Uint32, 1, 768, 45},
        {DType::Uint32, 1, 4096, 99},
        {DType::Uint32, 1, 12288, 227.1},
        {DType::Uint32, 1, 32, 34},
        {DType::Uint32, 1, 96, 35},
        {DType::Uint32, 1, 32736, 1024.1},
        {DType::Uint32, 1, 8, 34},
        {DType::Uint32, 1, 976, 49},
        {DType::Uint32, 1, 992, 49},
        {DType::Uint32, 1, 1008, 49},
        {DType::Uint32, 1, 2000, 68},
        {DType::Uint32, 1, 2016, 68},
        {DType::Uint32, 1, 2032, 68},
        {DType::Uint32, 1, 4048, 128},
        {DType::Uint32, 1, 4064, 128},
        {DType::Uint32, 1, 4080, 128},
        {DType::Uint32, 1, 6096, 192},
        {DType::Uint32, 1, 6112, 192},
        {DType::Uint32, 1, 6128, 192},
        {DType::Uint32, 1, 8144, 256},
        {DType::Uint32, 1, 8160, 256},
        {DType::Uint32, 1, 8176, 256},
        {DType::Uint32, 1, 12240, 384.1},
        {DType::Uint32, 1, 12256, 384.1},
        {DType::Uint32, 1, 12272, 384.1},
        {DType::Uint32, 1, 16336, 512.1},
        {DType::Uint32, 1, 16352, 512.1},
        {DType::Uint32, 1, 16368, 512.1},
        {DType::Uint32, 2, 8, 35},
        {DType::Uint32, 2, 64, 35},
        {DType::Uint32, 8, 8, 66},
        {DType::Uint32, 8, 64, 66},
        {DType::Uint32, 32, 8, 210.1},
        {DType::Uint32, 32, 64, 210.1},
        {DType::Uint32, 128, 8, 786.2},
        {DType::Uint32, 128, 64, 786.2},
        {DType::Uint32, 1024, 8, 6163.5},
        {DType::Uint32, 3, 8, 36},
        {DType::Uint32, 3, 32, 36},
        {DType::Uint32, 17, 8, 120},
        {DType::Uint32, 17, 32, 120},
        {DType::Uint32, 64, 8, 402.1},
        {DType::Uint32, 64, 32, 402.1},
        {DType::Uint32, 2048, 8, 12309},
        {DType::Uint32, 16, 64, 114},
        {DType::Uint32, 4092, 8, 24576},
        {DType::Uint32, 4, 16, 42},
        {DType::Uint32, 4, 64, 42},
        {DType::Uint32, 4, 128, 46},
        {DType::Uint32, 4, 256, 50},
        {DType::Uint32, 5, 32, 48},
        {DType::Uint32, 5, 192, 53},
        {DType::Uint32, 6, 16, 54},
        {DType::Uint32, 6, 64, 54},
        {DType::Uint32, 6, 128, 60},
        {DType::Uint32, 6, 256, 66},
        {DType::Uint32, 12, 16, 90},
        {DType::Uint32, 12, 64, 90},
        {DType::Uint32, 12, 128, 102},
        {DType::Uint32, 12, 256, 114},
        {DType::Uint32, 24, 32, 162},
        {DType::Uint32, 24, 192, 186.1},
        {DType::Uint32, 48, 16, 306.1},
        {DType::Uint32, 48, 64, 306.1},
        {DType::Uint32, 48, 128, 354.1},
        {DType::Uint32, 48, 256, 402.1},
    };
    for (const auto& p : profiles) {
        CostModelInput input{};
        input.op = PtoOpcode::TMOV;
        input.arch = CostModelArch::A5;
        input.dtype = p.dtype;
        input.rows = p.rows;
        input.cols = p.cols;
        CostModelResult move{}, scatter{};
        ASSERT_TRUE(EstimateCycles(input, move));
        EXPECT_NEAR(move.cycles, p.measured, p.measured * .10);
        input.op = PtoOpcode::TSCATTER;
        input.has_mask_pattern = true;
        input.mask_pattern = pto::MaskPattern::P1111;
        input.dst_rows = p.rows;
        input.dst_cols = p.cols;
        ASSERT_TRUE(EstimateCycles(input, scatter));
        EXPECT_EQ(move.cycles, scatter.cycles);
    }
}

TEST(A5ComparisonLayoutFit, DenseMoveCalibrationDomainIsolation)
{
    CostModelInput input{};
    input.op = PtoOpcode::TMOV;
    input.arch = CostModelArch::A5;
    input.dtype = DType::Float;
    input.rows = 4;
    input.cols = 128;
    const auto keyFor = [](const CostModelInput& value) {
        a5::fit::VfCurveKey key{};
        uint64_t loops = 0, outer = 0, inner = 0, tail = 0, rowInner = 0;
        EXPECT_TRUE(a5::BuildVfCurveKeyAndLoopCount(value, key, loops, outer, inner, tail, rowInner));
        return key;
    };
    EXPECT_EQ(keyFor(input).op_params, "dense_vec_move");
    EXPECT_EQ(keyFor(input).shape_path, a5::fit::ShapePath::Path2D);
    input.a5_op_params = "explicit_profile";
    EXPECT_EQ(keyFor(input).op_params, "explicit_profile");
    input.a5_op_params = {};
    input.valid_cols = 64;
    EXPECT_NE(keyFor(input).op_params, "dense_vec_move");
    input.valid_cols = 0;
    input.dst_cols = 256;
    EXPECT_NE(keyFor(input).op_params, "dense_vec_move");
    input.dst_cols = 0;
    input.rows = 1024;
    EXPECT_NE(keyFor(input).op_params, "dense_vec_move");
    input.rows = 1;
    input.cols = 1000;
    EXPECT_NE(keyFor(input).op_params, "dense_vec_move");
    input.cols = 65536;
    EXPECT_NE(keyFor(input).op_params, "dense_vec_move");
}

} // namespace
