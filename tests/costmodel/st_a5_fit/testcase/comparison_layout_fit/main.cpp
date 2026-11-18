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
#include <vector>

#include <pto/pto-inst.hpp>
#include <pto/costmodel/lightweight_costmodel.hpp>

#include "a5_cost_check.hpp"

namespace {

using namespace ::pto::mocker::lightweight;

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
    if constexpr (!Column && (Pattern == MaskPattern::P0001 || Pattern == MaskPattern::P0101)) {
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
    EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 0);
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

} // namespace
