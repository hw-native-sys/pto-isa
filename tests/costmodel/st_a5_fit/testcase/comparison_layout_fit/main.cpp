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
    // These fp32 pattern keys (including the fp32 TMOV alias) are not calibrated.
    EXPECT_EQ(pto::mocker::GetLastPtoInstrCycles(), 0);
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

} // namespace
