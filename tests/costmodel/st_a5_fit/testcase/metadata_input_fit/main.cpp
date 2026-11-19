/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <array>
#include <limits>
#include <type_traits>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <pto/pto-inst.hpp>

namespace {
using namespace pto;
using namespace pto::mocker::lightweight;
using DynamicTile =
    Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;

TEST(A5MetadataInputs, ComparisonCodegenScopesAndSnapshots)
{
    auto& ctx = perf_sim::GetCostModelCtx();
    const auto saved = ctx;
    ctx = {};
    ctx.npu_arch = perf_sim::kAscend950Arch;
    using S = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1>;
    using D = Tile<TileType::Vec, uint8_t, 16, 128, BLayout::RowMajor, -1, -1>;
    S a(16, 128), b(16, 128);
    D d(16, 128);
    TASSIGN(d, 0);
    TASSIGN(a, 32768);
    TASSIGN(b, 65536);
    MetadataGeometry folded, runtime;
    EXPECT_EQ(ctx.comparison_codegen, ComparisonCodegen::Unknown);
    try {
        perf_sim::ScopedComparisonCodegen scope(ComparisonCodegen::RuntimeDispatch);
        throw 1;
    } catch (int) {
        EXPECT_EQ(ctx.comparison_codegen, ComparisonCodegen::Unknown);
    }
    {
        perf_sim::ScopedComparisonCodegen outer(ComparisonCodegen::ConstantFolded);
        folded = mocker::ComparisonGeometryTag(d, a, b, CmpMode::EQ);
        TCMP(d, a, b, CmpMode::EQ);
        EXPECT_NEAR(mocker::GetLastPtoInstrCycles(), 130, 13);
        TCMPS(d, a, 2.0f, CmpMode::EQ);
        EXPECT_NEAR(mocker::GetLastPtoInstrCycles(), 130, 13);
        {
            perf_sim::ScopedComparisonCodegen inner(ComparisonCodegen::RuntimeDispatch);
            runtime = mocker::ComparisonGeometryTag(d, a, b, CmpMode::EQ);
            TCMP(d, a, b, CmpMode::EQ);
            EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
            TCMPS(d, a, 2.0f, CmpMode::EQ);
            EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
        }
        EXPECT_EQ(ctx.comparison_codegen, ComparisonCodegen::ConstantFolded);
        bool isolated = false;
        std::thread other([&] {
            isolated = perf_sim::GetCostModelCtx().comparison_codegen == ComparisonCodegen::Unknown;
            perf_sim::ScopedComparisonCodegen own(ComparisonCodegen::RuntimeDispatch);
        });
        other.join();
        EXPECT_TRUE(isolated);
        EXPECT_EQ(ctx.comparison_codegen, ComparisonCodegen::ConstantFolded);
    }
    EXPECT_EQ(ctx.comparison_codegen, ComparisonCodegen::Unknown);
    CostModelInput input{
        .op = PtoOpcode::TCMP, .dtype = DType::Uint8, .rows = 16, .cols = 32, .arch = CostModelArch::A5};
    input.metadata = folded;
    uint64_t cycles = 0;
    EXPECT_TRUE(a5::TryEstimateA5MetadataCycles(input, cycles));
    EXPECT_NEAR(cycles, 130, 13);
    input.metadata = runtime;
    EXPECT_FALSE(a5::TryEstimateA5MetadataCycles(input, cycles));
    EXPECT_EQ(cycles, 0);
    TCMP(d, a, b, CmpMode::EQ);
    EXPECT_GT(mocker::GetLastPtoInstrCycles(), 0); // Existing equal-width legacy route.
    ctx = saved;
}

template <typename T, int Rows, int Cols>
void CheckExplicitFoldedComparison(double measuredBinary, double measuredScalar)
{
    using S = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using D = Tile<TileType::Vec, uint8_t, Rows, Cols, BLayout::RowMajor, -1, -1>;
    S a(Rows, Cols), b(Rows, Cols);
    D d(Rows, Cols);
    TASSIGN(d, 0);
    TASSIGN(a, 65536);
    TASSIGN(b, 131072);
    perf_sim::ScopedComparisonCodegen cfg(ComparisonCodegen::ConstantFolded);
    TCMP(d, a, b, CmpMode::EQ);
    EXPECT_NEAR(mocker::GetLastPtoInstrCycles(), measuredBinary, measuredBinary * .10);
    TCMPS(d, a, T(2), CmpMode::EQ);
    EXPECT_NEAR(mocker::GetLastPtoInstrCycles(), measuredScalar, measuredScalar * .10);
    // Declaring constant folding does not extend the measured mode or RHS domain.
    for (const auto mode : {CmpMode::EQ, CmpMode::NE, CmpMode::GT, CmpMode::GE, CmpMode::LT, CmpMode::LE}) {
        TCMPS(d, a, b, mode);
        EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
        if (mode != CmpMode::EQ) {
            TCMP(d, a, b, mode);
            EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
            TCMPS(d, a, T(2), mode);
            EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
        }
    }
    TCMPS(d, a, T(3), CmpMode::EQ);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    d.SetValidShape(Rows, Cols / 8 - 1);
    TCMP(d, a, b, CmpMode::EQ);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    d.SetValidShape(Rows, Cols);
    using Packed = Tile<TileType::Vec, uint8_t, Rows, 32, BLayout::RowMajor, -1, -1>;
    Packed packed(Rows, Cols / 8);
    TASSIGN(packed, 0);
    TCMP(packed, a, b, CmpMode::EQ);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    TASSIGN(a, 1);
    TCMPS(d, a, T(2), CmpMode::EQ);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
}

TEST(A5MetadataInputs, ExplicitFoldedComparisonUsesPreHoistCalibration)
{
    auto& ctx = perf_sim::GetCostModelCtx();
    const auto saved = ctx;
    ctx = {};
    ctx.npu_arch = perf_sim::kAscend950Arch;
    // Existing pre-hoist hardware profiles, reused as regression evidence;
    // these are not new independent validation or runtime-dispatch fits.
    CheckExplicitFoldedComparison<float, 16, 128>(130, 130);
    CheckExplicitFoldedComparison<int32_t, 16, 128>(130, 130);
    CheckExplicitFoldedComparison<float, 16, 256>(182, 182);
    CheckExplicitFoldedComparison<int32_t, 16, 256>(182, 182);
    ctx = saved;
}

TEST(A5MetadataInputs, ExplicitComparisonNeverFallsBackToLegacy)
{
    auto& ctx = perf_sim::GetCostModelCtx();
    const auto saved = ctx;
    ctx = {};
    ctx.npu_arch = perf_sim::kAscend950Arch;
    using S = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1>;
    using D = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;
    S a(1, 192), b(1, 192);
    D d(1, 24);
    std::vector<float> source(S::Numel), other(S::Numel);
    std::vector<uint8_t> dest(D::Numel);
    TASSIGN(a, reinterpret_cast<uintptr_t>(source.data()));
    TASSIGN(b, reinterpret_cast<uintptr_t>(other.data()));
    TASSIGN(d, reinterpret_cast<uintptr_t>(dest.data()));
    for (const auto path :
         {ComparisonCodegen::ConstantFolded, ComparisonCodegen::RuntimeDispatch, static_cast<ComparisonCodegen>(99)}) {
        perf_sim::ScopedComparisonCodegen cfg(path);
        TCMP(d, a, b, CmpMode::EQ);
        EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
        TCMPS(d, a, 2.0f, CmpMode::EQ);
        EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    }
    ctx.comparison_codegen = ComparisonCodegen::ConstantFolded;
    ctx = {};
    EXPECT_EQ(ctx.comparison_codegen, ComparisonCodegen::Unknown);
    ctx = saved;
}

TEST(A5MetadataInputs, ExplicitAddressAndNestedGeometry)
{
    DynamicTile dst(16, 128), first(8, 64), second(16, 128);
    CostModelInput input{
        .op = PtoOpcode::TPARTADD, .dtype = DType::Float, .rows = 16, .cols = 128, .arch = CostModelArch::A5};
    std::array<double, 6> work{};
    std::string_view family;
    input.metadata = mocker::PartialGeometryTag(dst, first, second);
    EXPECT_FALSE(a5::BuildMetadataFeatures(input, work, family));
    TASSIGN(dst, 0);
    TASSIGN(first, 8192);
    TASSIGN(second, 16384);
    EXPECT_TRUE(dst.HasCostModelAssignedAddress());
    EXPECT_EQ(dst.GetCostModelAssignedAddress(), 0);
    input.metadata = mocker::PartialGeometryTag(dst, first, second);
    ASSERT_TRUE(a5::BuildMetadataFeatures(input, work, family));
    EXPECT_EQ(work, (std::array<double, 6>{8, 24, 8, 8, 1, 0}));
    for (auto address : {4096u, 8193u, 262144u}) {
        TASSIGN(first, address);
        input.metadata = mocker::PartialGeometryTag(dst, first, second);
        EXPECT_FALSE(a5::BuildMetadataFeatures(input, work, family));
    }
    TASSIGN(first, 8192);
    second.SetValidShape(16, 32);
    input.metadata = mocker::PartialGeometryTag(dst, first, second);
    EXPECT_FALSE(a5::BuildMetadataFeatures(input, work, family));
    second.SetValidShape(16, 128);
    first.SetValidShape(0, 128);
    input.metadata = mocker::PartialGeometryTag(dst, first, second);
    ASSERT_TRUE(a5::BuildMetadataFeatures(input, work, family));
    EXPECT_EQ(family, "metadata_partial_empty_v1");
    first.SetValidShape(16, 128);
    input.metadata = mocker::PartialGeometryTag(dst, first, second);
    ASSERT_TRUE(a5::BuildMetadataFeatures(input, work, family));
    EXPECT_EQ(family, "metadata_partial_equal_v1");
    input.metadata.runtime_valid_shape = false;
    EXPECT_FALSE(a5::BuildMetadataFeatures(input, work, family));
}

TEST(A5MetadataInputs, PreserveLegacyHalfPaddingRoute)
{
    auto& ctx = perf_sim::GetCostModelCtx();
    const auto saved = ctx;
    ctx.npu_arch = perf_sim::kAscend950Arch;
    using D = Tile<TileType::Vec, half, 1, 256, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    using S = Tile<TileType::Vec, half, 1, 256, BLayout::RowMajor, -1, -1>;
    using E = Tile<TileType::Vec, half, 1, 240, BLayout::RowMajor, -1, -1>;
    D dst(1, 240);
    S src(1, 240);
    E expandedSrc(1, 240);
    TASSIGN(dst, 0);
    TASSIGN(src, 32768);
    TASSIGN(expandedSrc, 32768);
    // Compatibility values from the unchanged canonical public path, not new
    // measurements and not evidence for general fp16 padding geometry.
    TFILLPAD(dst, src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 37);
    TFILLPAD_EXPAND(dst, expandedSrc);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 37);
    ctx = saved;
}

TEST(A5MetadataInputs, PublicPartialRejectsUnmeasuredInputs)
{
    auto& ctx = perf_sim::GetCostModelCtx();
    const auto saved = ctx;
    ctx.npu_arch = perf_sim::kAscend950Arch;
    DynamicTile dst(16, 128), a(8, 64), b(16, 128);
    TPARTADD(dst, a, b);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    TASSIGN(dst, 0);
    TASSIGN(a, 8192);
    TASSIGN(b, 16384);
    TPARTADD(dst, a, b);
    EXPECT_GT(mocker::GetLastPtoInstrCycles(), 0);
    TASSIGN(a, 4096);
    TPARTADD(dst, a, b);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    TASSIGN(a, 8192);
    b.SetValidShape(16, 32);
    TPARTADD(dst, a, b);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    b.SetValidShape(16, 128);
    a.SetValidShape(8, 0);
    TPARTADD(dst, a, b);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    using Static = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, 16, 128>;
    Static sd, sa, sb;
    TASSIGN(sd, 0);
    TASSIGN(sa, 8192);
    TASSIGN(sb, 16384);
    TPARTADD(sd, sa, sb);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    using Unsigned = Tile<TileType::Vec, uint16_t, 16, 128, BLayout::RowMajor, -1, -1>;
    Unsigned ud(16, 128), ua(8, 64), ub(16, 128);
    TASSIGN(ud, 0);
    TASSIGN(ua, 8192);
    TASSIGN(ub, 16384);
    // uint16 is also rejected by the mock device compiler, so exercise the
    // formula boundary without instantiating that unsupported device API.
    CostModelInput unsupported{
        .op = PtoOpcode::TPARTADD, .dtype = DType::Uint16, .rows = 16, .cols = 128, .arch = CostModelArch::A5};
    unsupported.metadata = mocker::PartialGeometryTag(ud, ua, ub);
    uint64_t cycles = 0;
    EXPECT_FALSE(a5::TryEstimateA5MetadataCycles(unsupported, cycles));
    ctx = saved;
}

TEST(A5MetadataInputs, PublicPaddingRejectsUnmeasuredInputs)
{
    auto& ctx = perf_sim::GetCostModelCtx();
    const auto saved = ctx;
    ctx.npu_arch = perf_sim::kAscend950Arch;
    DynamicTile dst(16, 128), src(8, 64);
    TFILLPAD(dst, src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    TASSIGN(dst, 0);
    TASSIGN(src, 8192);
    TFILLPAD(dst, src);
    EXPECT_GT(mocker::GetLastPtoInstrCycles(), 0);
    dst.SetValidShape(8, 128);
    TFILLPAD(dst, src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    dst.SetValidShape(16, 128);
    src.SetValidShape(0, 64);
    TFILLPAD(dst, src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    using Nonzero =
        Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Max>;
    Nonzero other(16, 128);
    TASSIGN(other, 0);
    src.SetValidShape(8, 64);
    TFILLPAD(other, src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    ctx = saved;
}

TEST(A5MetadataInputs, PublicMinimumRejectsUnmeasuredInputs)
{
    auto& ctx = perf_sim::GetCostModelCtx();
    const auto saved = ctx;
    ctx.npu_arch = perf_sim::kAscend950Arch;
    DynamicTile dst(16, 128), a(8, 64), b(16, 128);
    TPARTMIN(dst, a, b);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    TASSIGN(dst, 0);
    TASSIGN(a, 8192);
    TASSIGN(b, 16384);
    TPARTMIN(dst, a, b);
    EXPECT_GT(mocker::GetLastPtoInstrCycles(), 0);
    TASSIGN(a, 4096);
    TPARTMIN(dst, a, b);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    TASSIGN(a, 8192);
    b.SetValidShape(16, 32);
    TPARTMIN(dst, a, b);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    b.SetValidShape(16, 128);
    a.SetValidShape(8, 0);
    TPARTMIN(dst, a, b);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    using Static = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, 16, 128>;
    Static sd, sa, sb;
    TASSIGN(sd, 0);
    TASSIGN(sa, 8192);
    TASSIGN(sb, 16384);
    TPARTMIN(sd, sa, sb);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    ctx = saved;
}

TEST(A5MetadataInputs, PublicSelectRejectsUnknownStorageAndMaskGeometry)
{
    auto& ctx = perf_sim::GetCostModelCtx();
    const auto saved = ctx;
    ctx.npu_arch = perf_sim::kAscend950Arch;
    using Data = Tile<TileType::Vec, uint32_t, 12, 96, BLayout::RowMajor, -1, -1>;
    using Mask = Tile<TileType::Vec, uint8_t, 12, 32, BLayout::RowMajor, -1, -1>;
    Data dst(12, 96), a(12, 96), b(12, 96);
    Mask mask(12, 32), tmp(12, 32);
    auto predict = [&]() {
        TSEL(dst, mask, a, b, tmp);
        return mocker::GetLastPtoInstrCycles();
    };
    EXPECT_EQ(predict(), 0);
    TASSIGN(dst, 0);
    TASSIGN(a, 4608);
    TASSIGN(b, 9216);
    TASSIGN(mask, 13824);
    TASSIGN(tmp, 13824);
    EXPECT_EQ(predict(), 115);
    TASSIGN(mask, 9216);
    EXPECT_EQ(predict(), 0);
    TASSIGN(mask, 13825);
    EXPECT_EQ(predict(), 0);
    TASSIGN(mask, 262144);
    EXPECT_EQ(predict(), 0);
    TASSIGN(mask, 13824);
    mask.SetValidShape(12, 12);
    EXPECT_EQ(predict(), 0);
    mask.SetValidShape(12, 32);
    a.SetValidShape(11, 96);
    EXPECT_EQ(predict(), 0);
    a.SetValidShape(12, 96);
    TASSIGN(b, 4608);
    EXPECT_EQ(predict(), 0);
    TASSIGN(b, 9216);
    EXPECT_EQ(predict(), 115);
    using Static = Tile<TileType::Vec, uint32_t, 12, 96, BLayout::RowMajor, 12, 96>;
    Static sd, sa, sb;
    TASSIGN(sd, 0);
    TASSIGN(sa, 4608);
    TASSIGN(sb, 9216);
    TSEL(sd, mask, sa, sb, tmp);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    using Large = Tile<TileType::Vec, uint32_t, 16, 96, BLayout::RowMajor, -1, -1>;
    Large ld(16, 96), la(16, 96), lb(16, 96);
    TASSIGN(ld, 0);
    TASSIGN(la, 6144);
    TASSIGN(lb, 12288);
    TSEL(ld, mask, la, lb, tmp);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    ctx = saved;
}

TEST(A5MetadataInputs, TriangleDirectionDiagonalAndStorage)
{
    auto& ctx = perf_sim::GetCostModelCtx();
    const auto saved = ctx;
    ctx.npu_arch = perf_sim::kAscend950Arch;
    using D = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, -1, -1>;
    D dst(16, 128);
    TTRI<D, 0>(dst, 0);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    TASSIGN(dst, 0);
    TTRI<D, 0>(dst, 0);
    const auto full = mocker::GetLastPtoInstrCycles();
    EXPECT_GT(full, 0);
    TTRI<D, 0>(dst, -16);
    const auto empty = mocker::GetLastPtoInstrCycles();
    EXPECT_LT(empty, full);
    TTRI<D, 0>(dst, std::numeric_limits<int>::min());
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), empty);
    TTRI<D, 1>(dst, 128);
    const auto upper = mocker::GetLastPtoInstrCycles();
    EXPECT_GT(upper, empty);
    TTRI<D, 1>(dst, std::numeric_limits<int>::max());
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), upper);
    // The public device API rejects invalid directions at compile time.
    // Check the metadata boundary without instantiating that invalid API.
    EXPECT_FALSE(mocker::TriangleGeometryTag<2>(dst, 0).supported_layout);
    dst.SetValidShape(15, 128);
    TTRI<D, 0>(dst, 0);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    dst.SetValidShape(16, 128);
    TASSIGN(dst, 1);
    TTRI<D, 0>(dst, 0);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    TASSIGN(dst, 262144);
    TTRI<D, 0>(dst, 0);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    using Static = Tile<TileType::Vec, float, 16, 128, BLayout::RowMajor, 16, 128>;
    Static fixed;
    TASSIGN(fixed, 0);
    TTRI<Static, 0>(fixed, 0);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    ctx = saved;
}

template <class T, bool Binary>
void CheckUnidentifiedNativeComparison()
{
    using S = Tile<TileType::Vec, T, 12, 192, BLayout::RowMajor, -1, -1>;
    using D = Tile<TileType::Vec, uint8_t, 12, 32, BLayout::RowMajor, -1, -1>;
    S a(12, 192), b(12, 192);
    D dst(12, 24);
    TASSIGN(dst, 0);
    TASSIGN(a, 512);
    TASSIGN(b, 512 + 12 * 192 * sizeof(T));
    T scalar;
    if constexpr (std::is_same_v<T, bfloat16_t> && !std::is_same_v<T, half>) {
        scalar = std::bit_cast<T>(uint16_t(0x4040)); // BF16 3.0
    } else {
        scalar = T(3);
    }
    for (int mode = 0; mode < 6; ++mode) {
        if constexpr (Binary) {
            TCMP(dst, a, b, static_cast<CmpMode>(mode));
            EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
        }
        TCMPS(dst, a, scalar, static_cast<CmpMode>(mode));
        EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
        TCMPS(dst, a, b, static_cast<CmpMode>(mode));
        EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    }
    // A literal mode in the host compilation is not proof of NPU constant folding.
    if constexpr (Binary) {
        TCMP(dst, a, b, CmpMode::EQ);
        EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    }
    TCMPS(dst, a, scalar, CmpMode::EQ);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    TCMPS(dst, a, b, CmpMode::EQ);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    CostModelInput input;
    input.op = PtoOpcode::TCMPS;
    input.dtype = perf_sim::MapDType(mocker::CostModelTileDTypeString<S>());
    input.metadata = mocker::ComparisonGeometryTag(dst, a, b, CmpMode::EQ);
    uint64_t cycles = 123;
    EXPECT_FALSE(a5::TryEstimateA5MetadataCycles(input, cycles));
    EXPECT_EQ(cycles, 0);
}

TEST(A5MetadataInputs, ComparisonsWithoutNativeCodegenEvidenceAreUnsupported)
{
    auto& ctx = perf_sim::GetCostModelCtx();
    const auto saved = ctx;
    ctx.npu_arch = perf_sim::kAscend950Arch;
    CheckUnidentifiedNativeComparison<float, true>();
    CheckUnidentifiedNativeComparison<int32_t, true>();
    CheckUnidentifiedNativeComparison<bfloat16_t, true>();
    CheckUnidentifiedNativeComparison<int8_t, false>();
    CheckUnidentifiedNativeComparison<uint8_t, false>();
    CheckUnidentifiedNativeComparison<uint16_t, false>();
    CheckUnidentifiedNativeComparison<uint32_t, false>();
    ctx = saved;
}

TEST(A5MetadataInputs, LegacyComparisonHostBuffersRemainLimited)
{
    auto& ctx = perf_sim::GetCostModelCtx();
    const auto saved = ctx;
    ctx.npu_arch = perf_sim::kAscend950Arch;
    using S = Tile<TileType::Vec, float, 1, 192, BLayout::RowMajor, -1, -1>;
    using D = Tile<TileType::Vec, uint8_t, 1, 32, BLayout::RowMajor, -1, -1>;
    S src(1, 192);
    D dst(1, 24);
    std::vector<float> source(S::Numel);
    std::vector<uint8_t> output(D::Numel);
    TASSIGN(src, reinterpret_cast<uintptr_t>(source.data()));
    TASSIGN(dst, reinterpret_cast<uintptr_t>(output.data()));
    // Compatibility value from the pre-existing mock path, not a new sample.
    TCMPS(dst, src, 1.f, CmpMode::EQ);
    EXPECT_NEAR(mocker::GetLastPtoInstrCycles(), 46., 4.6);
    TCMPS(dst, src, 1.f, CmpMode::NE);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    TCMPS(dst, src, 3.f, CmpMode::EQ);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    ctx = saved;
}

#if defined(PTO_COSTMODEL_NATIVE_BFLOAT16)
TEST(A5MetadataInputs, BfloatTypeIdentity)
{
    auto& ctx = perf_sim::GetCostModelCtx();
    const auto saved = ctx;
    ctx.npu_arch = perf_sim::kAscend950Arch;
    static_assert(!std::is_same_v<bfloat16_t, half>);
    using Data = Tile<TileType::Vec, bfloat16_t, 12, 96, BLayout::RowMajor, -1, -1>;
    EXPECT_EQ(perf_sim::MapDType(mocker::CostModelTileDTypeString<Data>()), DType::BFloat16);
    ctx = saved;
}
#endif
} // namespace
