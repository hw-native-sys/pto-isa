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
#include <cmath>
#include <pto/pto-inst.hpp>

namespace {
using namespace pto;
using namespace pto::mocker::lightweight;

TEST(A5DefaultInputs, ConvolutionTilesDoNotInstantiateVectorMetadata)
{
    using Image = ConvTile<TileType::Mat, half, 1024, Layout::NC1HWC0, ConvTileShape<1, 2, 4, 4>>;
    using Weight = ConvTile<TileType::Mat, half, 512, Layout::FRACTAL_Z, ConvTileShape<1, 1, 16, 16>>;
    using Vec = Tile<TileType::Vec, half, 16, 16, BLayout::RowMajor, -1, -1>;
    static_assert(!mocker::IsApproximateDefaultLoad<Image>());
    static_assert(!mocker::IsApproximateDefaultLoad<Weight>());
    static_assert(mocker::IsApproximateDefaultLoad<Vec>());
    Image image;
    Weight weight;
    Vec vec(16, 16);
    EXPECT_EQ(mocker::SliceFitTag(vec, image, 0, 0).key, "unsupported");
    EXPECT_EQ(mocker::SliceFitTag(image, vec, 0, 0).key, "unsupported");
    EXPECT_EQ(mocker::SliceFitTag(vec, weight, 0, 0).key, "unsupported");
    EXPECT_EQ(mocker::SliceFitTag(weight, vec, 0, 0).key, "unsupported");
}

template <typename T, typename I, int Kind, int R = 16, int C = 32 / sizeof(T)>
void CheckIndexedPublic()
{
    constexpr int SC = Kind == 1 ? 32 / sizeof(T) : C;
    constexpr int SR = Kind == 1 ? 1 : R;
    constexpr int IC = Kind == 3 ? 64 : C;
    using D = Tile<TileType::Vec, T, R, C, BLayout::RowMajor, -1, -1>;
    using S = Tile<TileType::Vec, T, SR, SC, BLayout::RowMajor, -1, -1>;
    using IX = Tile<TileType::Vec, I, R, IC, BLayout::RowMajor, -1, -1>;
    D dst(R, C);
    S src(SR, SC);
    IX index(R, IC);
    auto& ctx = perf_sim::GetCostModelCtx();
    const auto saved = ctx;
    ctx.npu_arch = perf_sim::kAscend950Arch;
    ctx.assume_default_inputs = true;
    TASSIGN(index, 0);
    TASSIGN(src, R * IC * sizeof(I));
    TASSIGN(dst, R * IC * sizeof(I) + SR * SC * sizeof(T));
    constexpr auto profile = Kind == 3 ? DefaultInputProfile::IdentityBlocks : DefaultInputProfile::ZeroIndex;
    CostModelInput input{
        .op = Kind == 1 ? PtoOpcode::TGATHER :
              Kind == 2 ? PtoOpcode::TSCATTER :
                          PtoOpcode::TGATHERB,
        .dtype = perf_sim::MapDType(mocker::CostModelTileDTypeString<D>()),
        .rows = R,
        .cols = C,
        .arch = CostModelArch::A5};
    input.default_input = mocker::DefaultIndexedGeometryTag<profile>(dst, src, index);
    const auto expected = EstimateCycles(input);
    ASSERT_GT(expected.cycles, 0);
    const auto call = [&] {
        if constexpr (Kind == 1)
            TGATHER(dst, src, index, dst);
        else if constexpr (Kind == 2)
            TSCATTER(dst, src, index);
        else
            TGATHERB(dst, src, index);
    };
    call();
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), expected.cycles);
    EXPECT_EQ(ctx.last_assumed_input_profile, profile);
    ctx.assume_default_inputs = false;
    call();
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    EXPECT_EQ(ctx.last_assumed_input_profile, DefaultInputProfile::None);
    ctx.assume_default_inputs = true;
    TASSIGN(dst, 262144);
    call();
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    ctx = saved;
}

TEST(A5DefaultInputs, PublicIndexedRoutesAndStrictOptOut)
{
    CheckIndexedPublic<half, uint16_t, 1>();
    CheckIndexedPublic<half, uint32_t, 1>();
    CheckIndexedPublic<float, uint32_t, 1>();
    CheckIndexedPublic<int16_t, int16_t, 1>();
    CheckIndexedPublic<uint32_t, uint32_t, 1>();
    CheckIndexedPublic<half, uint16_t, 2>();
    CheckIndexedPublic<uint8_t, uint16_t, 2>();
    CheckIndexedPublic<float, uint32_t, 2>();
    CheckIndexedPublic<int8_t, uint16_t, 2>();
    CheckIndexedPublic<int32_t, int32_t, 2>();
    CheckIndexedPublic<half, uint32_t, 3>();
    CheckIndexedPublic<uint8_t, uint32_t, 3>();
    CheckIndexedPublic<float, uint32_t, 3>();
    CheckIndexedPublic<half, uint16_t, 1, 1, 256>();
    CheckIndexedPublic<half, uint32_t, 1, 1, 256>();
    CheckIndexedPublic<float, uint32_t, 1, 1, 256>();
    CheckIndexedPublic<half, uint16_t, 2, 1, 256>();
    CheckIndexedPublic<uint8_t, uint16_t, 2, 1, 256>();
    CheckIndexedPublic<float, uint32_t, 2, 1, 256>();
}

TEST(A5DefaultInputs, RejectUnknownOrUnmeasuredGeometry)
{
    using D = Tile<TileType::Vec, half, 16, 16, BLayout::RowMajor, -1, -1>;
    using S = Tile<TileType::Vec, half, 1, 16, BLayout::RowMajor, -1, -1>;
    using I = Tile<TileType::Vec, uint16_t, 16, 16, BLayout::RowMajor, -1, -1>;
    D dst(16, 16);
    S src(1, 16);
    I index(16, 16);
    CostModelInput input{
        .op = PtoOpcode::TGATHER, .dtype = DType::Half, .rows = 16, .cols = 16, .arch = CostModelArch::A5};
    input.default_input = mocker::DefaultIndexedGeometryTag<DefaultInputProfile::ZeroIndex>(dst, src, index);
    CostModelResult result;
    EXPECT_FALSE(EstimateCycles(input, result));
    TASSIGN(index, 0);
    TASSIGN(src, 512);
    TASSIGN(dst, 544);
    input.default_input = mocker::DefaultIndexedGeometryTag<DefaultInputProfile::ZeroIndex>(dst, src, index);
    ASSERT_TRUE(EstimateCycles(input, result));
    const auto valid = input;
    input.default_input.profile = DefaultInputProfile::None;
    EXPECT_FALSE(EstimateCycles(input, result));
    input = valid;
    input.default_input.dst.valid_cols = 8;
    EXPECT_FALSE(EstimateCycles(input, result));
    input = valid;
    input.default_input.dst.address += 32;
    EXPECT_FALSE(EstimateCycles(input, result));
    input = valid;
    input.default_input.index_dtype = DType::Float;
    EXPECT_FALSE(EstimateCycles(input, result));
    input = valid;
    input.default_input.runtime_valid_shape = false;
    EXPECT_FALSE(EstimateCycles(input, result));
    input = valid;
    input.default_input.supported_layout = false;
    EXPECT_FALSE(EstimateCycles(input, result));
    input = valid;
    input.op = PtoOpcode::TLOAD;
    EXPECT_FALSE(EstimateCycles(input, result));
}

template <typename T>
void CheckApproximateLoadPublic(uint64_t expected)
{
    using D = Tile<TileType::Vec, T, 16, 16, BLayout::RowMajor, -1, -1>;
    using G = GlobalTensor<T, Shape<1, 1, 1, 16, 16>, Stride<256, 256, 256, 16, 1>>;
    D dst(16, 16);
    G src(reinterpret_cast<T*>(0x100000));
    TASSIGN(dst, 0);
    auto& ctx = perf_sim::GetCostModelCtx();
    const auto saved = ctx;
    ctx.npu_arch = perf_sim::kAscend950Arch;
    ctx.assume_default_inputs = true;
    TLOAD(dst, src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), expected);
    EXPECT_EQ(ctx.last_assumed_input_profile, DefaultInputProfile::WarmSameAddressApproximate);
    TLOAD<TLoadL2Hint::NormalFirstVictim>(dst, src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), expected);
    TLOAD<TLoadL2Hint::NotAllocDrop>(dst, src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    G misaligned(reinterpret_cast<T*>(0x100020)); // UB alignment is not the measured GM alignment.
    TLOAD(dst, misaligned);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    G unknown(nullptr);
    TLOAD(dst, unknown);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    using BadStride = GlobalTensor<T, Shape<1, 1, 1, 16, 16>, Stride<512, 512, 512, 32, 1>>;
    BadStride padded(reinterpret_cast<T*>(0x100000));
    TLOAD(dst, padded);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    using BadShape = GlobalTensor<T, Shape<1, 1, 1, 32, 16>, Stride<512, 512, 512, 16, 1>>;
    BadShape larger(reinterpret_cast<T*>(0x100000));
    TLOAD(dst, larger);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    ctx.assume_default_inputs = false;
    TLOAD(dst, src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    EXPECT_EQ(ctx.last_assumed_input_profile, DefaultInputProfile::None);
    ctx.assume_default_inputs = true;
    TASSIGN(dst, 32);
    TLOAD(dst, src);
    EXPECT_EQ(mocker::GetLastPtoInstrCycles(), 0);
    ctx = saved;
}

TEST(A5DefaultInputs, ProvisionalLoadRoutesAndBoundaries)
{
    CheckApproximateLoadPublic<half>(153);
    CheckApproximateLoadPublic<float>(177);
    CheckApproximateLoadPublic<int32_t>(177);
}

TEST(A5DefaultInputs, ProvisionalLoadKnownAccuracyGapIsNotAnAccuracyPass)
{
    using D = Tile<TileType::Vec, half, 16, 16, BLayout::RowMajor, -1, -1>;
    using G = GlobalTensor<half, Shape<1, 1, 1, 16, 16>, Stride<256, 256, 256, 16, 1>>;
    D dst(16, 16);
    TASSIGN(dst, 0);
    G src(reinterpret_cast<half*>(0x100000));
    CostModelInput input{
        .op = PtoOpcode::TLOAD, .dtype = DType::Half, .rows = 16, .cols = 16, .arch = CostModelArch::A5};
    input.default_input = mocker::DefaultLoadGeometryTag<TLoadL2Hint::NormalFirstVictim>(dst, src);
    CostModelResult result;
    ASSERT_TRUE(EstimateCycles(input, result));
    EXPECT_EQ(result.cycles, 153);
    EXPECT_GT(std::abs(double(result.cycles) / 173.39183202449803 - 1.0), 0.1);
    for (const auto dtype : {DType::Float, DType::Int32}) {
        input.dtype = dtype;
        ASSERT_TRUE(EstimateCycles(input, result));
        EXPECT_EQ(result.cycles, 177);
        EXPECT_GT(std::abs(double(result.cycles) / 196.73759924641763 - 1.0), 0.1);
    }
    input.dtype = DType::Uint32;
    EXPECT_FALSE(EstimateCycles(input, result));
    input.dtype = DType::Half;
    input.default_input.profile = DefaultInputProfile::ZeroIndex;
    EXPECT_FALSE(EstimateCycles(input, result));
}

} // namespace
