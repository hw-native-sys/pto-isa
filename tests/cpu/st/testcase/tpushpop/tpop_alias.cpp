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
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <pto/pto-inst.hpp>

namespace {
using namespace pto;
using SourceTile = Tile<TileType::Vec, float, 2, 128>;
using RowTile = Tile<TileType::Vec, float, 1, 128>;

TEST(TPopAliasTest, base_and_interior_share_storage)
{
    SourceTile source;
    cpu_pipe::EnsureTileStorage(source);
    auto* original = source.data();
    ASSERT_NE(original, nullptr);
    cpu_pipe::EnsureTileStorage(source);
    ASSERT_EQ(source.data(), original);

    SourceTile alias;
    TASSIGN(alias, reinterpret_cast<std::uintptr_t>(original));
    ASSERT_EQ(alias.data(), original);
    RowTile row;
    TASSIGN(row, reinterpret_cast<std::uintptr_t>(original + SourceTile::Cols));
    ASSERT_EQ(row.data(), original + SourceTile::Cols);
    row.data()[0] = 17.0f;
    EXPECT_FLOAT_EQ(alias.data()[SourceTile::Cols], 17.0f);
    source.data()[SourceTile::Numel - 1] = 29.0f;
    EXPECT_FLOAT_EQ(row.data()[RowTile::Numel - 1], 29.0f);
}

template <TileType Location>
void checkRegionAliases()
{
    using RegionTile = Tile<Location, float, 2, 128>;
    using RegionRow = Tile<Location, float, 1, 128>;
    auto& memory = NPUMemoryModel::Instance();
    RegionTile source;
    TASSIGN(source, 512);
    EXPECT_EQ(source.data(), memory.GetPointer<RegionTile>(512));
    RegionTile alias;
    TASSIGN(alias, reinterpret_cast<std::uintptr_t>(source.data()));
    ASSERT_EQ(alias.data(), source.data());
    RegionRow row;
    TASSIGN(row, reinterpret_cast<std::uintptr_t>(source.data() + RegionTile::Cols));
    ASSERT_EQ(row.data(), source.data() + RegionTile::Cols);
    cpu_pipe::EnsureTileStorage(source);
    EXPECT_EQ(source.data(), alias.data());
    row.data()[0] = 31.0f;
    EXPECT_FLOAT_EQ(source.data()[RegionTile::Cols], 31.0f);
}

TEST(TPopAliasTest, offsets_and_region_aliases_keep_working)
{
    checkRegionAliases<TileType::Vec>();
    checkRegionAliases<TileType::Mat>();
    checkRegionAliases<TileType::Left>();
    checkRegionAliases<TileType::Right>();
    checkRegionAliases<TileType::Acc>();
    checkRegionAliases<TileType::ScaleLeft>();
    checkRegionAliases<TileType::ScaleRight>();
}

template <TileSplitAxis SplitAxis, bool UseGmWorkspace>
void checkPopSubviewMultiply()
{
    constexpr bool SPLIT_ROWS = SplitAxis == TileSplitAxis::TILE_UP_DOWN;
    constexpr bool SPLIT_COLS = SplitAxis == TileSplitAxis::TILE_LEFT_RIGHT;
    constexpr bool IS_SPLIT = SPLIT_ROWS || SPLIT_COLS;
    using ProducerTile = Tile<TileType::Mat, float, 4, 128>;
    using ConsumerTile = Tile<TileType::Vec, float, SPLIT_ROWS ? 2 : 4, SPLIT_COLS ? 64 : 128>;
    using ViewTile = Tile<TileType::Vec, float, 1, ConsumerTile::Cols>;
    using Pipe = TPipe<3, Direction::DIR_C2V, ProducerTile::GetSizeInBytes(), 2>;
    std::vector<float> gmWorkspace(UseGmWorkspace ? ProducerTile::Numel * Pipe::RingFiFo::SLOT_NUM : 0);
    Pipe::reset_for_cpu_sim();
    Pipe producer(UseGmWorkspace ? gmWorkspace.data() : nullptr, 0, 0);
    Pipe consumer0(UseGmWorkspace ? gmWorkspace.data() : nullptr, 0, 0);
    Pipe consumer1(UseGmWorkspace ? gmWorkspace.data() : nullptr, 0, 0);
    ProducerTile source;
    TASSIGN(source, 0);
    for (int i = 0; i < source.Numel; ++i) {
        source.data()[i] = static_cast<float>(i + 1);
    }
    {
        cpu_sim::ScopedExecutionContext ctx(0, 0, 1);
        TPUSH<Pipe, ProducerTile, SplitAxis>(producer, source);
    }
    for (uint32_t lane = 0; lane < (IS_SPLIT ? 2u : 1u); ++lane) {
        cpu_sim::ScopedExecutionContext ctx(0, lane, IS_SPLIT ? 2 : 1);
        auto& consumer = lane == 0 ? consumer0 : consumer1;
        ConsumerTile popped;
        ASSERT_EQ(popped.data(), nullptr);
        TPOP<Pipe, ConsumerTile, SplitAxis>(consumer, popped);
        ConsumerTile base;
        TASSIGN(base, reinterpret_cast<std::uintptr_t>(popped.data()));
        ASSERT_EQ(base.data(), popped.data());
        ViewTile view;
        constexpr int LAST_ROW = ConsumerTile::Rows - 1;
        TASSIGN(view, reinterpret_cast<std::uintptr_t>(popped.data() + LAST_ROW * ConsumerTile::Cols));
        ASSERT_EQ(view.data(), popped.data() + LAST_ROW * ConsumerTile::Cols);
        ViewTile scale;
        ViewTile result;
        TASSIGN(scale, 0);
        TASSIGN(result, ViewTile::GetSizeInBytes());
        for (int c = 0; c < ViewTile::Cols; ++c) {
            scale.data()[c] = 2.0f;
        }
        TMUL(result, view, scale);
        for (int c = 0; c < ViewTile::Cols; ++c) {
            const int sourceRow = LAST_ROW + (SPLIT_ROWS ? lane * ConsumerTile::Rows : 0);
            const int sourceCol = c + (SPLIT_COLS ? lane * ConsumerTile::Cols : 0);
            const float expected = static_cast<float>(sourceRow * ProducerTile::Cols + sourceCol + 1);
            EXPECT_FLOAT_EQ(view.data()[c], expected);
            EXPECT_FLOAT_EQ(result.data()[c], expected * 2.0f);
        }
        view.data()[0] = -7.0f;
        EXPECT_FLOAT_EQ(base.data()[LAST_ROW * ConsumerTile::Cols], -7.0f);
        TFREE<Pipe, SplitAxis>(consumer);
    }
}

TEST(TPopAliasTest, pop_subview_multiply_no_split)
{
    checkPopSubviewMultiply<TileSplitAxis::TILE_NO_SPLIT, false>();
    checkPopSubviewMultiply<TileSplitAxis::TILE_NO_SPLIT, true>();
}

TEST(TPopAliasTest, pop_subview_multiply_split_rows)
{
    checkPopSubviewMultiply<TileSplitAxis::TILE_UP_DOWN, false>();
    checkPopSubviewMultiply<TileSplitAxis::TILE_UP_DOWN, true>();
}

TEST(TPopAliasTest, pop_subview_multiply_split_columns)
{
    checkPopSubviewMultiply<TileSplitAxis::TILE_LEFT_RIGHT, false>();
    checkPopSubviewMultiply<TileSplitAxis::TILE_LEFT_RIGHT, true>();
}

TEST(TPopAliasTest, host_storage_survives_region_reset)
{
    // Use a fresh worker to cover allocation before region initialization and TLS cleanup.
    std::thread worker([] {
        SourceTile source;
        cpu_pipe::EnsureTileStorage(source);
        auto& memory = NPUMemoryModel::Instance();
        const auto address = reinterpret_cast<std::uintptr_t>(source.data());
        EXPECT_TRUE(memory.ContainsAddress(address));
        memory.Reset();
        SourceTile alias;
        TASSIGN(alias, address);
        EXPECT_EQ(alias.data(), source.data());
        memory.Initialize(NPUArch::A5);
        TASSIGN(alias, address);
        EXPECT_EQ(alias.data(), source.data());
    });
    worker.join();
}

TEST(TPopAliasTest, registration_does_not_keep_freed_storage_alive)
{
    auto& memory = NPUMemoryModel::Instance();
    auto storage = std::make_shared<std::vector<float>>(SourceTile::Numel);
    const auto address = reinterpret_cast<std::uintptr_t>(storage->data());
    memory.RegisterHostStorage(storage);
    EXPECT_TRUE(memory.ContainsAddress(address));
    EXPECT_TRUE(memory.ContainsAddress(address + SourceTile::GetSizeInBytes() - 1));
    EXPECT_FALSE(memory.ContainsAddress(address + SourceTile::GetSizeInBytes()));
    storage.reset();
    EXPECT_FALSE(memory.ContainsAddress(address));
}

#if GTEST_HAS_DEATH_TEST
TEST(TPopAliasDeathTest, rejects_view_extending_past_host_storage)
{
    SourceTile source;
    cpu_pipe::EnsureTileStorage(source);
    EXPECT_DEATH(
        {
            SourceTile oversized;
            TASSIGN(oversized, reinterpret_cast<std::uintptr_t>(source.data() + SourceTile::Cols));
        },
        "Tile assignment exceeds host storage capacity");
}
#endif
} // namespace
