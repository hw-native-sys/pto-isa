/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Compile-only tests. Invalid addresses must never be executed on an NPU.
// test_tassign_static.py selects one case per compiler invocation.
#include <cstddef>
#include <cstdint>

#include <pto/pto-inst.hpp>

using namespace pto;
using Ordinary = Tile<TileType::Vec, int32_t, 16, 32>;
using DefaultNZ = Tile<TileType::Vec, int32_t, 16, 32, BLayout::ColMajor, 16, 32, SLayout::RowMajor>;
using Normal = Tile<
    TileType::Vec, int32_t, 32, 32, BLayout::ColMajor, 19, 32, SLayout::RowMajor, TileConfig::fractalABSize,
    PadValue::Null, CompactMode::Normal>;
using RowPlusOne64 = Tile<
    TileType::Vec, int32_t, 64, 32, BLayout::ColMajor, 19, 32, SLayout::RowMajor, TileConfig::fractalABSize,
    PadValue::Null, CompactMode::RowPlusOne>;
using Packed = Tile<TileType::Vec, float4_e2m1x2_t, 16, 64>;
using PackedRowPlusOne64 = Tile<
    TileType::Vec, float4_e2m1x2_t, 64, 128, BLayout::ColMajor, 19, 128, SLayout::RowMajor, TileConfig::fractalABSize,
    PadValue::Null, CompactMode::RowPlusOne>;
using LegacyRowPlusOne33 = Tile<
    TileType::Vec, int32_t, 33, 32, BLayout::ColMajor, 32, 32, SLayout::RowMajor, TileConfig::fractalABSize,
    PadValue::Null, CompactMode::RowPlusOne>;
using LegacyPackedRowPlusOne33 = Tile<
    TileType::Vec, float4_e2m1x2_t, 33, 128, BLayout::ColMajor, 32, 128, SLayout::RowMajor, TileConfig::fractalABSize,
    PadValue::Null, CompactMode::RowPlusOne>;
using DynamicRowPlusOne33 = Tile<
    TileType::Vec, int32_t, 33, 32, BLayout::ColMajor, DYNAMIC, 32, SLayout::RowMajor, TileConfig::fractalABSize,
    PadValue::Null, CompactMode::RowPlusOne>;

static_assert(Ordinary::StorageBytes == 2048);
static_assert(DefaultNZ::StorageBytes == 2048 && DefaultNZ::Compact == CompactMode::Null);
static_assert(Normal::StorageBytes == 4096);
static_assert(RowPlusOne64::StorageBytes == 8320 && RowPlusOne64::Numel * sizeof(int32_t) == 8192);
static_assert(Packed::StorageBytes == 512);
static_assert(PackedRowPlusOne64::StorageBytes == 4160);
static_assert(LegacyRowPlusOne33::Rows == 33 && LegacyRowPlusOne33::StorageBytes == 4224);
static_assert(LegacyPackedRowPlusOne33::StorageBytes == 2112);
static_assert(DynamicRowPlusOne33::StorageBytes == 6272);

#if !defined(__CPU_SIM)
// Ordinary packed TASSIGN intentionally retains its baseline Rows*Cols*sizeof
// footprint, independent of packed Tile allocation. This patch only adds RPO.
static_assert(detail::tile_storage_bytes_v<Packed> == 1024);

#if PTO_TASSIGN_TEST_CASE == 1 || PTO_TASSIGN_TEST_CASE == 2 || PTO_TASSIGN_TEST_CASE == 3
using TestTile = Ordinary;
constexpr std::size_t kAddress =
    PTO_TASSIGN_TEST_CASE == 3 ? 0 : PTO_UBUF_SIZE_BYTES - 2048 + (PTO_TASSIGN_TEST_CASE == 2 ? 32 : 0);
#elif PTO_TASSIGN_TEST_CASE == 4
using TestTile = DefaultNZ;
constexpr std::size_t kAddress = PTO_UBUF_SIZE_BYTES - 2048;
#elif PTO_TASSIGN_TEST_CASE == 5
using TestTile = Normal;
constexpr std::size_t kAddress = PTO_UBUF_SIZE_BYTES - 4096;
#elif PTO_TASSIGN_TEST_CASE == 6 || PTO_TASSIGN_TEST_CASE == 7 || PTO_TASSIGN_TEST_CASE == 8
using TestTile = RowPlusOne64;
constexpr std::size_t kAddress = PTO_TASSIGN_TEST_CASE == 6 ? PTO_UBUF_SIZE_BYTES - 8320 :
                                 PTO_TASSIGN_TEST_CASE == 7 ? PTO_UBUF_SIZE_BYTES - 8192 :
                                                              0;
#elif PTO_TASSIGN_TEST_CASE == 9 || PTO_TASSIGN_TEST_CASE == 10
using TestTile = Packed;
constexpr std::size_t kAddress = PTO_UBUF_SIZE_BYTES - (PTO_TASSIGN_TEST_CASE == 9 ? 1024 : 512);
#elif PTO_TASSIGN_TEST_CASE == 11 || PTO_TASSIGN_TEST_CASE == 12
using TestTile = PackedRowPlusOne64;
constexpr std::size_t kAddress = PTO_UBUF_SIZE_BYTES - 4160 + (PTO_TASSIGN_TEST_CASE == 12 ? 32 : 0);
#elif PTO_TASSIGN_TEST_CASE == 13 || PTO_TASSIGN_TEST_CASE == 14
using TestTile = LegacyRowPlusOne33;
constexpr std::size_t kAddress = PTO_UBUF_SIZE_BYTES - 4224 + (PTO_TASSIGN_TEST_CASE == 14 ? 32 : 0);
#elif PTO_TASSIGN_TEST_CASE == 15 || PTO_TASSIGN_TEST_CASE == 16
using TestTile = LegacyPackedRowPlusOne33;
constexpr std::size_t kAddress = PTO_UBUF_SIZE_BYTES - 2112 + (PTO_TASSIGN_TEST_CASE == 16 ? 32 : 0);
#elif PTO_TASSIGN_TEST_CASE == 17 || PTO_TASSIGN_TEST_CASE == 18
using TestTile = DynamicRowPlusOne33;
constexpr std::size_t kAddress = PTO_UBUF_SIZE_BYTES - (PTO_TASSIGN_TEST_CASE == 17 ? 6272 : 4224);
#elif PTO_TASSIGN_TEST_CASE != 0
#error "Unknown PTO_TASSIGN_TEST_CASE"
#endif

#if PTO_TASSIGN_TEST_CASE != 0
extern "C" __global__ AICORE void tassign_static_compile_only()
{
#if PTO_TASSIGN_TEST_CASE == 17 || PTO_TASSIGN_TEST_CASE == 18
    TestTile tile(33);
#else
    TestTile tile;
#endif
    TASSIGN<kAddress>(tile);
}
#endif
#endif // !defined(__CPU_SIM)
