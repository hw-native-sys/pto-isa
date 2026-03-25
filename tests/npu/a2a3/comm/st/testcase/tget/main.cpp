/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Test TGET (remote read) operation via PTO with SHMEM backend
// Ring communication pattern: each rank reads data from next rank

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

#include "tget_kernel.h"
#include "../comm_mpi.h"

// ============================================================================
// 1D Vector Tile Tests
// ============================================================================
TEST(TGet, Vec_FloatSmall)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRing<float, 256>(2, 2, 0, 0)));
}
TEST(TGet, Vec_Int32Large)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRing<int32_t, 4096>(2, 2, 0, 0)));
}
TEST(TGet, Vec_Uint8Small)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRing<uint8_t, 512>(2, 2, 0, 0)));
}

// ============================================================================
// 2D Shape Tests (using Vec Tile with 2D shape)
// ============================================================================
TEST(TGet, Shape2D_Float16x16)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRing2D<float, 16, 16>(2, 2, 0, 0)));
}
TEST(TGet, Shape2D_Float8x32)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRing2D<float, 8, 32>(2, 2, 0, 0)));
}
TEST(TGet, Shape2D_Int32_4x64)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRing2D<int32_t, 4, 64>(2, 2, 0, 0)));
}

// ============================================================================
// Large Shape Chunked Tests
// GlobalTensor shape exceeds UB tile capacity, TGET_IMPL auto-chunks the transfer
// ============================================================================
TEST(TGet, LargeShape_Float_128x64_tile16)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRingLargeShape<float, 128, 64, 16>(2, 2, 0, 0)));
}
TEST(TGet, LargeShape_Int32_256x32_tile32)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRingLargeShape<int32_t, 256, 32, 32>(2, 2, 0, 0)));
}
TEST(TGet, LargeShape_Float_512x32_tile64)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRingLargeShape<float, 512, 32, 64>(2, 2, 0, 0)));
}
TEST(TGet, LargeShape_Float_2048x32_tile64)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRingLargeShape<float, 2048, 32, 64>(2, 2, 0, 0)));
}
TEST(TGet, LargeShape_Float_4096x32_tile64)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRingLargeShape<float, 4096, 32, 64>(2, 2, 0, 0)));
}
TEST(TGet, LargeShape_Float_2048x64_tile128)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRingLargeShape<float, 2048, 64, 128>(2, 2, 0, 0)));
}
TEST(TGet, LargeShape_Int32_4096x64_tile128)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRingLargeShape<int32_t, 4096, 64, 128>(2, 2, 0, 0)));
}

// ============================================================================
// Multi-Dimensional Chunked Tests
// GlobalTensor has outer dims > 1, TGET_IMPL iterates outer dims + chunks dim3
// ============================================================================
TEST(TGet, MultiDim_Float_2x2x1x32x32_tile16)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRingMultiDim<float, 2, 2, 1, 32, 32, 16>(2, 2, 0, 0)));
}
TEST(TGet, MultiDim_Int32_4x1x1x32x64_tile16)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRingMultiDim<int32_t, 4, 1, 1, 32, 64, 16>(2, 2, 0, 0)));
}

// ============================================================================
// Irregular Shape Chunked Tests (Non-power-of-2, non-divisible by tile rows)
// Tests partial last chunk path in TGET_IMPL via DYNAMIC ValidRow tiles
// ============================================================================
TEST(TGet, Irregular_Float_2047x32_tile64)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRingIrregularShape<float, 2047, 32, 64>(2, 2, 0, 0)));
}
TEST(TGet, Irregular_Int32_1025x32_tile64)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRingIrregularShape<int32_t, 1025, 32, 64>(2, 2, 0, 0)));
}
TEST(TGet, Irregular_Float_4095x32_tile128)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRingIrregularShape<float, 4095, 32, 128>(2, 2, 0, 0)));
}

// ============================================================================
// 2D Sliding Tests (both rows and cols exceed tile, TGET_IMPL auto 2D chunking)
// ============================================================================
// ---- Regular 2D sliding ----
TEST(TGet, Sliding2D_Float_64x128_tile16x32)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRing2DSliding<float, 64, 128, 16, 32>(2, 2, 0, 0)));
}
TEST(TGet, Sliding2D_Int32_128x256_tile32x64)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRing2DSliding<int32_t, 128, 256, 32, 64>(2, 2, 0, 0)));
}
TEST(TGet, Sliding2D_Float_256x512_tile64x128)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRing2DSliding<float, 256, 512, 64, 128>(2, 2, 0, 0)));
}
// ---- Irregular 2D sliding ----
TEST(TGet, Sliding2D_IrregRow_Float_65x64_tile16x32)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRing2DSliding<float, 65, 64, 16, 32>(2, 2, 0, 0)));
}
TEST(TGet, Sliding2D_IrregCol_Float_64x104_tile16x32)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRing2DSliding<float, 64, 104, 16, 32>(2, 2, 0, 0)));
}
TEST(TGet, Sliding2D_IrregBoth_Float_65x104_tile16x32)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRing2DSliding<float, 65, 104, 16, 32>(2, 2, 0, 0)));
}

// ============================================================================
// Ping-Pong Double Buffering Tests
// TGET with two staging tiles: overlap TLOAD(chunk N+1) with TSTORE(chunk N)
// ============================================================================
TEST(TGet, PingPong_Float_128x128_tile16x32)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRingPingPong<float, 128, 128, 16, 32>(2, 2, 0, 0)));
}
TEST(TGet, PingPong_Int32_256x256_tile32x64)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRingPingPong<int32_t, 256, 256, 32, 64>(2, 2, 0, 0)));
}
TEST(TGet, PingPong_Irregular_Float_65x104_tile16x32)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunGetRingPingPong<float, 65, 104, 16, 32>(2, 2, 0, 0)));
}
int main(int argc, char **argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
