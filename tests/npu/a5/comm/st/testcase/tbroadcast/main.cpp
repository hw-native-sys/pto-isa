/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

#include "tbroadcast_kernel.h"
#include "../comm_mpi.h"

// ============================================================================
// TBROADCAST Tests - Basic (data fits in single UB Tile)
// ============================================================================
TEST(TBroadCast, FloatSmallRoot0_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    EXPECT_TRUE((RunBroadCast<float, 256>(4, 4, 0, 0, 0)));
}

TEST(TBroadCast, Int32LargeRoot1)
{
    SKIP_IF_RANKS_LT(2);
    EXPECT_TRUE((RunBroadCast<int32_t, 4096>(2, 2, 0, 0, 1)));
}
// ============================================================================
// TBROADCAST Tests - Large Shape Chunked (GlobalTensor > UB Tile, auto-chunked)
// ============================================================================
// int32: 128x32, tile 16 rows → 8 chunks, root=0, 2 ranks
TEST(TBroadCast, LargeShape_Int32_128x32_tile16_Root0)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunBroadCastLargeShape_Int32_128x32_tile16(2, 2, 0, 0, 0)));
}
// int32: 128x32, tile 16 rows → 8 chunks, root=0, 4 ranks
TEST(TBroadCast, LargeShape_Int32_128x32_tile16_Root0_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE((RunBroadCastLargeShape_Int32_128x32_tile16(4, 4, 0, 0, 0)));
}
// float: 256x64, tile 32 rows → 8 chunks, root=0, 2 ranks
TEST(TBroadCast, LargeShape_Float_256x64_tile32_Root0)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunBroadCastLargeShape_Float_256x64_tile32(2, 2, 0, 0, 0)));
}
// int32: 512x32, tile 64 rows → 8 chunks, root=0, 2 ranks (larger data)
TEST(TBroadCast, LargeShape_Int32_512x32_tile64_Root0)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunBroadCastLargeShape_Int32_512x32_tile64(2, 2, 0, 0, 0)));
}
// int32: 512x32, tile 64 rows → 8 chunks, root=0, 8 ranks
TEST(TBroadCast, LargeShape_Int32_512x32_tile64_Root0_8Ranks)
{
    SKIP_IF_RANKS_LT(8);
    ASSERT_TRUE((RunBroadCastLargeShape_Int32_512x32_tile64(8, 8, 0, 0, 0)));
}
// int32: 128x32, tile 16 rows → 8 chunks, root=1, 2 ranks (non-zero root)
TEST(TBroadCast, LargeShape_Int32_128x32_tile16_Root1)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunBroadCastLargeShape_Int32_128x32_tile16(2, 2, 0, 0, 1)));
}

// ============================================================================
// TBROADCAST Tests - Ping-Pong Double Buffering (2 UB Tiles: ping + pong)
// ============================================================================
// int32: 128x32, tile 16 rows → 8 chunks, root=0, 2 ranks
TEST(TBroadCast, PingPong_Int32_128x32_tile16_Root0)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunBroadCastPingPong_Int32_128x32_tile16(2, 2, 0, 0, 0)));
}
// int32: 128x32, tile 16 rows → 8 chunks, root=0, 4 ranks
TEST(TBroadCast, PingPong_Int32_128x32_tile16_Root0_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE((RunBroadCastPingPong_Int32_128x32_tile16(4, 4, 0, 0, 0)));
}
// float: 256x64, tile 32 rows → 8 chunks, root=0, 2 ranks
TEST(TBroadCast, PingPong_Float_256x64_tile32_Root0)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunBroadCastPingPong_Float_256x64_tile32(2, 2, 0, 0, 0)));
}
// int32: 128x32, tile 16 rows → 8 chunks, root=1, 2 ranks (non-zero root)
TEST(TBroadCast, PingPong_Int32_128x32_tile16_Root1)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunBroadCastPingPong_Int32_128x32_tile16(2, 2, 0, 0, 1)));
}
// float: 256x64, tile 32 rows → 8 chunks, root=0, 8 ranks
TEST(TBroadCast, PingPong_Float_256x64_tile32_Root0_8Ranks)
{
    SKIP_IF_RANKS_LT(8);
    ASSERT_TRUE((RunBroadCastPingPong_Float_256x64_tile32(8, 8, 0, 0, 0)));
}
int main(int argc, char **argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
