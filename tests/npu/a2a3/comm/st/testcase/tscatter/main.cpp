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

// G.EXP.05-CPP: Include shared header instead of forward declarations
#include "tscatter_kernel.h"
#include "../comm_mpi.h"

// ============================================================================
// TSCATTER Tests - Basic: Root scatters data to all ranks
// ============================================================================
TEST(TScatter, FloatSmall_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE((RunScatter<float, 256>(4, 4, 0, 0)));
}
TEST(TScatter, Int32Large)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunScatter<int32_t, 4096>(2, 2, 0, 0)));
}
TEST(TScatter, Uint8Small)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunScatter<uint8_t, 512>(2, 2, 0, 0)));
}
// TEST(TScatter, SingleRank)
// {
//     ASSERT_TRUE((RunScatter<float, 256>(1, 1, 0, 0)));
// }
TEST(TScatter, Root1_FloatSmall)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunScatterWithRoot<float, 256>(2, 2, 0, 0, 1)));
}
TEST(TScatter, EmptyRows_FloatSmall)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunScatterEmpty<float, 256>(2, 2, 0, 0, 0)));
}

// ============================================================================
// TSCATTER Tests - Large Shape (chunked): per-rank data > single UB tile
// ============================================================================
TEST(TScatterLargeShape, Int32_128x32_tile16_2ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunScatterLargeShape_Int32_128x32_tile16(2, 2, 0, 0));
}
TEST(TScatterLargeShape, Int32_128x32_tile16_4ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunScatterLargeShape_Int32_128x32_tile16(4, 4, 0, 0));
}
TEST(TScatterLargeShape, Float_256x64_tile32_2ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunScatterLargeShape_Float_256x64_tile32(2, 2, 0, 0));
}
TEST(TScatterLargeShape, Int32_512x32_tile64_2ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunScatterLargeShape_Int32_512x32_tile64(2, 2, 0, 0));
}

// ============================================================================
// TSCATTER Tests - PingPong: double-buffered chunked TSCATTER
// ============================================================================
TEST(TScatterPingPong, Int32_128x32_tile16_2ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunScatterPingPong_Int32_128x32_tile16(2, 2, 0, 0));
}
TEST(TScatterPingPong, Int32_128x32_tile16_4ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunScatterPingPong_Int32_128x32_tile16(4, 4, 0, 0));
}
TEST(TScatterPingPong, Float_256x64_tile32_2ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunScatterPingPong_Float_256x64_tile32(2, 2, 0, 0));
}
int main(int argc, char **argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
