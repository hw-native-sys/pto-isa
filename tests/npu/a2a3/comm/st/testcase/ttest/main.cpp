/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Test TTEST (non-blocking signal test) operations via PTO (HCCL backend)
// TTEST returns true if signal meets comparison condition, false otherwise

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

// G.EXP.05-CPP: Include shared header instead of forward declarations
#include "ttest_kernel.h"
#include "../comm_mpi.h"

// ============================================================================
// TTEST Basic Tests
// ============================================================================

// Test TTEST returns true when signal == expected value
TEST(TTest, True_EQ)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTTestTrue(2, 2, 0, 0));
}

// Test TTEST returns false when signal != expected value
TEST(TTest, False_EQ)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTTestFalse(2, 2, 0, 0));
}

// Test TTEST with NE (not equal) comparison
TEST(TTest, NE_True)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTTestNE(2, 2, 0, 0));
}

// ============================================================================
// TTEST Comparison Operator Tests
// ============================================================================

// GE (>=): signal=100, test >= 50 -> should be true
TEST(TTest, GE_True)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTTestCompare_GE(2, 2, 0, 0, 100, 50, true));
}

// GE (>=): signal=100, test >= 100 -> should be true (equal case)
TEST(TTest, GE_Equal)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTTestCompare_GE(2, 2, 0, 0, 100, 100, true));
}

// GT (>): signal=100, test > 50 -> should be true
TEST(TTest, GT_True)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTTestCompare_GT(2, 2, 0, 0, 100, 50, true));
}

// GT (>): signal=100, test > 100 -> should be false (equal, not greater)
TEST(TTest, GT_False)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTTestCompare_GT(2, 2, 0, 0, 100, 100, false));
}

// LE (<=): signal=50, test <= 100 -> should be true
TEST(TTest, LE_True)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTTestCompare_LE(2, 2, 0, 0, 50, 100, true));
}

// LT (<): signal=50, test < 100 -> should be true
TEST(TTest, LT_True)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTTestCompare_LT(2, 2, 0, 0, 50, 100, true));
}

// LT (<): signal=100, test < 100 -> should be false (equal, not less)
TEST(TTest, LT_False)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTTestCompare_LT(2, 2, 0, 0, 100, 100, false));
}

// ============================================================================
// TTEST Polling Pattern Tests
// ============================================================================

// Test polling loop with TTEST until signal arrives
TEST(TTest, PollingTimeout)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTTestPollingTimeout(2, 2, 0, 0));
}
// Test polling loop timeout when signal is delayed too long
TEST(TTest, PollingTimeoutMiss)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTTestPollingTimeoutMiss(2, 2, 0, 0));
}

// Test sub-region of signal matrix
TEST(TTest, SubRegion_4x8_of_16)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunTTestSubRegion<16, 4, 8>(2, 2, 0, 0)));
}
int main(int argc, char **argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
