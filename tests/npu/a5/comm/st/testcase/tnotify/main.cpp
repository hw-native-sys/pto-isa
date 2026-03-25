/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Test TNOTIFY (remote signal notification) via PTO with SHMEM backend
// Tests AtomicAdd and Set operations

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

#include "tnotify_kernel.h"
#include "../comm_mpi.h"

// ============================================================================
// AtomicAdd Tests
// All ranks perform atomic add 1 to rank 0's counter
// Expected result: counter = n_ranks
// ============================================================================
TEST(TNotify, AtomicAdd_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunNotifyAtomicAdd(2, 2, 0, 0));
}

// ============================================================================
// Set Tests (Ring pattern)
// Each rank sets next rank's signal to its own rank_id + 100
// ============================================================================
TEST(TNotify, Set_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunNotifySet(2, 2, 0, 0));
}

// ============================================================================
// Scoreboard Tests
// Each rank notifies its slot in rank 0's scoreboard
// ============================================================================
TEST(TNotify, Scoreboard_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunNotifyScoreboard<4>(2, 2, 0, 0)));
}

// ============================================================================
// Runtime NotifyOp Tests (Set operation)
// ============================================================================
TEST(TNotify, RuntimeOp_Set)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunNotifyRuntimeOp(2, 2, 0, 0));
}
int main(int argc, char **argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
