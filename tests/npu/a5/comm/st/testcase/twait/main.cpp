/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// Test TWAIT and TTEST operations via PTO (HCCL backend)
// TWAIT: Blocking wait for signal condition
// TTEST: Non-blocking test of signal condition

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

// G.EXP.05-CPP: Include shared header instead of forward declarations
#include "twait_kernel.h"
#include "../comm_mpi.h"

// ============================================================================
// TWAIT Tests
// ============================================================================

// Basic TWAIT test: wait for signal == expected value
TEST(TWait, Basic_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTWaitBasic(2, 2, 0, 0));
}

// TWAIT with GE comparison: wait for signal >= 100
TEST(TWait, Compare_GE_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTWaitCompare(2, 2, 0, 0, 150));
}

// TWAIT with atomic add: multiple ranks contribute, one waits for threshold
TEST(TWait, Atomic_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTWaitAtomic(2, 2, 0, 0));
}
TEST(TWait, Atomic_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE(RunTWaitAtomic(4, 4, 0, 0));
}

// TWAIT 2D signal matrix
TEST(TWait, Matrix2D_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunTWaitMatrix<4, 8>(2, 2, 0, 0)));
}
TEST(TWait, Matrix2D_2Ranks_Large)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunTWaitMatrix<7, 13>(2, 2, 0, 0)));
}

// TWAIT multi-phase update
TEST(TWait, MultiPhase_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE(RunTWaitMultiPhase(2, 2, 0, 0));
}

// TWAIT sub-region of signal matrix
TEST(TWait, SubRegion_4x8_of_16)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunTWaitSubRegion<16, 4, 8>(2, 2, 0, 0)));
}
int main(int argc, char **argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
