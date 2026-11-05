/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>

#include "../comm_mpi.h"
#include "tput_async_urma_batch_kernel.h"

TEST(TPutAsyncUrmaBatch, PerPeerSingleBatchFourOperationsConsumed)
{
    if (CommMpiSize() != 2) {
        GTEST_SKIP() << "Requires exactly 2 MPI ranks";
    }
    ASSERT_TRUE(RunTPutAsyncUrmaBatchBasic(2, 2, 0, 0, false));
}

TEST(TPutAsyncUrmaBatch, PerPeerDocumentationKernelTwoWrites)
{
    if (CommMpiSize() != 2) {
        GTEST_SKIP() << "Requires exactly 2 MPI ranks";
    }
    ASSERT_TRUE(RunTPutAsyncUrmaBatchDocExample(2, 2, 0, 0));
}

TEST(TPutAsyncUrmaBatch, SharedPoolSingleBatchFourOperationsConsumed)
{
    if (CommMpiSize() != 2) {
        GTEST_SKIP() << "Requires exactly 2 MPI ranks";
    }
    ASSERT_TRUE(RunTPutAsyncUrmaBatchBasic(2, 2, 0, 0, true));
}

TEST(TPutAsyncUrmaBatch, SharedPoolRoundRobinAcrossTwoJetties)
{
    if (CommMpiSize() != 2) {
        GTEST_SKIP() << "Requires exactly 2 MPI ranks";
    }
    ASSERT_TRUE(RunTPutAsyncUrmaBatchMultiJetty(2, 2, 0, 0));
}

TEST(TPutAsyncUrmaBatch, SessionPolicyAndImplicitFlushBoundaries)
{
    if (CommMpiSize() != 2) {
        GTEST_SKIP() << "Requires exactly 2 MPI ranks";
    }
    ASSERT_TRUE(RunTPutAsyncUrmaBatchPolicySuite(2, 2, 0, 0));
}

TEST(TPutAsyncUrmaBatch, SinglePutSplitsAcrossTwoJetties)
{
    if (CommMpiSize() != 2) {
        GTEST_SKIP() << "Requires exactly 2 MPI ranks";
    }
    ASSERT_TRUE(RunTPutAsyncUrmaLargeMultiWqe(2, 2, 0, 0));
}

TEST(TPutAsyncUrmaBatch, PerPeer65UniqueBatchesWqWrapAndConsume)
{
    if (CommMpiSize() != 2) {
        GTEST_SKIP() << "Requires exactly 2 MPI ranks";
    }
    ASSERT_TRUE(RunTPutAsyncUrmaBatchConsume(2, 2, 0, 0, false));
}

TEST(TPutAsyncUrmaBatch, SharedPool65UniqueBatchesWqWrapAndConsume)
{
    if (CommMpiSize() != 2) {
        GTEST_SKIP() << "Requires exactly 2 MPI ranks";
    }
    ASSERT_TRUE(RunTPutAsyncUrmaBatchConsume(2, 2, 0, 0, true));
}

TEST(TPutAsyncUrmaBatch, PerPeerBatchAfterUnwaitedAtomicNotifyPrefix)
{
    if (CommMpiSize() != 2) {
        GTEST_SKIP() << "Requires exactly 2 MPI ranks";
    }
    ASSERT_TRUE(RunTPutAsyncUrmaBatchConsume(2, 2, 0, 0, false, true));
}

int main(int argc, char** argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
