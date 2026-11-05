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

#include "tput_async_notify_kernel.h"
#include "../comm_mpi.h"

// Verify repeated A5 MTE payload transfers followed by Scalar SET.
TEST(TPutAsyncNotify, RepeatedMteScalarSetMultiChunk_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncNotifySet<int32_t, 4096>(2, 2, 0, 0)));
}

// Verify the receiver consumes the MTE payload after observing the Scalar notification.
TEST(TPutAsyncNotify, MteReceiverConsumesPayload_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncNotifyConsumeSet<int32_t, 4096>(2, 2, 0, 0)));
}

// Verify repeated A5 MTE payload transfers followed by Scalar AtomicAdd.
TEST(TPutAsyncNotify, RepeatedMteScalarAddMultiChunk_2Ranks)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncNotifyAdd<int32_t, 4096>(2, 2, 0, 0)));
}

int main(int argc, char** argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
