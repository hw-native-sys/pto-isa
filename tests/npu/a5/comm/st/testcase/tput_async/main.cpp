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

#include "tput_async_kernel.h"
#include "../comm_mpi.h"

// ============================================================================
// Basic correctness (default 256B scratch tile, multi-chunk MTE fallback path)
// ============================================================================
TEST(TPutAsync, Vec_FloatSmall)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncRootPut<float, 256>(2, 2, 0, 0)));
}
TEST(TPutAsync, Vec_Int32Large)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncRootPut<int32_t, 4096>(2, 2, 0, 0)));
}
TEST(TPutAsync, Vec_Uint8Small)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncRootPut<uint8_t, 512>(2, 2, 0, 0)));
}

// ============================================================================
// Boundary scenarios for MTE fallback chunking
// ============================================================================
// 64 uint8_t = 64B, smaller than the 256B UB scratch buffer -> single chunk
TEST(TPutAsync, Vec_Uint8_SingleChunk)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncRootPut<uint8_t, 64>(2, 2, 0, 0)));
}
// 64 float = 256B, exactly equals the 256B UB scratch buffer -> exactly 1 chunk
TEST(TPutAsync, Vec_Float_ExactChunk)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncRootPut<float, 64>(2, 2, 0, 0)));
}

// ============================================================================
// Multi-rank broadcast
// ============================================================================
TEST(TPutAsync, Vec_FloatSmall_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE((RunPutAsyncRootPut<float, 256>(4, 4, 0, 0)));
}

int main(int argc, char** argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
