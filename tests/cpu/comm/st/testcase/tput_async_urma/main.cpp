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

#include "tput_async_urma_kernel.h"
#include "pto/common/cpu_stub.hpp"

// ============================================================================
// Basic correctness (URMA true async PUT on A5 3510)
// ============================================================================
TEST(TPutAsyncUrma, Vec_FloatSmall)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncUrmaRootPut<float, 256>(2, 2, 0, 0)));
}
TEST(TPutAsyncUrma, Vec_Int32Large)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncUrmaRootPut<int32_t, 4096>(2, 2, 0, 0)));
}
TEST(TPutAsyncUrma, Vec_Uint8Small)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncUrmaRootPut<uint8_t, 512>(2, 2, 0, 0)));
}

// ============================================================================
// Boundary scenarios
// ============================================================================
TEST(TPutAsyncUrma, Vec_Uint8_SingleChunk)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncUrmaRootPut<uint8_t, 64>(2, 2, 0, 0)));
}
TEST(TPutAsyncUrma, Vec_Float_ExactChunk)
{
    SKIP_IF_RANKS_LT(2);
    ASSERT_TRUE((RunPutAsyncUrmaRootPut<float, 64>(2, 2, 0, 0)));
}

// ============================================================================
// Multi-rank broadcast
// ============================================================================
TEST(TPutAsyncUrma, Vec_FloatSmall_4Ranks)
{
    SKIP_IF_RANKS_LT(4);
    ASSERT_TRUE((RunPutAsyncUrmaRootPut<float, 256>(4, 4, 0, 0)));
}

// ============================================================================
// Large MR boundary tests
// ============================================================================
TEST(TPutAsyncUrma, Vec_Float_MR_6MB)
{
    SKIP_IF_RANKS_LT(2);
    // 512K floats → commBytesNeeded ≈ 4MB+256B, allocSize = 6MB (>2MB)
    ASSERT_TRUE((RunPutAsyncUrmaRootPut<float, 524288>(2, 2, 0, 0)));
}
TEST(TPutAsyncUrma, Vec_Int32_MR_Over512MB)
{
    SKIP_IF_RANKS_LT(2);
    // 64M int32 → commBytesNeeded ≈ 512MB+256B, allocSize ≈ 514MB (>512MB)
    ASSERT_TRUE((RunPutAsyncUrmaRootPut<int32_t, 67108864>(2, 2, 0, 0)));
}

int main(int argc, char **argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
