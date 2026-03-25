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
// 1D Vector Tile Tests
// ============================================================================
TEST(TPutAsync, Vec_FloatSmall_4Ranks)
{
    ASSERT_TRUE((RunPutAsyncRootPut<float, 256>(4, 4, 0, 0)));
}
TEST(TPutAsync, Vec_Int32Large)
{
    ASSERT_TRUE((RunPutAsyncRootPut<int32_t, 4096>(2, 2, 0, 0)));
}
TEST(TPutAsync, Vec_Uint8Small_8Ranks)
{
    ASSERT_TRUE((RunPutAsyncRootPut<uint8_t, 512>(8, 8, 0, 0)));
}
int main(int argc, char **argv)
{
    CommMpiInit(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    CommMpiFinalize();
    return ret;
}
