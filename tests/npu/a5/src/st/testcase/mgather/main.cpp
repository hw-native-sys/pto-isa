/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class MGATHERTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

static std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return std::string("../") + testInfo->test_suite_name() + "." + testInfo->name();
}

template <typename T, typename TIdx, typename Launcher>
void RunMGatherTest(size_t tableCount, size_t idxCount, size_t outCount, Launcher launcher)
{
    size_t tableByteSize = tableCount * sizeof(T);
    size_t idxByteSize = idxCount * sizeof(TIdx);
    size_t outByteSize = outCount * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *tableHost, *outHost;
    TIdx *idxHost;
    T *tableDevice, *outDevice;
    TIdx *idxDevice;

    aclrtMallocHost((void **)(&tableHost), tableByteSize);
    aclrtMallocHost((void **)(&idxHost), idxByteSize);
    aclrtMallocHost((void **)(&outHost), outByteSize);

    aclrtMalloc((void **)&tableDevice, tableByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&idxDevice, idxByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outDevice, outByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/table.bin", tableByteSize, tableHost, tableByteSize);
    ReadFile(GetGoldenDir() + "/indices.bin", idxByteSize, idxHost, idxByteSize);

    aclrtMemcpy(tableDevice, tableByteSize, tableHost, tableByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(idxDevice, idxByteSize, idxHost, idxByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtMemset(outDevice, outByteSize, 0, outByteSize);

    launcher(outDevice, tableDevice, idxDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, outByteSize, outDevice, outByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", outHost, outByteSize);

    aclrtFree(tableDevice);
    aclrtFree(idxDevice);
    aclrtFree(outDevice);

    aclrtFreeHost(tableHost);
    aclrtFreeHost(idxHost);
    aclrtFreeHost(outHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(outCount);
    std::vector<T> devFinal(outCount);
    ReadFile(GetGoldenDir() + "/golden.bin", outByteSize, golden.data(), outByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", outByteSize, devFinal.data(), outByteSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.0f);
    EXPECT_TRUE(ret);
}

#define DECLARE_LAUNCH(NAME, THOST, TIDX) void Launch_##NAME(THOST *out, THOST *table, TIDX *indices, void *stream);

#define ROW_TEST(NAME, THOST, TIDX, R, C, TR)                                                 \
    TEST_F(MGATHERTest, case_##NAME)                                                          \
    {                                                                                         \
        RunMGatherTest<THOST, TIDX>((size_t)TR * C, (size_t)R, (size_t)R * C, Launch_##NAME); \
    }

#define ELEM_TEST(NAME, THOST, TIDX, N, TS)                                           \
    TEST_F(MGATHERTest, case_##NAME)                                                  \
    {                                                                                 \
        RunMGatherTest<THOST, TIDX>((size_t)TS, (size_t)N, (size_t)N, Launch_##NAME); \
    }

#define ELEM2D_TEST(NAME, THOST, TIDX, R, C, TS)                                              \
    TEST_F(MGATHERTest, case_##NAME)                                                          \
    {                                                                                         \
        RunMGatherTest<THOST, TIDX>((size_t)TS, (size_t)R * C, (size_t)R * C, Launch_##NAME); \
    }

#define ELEM2D_DYN_TEST(NAME, THOST, TIDX, RVR, RVC, RTR, RTC)                                               \
    TEST_F(MGATHERTest, case_##NAME)                                                                         \
    {                                                                                                        \
        RunMGatherTest<THOST, TIDX>((size_t)RTR * RTC, (size_t)RVR * RVC, (size_t)RVR * RVC, Launch_##NAME); \
    }

// --- Launcher declarations ------------------------------------------------------------------------------

DECLARE_LAUNCH(row_float_8x32_64rows, float, int32_t)
DECLARE_LAUNCH(row_half_16x64_64rows, aclFloat16, int32_t)
DECLARE_LAUNCH(row_int32_8x16_32rows, int32_t, int32_t)
DECLARE_LAUNCH(row_uint8_8x32_32rows, uint8_t, int32_t)
DECLARE_LAUNCH(row_int16_8x16_32rows, int16_t, int32_t)
DECLARE_LAUNCH(row_float_clamp_8x32_8rows, float, int32_t)
DECLARE_LAUNCH(row_int32_wrap_8x16_8rows, int32_t, int32_t)
DECLARE_LAUNCH(row_half_zero_8x32_8rows, aclFloat16, int32_t)

DECLARE_LAUNCH(row_colidx_float_8x32_64rows, float, int32_t)
DECLARE_LAUNCH(row_colidx_int32_clamp_8x16_8rows, int32_t, int32_t)
DECLARE_LAUNCH(row_colidx_half_16x64_64rows, aclFloat16, int32_t)

DECLARE_LAUNCH(elem_float_64_128size, float, int32_t)
DECLARE_LAUNCH(elem_half_64_128size, aclFloat16, int32_t)
DECLARE_LAUNCH(elem_int32_32_64size, int32_t, int32_t)
DECLARE_LAUNCH(elem_uint8_64_128size, uint8_t, int32_t)
DECLARE_LAUNCH(elem_int16_32_64size, int16_t, int32_t)
DECLARE_LAUNCH(elem_float_clamp_32_16size, float, int32_t)
DECLARE_LAUNCH(elem_int32_wrap_32_16size, int32_t, int32_t)
DECLARE_LAUNCH(elem_half_zero_32_16size, aclFloat16, int32_t)

DECLARE_LAUNCH(elem2d_float_8x32_256size, float, int32_t)
DECLARE_LAUNCH(elem2d_int32_8x16_256size, int32_t, int32_t)
DECLARE_LAUNCH(elem2d_half_4x32_256size, aclFloat16, int32_t)
DECLARE_LAUNCH(elem2d_int32_unaligned_3x8_64size, int32_t, int32_t)
DECLARE_LAUNCH(elem2d_uint8_unaligned_3x32_256size, uint8_t, int32_t)
DECLARE_LAUNCH(elem2d_int32_unaligned_3x3_in_3x8_64size, int32_t, int32_t)
DECLARE_LAUNCH(elem2d_int32_unaligned_9x9_in_9x16_256size, int32_t, int32_t)
DECLARE_LAUNCH(elem2d_int32_scalar_1x1_in_1x8_8size, int32_t, int32_t)
DECLARE_LAUNCH(row_int32_unaligned_3x8_8rows, int32_t, int32_t)
DECLARE_LAUNCH(row_int32_unaligned_9x16_16rows, int32_t, int32_t)

DECLARE_LAUNCH(elem2d_dyn_user_float_1x9_in_1x16_3x10, float, int32_t)
DECLARE_LAUNCH(elem2d_dyn_int32_4x8_in_4x8_64size, int32_t, int32_t)
DECLARE_LAUNCH(elem2d_dyn_float_3x3_in_3x8_64size, float, int32_t)
DECLARE_LAUNCH(elem2d_dyn_half_8x16_in_8x16_4x32, aclFloat16, int32_t)
DECLARE_LAUNCH(row_dyn_int32_3x16_8rows, int32_t, int32_t)
DECLARE_LAUNCH(row_dyn_half_4x32_16rows, aclFloat16, int32_t)

// --- Row coalesce: 1-D index `[1, R]` --------------------------------------------------------------------

ROW_TEST(row_float_8x32_64rows, float, int32_t, 8, 32, 64)
ROW_TEST(row_half_16x64_64rows, aclFloat16, int32_t, 16, 64, 64)
ROW_TEST(row_int32_8x16_32rows, int32_t, int32_t, 8, 16, 32)
ROW_TEST(row_uint8_8x32_32rows, uint8_t, int32_t, 8, 32, 32)
ROW_TEST(row_int16_8x16_32rows, int16_t, int32_t, 8, 16, 32)
ROW_TEST(row_float_clamp_8x32_8rows, float, int32_t, 8, 32, 8)
ROW_TEST(row_int32_wrap_8x16_8rows, int32_t, int32_t, 8, 16, 8)
ROW_TEST(row_half_zero_8x32_8rows, aclFloat16, int32_t, 8, 32, 8)

// --- Row coalesce: 1-D index `[R, 1]` (ColMajor + DN) ----------------------------------------------------

ROW_TEST(row_colidx_float_8x32_64rows, float, int32_t, 8, 32, 64)
ROW_TEST(row_colidx_int32_clamp_8x16_8rows, int32_t, int32_t, 8, 16, 8)
ROW_TEST(row_colidx_half_16x64_64rows, aclFloat16, int32_t, 16, 64, 64)

// --- Element coalesce: 1-D destination `[1, N]` ----------------------------------------------------------

ELEM_TEST(elem_float_64_128size, float, int32_t, 64, 128)
ELEM_TEST(elem_half_64_128size, aclFloat16, int32_t, 64, 128)
ELEM_TEST(elem_int32_32_64size, int32_t, int32_t, 32, 64)
ELEM_TEST(elem_uint8_64_128size, uint8_t, int32_t, 64, 128)
ELEM_TEST(elem_int16_32_64size, int16_t, int32_t, 32, 64)
ELEM_TEST(elem_float_clamp_32_16size, float, int32_t, 32, 16)
ELEM_TEST(elem_int32_wrap_32_16size, int32_t, int32_t, 32, 16)
ELEM_TEST(elem_half_zero_32_16size, aclFloat16, int32_t, 32, 16)

// --- Element coalesce: 2-D destination `[R, C]` ----------------------------------------------------------

ELEM2D_TEST(elem2d_float_8x32_256size, float, int32_t, 8, 32, 256)
ELEM2D_TEST(elem2d_int32_8x16_256size, int32_t, int32_t, 8, 16, 256)
ELEM2D_TEST(elem2d_half_4x32_256size, aclFloat16, int32_t, 4, 32, 256)
ELEM2D_TEST(elem2d_int32_unaligned_3x8_64size, int32_t, int32_t, 3, 8, 64)
ELEM2D_TEST(elem2d_uint8_unaligned_3x32_256size, uint8_t, int32_t, 3, 32, 256)
ELEM2D_TEST(elem2d_int32_unaligned_3x3_in_3x8_64size, int32_t, int32_t, 3, 3, 64)
ELEM2D_TEST(elem2d_int32_unaligned_9x9_in_9x16_256size, int32_t, int32_t, 9, 9, 256)
ELEM2D_TEST(elem2d_int32_scalar_1x1_in_1x8_8size, int32_t, int32_t, 1, 1, 8)
ROW_TEST(row_int32_unaligned_3x8_8rows, int32_t, int32_t, 3, 8, 8)
ROW_TEST(row_int32_unaligned_9x16_16rows, int32_t, int32_t, 9, 16, 16)

// --- Dynamic runtime shapes ------------------------------------------------------------------------------

ELEM2D_DYN_TEST(elem2d_dyn_user_float_1x9_in_1x16_3x10, float, int32_t, 1, 9, 3, 10)
ELEM2D_DYN_TEST(elem2d_dyn_int32_4x8_in_4x8_64size, int32_t, int32_t, 4, 8, 1, 64)
ELEM2D_DYN_TEST(elem2d_dyn_float_3x3_in_3x8_64size, float, int32_t, 3, 3, 1, 64)
ELEM2D_DYN_TEST(elem2d_dyn_half_8x16_in_8x16_4x32, aclFloat16, int32_t, 8, 16, 4, 32)
ROW_TEST(row_dyn_int32_3x16_8rows, int32_t, int32_t, 3, 16, 8)
ROW_TEST(row_dyn_half_4x32_16rows, aclFloat16, int32_t, 4, 32, 16)
