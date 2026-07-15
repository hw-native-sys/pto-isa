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

class MSCATTERTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

static std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return std::string("../") + testInfo->test_suite_name() + "." + testInfo->name();
}

template <typename T, typename TIdx, typename Launcher>
void RunMScatterTest(size_t srcCount, size_t idxCount, size_t outCount, Launcher launcher)
{
    size_t srcByteSize = srcCount * sizeof(T);
    size_t idxByteSize = idxCount * sizeof(TIdx);
    size_t outByteSize = outCount * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *srcHost, *outHost;
    TIdx* idxHost;
    T *srcDevice, *outDevice;
    TIdx* idxDevice;

    aclrtMallocHost((void**)(&srcHost), srcByteSize);
    aclrtMallocHost((void**)(&idxHost), idxByteSize);
    aclrtMallocHost((void**)(&outHost), outByteSize);

    aclrtMalloc((void**)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&idxDevice, idxByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&outDevice, outByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/src.bin", srcByteSize, srcHost, srcByteSize);
    ReadFile(GetGoldenDir() + "/indices.bin", idxByteSize, idxHost, idxByteSize);

    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(idxDevice, idxByteSize, idxHost, idxByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtMemset(outDevice, outByteSize, 0, outByteSize);

    launcher(outDevice, srcDevice, idxDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, outByteSize, outDevice, outByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", outHost, outByteSize);

    aclrtFree(srcDevice);
    aclrtFree(idxDevice);
    aclrtFree(outDevice);

    aclrtFreeHost(srcHost);
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

#define DECLARE_LAUNCH(NAME, THOST, TIDX) void Launch_##NAME(THOST* out, THOST* src, TIDX* indices, void* stream);

#define ROW_TEST(NAME, THOST, TIDX, R, C, TR)                                                  \
    TEST_F(MSCATTERTest, case_##NAME)                                                          \
    {                                                                                          \
        RunMScatterTest<THOST, TIDX>((size_t)R * C, (size_t)R, (size_t)TR * C, Launch_##NAME); \
    }

#define ELEM_TEST(NAME, THOST, TIDX, N, TS) \
    TEST_F(MSCATTERTest, case_##NAME) { RunMScatterTest<THOST, TIDX>((size_t)N, (size_t)N, (size_t)TS, Launch_##NAME); }

#define ELEM2D_TEST(NAME, THOST, TIDX, R, C, TS)                                               \
    TEST_F(MSCATTERTest, case_##NAME)                                                          \
    {                                                                                          \
        RunMScatterTest<THOST, TIDX>((size_t)R * C, (size_t)R * C, (size_t)TS, Launch_##NAME); \
    }

#define ELEM2D_DYN_TEST(NAME, THOST, TIDX, RVR, RVC, RTR, RTC)                                                \
    TEST_F(MSCATTERTest, case_##NAME)                                                                         \
    {                                                                                                         \
        RunMScatterTest<THOST, TIDX>((size_t)RVR * RVC, (size_t)RVR * RVC, (size_t)RTR * RTC, Launch_##NAME); \
    }

// --- Launcher declarations ------------------------------------------------------------------------------

DECLARE_LAUNCH(row_float_random_8x32_64rows, float, int32_t)
DECLARE_LAUNCH(row_float_same_8x32_16rows, float, int32_t)
DECLARE_LAUNCH(row_half_random_16x64_64rows, aclFloat16, int32_t)
DECLARE_LAUNCH(row_int32_random_8x16_32rows, int32_t, int32_t)
DECLARE_LAUNCH(row_uint8_random_8x32_32rows, uint8_t, int32_t)
DECLARE_LAUNCH(row_int16_random_8x16_32rows, int16_t, int32_t)
DECLARE_LAUNCH(row_float_atomicadd_8x32_8rows, float, int32_t)
DECLARE_LAUNCH(row_float_skip_8x32_8rows, float, int32_t)
DECLARE_LAUNCH(row_int32_clamp_8x16_8rows, int32_t, int32_t)
DECLARE_LAUNCH(row_half_wrap_8x32_8rows, aclFloat16, int32_t)

DECLARE_LAUNCH(row_colidx_float_random_8x32_64rows, float, int32_t)
DECLARE_LAUNCH(row_colidx_int32_clamp_8x16_8rows, int32_t, int32_t)
DECLARE_LAUNCH(row_colidx_half_random_16x64_64rows, aclFloat16, int32_t)

DECLARE_LAUNCH(elem_float_random_64_128size, float, int32_t)
DECLARE_LAUNCH(elem_float_same_64_8size, float, int32_t)
DECLARE_LAUNCH(elem_float_seq_32_32size, float, int32_t)
DECLARE_LAUNCH(elem_half_random_64_128size, aclFloat16, int32_t)
DECLARE_LAUNCH(elem_int32_random_32_64size, int32_t, int32_t)
DECLARE_LAUNCH(elem_uint8_random_64_128size, uint8_t, int32_t)
DECLARE_LAUNCH(elem_int16_random_32_64size, int16_t, int32_t)
DECLARE_LAUNCH(elem_float_atomicadd_32_32size, float, int32_t)
DECLARE_LAUNCH(elem_int32_atomicadd_skip_32_16size, int32_t, int32_t)
DECLARE_LAUNCH(elem_float_skip_32_16size, float, int32_t)
DECLARE_LAUNCH(elem_int32_clamp_32_16size, int32_t, int32_t)
DECLARE_LAUNCH(elem_half_wrap_32_16size, aclFloat16, int32_t)
DECLARE_LAUNCH(elem_float_default_seq_32_32size, float, int32_t)
DECLARE_LAUNCH(elem_float_small_16_32size, float, int32_t)
DECLARE_LAUNCH(elem_int32_atomicmax_random_32_32size, int32_t, int32_t)
DECLARE_LAUNCH(elem_float_atomicmin_random_32_32size, float, int32_t)
DECLARE_LAUNCH(elem_float_last_same_32_8size, float, int32_t)
DECLARE_LAUNCH(elem_int32_last_seq_32_32size, int32_t, int32_t)
DECLARE_LAUNCH(elem_float_clamp_no_dup_32_16size, float, int32_t)
DECLARE_LAUNCH(elem_uint8_wrap_64_16size, uint8_t, int32_t)
DECLARE_LAUNCH(elem_int16_clamp_32_16size, int16_t, int32_t)

DECLARE_LAUNCH(elem2d_float_8x32_random_256size, float, int32_t)
DECLARE_LAUNCH(elem2d_int32_8x16_random_256size, int32_t, int32_t)
DECLARE_LAUNCH(elem2d_half_4x32_random_256size, aclFloat16, int32_t)
DECLARE_LAUNCH(elem2d_int32_unaligned_3x8_64size, int32_t, int32_t)
DECLARE_LAUNCH(elem2d_uint8_unaligned_3x32_256size, uint8_t, int32_t)
DECLARE_LAUNCH(elem2d_int32_unaligned_3x3_in_3x8_64size, int32_t, int32_t)
DECLARE_LAUNCH(elem2d_int32_unaligned_9x9_in_9x16_256size, int32_t, int32_t)
DECLARE_LAUNCH(elem2d_int32_scalar_1x1_in_1x8_8size, int32_t, int32_t)
DECLARE_LAUNCH(row_int32_unaligned_3x8_8rows, int32_t, int32_t)
DECLARE_LAUNCH(row_int32_unaligned_9x16_16rows, int32_t, int32_t)

DECLARE_LAUNCH(elem2d_float_2048x8_last_256size, float, int32_t)
DECLARE_LAUNCH(elem2d_float_2048x8_default_16384size, float, int32_t)

DECLARE_LAUNCH(elem2d_float_2304x8_last_256size, float, int32_t)
DECLARE_LAUNCH(elem2d_float_2304x8_default_18432size, float, int32_t)
DECLARE_LAUNCH(elem2d_float_2560x8_last_256size, float, int32_t)
DECLARE_LAUNCH(elem2d_float_2560x8_default_20480size, float, int32_t)
DECLARE_LAUNCH(elem2d_float_2816x8_last_256size, float, int32_t)
DECLARE_LAUNCH(elem2d_float_2816x8_default_22528size, float, int32_t)
DECLARE_LAUNCH(elem2d_float_3072x8_last_256size, float, int32_t)
DECLARE_LAUNCH(elem2d_float_3072x8_default_24576size, float, int32_t)
DECLARE_LAUNCH(elem2d_float_3200x8_last_256size, float, int32_t)
DECLARE_LAUNCH(elem2d_float_3200x8_default_25600size, float, int32_t)
DECLARE_LAUNCH(elem2d_float_3456x8_last_256size, float, int32_t)
DECLARE_LAUNCH(elem2d_float_3456x8_default_27648size, float, int32_t)

DECLARE_LAUNCH(elem2d_dyn_user_float_1x9_in_1x16_3x10, float, int32_t)
DECLARE_LAUNCH(elem2d_dyn_int32_4x8_in_4x8_64size, int32_t, int32_t)
DECLARE_LAUNCH(elem2d_dyn_float_3x3_in_3x8_64size, float, int32_t)
DECLARE_LAUNCH(elem2d_dyn_half_8x16_in_8x16_4x32, aclFloat16, int32_t)
DECLARE_LAUNCH(row_dyn_int32_3x16_8rows, int32_t, int32_t)
DECLARE_LAUNCH(row_dyn_half_4x32_16rows, aclFloat16, int32_t)

// --- Row coalesce: 1-D index `[1, R]` --------------------------------------------------------------------

ROW_TEST(row_float_random_8x32_64rows, float, int32_t, 8, 32, 64)
ROW_TEST(row_float_same_8x32_16rows, float, int32_t, 8, 32, 16)
ROW_TEST(row_half_random_16x64_64rows, aclFloat16, int32_t, 16, 64, 64)
ROW_TEST(row_int32_random_8x16_32rows, int32_t, int32_t, 8, 16, 32)
ROW_TEST(row_uint8_random_8x32_32rows, uint8_t, int32_t, 8, 32, 32)
ROW_TEST(row_int16_random_8x16_32rows, int16_t, int32_t, 8, 16, 32)
ROW_TEST(row_float_atomicadd_8x32_8rows, float, int32_t, 8, 32, 8)
ROW_TEST(row_float_skip_8x32_8rows, float, int32_t, 8, 32, 8)
ROW_TEST(row_int32_clamp_8x16_8rows, int32_t, int32_t, 8, 16, 8)
ROW_TEST(row_half_wrap_8x32_8rows, aclFloat16, int32_t, 8, 32, 8)

// --- Row coalesce: 1-D index `[R, 1]` (ColMajor + DN) ----------------------------------------------------

ROW_TEST(row_colidx_float_random_8x32_64rows, float, int32_t, 8, 32, 64)
ROW_TEST(row_colidx_int32_clamp_8x16_8rows, int32_t, int32_t, 8, 16, 8)
ROW_TEST(row_colidx_half_random_16x64_64rows, aclFloat16, int32_t, 16, 64, 64)

// --- Element coalesce: 1-D source `[1, N]` ---------------------------------------------------------------

ELEM_TEST(elem_float_random_64_128size, float, int32_t, 64, 128)
ELEM_TEST(elem_float_same_64_8size, float, int32_t, 64, 8)
ELEM_TEST(elem_float_seq_32_32size, float, int32_t, 32, 32)
ELEM_TEST(elem_half_random_64_128size, aclFloat16, int32_t, 64, 128)
ELEM_TEST(elem_int32_random_32_64size, int32_t, int32_t, 32, 64)
ELEM_TEST(elem_uint8_random_64_128size, uint8_t, int32_t, 64, 128)
ELEM_TEST(elem_int16_random_32_64size, int16_t, int32_t, 32, 64)
ELEM_TEST(elem_float_atomicadd_32_32size, float, int32_t, 32, 32)
ELEM_TEST(elem_int32_atomicadd_skip_32_16size, int32_t, int32_t, 32, 16)
ELEM_TEST(elem_float_skip_32_16size, float, int32_t, 32, 16)
ELEM_TEST(elem_int32_clamp_32_16size, int32_t, int32_t, 32, 16)
ELEM_TEST(elem_half_wrap_32_16size, aclFloat16, int32_t, 32, 16)
ELEM_TEST(elem_float_default_seq_32_32size, float, int32_t, 32, 32)
ELEM_TEST(elem_float_small_16_32size, float, int32_t, 16, 32)
ELEM_TEST(elem_int32_atomicmax_random_32_32size, int32_t, int32_t, 32, 32)
ELEM_TEST(elem_float_atomicmin_random_32_32size, float, int32_t, 32, 32)
ELEM_TEST(elem_float_last_same_32_8size, float, int32_t, 32, 8)
ELEM_TEST(elem_int32_last_seq_32_32size, int32_t, int32_t, 32, 32)
ELEM_TEST(elem_float_clamp_no_dup_32_16size, float, int32_t, 32, 16)
ELEM_TEST(elem_uint8_wrap_64_16size, uint8_t, int32_t, 64, 16)
ELEM_TEST(elem_int16_clamp_32_16size, int16_t, int32_t, 32, 16)

// --- Element coalesce: 2-D source `[R, C]` ---------------------------------------------------------------

ELEM2D_TEST(elem2d_float_8x32_random_256size, float, int32_t, 8, 32, 256)
ELEM2D_TEST(elem2d_int32_8x16_random_256size, int32_t, int32_t, 8, 16, 256)
ELEM2D_TEST(elem2d_half_4x32_random_256size, aclFloat16, int32_t, 4, 32, 256)
ELEM2D_TEST(elem2d_int32_unaligned_3x8_64size, int32_t, int32_t, 3, 8, 64)
ELEM2D_TEST(elem2d_uint8_unaligned_3x32_256size, uint8_t, int32_t, 3, 32, 256)
ELEM2D_TEST(elem2d_int32_unaligned_3x3_in_3x8_64size, int32_t, int32_t, 3, 3, 64)
ELEM2D_TEST(elem2d_int32_unaligned_9x9_in_9x16_256size, int32_t, int32_t, 9, 9, 256)
ELEM2D_TEST(elem2d_int32_scalar_1x1_in_1x8_8size, int32_t, int32_t, 1, 1, 8)
ROW_TEST(row_int32_unaligned_3x8_8rows, int32_t, int32_t, 3, 8, 8)
ROW_TEST(row_int32_unaligned_9x16_16rows, int32_t, int32_t, 9, 16, 16)

// --- Element coalesce: tiled iteration (large source, default UB budget) ---------------------------------

ELEM2D_TEST(elem2d_float_2048x8_last_256size, float, int32_t, 2048, 8, 256)
ELEM2D_TEST(elem2d_float_2048x8_default_16384size, float, int32_t, 2048, 8, 16384)

// --- Element coalesce: extended UB (`dynUBufSize`) single-shot, 144 KB .. 216 KB -------------------------

ELEM2D_TEST(elem2d_float_2304x8_last_256size, float, int32_t, 2304, 8, 256)
ELEM2D_TEST(elem2d_float_2304x8_default_18432size, float, int32_t, 2304, 8, 18432)
ELEM2D_TEST(elem2d_float_2560x8_last_256size, float, int32_t, 2560, 8, 256)
ELEM2D_TEST(elem2d_float_2560x8_default_20480size, float, int32_t, 2560, 8, 20480)
ELEM2D_TEST(elem2d_float_2816x8_last_256size, float, int32_t, 2816, 8, 256)
ELEM2D_TEST(elem2d_float_2816x8_default_22528size, float, int32_t, 2816, 8, 22528)
ELEM2D_TEST(elem2d_float_3072x8_last_256size, float, int32_t, 3072, 8, 256)
ELEM2D_TEST(elem2d_float_3072x8_default_24576size, float, int32_t, 3072, 8, 24576)
ELEM2D_TEST(elem2d_float_3200x8_last_256size, float, int32_t, 3200, 8, 256)
ELEM2D_TEST(elem2d_float_3200x8_default_25600size, float, int32_t, 3200, 8, 25600)
ELEM2D_TEST(elem2d_float_3456x8_last_256size, float, int32_t, 3456, 8, 256)
ELEM2D_TEST(elem2d_float_3456x8_default_27648size, float, int32_t, 3456, 8, 27648)

// --- Dynamic runtime shapes ------------------------------------------------------------------------------

ELEM2D_DYN_TEST(elem2d_dyn_user_float_1x9_in_1x16_3x10, float, int32_t, 1, 9, 3, 10)
ELEM2D_DYN_TEST(elem2d_dyn_int32_4x8_in_4x8_64size, int32_t, int32_t, 4, 8, 1, 64)
ELEM2D_DYN_TEST(elem2d_dyn_float_3x3_in_3x8_64size, float, int32_t, 3, 3, 1, 64)
ELEM2D_DYN_TEST(elem2d_dyn_half_8x16_in_8x16_4x32, aclFloat16, int32_t, 8, 16, 4, 32)
ROW_TEST(row_dyn_int32_3x16_8rows, int32_t, int32_t, 3, 16, 8)
ROW_TEST(row_dyn_half_4x32_16rows, aclFloat16, int32_t, 4, 32, 16)
