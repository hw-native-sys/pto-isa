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
void run_mgather_test(size_t tableCount, size_t idxCount, size_t outCount, Launcher launcher)
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

DECLARE_LAUNCH(row_float_8x32_64rows, float, int32_t)
DECLARE_LAUNCH(row_half_16x64_64rows, aclFloat16, int32_t)
DECLARE_LAUNCH(row_bfloat16_16x16_64rows, uint16_t, int32_t)
DECLARE_LAUNCH(row_int32_8x16_32rows, int32_t, int32_t)
DECLARE_LAUNCH(row_uint32_8x16_32rows, uint32_t, int32_t)
DECLARE_LAUNCH(row_int16_8x16_32rows, int16_t, int32_t)
DECLARE_LAUNCH(row_uint16_8x16_32rows, uint16_t, int32_t)
DECLARE_LAUNCH(row_int8_8x32_32rows, int8_t, int32_t)
DECLARE_LAUNCH(row_uint8_8x32_32rows, uint8_t, int32_t)
DECLARE_LAUNCH(row_float_clamp_8x32_8rows, float, int32_t)
DECLARE_LAUNCH(row_int32_wrap_8x16_8rows, int32_t, int32_t)
DECLARE_LAUNCH(row_half_zero_8x32_8rows, aclFloat16, int32_t)

DECLARE_LAUNCH(row_int32_unaligned_3x8_8rows, int32_t, int32_t)
DECLARE_LAUNCH(row_float_partial_4x16_in_8x16, float, int32_t)
DECLARE_LAUNCH(row_half_partial_5x32_in_8x32, aclFloat16, int32_t)
DECLARE_LAUNCH(row_uint8_unaligned_3x32_32rows, uint8_t, int32_t)
DECLARE_LAUNCH(row_int16_partial_3x16_in_4x16, int16_t, int32_t)

DECLARE_LAUNCH(elem_float_64_128size, float, int32_t)
DECLARE_LAUNCH(elem_half_64_128size, aclFloat16, int32_t)
DECLARE_LAUNCH(elem_bfloat16_64_128size, uint16_t, int32_t)
DECLARE_LAUNCH(elem_int32_32_64size, int32_t, int32_t)
DECLARE_LAUNCH(elem_uint32_32_64size, uint32_t, int32_t)
DECLARE_LAUNCH(elem_int16_32_64size, int16_t, int32_t)
DECLARE_LAUNCH(elem_uint16_32_64size, uint16_t, int32_t)
DECLARE_LAUNCH(elem_int8_64_128size, int8_t, int32_t)
DECLARE_LAUNCH(elem_uint8_64_128size, uint8_t, int32_t)
DECLARE_LAUNCH(elem_float_clamp_32_16size, float, int32_t)
DECLARE_LAUNCH(elem_int32_wrap_32_16size, int32_t, int32_t)
DECLARE_LAUNCH(elem_half_zero_32_16size, aclFloat16, int32_t)

DECLARE_LAUNCH(elem2d_float_8x32_256size, float, int32_t)
DECLARE_LAUNCH(elem2d_int32_8x16_256size, int32_t, int32_t)
DECLARE_LAUNCH(elem2d_half_4x32_256size, aclFloat16, int32_t)
DECLARE_LAUNCH(elem2d_bfloat16_4x32_256size, uint16_t, int32_t)
DECLARE_LAUNCH(elem2d_uint8_4x64_256size, uint8_t, int32_t)
DECLARE_LAUNCH(elem2d_int8_4x64_256size, int8_t, int32_t)
DECLARE_LAUNCH(elem2d_int16_4x32_256size, int16_t, int32_t)
DECLARE_LAUNCH(elem2d_uint16_4x32_256size, uint16_t, int32_t)
DECLARE_LAUNCH(elem2d_uint32_8x16_256size, uint32_t, int32_t)
DECLARE_LAUNCH(elem2d_float_wrap_4x16_64size, float, int32_t)
DECLARE_LAUNCH(elem2d_int32_clamp_4x8_32size, int32_t, int32_t)
DECLARE_LAUNCH(elem2d_half_zero_4x32_64size, aclFloat16, int32_t)

DECLARE_LAUNCH(elem2d_int32_unaligned_3x3_in_3x8_64size, int32_t, int32_t)
DECLARE_LAUNCH(elem2d_float_unaligned_5x5_in_5x8_64size, float, int32_t)
DECLARE_LAUNCH(elem2d_half_unaligned_3x9_in_3x16_64size, aclFloat16, int32_t)
DECLARE_LAUNCH(elem2d_int8_unaligned_3x17_in_3x32_64size, int8_t, int32_t)

DECLARE_LAUNCH(elem_scalar_float_1x1_in_1x8_8size, float, int32_t)
DECLARE_LAUNCH(elem_scalar_int32_1x1_in_1x8_8size, int32_t, int32_t)
DECLARE_LAUNCH(elem_scalar_half_1x1_in_1x16_16size, aclFloat16, int32_t)

DECLARE_LAUNCH(elem2d_dyn_float_4x8_64size, float, int32_t)
DECLARE_LAUNCH(elem2d_dyn_int32_3x3_in_3x8_64size, int32_t, int32_t)
DECLARE_LAUNCH(row_dyn_int32_3x16_8rows, int32_t, int32_t)
DECLARE_LAUNCH(row_dyn_half_4x32_16rows, aclFloat16, int32_t)

DECLARE_LAUNCH(row_nz_float_16x16_2blk, float, int32_t)
DECLARE_LAUNCH(row_nz_half_32x16_2blk, aclFloat16, int32_t)
DECLARE_LAUNCH(row_nz_int32_16x16_2blk, int32_t, int32_t)
DECLARE_LAUNCH(row_nz_int16_32x16_1blk, int16_t, int32_t)
DECLARE_LAUNCH(row_nz_int8_16x32_1blk, int8_t, int32_t)
DECLARE_LAUNCH(row_nz_float_clamp_16x8_1blk, float, int32_t)
DECLARE_LAUNCH(row_nz_half_zero_16x16_2blk, aclFloat16, int32_t)
DECLARE_LAUNCH(elem2d_nz_float_16x16_2blk, float, int32_t)
DECLARE_LAUNCH(elem2d_nz_half_16x16_1blk, aclFloat16, int32_t)
DECLARE_LAUNCH(elem2d_nz_int32_16x8_1blk, int32_t, int32_t)
DECLARE_LAUNCH(elem2d_nz_half_zero_16x16_1blk, aclFloat16, int32_t)

#define ROW_TEST(NAME, THOST, TIDX, R, C, TR)                                                   \
    TEST_F(MGATHERTest, case_##NAME)                                                            \
    {                                                                                           \
        run_mgather_test<THOST, TIDX>((size_t)TR * C, (size_t)R, (size_t)R * C, Launch_##NAME); \
    }

#define ELEM_TEST(NAME, THOST, TIDX, N, TS)                                             \
    TEST_F(MGATHERTest, case_##NAME)                                                    \
    {                                                                                   \
        run_mgather_test<THOST, TIDX>((size_t)TS, (size_t)N, (size_t)N, Launch_##NAME); \
    }

#define ELEM2D_TEST(NAME, THOST, TIDX, R, C, TS)                                                \
    TEST_F(MGATHERTest, case_##NAME)                                                            \
    {                                                                                           \
        run_mgather_test<THOST, TIDX>((size_t)TS, (size_t)R * C, (size_t)R * C, Launch_##NAME); \
    }

#define SCALAR_TEST(NAME, THOST, TIDX, TS)                                              \
    TEST_F(MGATHERTest, case_##NAME)                                                    \
    {                                                                                   \
        run_mgather_test<THOST, TIDX>((size_t)TS, (size_t)1, (size_t)1, Launch_##NAME); \
    }

ROW_TEST(row_float_8x32_64rows, float, int32_t, 8, 32, 64)
ROW_TEST(row_half_16x64_64rows, aclFloat16, int32_t, 16, 64, 64)
ROW_TEST(row_bfloat16_16x16_64rows, uint16_t, int32_t, 16, 16, 64)
ROW_TEST(row_int32_8x16_32rows, int32_t, int32_t, 8, 16, 32)
ROW_TEST(row_uint32_8x16_32rows, uint32_t, int32_t, 8, 16, 32)
ROW_TEST(row_int16_8x16_32rows, int16_t, int32_t, 8, 16, 32)
ROW_TEST(row_uint16_8x16_32rows, uint16_t, int32_t, 8, 16, 32)
ROW_TEST(row_int8_8x32_32rows, int8_t, int32_t, 8, 32, 32)
ROW_TEST(row_uint8_8x32_32rows, uint8_t, int32_t, 8, 32, 32)
ROW_TEST(row_float_clamp_8x32_8rows, float, int32_t, 8, 32, 8)
ROW_TEST(row_int32_wrap_8x16_8rows, int32_t, int32_t, 8, 16, 8)
ROW_TEST(row_half_zero_8x32_8rows, aclFloat16, int32_t, 8, 32, 8)

ROW_TEST(row_int32_unaligned_3x8_8rows, int32_t, int32_t, 3, 8, 8)
ROW_TEST(row_float_partial_4x16_in_8x16, float, int32_t, 4, 16, 8)
ROW_TEST(row_half_partial_5x32_in_8x32, aclFloat16, int32_t, 5, 32, 8)
ROW_TEST(row_uint8_unaligned_3x32_32rows, uint8_t, int32_t, 3, 32, 8)
ROW_TEST(row_int16_partial_3x16_in_4x16, int16_t, int32_t, 3, 16, 8)

ELEM_TEST(elem_float_64_128size, float, int32_t, 64, 128)
ELEM_TEST(elem_half_64_128size, aclFloat16, int32_t, 64, 128)
ELEM_TEST(elem_bfloat16_64_128size, uint16_t, int32_t, 64, 128)
ELEM_TEST(elem_int32_32_64size, int32_t, int32_t, 32, 64)
ELEM_TEST(elem_uint32_32_64size, uint32_t, int32_t, 32, 64)
ELEM_TEST(elem_int16_32_64size, int16_t, int32_t, 32, 64)
ELEM_TEST(elem_uint16_32_64size, uint16_t, int32_t, 32, 64)
ELEM_TEST(elem_int8_64_128size, int8_t, int32_t, 64, 128)
ELEM_TEST(elem_uint8_64_128size, uint8_t, int32_t, 64, 128)
ELEM_TEST(elem_float_clamp_32_16size, float, int32_t, 32, 16)
ELEM_TEST(elem_int32_wrap_32_16size, int32_t, int32_t, 32, 16)
ELEM_TEST(elem_half_zero_32_16size, aclFloat16, int32_t, 32, 16)

ELEM2D_TEST(elem2d_float_8x32_256size, float, int32_t, 8, 32, 256)
ELEM2D_TEST(elem2d_int32_8x16_256size, int32_t, int32_t, 8, 16, 256)
ELEM2D_TEST(elem2d_half_4x32_256size, aclFloat16, int32_t, 4, 32, 256)
ELEM2D_TEST(elem2d_bfloat16_4x32_256size, uint16_t, int32_t, 4, 32, 256)
ELEM2D_TEST(elem2d_uint8_4x64_256size, uint8_t, int32_t, 4, 64, 256)
ELEM2D_TEST(elem2d_int8_4x64_256size, int8_t, int32_t, 4, 64, 256)
ELEM2D_TEST(elem2d_int16_4x32_256size, int16_t, int32_t, 4, 32, 256)
ELEM2D_TEST(elem2d_uint16_4x32_256size, uint16_t, int32_t, 4, 32, 256)
ELEM2D_TEST(elem2d_uint32_8x16_256size, uint32_t, int32_t, 8, 16, 256)
ELEM2D_TEST(elem2d_float_wrap_4x16_64size, float, int32_t, 4, 16, 64)
ELEM2D_TEST(elem2d_int32_clamp_4x8_32size, int32_t, int32_t, 4, 8, 32)
ELEM2D_TEST(elem2d_half_zero_4x32_64size, aclFloat16, int32_t, 4, 32, 64)

ELEM2D_TEST(elem2d_int32_unaligned_3x3_in_3x8_64size, int32_t, int32_t, 3, 3, 64)
ELEM2D_TEST(elem2d_float_unaligned_5x5_in_5x8_64size, float, int32_t, 5, 5, 64)
ELEM2D_TEST(elem2d_half_unaligned_3x9_in_3x16_64size, aclFloat16, int32_t, 3, 9, 64)
ELEM2D_TEST(elem2d_int8_unaligned_3x17_in_3x32_64size, int8_t, int32_t, 3, 17, 64)

SCALAR_TEST(elem_scalar_float_1x1_in_1x8_8size, float, int32_t, 8)
SCALAR_TEST(elem_scalar_int32_1x1_in_1x8_8size, int32_t, int32_t, 8)
SCALAR_TEST(elem_scalar_half_1x1_in_1x16_16size, aclFloat16, int32_t, 16)

#define ELEM2D_DYN_TEST(NAME, THOST, TIDX, RVR, RVC, RTS)                                                \
    TEST_F(MGATHERTest, case_##NAME)                                                                     \
    {                                                                                                    \
        run_mgather_test<THOST, TIDX>((size_t)RTS, (size_t)RVR * RVC, (size_t)RVR * RVC, Launch_##NAME); \
    }

ELEM2D_DYN_TEST(elem2d_dyn_float_4x8_64size, float, int32_t, 4, 8, 64)
ELEM2D_DYN_TEST(elem2d_dyn_int32_3x3_in_3x8_64size, int32_t, int32_t, 3, 3, 64)
ROW_TEST(row_dyn_int32_3x16_8rows, int32_t, int32_t, 3, 16, 8)
ROW_TEST(row_dyn_half_4x32_16rows, aclFloat16, int32_t, 4, 32, 16)

#define ROW_NZ_TEST(NAME, THOST, TIDX, R, C, BR, BC, C0)                                                   \
    TEST_F(MGATHERTest, case_##NAME)                                                                       \
    {                                                                                                      \
        run_mgather_test<THOST, TIDX>((size_t)BR * 16 * BC * C0, (size_t)R, (size_t)R * C, Launch_##NAME); \
    }

#define ELEM2D_NZ_TEST(NAME, THOST, TIDX, R, C, BR, BC, C0)                                                    \
    TEST_F(MGATHERTest, case_##NAME)                                                                           \
    {                                                                                                          \
        run_mgather_test<THOST, TIDX>((size_t)BR * 16 * BC * C0, (size_t)R * C, (size_t)R * C, Launch_##NAME); \
    }

ROW_NZ_TEST(row_nz_float_16x16_2blk, float, int32_t, 16, 16, 2, 2, 8)
ROW_NZ_TEST(row_nz_half_32x16_2blk, aclFloat16, int32_t, 32, 16, 2, 1, 16)
ROW_NZ_TEST(row_nz_int32_16x16_2blk, int32_t, int32_t, 16, 16, 2, 2, 8)
ROW_NZ_TEST(row_nz_int16_32x16_1blk, int16_t, int32_t, 32, 16, 2, 1, 16)
ROW_NZ_TEST(row_nz_int8_16x32_1blk, int8_t, int32_t, 16, 32, 2, 1, 32)
ROW_NZ_TEST(row_nz_float_clamp_16x8_1blk, float, int32_t, 16, 8, 2, 1, 8)
ROW_NZ_TEST(row_nz_half_zero_16x16_2blk, aclFloat16, int32_t, 16, 16, 2, 1, 16)

ELEM2D_NZ_TEST(elem2d_nz_float_16x16_2blk, float, int32_t, 16, 16, 2, 2, 8)
ELEM2D_NZ_TEST(elem2d_nz_half_16x16_1blk, aclFloat16, int32_t, 16, 16, 2, 1, 16)
ELEM2D_NZ_TEST(elem2d_nz_int32_16x8_1blk, int32_t, int32_t, 16, 8, 2, 1, 8)
ELEM2D_NZ_TEST(elem2d_nz_half_zero_16x16_1blk, aclFloat16, int32_t, 16, 16, 2, 1, 16)
