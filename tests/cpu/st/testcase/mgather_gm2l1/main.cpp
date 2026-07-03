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
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class MGATHERGM2L1Test : public testing::Test {
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
void run_gm2l1_test(size_t tableCount, size_t idxCount, size_t outCount, Launcher launcher)
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
    T *tableDevice, *outDevice, *scratchDevice;
    TIdx *idxDevice;

    aclrtMallocHost((void **)(&tableHost), tableByteSize);
    aclrtMallocHost((void **)(&idxHost), idxByteSize);
    aclrtMallocHost((void **)(&outHost), outByteSize);

    aclrtMalloc((void **)&tableDevice, tableByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&idxDevice, idxByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outDevice, outByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&scratchDevice, outByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/table.bin", tableByteSize, tableHost, tableByteSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/indices.bin", idxByteSize, idxHost, idxByteSize));

    aclrtMemcpy(tableDevice, tableByteSize, tableHost, tableByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(idxDevice, idxByteSize, idxHost, idxByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtMemset(outDevice, outByteSize, 0, outByteSize);
    aclrtMemset(scratchDevice, outByteSize, 0, outByteSize);

    launcher(outDevice, tableDevice, idxDevice, scratchDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, outByteSize, outDevice, outByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", outHost, outByteSize);

    aclrtFree(tableDevice);
    aclrtFree(idxDevice);
    aclrtFree(outDevice);
    aclrtFree(scratchDevice);

    aclrtFreeHost(tableHost);
    aclrtFreeHost(idxHost);
    aclrtFreeHost(outHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(outCount);
    std::vector<T> devFinal(outCount);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", outByteSize, golden.data(), outByteSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", outByteSize, devFinal.data(), outByteSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.0f);
    EXPECT_TRUE(ret);
}

#define DECLARE_LAUNCH(NAME, THOST, TIDX) \
    void Launch_##NAME(THOST *out, THOST *table, TIDX *indices, THOST *scratch, void *stream);

DECLARE_LAUNCH(row_float_16x16_64rows, float, int32_t)
DECLARE_LAUNCH(row_half_16x32_64rows, aclFloat16, int32_t)
DECLARE_LAUNCH(row_bfloat16_16x16_64rows, uint16_t, int32_t)
DECLARE_LAUNCH(row_int32_16x8_32rows, int32_t, int32_t)
DECLARE_LAUNCH(row_uint32_16x16_64rows, uint32_t, int32_t)
DECLARE_LAUNCH(row_int16_16x16_32rows, int16_t, int32_t)
DECLARE_LAUNCH(row_uint16_16x32_48rows, uint16_t, int32_t)
DECLARE_LAUNCH(row_int8_16x32_64rows, int8_t, int32_t)
DECLARE_LAUNCH(row_uint8_32x32_64rows, uint8_t, int32_t)
DECLARE_LAUNCH(row_float_clamp_16x16_8rows, float, int32_t)
DECLARE_LAUNCH(row_int32_wrap_16x8_8rows, int32_t, int32_t)
DECLARE_LAUNCH(row_half_zero_16x16_8rows, aclFloat16, int32_t)

DECLARE_LAUNCH(elem_float_16x16_256size, float, int32_t)
DECLARE_LAUNCH(elem_half_16x16_256size, aclFloat16, int32_t)
DECLARE_LAUNCH(elem_bfloat16_16x16_256size, uint16_t, int32_t)
DECLARE_LAUNCH(elem_int32_16x8_128size, int32_t, int32_t)
DECLARE_LAUNCH(elem_uint32_16x16_256size, uint32_t, int32_t)
DECLARE_LAUNCH(elem_int16_16x16_256size, int16_t, int32_t)
DECLARE_LAUNCH(elem_uint16_16x32_512size, uint16_t, int32_t)
DECLARE_LAUNCH(elem_int8_16x32_512size, int8_t, int32_t)
DECLARE_LAUNCH(elem_uint8_32x32_1024size, uint8_t, int32_t)
DECLARE_LAUNCH(elem_float_clamp_16x16_64size, float, int32_t)
DECLARE_LAUNCH(elem_int32_wrap_16x8_32size, int32_t, int32_t)
DECLARE_LAUNCH(elem_half_zero_16x16_64size, aclFloat16, int32_t)

DECLARE_LAUNCH(elem_simt_float_16x16_256size, float, int32_t)
DECLARE_LAUNCH(elem_simt_half_16x16_256size, aclFloat16, int32_t)
DECLARE_LAUNCH(elem_simt_bfloat16_16x16_256size, uint16_t, int32_t)
DECLARE_LAUNCH(elem_simt_int32_16x8_128size, int32_t, int32_t)
DECLARE_LAUNCH(elem_simt_uint32_16x16_256size, uint32_t, int32_t)
DECLARE_LAUNCH(elem_simt_int16_16x16_256size, int16_t, int32_t)
DECLARE_LAUNCH(elem_simt_uint16_16x32_512size, uint16_t, int32_t)
DECLARE_LAUNCH(elem_simt_int8_16x32_512size, int8_t, int32_t)
DECLARE_LAUNCH(elem_simt_uint8_32x32_1024size, uint8_t, int32_t)
DECLARE_LAUNCH(elem_simt_float_clamp_16x16_64size, float, int32_t)
DECLARE_LAUNCH(elem_simt_int32_wrap_16x8_32size, int32_t, int32_t)
DECLARE_LAUNCH(elem_simt_half_zero_16x16_64size, aclFloat16, int32_t)

#define ROW_TEST(NAME, THOST, TIDX, R, C, TR)                                                 \
    TEST_F(MGATHERGM2L1Test, case_##NAME)                                                     \
    {                                                                                         \
        run_gm2l1_test<THOST, TIDX>((size_t)TR * C, (size_t)R, (size_t)R * C, Launch_##NAME); \
    }

#define ELEM_TEST(NAME, THOST, TIDX, R, C, TS)                                                \
    TEST_F(MGATHERGM2L1Test, case_##NAME)                                                     \
    {                                                                                         \
        run_gm2l1_test<THOST, TIDX>((size_t)TS, (size_t)R * C, (size_t)R * C, Launch_##NAME); \
    }

ROW_TEST(row_float_16x16_64rows, float, int32_t, 16, 16, 64)
ROW_TEST(row_half_16x32_64rows, aclFloat16, int32_t, 16, 32, 64)
ROW_TEST(row_bfloat16_16x16_64rows, uint16_t, int32_t, 16, 16, 64)
ROW_TEST(row_int32_16x8_32rows, int32_t, int32_t, 16, 8, 32)
ROW_TEST(row_uint32_16x16_64rows, uint32_t, int32_t, 16, 16, 64)
ROW_TEST(row_int16_16x16_32rows, int16_t, int32_t, 16, 16, 32)
ROW_TEST(row_uint16_16x32_48rows, uint16_t, int32_t, 16, 32, 48)
ROW_TEST(row_int8_16x32_64rows, int8_t, int32_t, 16, 32, 64)
ROW_TEST(row_uint8_32x32_64rows, uint8_t, int32_t, 32, 32, 64)
ROW_TEST(row_float_clamp_16x16_8rows, float, int32_t, 16, 16, 8)
ROW_TEST(row_int32_wrap_16x8_8rows, int32_t, int32_t, 16, 8, 8)
ROW_TEST(row_half_zero_16x16_8rows, aclFloat16, int32_t, 16, 16, 8)

ELEM_TEST(elem_float_16x16_256size, float, int32_t, 16, 16, 256)
ELEM_TEST(elem_half_16x16_256size, aclFloat16, int32_t, 16, 16, 256)
ELEM_TEST(elem_bfloat16_16x16_256size, uint16_t, int32_t, 16, 16, 256)
ELEM_TEST(elem_int32_16x8_128size, int32_t, int32_t, 16, 8, 128)
ELEM_TEST(elem_uint32_16x16_256size, uint32_t, int32_t, 16, 16, 256)
ELEM_TEST(elem_int16_16x16_256size, int16_t, int32_t, 16, 16, 256)
ELEM_TEST(elem_uint16_16x32_512size, uint16_t, int32_t, 16, 32, 512)
ELEM_TEST(elem_int8_16x32_512size, int8_t, int32_t, 16, 32, 512)
ELEM_TEST(elem_uint8_32x32_1024size, uint8_t, int32_t, 32, 32, 1024)
ELEM_TEST(elem_float_clamp_16x16_64size, float, int32_t, 16, 16, 64)
ELEM_TEST(elem_int32_wrap_16x8_32size, int32_t, int32_t, 16, 8, 32)
ELEM_TEST(elem_half_zero_16x16_64size, aclFloat16, int32_t, 16, 16, 64)

ELEM_TEST(elem_simt_float_16x16_256size, float, int32_t, 16, 16, 256)
ELEM_TEST(elem_simt_half_16x16_256size, aclFloat16, int32_t, 16, 16, 256)
ELEM_TEST(elem_simt_bfloat16_16x16_256size, uint16_t, int32_t, 16, 16, 256)
ELEM_TEST(elem_simt_int32_16x8_128size, int32_t, int32_t, 16, 8, 128)
ELEM_TEST(elem_simt_uint32_16x16_256size, uint32_t, int32_t, 16, 16, 256)
ELEM_TEST(elem_simt_int16_16x16_256size, int16_t, int32_t, 16, 16, 256)
ELEM_TEST(elem_simt_uint16_16x32_512size, uint16_t, int32_t, 16, 32, 512)
ELEM_TEST(elem_simt_int8_16x32_512size, int8_t, int32_t, 16, 32, 512)
ELEM_TEST(elem_simt_uint8_32x32_1024size, uint8_t, int32_t, 32, 32, 1024)
ELEM_TEST(elem_simt_float_clamp_16x16_64size, float, int32_t, 16, 16, 64)
ELEM_TEST(elem_simt_int32_wrap_16x8_32size, int32_t, int32_t, 16, 8, 32)
ELEM_TEST(elem_simt_half_zero_16x16_64size, aclFloat16, int32_t, 16, 16, 64)
