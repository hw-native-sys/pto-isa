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

class TINTERLEAVETest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, int tileH, int tileW, int vRows, int vCols>
void LaunchTInterleave(T* out0, T* out1, T* src0, T* src1, void* stream);

template <int tileH, int tileW, int vRows, int vCols>
void LaunchTInterleaveHalf(aclFloat16* out0, aclFloat16* out1, aclFloat16* src0, aclFloat16* src1, void* stream);

template <typename T, int tileH, int tileW, int vRows, int vCols, bool isHalf = false>
void test_tinterleave()
{
    size_t fileSizeTile = tileH * tileW * sizeof(T);

    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
    ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    aclrtStream stream;
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS);

    T *dst0Host, *dst1Host, *src0Host, *src1Host;
    T *dst0Device, *dst1Device, *src0Device, *src1Device;

    ASSERT_EQ(aclrtMallocHost((void**)(&dst0Host), fileSizeTile), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost((void**)(&dst1Host), fileSizeTile), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost((void**)(&src0Host), fileSizeTile), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost((void**)(&src1Host), fileSizeTile), ACL_SUCCESS);

    ASSERT_EQ(aclrtMalloc((void**)&dst0Device, fileSizeTile, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc((void**)&dst1Device, fileSizeTile, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc((void**)&src0Device, fileSizeTile, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc((void**)&src1Device, fileSizeTile, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);

    memset(dst0Host, 0, fileSizeTile);
    memset(dst1Host, 0, fileSizeTile);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", fileSizeTile, src0Host, fileSizeTile));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input2.bin", fileSizeTile, src1Host, fileSizeTile));

    ASSERT_EQ(aclrtMemcpy(dst0Device, fileSizeTile, dst0Host, fileSizeTile, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dst1Device, fileSizeTile, dst1Host, fileSizeTile, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(src0Device, fileSizeTile, src0Host, fileSizeTile, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(src1Device, fileSizeTile, src1Host, fileSizeTile, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);

    if constexpr (isHalf) {
        LaunchTInterleaveHalf<tileH, tileW, vRows, vCols>(dst0Device, dst1Device, src0Device, src1Device, stream);
    } else {
        LaunchTInterleave<T, tileH, tileW, vRows, vCols>(dst0Device, dst1Device, src0Device, src1Device, stream);
    }

    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dst0Host, fileSizeTile, dst0Device, fileSizeTile, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dst1Host, fileSizeTile, dst1Device, fileSizeTile, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);

    CHECK_RESULT_GTEST(WriteFile(GetGoldenDir() + "/output0.bin", dst0Host, fileSizeTile));
    CHECK_RESULT_GTEST(WriteFile(GetGoldenDir() + "/output1.bin", dst1Host, fileSizeTile));

    aclrtFree(dst0Device);
    aclrtFree(dst1Device);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dst0Host);
    aclrtFreeHost(dst1Host);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden0(fileSizeTile / sizeof(T));
    std::vector<T> golden1(fileSizeTile / sizeof(T));
    std::vector<T> devFinal0(fileSizeTile / sizeof(T));
    std::vector<T> devFinal1(fileSizeTile / sizeof(T));
    // Compare only after both golden and output have been read successfully, otherwise two
    // zero-initialised vectors would make ResultCmp trivially pass.
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden0.bin", fileSizeTile, golden0.data(), fileSizeTile));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden1.bin", fileSizeTile, golden1.data(), fileSizeTile));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output0.bin", fileSizeTile, devFinal0.data(), fileSizeTile));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output1.bin", fileSizeTile, devFinal1.data(), fileSizeTile));

    bool ret0 = ResultCmp<T>(golden0, devFinal0, 0.001f);
    bool ret1 = ResultCmp<T>(golden1, devFinal1, 0.001f);

    EXPECT_TRUE(ret0);
    EXPECT_TRUE(ret1);
}

TEST_F(TINTERLEAVETest, case_float_64x64_64x64) { test_tinterleave<float, 64, 64, 64, 64>(); }
TEST_F(TINTERLEAVETest, case_int32_64x64_64x64) { test_tinterleave<int32_t, 64, 64, 64, 64>(); }
TEST_F(TINTERLEAVETest, case_int16_64x64_64x64) { test_tinterleave<int16_t, 64, 64, 64, 64>(); }
TEST_F(TINTERLEAVETest, case_half_16x256_16x256) { test_tinterleave<aclFloat16, 16, 256, 16, 256, true>(); }
TEST_F(TINTERLEAVETest, case_int8_32x256_32x256) { test_tinterleave<int8_t, 32, 256, 32, 256>(); }
TEST_F(TINTERLEAVETest, case_uint8_32x256_32x256) { test_tinterleave<uint8_t, 32, 256, 32, 256>(); }
// odd valid column cases
TEST_F(TINTERLEAVETest, case_float_64x64_64x63) { test_tinterleave<float, 64, 64, 64, 63>(); }
TEST_F(TINTERLEAVETest, case_uint8_32x256_32x255) { test_tinterleave<uint8_t, 32, 256, 32, 255>(); }
