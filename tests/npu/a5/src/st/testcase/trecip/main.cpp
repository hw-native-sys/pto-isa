/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

class TRECIPTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, bool isInPlace = false,
          bool highPrecision = false>
void LaunchTRecip(T *out, T *src, void *stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, bool isInPlace = false,
          bool highPrecision = false>
void test_trecip()
{
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&srcHost), fileSize);

    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, srcHost, fileSize);

    aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTRecip<T, kGRows_, kGCols_, kTRows_, kTCols_, isInPlace, highPrecision>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize);
    std::vector<T> devFinal(fileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize);

    float eps = 0.0f;
    if constexpr (std::is_same_v<T, float>) {
        eps = 0.000001f;
    } else if constexpr (std::is_same_v<T, aclFloat16>) {
        eps = 0.00001f;
    }
    bool ret = ResultCmp<T>(golden, devFinal, eps);

    EXPECT_TRUE(ret);
}

TEST_F(TRECIPTest, case_float_64x64_64x64_64x64_inPlace)
{
    test_trecip<float, 64, 64, 64, 64, true>();
}
TEST_F(TRECIPTest, case_float_64x64_64x64_64x64)
{
    test_trecip<float, 64, 64, 64, 64, false>();
}
TEST_F(TRECIPTest, case_half_64x64_64x64_64x64_inPlace)
{
    test_trecip<aclFloat16, 64, 64, 64, 64, true>();
}
TEST_F(TRECIPTest, case_half_64x64_64x64_64x64)
{
    test_trecip<aclFloat16, 64, 64, 64, 64, false>();
}
TEST_F(TRECIPTest, case_float_64x64_66x72_64x64)
{
    test_trecip<float, 64, 64, 66, 72, false>();
}
TEST_F(TRECIPTest, case_float_58x70_66x72_58x70)
{
    test_trecip<float, 58, 70, 66, 72, false>();
}
TEST_F(TRECIPTest, case_float_1x70_1x72_1x70)
{
    test_trecip<float, 1, 70, 1, 72, false>();
}
TEST_F(TRECIPTest, case_float_hp_2x16_2x16_2x16)
{
    test_trecip<float, 2, 16, 2, 16, false, true>();
}
TEST_F(TRECIPTest, case_half_hp_2x32_2x32_2x32)
{
    test_trecip<aclFloat16, 2, 32, 2, 32, false, true>();
}
