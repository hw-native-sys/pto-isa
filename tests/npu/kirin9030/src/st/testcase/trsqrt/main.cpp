/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <type_traits>
#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TRSQRTTest : public testing::Test {
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

template <typename T, int dstRow, int dstCol, int srcRow, int srcCol, int validRow, int validCol,
          bool isInPlace = false>
void LaunchTRsqrt(T *out, T *src, void *stream);

template <typename T, int dstRow, int dstCol, int srcRow, int srcCol, int validRow, int validCol,
          bool isInPlace = false>
void test_trsqrt()
{
    size_t srcFileSize = srcRow * srcCol * sizeof(T);
    size_t dstFileSize = dstRow * dstCol * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);
    aclrtMemset(dstHost, dstFileSize, 0, dstFileSize);

    aclrtMemcpy(dstDevice, dstFileSize, dstHost, dstFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTRsqrt<T, dstRow, dstCol, srcRow, srcCol, validRow, validCol, isInPlace>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize);
    std::vector<T> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    float eps = 0.0005f;
    if constexpr (std::is_same_v<T, float>) {
        eps = 0.00005f;
    }
    bool ret = ResultCmp(golden, devFinal, eps);

    EXPECT_TRUE(ret);
}

TEST_F(TRSQRTTest, case1)
{
    test_trsqrt<float, 64, 64, 64, 64, 64, 64, true>();
}
TEST_F(TRSQRTTest, case2)
{
    test_trsqrt<float, 64, 64, 64, 64, 64, 64, false>();
}
TEST_F(TRSQRTTest, case3)
{
    test_trsqrt<aclFloat16, 64, 64, 64, 64, 64, 64, true>();
}
TEST_F(TRSQRTTest, case4)
{
    test_trsqrt<aclFloat16, 64, 64, 64, 64, 64, 64, false>();
}
TEST_F(TRSQRTTest, case5)
{
    test_trsqrt<float, 128, 128, 64, 64, 64, 64>();
}
TEST_F(TRSQRTTest, case6)
{
    test_trsqrt<float, 64, 64, 128, 128, 32, 32>();
}
TEST_F(TRSQRTTest, case7)
{
    test_trsqrt<aclFloat16, 128, 256, 64, 64, 64, 64>();
}
TEST_F(TRSQRTTest, case8)
{
    test_trsqrt<aclFloat16, 64, 64, 128, 256, 32, 32>();
}