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

class TSQRTTest : public testing::Test {
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

template <typename T, int dstRow, int dstCol, int srcRow, int srcCol, int validRow, int validCol, bool highPrecision,
          bool isInPlace = false>
void LaunchTSqrt(T *out, T *src, void *stream);

template <typename T, int dstRow, int dstCol, int srcRow, int srcCol, int validRow, int validCol,
          bool highPrecision = false, bool isInPlace = false>
void test_tsqrt()
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
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTSqrt<T, dstRow, dstCol, srcRow, srcCol, validRow, validCol, highPrecision, isInPlace>(dstDevice, srcDevice,
                                                                                                 stream);

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
    eps = highPrecision ? 0.0000001f : eps;
    bool ret = ResultCmp(golden, devFinal, eps);

    EXPECT_TRUE(ret);
}

TEST_F(TSQRTTest, case1)
{
    test_tsqrt<float, 64, 64, 64, 64, 64, 64, true, true>();
}
TEST_F(TSQRTTest, case2)
{
    test_tsqrt<float, 64, 64, 64, 64, 64, 64, true, false>();
}
TEST_F(TSQRTTest, case3)
{
    test_tsqrt<aclFloat16, 64, 64, 64, 64, 64, 64, true, true>();
}
TEST_F(TSQRTTest, case4)
{
    test_tsqrt<aclFloat16, 64, 64, 64, 64, 64, 64, true, false>();
}
TEST_F(TSQRTTest, case5)
{
    test_tsqrt<float, 128, 128, 64, 64, 64, 64>();
}
TEST_F(TSQRTTest, case6)
{
    test_tsqrt<float, 64, 64, 128, 128, 32, 32>();
}
TEST_F(TSQRTTest, case7)
{
    test_tsqrt<aclFloat16, 128, 256, 64, 64, 64, 64>();
}
TEST_F(TSQRTTest, case8)
{
    test_tsqrt<aclFloat16, 64, 64, 128, 256, 32, 32>();
}
