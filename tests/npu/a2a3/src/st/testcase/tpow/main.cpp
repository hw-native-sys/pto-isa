/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <filesystem>
#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TPOWTest : public testing::Test {
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

template <typename T, int TRow, int TCol, int validRow, int validCol>
void LaunchTPow(T *out, T *base, T *exp, void *stream);

template <typename T, int Row, int Col, int validRow, int validCol>
void test_tpow()
{
    size_t fileSize = Row * Col * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *baseHost, *expHost;
    T *dstDevice, *baseDevice, *expDevice;

    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&baseHost), fileSize);
    aclrtMallocHost((void **)(&expHost), fileSize);
    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&baseDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&expDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/base.bin", fileSize, baseHost, fileSize);
    ReadFile(GetGoldenDir() + "/exp.bin", fileSize, expHost, fileSize);
    aclrtMemcpy(baseDevice, fileSize, baseHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(expDevice, fileSize, expHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTPow<T, Row, Col, validRow, validCol>(dstDevice, baseDevice, expDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(baseDevice);
    aclrtFree(expDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(baseHost);
    aclrtFreeHost(expHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize / sizeof(T));
    std::vector<T> devFinal(fileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize);

    constexpr float eps = std::is_same_v<T, float> ? 0.0005f : 0.00005f;
    bool ret = ResultCmp<T>(golden, devFinal, eps, 0, 1000);

    EXPECT_TRUE(ret);
}

TEST_F(TPOWTest, case1)
{
    test_tpow<float, 16, 64, 15, 63>();
}
TEST_F(TPOWTest, case2)
{
    test_tpow<uint16_t, 16, 64, 15, 63>();
}
TEST_F(TPOWTest, case3)
{
    test_tpow<int32_t, 16, 64, 15, 63>();
}
TEST_F(TPOWTest, case4)
{
    test_tpow<int16_t, 16, 64, 15, 63>();
}
TEST_F(TPOWTest, case5)
{
    test_tpow<int8_t, 16, 64, 15, 63>();
}
TEST_F(TPOWTest, case6)
{
    test_tpow<uint32_t, 16, 64, 15, 63>();
}
TEST_F(TPOWTest, case7)
{
    test_tpow<uint8_t, 16, 64, 15, 63>();
}
TEST_F(TPOWTest, case8)
{
    test_tpow<float, 16, 64, 15, 63>();
}
TEST_F(TPOWTest, case9)
{
    test_tpow<uint16_t, 16, 64, 15, 63>();
}
TEST_F(TPOWTest, case10)
{
    test_tpow<float, 16, 256, 15, 231>();
}
TEST_F(TPOWTest, case11)
{
    test_tpow<uint16_t, 16, 512, 16, 400>();
}
TEST_F(TPOWTest, case12)
{
    test_tpow<float, 1, 64, 1, 64>();
}
TEST_F(TPOWTest, case13)
{
    test_tpow<float, 1, 64, 1, 64>();
}
TEST_F(TPOWTest, case14)
{
    test_tpow<float, 1, 64, 1, 64>();
}
TEST_F(TPOWTest, case15)
{
    test_tpow<float, 1, 64, 1, 64>();
}
