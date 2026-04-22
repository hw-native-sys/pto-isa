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

class TPOWSTest : public testing::Test {
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
void LaunchTPows(T *out, T *base, T *exp, void *stream);

template <typename T, int Row, int Col, int validRow, int validCol>
void test_tpows()
{
    size_t fileSize = Row * Col * sizeof(T);
    size_t expSize = sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *baseHost, *expHost;
    T *dstDevice, *baseDevice, *expDevice;

    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&baseHost), fileSize);
    aclrtMallocHost((void **)(&expHost), expSize);
    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&baseDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&expDevice, expSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/base.bin", fileSize, baseHost, fileSize);
    ReadFile(GetGoldenDir() + "/exp.bin", expSize, expHost, expSize);
    aclrtMemcpy(baseDevice, fileSize, baseHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(expDevice, expSize, expHost, expSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTPows<T, Row, Col, validRow, validCol>(dstDevice, baseDevice, expDevice, stream);

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
    bool ret = ResultCmp<T>(golden, devFinal, eps);

    EXPECT_TRUE(ret);
}

TEST_F(TPOWSTest, case1)
{
    test_tpows<float, 16, 64, 15, 63>();
}
TEST_F(TPOWSTest, case2)
{
    test_tpows<uint16_t, 16, 64, 15, 63>();
}
TEST_F(TPOWSTest, case3)
{
    test_tpows<int32_t, 16, 64, 15, 63>();
}
TEST_F(TPOWSTest, case4)
{
    test_tpows<int16_t, 16, 64, 15, 63>();
}
TEST_F(TPOWSTest, case5)
{
    test_tpows<int8_t, 16, 64, 15, 63>();
}
TEST_F(TPOWSTest, case6)
{
    test_tpows<uint32_t, 16, 64, 15, 63>();
}
TEST_F(TPOWSTest, case7)
{
    test_tpows<uint8_t, 16, 64, 15, 63>();
}
TEST_F(TPOWSTest, case8)
{
    test_tpows<float, 16, 64, 15, 63>();
}
TEST_F(TPOWSTest, case9)
{
    test_tpows<uint16_t, 16, 64, 15, 63>();
}
TEST_F(TPOWSTest, case10)
{
    test_tpows<float, 16, 256, 15, 231>();
}
TEST_F(TPOWSTest, case11)
{
    test_tpows<uint16_t, 16, 512, 16, 400>();
}
