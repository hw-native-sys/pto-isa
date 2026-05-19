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

template <int32_t testKey>
void launchTEXPANDS_MAT(uint8_t *out, void *stream);

class TEXPANDSTest : public testing::Test {
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

template <int32_t testKey, typename T, typename... Dims>
void texpands_test(Dims... dims)
{
    size_t totalElements = (1 * ... * dims);
    size_t fileSize = totalElements * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost;
    uint8_t *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), fileSize);

    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(dstDevice, fileSize, dstHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTEXPANDS_MAT<testKey>((uint8_t *)dstDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);

    aclrtFreeHost(dstHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize);
    std::vector<T> devFinal(fileSize);

    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize);

    bool ret = ResultCmp(golden, devFinal, 0);
    EXPECT_TRUE(ret);
}

TEST_F(TEXPANDSTest, case1)
{
    texpands_test<1, uint16_t>(128, 128); // uint16_t represent half
}

TEST_F(TEXPANDSTest, case2)
{
    texpands_test<2, int16_t>(32, 64);
}

TEST_F(TEXPANDSTest, case3)
{
    texpands_test<3, float>(32, 32);
}

TEST_F(TEXPANDSTest, case4)
{
    texpands_test<4, int8_t>(32, 32);
}

TEST_F(TEXPANDSTest, case5)
{
    texpands_test<5, uint16_t>(256, 256);
}

TEST_F(TEXPANDSTest, case6)
{
    texpands_test<6, uint16_t>(1, 16, 7, 7, 16);
}

TEST_F(TEXPANDSTest, case7)
{
    texpands_test<7, int16_t>(2, 5, 2, 3, 8);
}

TEST_F(TEXPANDSTest, case8)
{
    texpands_test<8, int32_t>(2, 2, 3, 2, 1, 8);
}

TEST_F(TEXPANDSTest, case9)
{
    texpands_test<9, uint32_t>(2, 3, 4, 1, 2, 8);
}