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
#include "acl/acl.h"

using namespace std;
using namespace PtoTestCommon;

template <typename T, int KGRows_, int KGCols_, int KTRows_, int KTCols_, int reverse, int mode>
void LaunchTci(T *out, T S, void *stream);

class TCITest : public testing::Test {
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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int reverse, int mode>
void test_tci(T S)
{
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost;
    T *dstDevice;
    aclrtMallocHost((void **)(&dstHost), fileSize);

    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    LaunchTci<T, kGRows_, kGCols_, kTRows_, kTCols_, reverse, mode>(dstDevice, S, stream);

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

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TCITest, case1)
{
    test_tci<int32_t, 1, 128, 1, 128, 1, 0>(100);
}
TEST_F(TCITest, case2)
{
    test_tci<int16_t, 1, 128, 1, 128, 0, 0>(-1);
}
TEST_F(TCITest, case3)
{
    test_tci<int16_t, 1, 128, 1, 128, 1, 0>(-1);
}
TEST_F(TCITest, case4)
{
    test_tci<int16_t, 1, 192, 1, 192, 1, 0>(-1);
}
TEST_F(TCITest, case5)
{
    test_tci<int32_t, 1, 192, 1, 192, 1, 0>(-1);
}
TEST_F(TCITest, case6)
{
    test_tci<int32_t, 1, 600, 1, 600, 1, 0>(0);
}
TEST_F(TCITest, case7)
{
    test_tci<int16_t, 1, 800, 1, 800, 0, 0>(0);
}
TEST_F(TCITest, case8)
{
    test_tci<int32_t, 1, 2560, 1, 2560, 1, 0>(0);
}
TEST_F(TCITest, case9)
{
    test_tci<int32_t, 1, 3200, 1, 3200, 0, 0>(0);
}
TEST_F(TCITest, case10)
{
    test_tci<int32_t, 1, 8, 1, 8, 0, 0>(0);
}
TEST_F(TCITest, case11)
{
    test_tci<int32_t, 1, 128, 1, 128, 1, 1>(100);
}
TEST_F(TCITest, case12)
{
    test_tci<int32_t, 1, 3200, 1, 3200, 0, 1>(0);
}
TEST_F(TCITest, case13)
{
    test_tci<int16_t, 1, 128, 1, 128, 1, 1>(-1);
}
TEST_F(TCITest, case14)
{
    test_tci<int16_t, 1, 800, 1, 800, 0, 1>(0);
}
TEST_F(TCITest, case15)
{
    test_tci<int16_t, 1, 3840, 1, 3840, 1, 1>(20);
}
TEST_F(TCITest, case16)
{
    test_tci<int16_t, 1, 1408, 1, 1408, 0, 1>(50);
}