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
#include <gtest/gtest.h>
#include <acl/acl.h>

using namespace std;
using namespace PtoTestCommon;

template <typename T, int cols, int src_row, int src_validRow>
void launchTCOLMIN(T* out, T* src, void* stream);

class TCOLMINTest : public testing::Test {
public:
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

template <typename T, int cols, int src_row, int src_validRow>
bool TCOLMinTestFramework()
{
    size_t dstByteSize = 1 * cols * sizeof(T);
    size_t srcByteSize = src_validRow * cols * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstByteSize);
    aclrtMallocHost((void**)(&srcHost), srcByteSize);

    aclrtMalloc((void**)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);

    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTCOLMIN<T, cols, src_row, src_validRow>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstByteSize);
    std::vector<T> devFinal(dstByteSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);

    return ResultCmp(golden, devFinal, 0.001f);
}

TEST_F(TCOLMINTest, case1)
{
    bool ret = TCOLMinTestFramework<int16_t, 16, 16, 8>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLMINTest, case2)
{
    bool ret = TCOLMinTestFramework<int32_t, 16, 16, 8>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLMINTest, case3)
{
    bool ret = TCOLMinTestFramework<float, 16, 16, 8>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLMINTest, case4)
{
    bool ret = TCOLMinTestFramework<int16_t, 128, 16, 8>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLMINTest, case5)
{
    bool ret = TCOLMinTestFramework<int32_t, 64, 16, 8>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLMINTest, case6)
{
    bool ret = TCOLMinTestFramework<float, 64, 16, 8>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLMINTest, case7)
{
    bool ret = TCOLMinTestFramework<int16_t, 512, 16, 8>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLMINTest, case8)
{
    bool ret = TCOLMinTestFramework<int32_t, 256, 16, 8>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLMINTest, case9)
{
    bool ret = TCOLMinTestFramework<float, 256, 16, 8>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLMINTest, case10)
{
    bool ret = TCOLMinTestFramework<int16_t, 512, 16, 7>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLMINTest, case11)
{
    bool ret = TCOLMinTestFramework<int32_t, 256, 32, 31>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLMINTest, case12)
{
    bool ret = TCOLMinTestFramework<float, 256, 32, 31>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLMINTest, case13)
{
    bool ret = TCOLMinTestFramework<float, 256, 16, 1>();
    EXPECT_TRUE(ret);
}
