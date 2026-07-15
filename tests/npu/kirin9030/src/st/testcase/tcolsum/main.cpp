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

template <uint32_t caseId>
void launchTCOLSUMTestCase(void* out, void* src, aclrtStream stream);

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

class TCOLSUMTest : public testing::Test {
public:
    aclrtStream stream;
    void* dstHost;
    void* srcHost;
    void* dstDevice;
    void* srcDevice;

protected:
    void SetUp() override
    {
        aclInit(nullptr);
        aclrtSetDevice(0);
        aclrtCreateStream(&stream);
    }

    void TearDown() override
    {
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
    }

    template <typename T>
    bool CompareGolden(size_t dstByteSize, bool printAllEn = false)
    {
        std::vector<T> golden(dstByteSize);
        std::vector<T> result(dstByteSize);
        float eps = sizeof(T) == 4 ? 0.001f : 0.005f;
        ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
        ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, result.data(), dstByteSize);
        if (printAllEn) {
            return ResultCmp(golden, result, eps, 0, 1000, true);
        }
        return ResultCmp(golden, result, eps, 0, 1000, false, true);
    }

    template <uint32_t caseId, typename T, int srcRow, int srcValidRow, int dstRow, int col, int validCol>
    bool TCOLSUMTestFramework()
    {
        size_t dstByteSize = dstRow * col * sizeof(T);
        size_t srcByteSize = srcRow * col * sizeof(T);
        aclrtMallocHost(&dstHost, dstByteSize);
        aclrtMallocHost(&srcHost, srcByteSize);
        aclrtMalloc(&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

        ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
        aclrtMemset(dstHost, dstByteSize, 0, dstByteSize);
        aclrtMemcpy(dstDevice, dstByteSize, dstHost, dstByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        launchTCOLSUMTestCase<caseId>(dstDevice, srcDevice, stream);
        aclrtSynchronizeStream(stream);

        aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

        aclrtFree(dstDevice);
        aclrtFree(srcDevice);
        aclrtFreeHost(dstHost);
        aclrtFreeHost(srcHost);

        return CompareGolden<T>(dstByteSize);
    }
};

TEST_F(TCOLSUMTest, case01)
{
    bool ret = TCOLSUMTestFramework<1, float, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLSUMTest, case02)
{
    bool ret = TCOLSUMTestFramework<2, float, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLSUMTest, case03)
{
    bool ret = TCOLSUMTestFramework<3, float, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLSUMTest, case04)
{
    bool ret = TCOLSUMTestFramework<4, float, 64, 63, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLSUMTest, case05)
{
    bool ret = TCOLSUMTestFramework<5, float, 64, 64, 1, 128, 128>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLSUMTest, case11)
{
    bool ret = TCOLSUMTestFramework<11, int16_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLSUMTest, case12)
{
    bool ret = TCOLSUMTestFramework<12, int16_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLSUMTest, case13)
{
    bool ret = TCOLSUMTestFramework<13, int16_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLSUMTest, case14)
{
    bool ret = TCOLSUMTestFramework<14, int16_t, 64, 63, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLSUMTest, case15)
{
    bool ret = TCOLSUMTestFramework<15, int16_t, 64, 64, 1, 128, 128>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLSUMTest, case21)
{
    bool ret = TCOLSUMTestFramework<21, int8_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLSUMTest, case22)
{
    bool ret = TCOLSUMTestFramework<22, int8_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLSUMTest, case23)
{
    bool ret = TCOLSUMTestFramework<23, int8_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLSUMTest, case24)
{
    bool ret = TCOLSUMTestFramework<24, int8_t, 64, 63, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLSUMTest, case25)
{
    bool ret = TCOLSUMTestFramework<25, int8_t, 64, 64, 1, 128, 128>();
    EXPECT_TRUE(ret);
}
