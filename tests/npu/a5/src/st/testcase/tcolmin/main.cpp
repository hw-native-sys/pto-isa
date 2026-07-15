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
void launchTCOLMINTestCase(void* out, void* src, aclrtStream stream);

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

class TCOLMINTest : public testing::Test {
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
    bool TCOLMINTestFramework()
    {
        size_t dstByteSize = dstRow * col * sizeof(T);
        size_t srcByteSize = srcRow * col * sizeof(T);
        aclrtMallocHost(&dstHost, dstByteSize);
        aclrtMallocHost(&srcHost, srcByteSize);
        aclrtMalloc(&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

        ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
        aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        launchTCOLMINTestCase<caseId>(dstDevice, srcDevice, stream);
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

TEST_F(TCOLMINTest, case01)
{
    bool ret = TCOLMINTestFramework<1, float, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case02)
{
    bool ret = TCOLMINTestFramework<2, float, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case03)
{
    bool ret = TCOLMINTestFramework<3, float, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case11)
{
    bool ret = TCOLMINTestFramework<11, aclFloat16, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case12)
{
    bool ret = TCOLMINTestFramework<12, aclFloat16, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case13)
{
    bool ret = TCOLMINTestFramework<13, aclFloat16, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case21)
{
    bool ret = TCOLMINTestFramework<21, int8_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case22)
{
    bool ret = TCOLMINTestFramework<22, int8_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case23)
{
    bool ret = TCOLMINTestFramework<23, int8_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case31)
{
    bool ret = TCOLMINTestFramework<31, uint8_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case32)
{
    bool ret = TCOLMINTestFramework<32, uint8_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case33)
{
    bool ret = TCOLMINTestFramework<33, uint8_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLMINTest, case41)
{
    bool ret = TCOLMINTestFramework<41, int16_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case42)
{
    bool ret = TCOLMINTestFramework<42, int16_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case43)
{
    bool ret = TCOLMINTestFramework<43, int16_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case51)
{
    bool ret = TCOLMINTestFramework<51, uint16_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case52)
{
    bool ret = TCOLMINTestFramework<52, uint16_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case53)
{
    bool ret = TCOLMINTestFramework<53, uint16_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case61)
{
    bool ret = TCOLMINTestFramework<61, int32_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case62)
{
    bool ret = TCOLMINTestFramework<62, int32_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case63)
{
    bool ret = TCOLMINTestFramework<63, int32_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case71)
{
    bool ret = TCOLMINTestFramework<71, uint32_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case72)
{
    bool ret = TCOLMINTestFramework<72, uint32_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLMINTest, case73)
{
    bool ret = TCOLMINTestFramework<73, uint32_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
