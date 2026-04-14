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
void launchTCOLCMAXTestCase(void *out, void *src, aclrtStream stream);

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

class TCOLCMAXTest : public testing::Test {
public:
    aclrtStream stream;
    void *dstHost;
    void *srcHost;
    void *dstDevice;
    void *srcDevice;

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

    // template <typename T>
    bool CompareGolden(size_t dstByteSize, bool printAllEn = false)
    {
        std::vector<uint32_t> golden(dstByteSize);
        std::vector<uint32_t> result(dstByteSize);
        float eps = 0.001f;
        ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
        ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, result.data(), dstByteSize);
        if (printAllEn) {
            return ResultCmp(golden, result, eps, 0, 1000, true);
        }
        return ResultCmp(golden, result, eps, 0, 1000, false, true);
    }

    template <uint32_t caseId, typename T, int srcRow, int srcValidRow, int dstRow, int col, int validCol>
    bool TCOLCMAXTestFramework()
    {
        int dstCol = (validCol + 7) / 8 * 8;
        size_t dstByteSize = dstRow * dstCol * sizeof(uint32_t);
        size_t srcByteSize = srcRow * col * sizeof(T);
        aclrtMallocHost(&dstHost, dstByteSize);
        aclrtMallocHost(&srcHost, srcByteSize);
        aclrtMalloc(&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

        ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
        aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        launchTCOLCMAXTestCase<caseId>(dstDevice, srcDevice, stream);
        aclrtSynchronizeStream(stream);

        aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

        aclrtFree(dstDevice);
        aclrtFree(srcDevice);
        aclrtFreeHost(dstHost);
        aclrtFreeHost(srcHost);

        return CompareGolden(dstByteSize);
    }
};

TEST_F(TCOLCMAXTest, case01)
{
    bool ret = TCOLCMAXTestFramework<1, float, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case02)
{
    bool ret = TCOLCMAXTestFramework<2, float, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case03)
{
    bool ret = TCOLCMAXTestFramework<3, float, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case11)
{
    bool ret = TCOLCMAXTestFramework<11, aclFloat16, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case12)
{
    bool ret = TCOLCMAXTestFramework<12, aclFloat16, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case13)
{
    bool ret = TCOLCMAXTestFramework<13, aclFloat16, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case51)
{
    bool ret = TCOLCMAXTestFramework<51, uint16_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case52)
{
    bool ret = TCOLCMAXTestFramework<52, uint16_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case53)
{
    bool ret = TCOLCMAXTestFramework<53, uint16_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case71)
{
    bool ret = TCOLCMAXTestFramework<71, uint32_t, 1, 1, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case72)
{
    bool ret = TCOLCMAXTestFramework<72, uint32_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case73)
{
    bool ret = TCOLCMAXTestFramework<73, uint32_t, 16, 15, 1, 256, 255>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case81)
{
    bool ret = TCOLCMAXTestFramework<81, aclFloat16, 16, 16, 1, 32, 32>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case82)
{
    bool ret = TCOLCMAXTestFramework<82, uint16_t, 16, 16, 1, 32, 32>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case83)
{
    bool ret = TCOLCMAXTestFramework<83, uint32_t, 16, 16, 1, 32, 31>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case84)
{
    bool ret = TCOLCMAXTestFramework<84, float, 16, 16, 1, 32, 31>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case91)
{
    bool ret = TCOLCMAXTestFramework<91, uint16_t, 16, 16, 1, 128, 120>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case92)
{
    bool ret = TCOLCMAXTestFramework<92, aclFloat16, 16, 16, 1, 96, 88>();
    EXPECT_TRUE(ret);
}
TEST_F(TCOLCMAXTest, case93)
{
    bool ret = TCOLCMAXTestFramework<93, uint16_t, 4, 4, 1, 48, 34>();
    EXPECT_TRUE(ret);
}
