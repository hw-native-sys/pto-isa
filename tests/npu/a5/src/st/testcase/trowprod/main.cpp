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
#include <gtest/gtest.h>
#include <acl/acl.h>

using namespace std;
using namespace PtoTestCommon;

template <uint32_t caseId>
void launchTROWPRODTestCase(void *out, void *src, aclrtStream stream);

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

class TROWPRODTest : public testing::Test {
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

    template <uint32_t caseId, typename T, int row, int vaildRow, int srcCol, int srcVaildCol, int dstCol>
    bool TRowProdTestFramework()
    {
        size_t dstByteSize = row * dstCol * sizeof(T);
        size_t srcByteSize = row * srcCol * sizeof(T);
        aclrtMallocHost(&dstHost, dstByteSize);
        aclrtMallocHost(&srcHost, srcByteSize);
        aclrtMalloc(&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

        ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
        aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        launchTROWPRODTestCase<caseId>(dstDevice, srcDevice, stream);
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

TEST_F(TROWPRODTest, case1)
{
    bool ret = TRowProdTestFramework<1, float, 127, 127, 64, 63, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWPRODTest, case2)
{
    bool ret = TRowProdTestFramework<2, float, 63, 63, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWPRODTest, case3)
{
    bool ret = TRowProdTestFramework<3, float, 31, 31, 128, 127, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWPRODTest, case4)
{
    bool ret = TRowProdTestFramework<4, float, 15, 15, 192, 192, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWPRODTest, case5)
{
    bool ret = TRowProdTestFramework<5, float, 7, 7, 448, 448, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWPRODTest, case6)
{
    bool ret = TRowProdTestFramework<6, aclFloat16, 256, 256, 16, 15, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWPRODTest, case7)
{
    bool ret = TRowProdTestFramework<7, float, 64, 64, 128, 128, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWPRODTest, case8)
{
    bool ret = TRowProdTestFramework<8, float, 32, 32, 256, 256, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWPRODTest, case9)
{
    bool ret = TRowProdTestFramework<9, float, 16, 16, 512, 512, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWPRODTest, case10)
{
    bool ret = TRowProdTestFramework<10, float, 8, 8, 1024, 1024, 1>();
    EXPECT_TRUE(ret);
}