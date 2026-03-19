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
#include <acl/acl.h>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <uint32_t caseId>
void launchTROWMINTestCase(void *out, void *src, aclrtStream stream);

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

class TROWMINTest : public testing::Test {
public:
    aclrtStream stream;
    void *srcHost;
    void *dstHost;
    void *srcDevice;
    void *dstDevice;

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
    bool CompareGolden(size_t byteSize, bool printAllEn = false)
    {
        std::vector<T> golden(byteSize);
        std::vector<T> result(byteSize);
        float eps = sizeof(T) == 4 ? 0.00005f : 0.0005f;
        ReadFile(GetGoldenDir() + "/golden.bin", byteSize, golden.data(), byteSize);
        ReadFile(GetGoldenDir() + "/output.bin", byteSize, result.data(), byteSize);
        if (printAllEn) {
            return ResultCmp<T>(golden, result, eps, 0, 1000, true);
        }
        return ResultCmp<T>(golden, result, eps, 0, 1000, false, true);
    }

    template <uint32_t caseId, typename T, int row, int validRow, int srcCol, int srcVaildCol, int dstCol>
    bool TRowMinTestFramework()
    {
        size_t dstByteSize = row * dstCol * sizeof(T);
        size_t srcByteSize = row * srcCol * sizeof(T);
        aclrtMallocHost(&dstHost, dstByteSize);
        aclrtMallocHost(&srcHost, srcByteSize);
        aclrtMalloc(&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

        ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
        aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

        launchTROWMINTestCase<caseId>(dstDevice, srcDevice, stream);
        aclrtSynchronizeStream(stream);

        aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
        WriteFile(GetGoldenDir() + "/output.bin", dstHost, validRow * dstCol * sizeof(T));

        aclrtFree(dstDevice);
        aclrtFree(srcDevice);
        aclrtFreeHost(dstHost);
        aclrtFreeHost(srcHost);

        return CompareGolden<T>(dstByteSize);
    }
};

TEST_F(TROWMINTest, case1)
{
    bool ret = TRowMinTestFramework<1, float, 127, 127, 64, 63, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case2)
{
    bool ret = TRowMinTestFramework<2, float, 63, 63, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case3)
{
    bool ret = TRowMinTestFramework<3, float, 31, 31, 128, 127, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case4)
{
    bool ret = TRowMinTestFramework<4, float, 15, 15, 192, 192, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case5)
{
    bool ret = TRowMinTestFramework<5, float, 7, 7, 448, 447, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case6)
{
    bool ret = TRowMinTestFramework<6, aclFloat16, 256, 256, 16, 15, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case7)
{
    bool ret = TRowMinTestFramework<7, float, 30, 30, 216, 216, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case8)
{
    bool ret = TRowMinTestFramework<8, float, 30, 30, 216, 24, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case9)
{
    bool ret = TRowMinTestFramework<9, float, 30, 11, 216, 216, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case10)
{
    bool ret = TRowMinTestFramework<10, float, 30, 11, 216, 24, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case11)
{
    bool ret = TRowMinTestFramework<11, float, 238, 238, 40, 40, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case12)
{
    bool ret = TRowMinTestFramework<12, float, 238, 238, 40, 16, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case13)
{
    bool ret = TRowMinTestFramework<13, float, 238, 121, 40, 40, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case14)
{
    bool ret = TRowMinTestFramework<14, float, 238, 121, 40, 16, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case15)
{
    bool ret = TRowMinTestFramework<15, float, 64, 64, 128, 128, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case16)
{
    bool ret = TRowMinTestFramework<16, float, 32, 32, 256, 256, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case17)
{
    bool ret = TRowMinTestFramework<17, float, 16, 16, 512, 512, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case18)
{
    bool ret = TRowMinTestFramework<18, float, 8, 8, 1024, 1024, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case19)
{
    bool ret = TRowMinTestFramework<19, int32_t, 127, 127, 64, 63, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case20)
{
    bool ret = TRowMinTestFramework<20, int32_t, 63, 63, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case21)
{
    bool ret = TRowMinTestFramework<21, int32_t, 31, 31, 128, 127, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case22)
{
    bool ret = TRowMinTestFramework<22, int32_t, 15, 15, 192, 192, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case23)
{
    bool ret = TRowMinTestFramework<23, int32_t, 7, 7, 448, 447, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case24)
{
    bool ret = TRowMinTestFramework<24, int16_t, 128, 128, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case25)
{
    bool ret = TRowMinTestFramework<25, int16_t, 64, 64, 64, 64, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case26)
{
    bool ret = TRowMinTestFramework<26, int16_t, 32, 32, 128, 128, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case27)
{
    bool ret = TRowMinTestFramework<27, int16_t, 16, 16, 192, 192, 1>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWMINTest, case28)
{
    bool ret = TRowMinTestFramework<28, int16_t, 8, 8, 448, 448, 1>();
    EXPECT_TRUE(ret);
}