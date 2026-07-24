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

template <typename T, int rows, int src_col, int src_validCol, int dst_col, int dst_validCol>
void launchTROWEXPAND(T* out, T* src, void* stream);

class TROWEXPANDTest : public testing::Test {
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

template <typename T, int rows, int srcCols, int srcValidCols, int dstCols, int dstValidCols>
bool TRowExpandFramework()
{
    size_t inputFileSize = rows * srcCols * sizeof(T);
    size_t outputFileSize = rows * dstCols * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host;
    T *dstDevice, *src0Device;

    aclrtMallocHost((void**)(&dstHost), outputFileSize);
    aclrtMallocHost((void**)(&src0Host), inputFileSize);

    aclrtMalloc((void**)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", inputFileSize, src0Host, inputFileSize);

    aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTROWEXPAND<T, rows, srcCols, srcValidCols, dstCols, dstValidCols>(dstDevice, src0Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(outputFileSize);
    std::vector<T> devFinal(outputFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);

    return ResultCmp(golden, devFinal, 0.001f, 0, 1000, false, true);
}

TEST_F(TROWEXPANDTest, case0)
{
    bool ret = TRowExpandFramework<uint16_t, 16, 16, 1, 512, 512>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case1)
{
    bool ret = TRowExpandFramework<uint8_t, 16, 32, 1, 256, 256>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case2)
{
    bool ret = TRowExpandFramework<uint32_t, 16, 8, 1, 128, 128>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case3)
{
    bool ret = TRowExpandFramework<float, 16, 32, 1, 512, 512>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case4)
{
    bool ret = TRowExpandFramework<uint16_t, 16, 16, 1, 256, 255>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case5)
{
    bool ret = TRowExpandFramework<uint8_t, 16, 32, 1, 512, 511>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case6)
{
    bool ret = TRowExpandFramework<uint32_t, 16, 8, 1, 128, 127>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case7)
{
    bool ret = TRowExpandFramework<uint16_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case8)
{
    bool ret = TRowExpandFramework<uint8_t, 2, 32, 1, 64, 63>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case9)
{
    bool ret = TRowExpandFramework<uint16_t, 4080, 1, 1, 16, 16>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case10)
{
    bool ret = TRowExpandFramework<uint16_t, 16, 1, 1, 16, 16>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case11)
{
    bool ret = TRowExpandFramework<uint32_t, 4080, 1, 1, 8, 8>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case12)
{
    bool ret = TRowExpandFramework<uint32_t, 16, 1, 1, 8, 8>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case13)
{
    bool ret = TRowExpandFramework<float, 4080, 1, 1, 8, 8>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case14)
{
    bool ret = TRowExpandFramework<float, 16, 1, 1, 8, 8>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case15)
{
    bool ret = TRowExpandFramework<int16_t, 16, 16, 1, 512, 512>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case16)
{
    bool ret = TRowExpandFramework<int8_t, 16, 32, 1, 256, 256>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case17)
{
    bool ret = TRowExpandFramework<int32_t, 16, 8, 1, 128, 128>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case18)
{
    bool ret = TRowExpandFramework<int16_t, 16, 16, 1, 256, 255>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case19)
{
    bool ret = TRowExpandFramework<int8_t, 16, 32, 1, 512, 511>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case20)
{
    bool ret = TRowExpandFramework<int32_t, 16, 8, 1, 128, 127>();
    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case21)
{
    bool ret = TRowExpandFramework<int16_t, 16, 16, 1, 128, 127>();
    EXPECT_TRUE(ret);
}
TEST_F(TROWEXPANDTest, case22)
{
    bool ret = TRowExpandFramework<int8_t, 2, 32, 1, 64, 63>();
    EXPECT_TRUE(ret);
}
