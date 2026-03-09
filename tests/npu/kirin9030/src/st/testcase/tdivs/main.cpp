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
void launchTDIVSTestCase(void *out, void *src, float scalar, aclrtStream stream);

class TDIVSTest : public testing::Test {
public:
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

template <uint32_t caseId, typename T, int dstTileRow, int dstTileCol, int row, int vaildRow, int col, int srcVaildCol>
bool TDivSTestFramework()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t dstByteSize = dstTileRow * dstTileCol * sizeof(T);
    size_t srcByteSize = row * col * sizeof(T);
    T *dstHost;
    T *srcHost;
    T *dstDevice;
    T *srcDevice;
    float scalar;

    aclrtMallocHost((void **)(&dstHost), dstByteSize);
    aclrtMallocHost((void **)(&srcHost), srcByteSize);

    aclrtMalloc((void **)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
    std::string scalar_file = GetGoldenDir() + "/divider.bin";
    std::ifstream file(scalar_file, std::ios::binary);

    file.read(reinterpret_cast<char *>(&scalar), 4);
    file.close();
    aclrtMemset(dstHost, dstByteSize, 0, dstByteSize);

    aclrtMemcpy(dstDevice, dstByteSize, dstHost, dstByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTDIVSTestCase<caseId>(dstDevice, srcDevice, scalar, stream);
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

    return ResultCmp<T>(golden, devFinal, 0.001f);
}

TEST_F(TDIVSTest, case1)
{
    bool ret = TDivSTestFramework<1, float, 32, 128, 32, 32, 64, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TDIVSTest, case2)
{
    bool ret = TDivSTestFramework<2, aclFloat16, 63, 128, 63, 63, 64, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TDIVSTest, case3)
{
    bool ret = TDivSTestFramework<3, int32_t, 31, 256, 31, 31, 128, 128>();
    EXPECT_TRUE(ret);
}

TEST_F(TDIVSTest, case4)
{
    bool ret = TDivSTestFramework<4, int16_t, 15, 192, 15, 15, 192, 192>();
    EXPECT_TRUE(ret);
}

TEST_F(TDIVSTest, case5)
{
    bool ret = TDivSTestFramework<5, float, 7, 512, 7, 7, 448, 448>();
    EXPECT_TRUE(ret);
}

TEST_F(TDIVSTest, case6)
{
    bool ret = TDivSTestFramework<6, float, 256, 32, 256, 256, 16, 16>();
    EXPECT_TRUE(ret);
}

TEST_F(TDIVSTest, case7)
{
    bool ret = TDivSTestFramework<7, float, 32, 128, 32, 32, 64, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TDIVSTest, case8)
{
    bool ret = TDivSTestFramework<8, aclFloat16, 63, 128, 63, 63, 64, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TDIVSTest, case9)
{
    bool ret = TDivSTestFramework<9, int32_t, 31, 256, 31, 31, 128, 128>();
    EXPECT_TRUE(ret);
}

TEST_F(TDIVSTest, case10)
{
    bool ret = TDivSTestFramework<10, int16_t, 15, 192, 15, 15, 192, 192>();
    EXPECT_TRUE(ret);
}

TEST_F(TDIVSTest, case11)
{
    bool ret = TDivSTestFramework<11, float, 7, 512, 7, 7, 448, 448>();
    EXPECT_TRUE(ret);
}

TEST_F(TDIVSTest, case12)
{
    bool ret = TDivSTestFramework<12, float, 256, 32, 256, 256, 16, 16>();
    EXPECT_TRUE(ret);
}