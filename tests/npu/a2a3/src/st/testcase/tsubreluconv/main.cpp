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
void launchTSUBRELUCONVTestCase(void* out, void* src0, void* src1, aclrtStream stream);

class TSUBRELUCONVTest : public testing::Test {
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

template <uint32_t caseId, typename DstT, typename SrcT, int row, int col, int toleranceX1000 = 1>
bool TSUBRELUCONVTestFramework()
{
    float tolerance = static_cast<float>(toleranceX1000) / 1000.0f;

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t srcByteSize = row * col * sizeof(SrcT);
    size_t dstByteSize = row * col * sizeof(DstT);
    SrcT* src0Host;
    SrcT* src1Host;
    DstT* dstHost;
    SrcT* src0Device;
    SrcT* src1Device;
    DstT* dstDevice;

    aclrtMallocHost((void**)(&src0Host), srcByteSize);
    aclrtMallocHost((void**)(&src1Host), srcByteSize);
    aclrtMallocHost((void**)(&dstHost), dstByteSize);

    aclrtMalloc((void**)&src0Device, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", srcByteSize, src0Host, srcByteSize);
    ReadFile(GetGoldenDir() + "/input1.bin", srcByteSize, src1Host, srcByteSize);

    aclrtMemcpy(src0Device, srcByteSize, src0Host, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, srcByteSize, src1Host, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTSUBRELUCONVTestCase<caseId>(dstDevice, src0Device, src1Device, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(dstDevice);

    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(dstHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<DstT> golden(row * col);
    std::vector<DstT> devFinal(row * col);
    ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);

    return ResultCmp<DstT>(golden, devFinal, tolerance);
}

TEST_F(TSUBRELUCONVTest, case1)
{
    bool ret = TSUBRELUCONVTestFramework<1, aclFloat16, float, 32, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TSUBRELUCONVTest, case2)
{
    bool ret = TSUBRELUCONVTestFramework<2, aclFloat16, float, 16, 128>();
    EXPECT_TRUE(ret);
}

TEST_F(TSUBRELUCONVTest, case3)
{
    bool ret = TSUBRELUCONVTestFramework<3, aclFloat16, float, 31, 96>();
    EXPECT_TRUE(ret);
}

TEST_F(TSUBRELUCONVTest, case4)
{
    bool ret = TSUBRELUCONVTestFramework<4, aclFloat16, float, 7, 192>();
    EXPECT_TRUE(ret);
}

TEST_F(TSUBRELUCONVTest, case5)
{
    bool ret = TSUBRELUCONVTestFramework<5, aclFloat16, float, 64, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TSUBRELUCONVTest, case6)
{
    bool ret = TSUBRELUCONVTestFramework<6, aclFloat16, float, 13, 48>();
    EXPECT_TRUE(ret);
}

TEST_F(TSUBRELUCONVTest, case7)
{
    bool ret = TSUBRELUCONVTestFramework<7, aclFloat16, float, 16, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TSUBRELUCONVTest, case8)
{
    bool ret = TSUBRELUCONVTestFramework<8, aclFloat16, float, 8, 128>();
    EXPECT_TRUE(ret);
}

TEST_F(TSUBRELUCONVTest, case9)
{
    bool ret = TSUBRELUCONVTestFramework<9, aclFloat16, float, 4, 256>();
    EXPECT_TRUE(ret);
}

TEST_F(TSUBRELUCONVTest, case10)
{
    bool ret = TSUBRELUCONVTestFramework<10, aclFloat16, float, 16, 32>();
    EXPECT_TRUE(ret);
}

TEST_F(TSUBRELUCONVTest, case11)
{
    bool ret = TSUBRELUCONVTestFramework<11, int8_t, aclFloat16, 16, 128, 1000>();
    EXPECT_TRUE(ret);
}

TEST_F(TSUBRELUCONVTest, case12)
{
    bool ret = TSUBRELUCONVTestFramework<12, int8_t, aclFloat16, 8, 64, 1000>();
    EXPECT_TRUE(ret);
}

TEST_F(TSUBRELUCONVTest, case13)
{
    bool ret = TSUBRELUCONVTestFramework<13, int8_t, aclFloat16, 8, 128, 1000>();
    EXPECT_TRUE(ret);
}

TEST_F(TSUBRELUCONVTest, case14)
{
    bool ret = TSUBRELUCONVTestFramework<14, int8_t, aclFloat16, 8, 64, 1000>();
    EXPECT_TRUE(ret);
}

TEST_F(TSUBRELUCONVTest, case15)
{
    bool ret = TSUBRELUCONVTestFramework<15, int8_t, int16_t, 16, 128, 0>();
    EXPECT_TRUE(ret);
}

TEST_F(TSUBRELUCONVTest, case16)
{
    bool ret = TSUBRELUCONVTestFramework<16, int8_t, int16_t, 8, 64, 0>();
    EXPECT_TRUE(ret);
}

TEST_F(TSUBRELUCONVTest, case17)
{
    bool ret = TSUBRELUCONVTestFramework<17, int8_t, int16_t, 8, 128, 0>();
    EXPECT_TRUE(ret);
}

TEST_F(TSUBRELUCONVTest, case18)
{
    bool ret = TSUBRELUCONVTestFramework<18, int8_t, int16_t, 8, 64, 0>();
    EXPECT_TRUE(ret);
}