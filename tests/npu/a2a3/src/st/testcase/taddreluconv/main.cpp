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
void launchTADDRELUCONVTestCase(void* out, void* src0, void* src1, aclrtStream stream);

class TADDRELUCONVTest : public testing::Test {
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
bool TADDRELUCONVTestFramework()
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

    launchTADDRELUCONVTestCase<caseId>(dstDevice, src0Device, src1Device, stream);
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

TEST_F(TADDRELUCONVTest, case1)
{
    bool ret = TADDRELUCONVTestFramework<1, aclFloat16, float, 32, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case2)
{
    bool ret = TADDRELUCONVTestFramework<2, aclFloat16, float, 16, 128>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case3)
{
    bool ret = TADDRELUCONVTestFramework<3, aclFloat16, float, 31, 96>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case4)
{
    bool ret = TADDRELUCONVTestFramework<4, aclFloat16, float, 7, 192>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case5)
{
    bool ret = TADDRELUCONVTestFramework<5, aclFloat16, float, 64, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case6)
{
    bool ret = TADDRELUCONVTestFramework<6, aclFloat16, float, 13, 48>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case7)
{
    bool ret = TADDRELUCONVTestFramework<7, aclFloat16, float, 16, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case8)
{
    bool ret = TADDRELUCONVTestFramework<8, aclFloat16, float, 8, 128>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case9)
{
    bool ret = TADDRELUCONVTestFramework<9, aclFloat16, float, 4, 256>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case10)
{
    bool ret = TADDRELUCONVTestFramework<10, aclFloat16, float, 16, 32>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case11)
{
    bool ret = TADDRELUCONVTestFramework<11, int8_t, aclFloat16, 16, 128, 1000>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case12)
{
    bool ret = TADDRELUCONVTestFramework<12, int8_t, aclFloat16, 8, 64, 1000>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case13)
{
    bool ret = TADDRELUCONVTestFramework<13, int8_t, aclFloat16, 8, 128, 1000>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case14)
{
    bool ret = TADDRELUCONVTestFramework<14, int8_t, aclFloat16, 8, 64, 1000>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case15)
{
    bool ret = TADDRELUCONVTestFramework<15, int8_t, int16_t, 16, 128, 0>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case16)
{
    bool ret = TADDRELUCONVTestFramework<16, int8_t, int16_t, 8, 64, 0>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case17)
{
    bool ret = TADDRELUCONVTestFramework<17, int8_t, int16_t, 8, 128, 0>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case18)
{
    bool ret = TADDRELUCONVTestFramework<18, int8_t, int16_t, 8, 64, 0>();
    EXPECT_TRUE(ret);
}
