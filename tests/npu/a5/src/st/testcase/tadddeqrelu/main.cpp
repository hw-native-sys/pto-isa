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

using namespace PtoTestCommon;

template <uint32_t caseId>
void dispatchTADDDEQRELUTestCase(void *out, void *src0, void *src1, aclrtStream stream);

class TADDDEQRELUTest : public testing::Test {
protected:
    void SetUp() override
    {}

    void TearDown() override
    {}
};

static std::string goldenDirPath()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    const std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <uint32_t caseId, int row, int validRow, int col, int validCol, int toleranceX1000 = 2>
bool runTAddDeqReluTest()
{
    float tolerance = static_cast<float>(toleranceX1000) / 1000.0f;

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t srcByteSize = row * col * sizeof(int32_t);
    size_t dstByteSize = row * col * sizeof(aclFloat16);
    aclFloat16 *dstHost;
    aclFloat16 *dstDevice;
    int32_t *src0Host;
    int32_t *src1Host;
    int32_t *src0Device;
    int32_t *src1Device;

    aclrtMallocHost((void **)(&dstHost), dstByteSize);
    aclrtMallocHost((void **)(&src0Host), srcByteSize);
    aclrtMallocHost((void **)(&src1Host), srcByteSize);

    aclrtMalloc((void **)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(goldenDirPath() + "/input0.bin", srcByteSize, src0Host, srcByteSize);
    ReadFile(goldenDirPath() + "/input1.bin", srcByteSize, src1Host, srcByteSize);

    aclrtMemcpy(src0Device, srcByteSize, src0Host, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, srcByteSize, src1Host, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    dispatchTADDDEQRELUTestCase<caseId>(dstDevice, src0Device, src1Device, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(goldenDirPath() + "/output.bin", dstHost, dstByteSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<aclFloat16> goldenData(row * col);
    std::vector<aclFloat16> deviceOutput(row * col);
    ReadFile(goldenDirPath() + "/golden.bin", dstByteSize, goldenData.data(), dstByteSize);
    ReadFile(goldenDirPath() + "/output.bin", dstByteSize, deviceOutput.data(), dstByteSize);

    return ResultCmp<aclFloat16>(goldenData, deviceOutput, tolerance);
}

TEST_F(TADDDEQRELUTest, case1)
{
    bool ret = runTAddDeqReluTest<1, 32, 32, 64, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDDEQRELUTest, case2)
{
    bool ret = runTAddDeqReluTest<2, 64, 64, 64, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDDEQRELUTest, case3)
{
    bool ret = runTAddDeqReluTest<3, 1, 1, 2048, 2048>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDDEQRELUTest, case4)
{
    bool ret = runTAddDeqReluTest<4, 64, 64, 128, 128>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDDEQRELUTest, case5)
{
    bool ret = runTAddDeqReluTest<5, 32, 31, 128, 128>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDDEQRELUTest, case6)
{
    bool ret = runTAddDeqReluTest<6, 32, 32, 128, 127>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDDEQRELUTest, case7)
{
    bool ret = runTAddDeqReluTest<7, 16, 16, 64, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDDEQRELUTest, case8)
{
    bool ret = runTAddDeqReluTest<8, 32, 32, 64, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDDEQRELUTest, case9)
{
    bool ret = runTAddDeqReluTest<9, 16, 16, 128, 128>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDDEQRELUTest, case10)
{
    bool ret = runTAddDeqReluTest<10, 16, 16, 128, 128>();
    EXPECT_TRUE(ret);
}
