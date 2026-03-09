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

template <int32_t testKey>
void launchTmovUb2l1(uint64_t *out, uint64_t *src, void *stream);

class TMovUb2l1Test : public testing::Test {
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

template <int32_t testKey, typename dType>
void testTMovUb2L1(int32_t srcRows, int32_t srcCols, int32_t dstRows, int32_t dstCols)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t srcByteSize = srcRows * srcCols * sizeof(dType);
    size_t dstByteSize = dstRows * dstCols * sizeof(dType);
    uint64_t *dstHost, *srcHost, *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstByteSize);
    aclrtMallocHost((void **)(&srcHost), srcByteSize);
    aclrtMalloc((void **)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input_arr.bin", srcByteSize, srcHost, srcByteSize);
    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTmovUb2l1<testKey>(dstDevice, srcDevice, stream);

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

    std::vector<dType> golden(dstByteSize);
    std::vector<dType> devFinal(dstByteSize);
    ReadFile(GetGoldenDir() + "/golden_output.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);
    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TMovUb2l1Test, case1)
{
    testTMovUb2L1<1, uint16_t>(16, 32, 16, 32);
}

TEST_F(TMovUb2l1Test, case2)
{
    testTMovUb2L1<2, uint16_t>(64, 256, 64, 256);
}

TEST_F(TMovUb2l1Test, case3)
{
    testTMovUb2L1<3, int32_t>(48, 72, 48, 72);
}

TEST_F(TMovUb2l1Test, case4)
{
    testTMovUb2L1<4, int32_t>(96, 8, 96, 8);
}

TEST_F(TMovUb2l1Test, case5)
{
    testTMovUb2L1<5, int8_t>(32, 512, 32, 512);
}

TEST_F(TMovUb2l1Test, case6)
{
    testTMovUb2L1<6, int8_t>(64, 96, 64, 96);
}

TEST_F(TMovUb2l1Test, case7)
{
    testTMovUb2L1<7, uint16_t>(64, 64, 48, 48);
}

TEST_F(TMovUb2l1Test, case8)
{
    testTMovUb2L1<8, int32_t>(128, 128, 64, 64);
}

TEST_F(TMovUb2l1Test, case9)
{
    testTMovUb2L1<9, int8_t>(256, 256, 32, 32);
}
