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
void launchTADDDEQRELUTestCase(void* out, void* src0, void* src1, aclrtStream stream);

class TADDDEQRELUTest : public testing::Test {
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

template <uint32_t caseId, int row, int validRow, int col, int validCol, int toleranceX1000 = 2>
bool TADDDEQRELUTestFramework()
{
    float tolerance = static_cast<float>(toleranceX1000) / 1000.0f;

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t srcByteSize = row * col * sizeof(int32_t);
    size_t dstByteSize = row * col * sizeof(aclFloat16);
    int32_t* src0Host;
    int32_t* src1Host;
    aclFloat16* dstHost;
    int32_t* src0Device;
    int32_t* src1Device;
    aclFloat16* dstDevice;

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

    launchTADDDEQRELUTestCase<caseId>(dstDevice, src0Device, src1Device, stream);
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

    std::vector<aclFloat16> golden(row * col);
    std::vector<aclFloat16> devFinal(row * col);
    ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);

    return ResultCmp<aclFloat16>(golden, devFinal, tolerance);
}

#define TEST_CASE(N, R, VR, C, VC, ...)                                    \
    TEST_F(TADDDEQRELUTest, case##N)                                       \
    {                                                                      \
        bool ret = TADDDEQRELUTestFramework<N, R, VR, C, VC>(__VA_ARGS__); \
        EXPECT_TRUE(ret);                                                  \
    }

TEST_CASE(1, 32, 32, 64, 64)
TEST_CASE(2, 64, 64, 64, 64)
TEST_CASE(3, 1, 1, 2048, 2048)
TEST_CASE(4, 64, 64, 128, 128)
TEST_CASE(5, 32, 31, 128, 128)
TEST_CASE(6, 32, 32, 128, 127)
TEST_CASE(7, 16, 16, 64, 64)
TEST_CASE(8, 32, 32, 64, 64)
TEST_CASE(9, 16, 16, 128, 128)
TEST_CASE(10, 16, 16, 128, 128)
