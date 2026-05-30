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

namespace TColExpandTest {
template <typename T, uint32_t srcRows, uint32_t dstRows, uint32_t cols, uint32_t validCols>
void launchTCOLEXPAND(T *out, T *src, void *stream);

class TCOLEXPANDTest : public testing::Test {
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

template <typename T, uint32_t srcRows, uint32_t dstRows, uint32_t cols, uint32_t validCols>
void test_tcolexpand()
{
    size_t inputFileSize = srcRows * cols * sizeof(T);
    size_t outputFileSize = dstRows * cols * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host;
    T *dstDevice, *src0Device;

    aclrtMallocHost((void **)(&dstHost), outputFileSize);
    aclrtMallocHost((void **)(&src0Host), inputFileSize);

    aclrtMalloc((void **)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", inputFileSize, src0Host, inputFileSize);
    aclrtMemset(dstHost, outputFileSize, 0, outputFileSize);
    aclrtMemcpy(dstDevice, outputFileSize, dstHost, outputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTCOLEXPAND<T, srcRows, dstRows, cols, validCols>(dstDevice, src0Device, stream);

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
    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TCOLEXPANDTest, case_half_1_16_512_512)
{
    test_tcolexpand<aclFloat16, 1, 16, 512, 512>();
}
TEST_F(TCOLEXPANDTest, case_int8_2_32_256_255)
{
    test_tcolexpand<int8_t, 2, 32, 256, 255>();
}
TEST_F(TCOLEXPANDTest, case_float_1_8_128_63)
{
    test_tcolexpand<float, 1, 8, 128, 63>();
}
TEST_F(TCOLEXPANDTest, case_half_1_33_512_512)
{
    test_tcolexpand<aclFloat16, 1, 33, 512, 512>();
}
TEST_F(TCOLEXPANDTest, case_int8_2_17_256_44)
{
    test_tcolexpand<int8_t, 2, 17, 256, 44>();
}
TEST_F(TCOLEXPANDTest, case_float_1_54_64_63)
{
    test_tcolexpand<float, 1, 54, 64, 63>();
}
} // namespace TColExpandTest
