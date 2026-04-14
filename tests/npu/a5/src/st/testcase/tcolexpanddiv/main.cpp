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

namespace TColExpandDivTest {
template <typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col, bool highPrecision>
void launchTColExpandDiv(T *out, T *src0, T *src1, void *stream);

class TColExpandDivTest : public testing::Test {
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

template <typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col, bool highPrecision = false>
void test_tcolexpanddiv()
{
    size_t inputFileSize = src1Row * src1Col * sizeof(T);
    size_t outputFileSize = dstRow * dstCol * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), outputFileSize);
    aclrtMallocHost((void **)(&src0Host), outputFileSize);
    aclrtMallocHost((void **)(&src1Host), inputFileSize);

    aclrtMalloc((void **)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", outputFileSize, src0Host, outputFileSize);
    ReadFile(GetGoldenDir() + "/input1.bin", inputFileSize, src1Host, inputFileSize);

    aclrtMemcpy(src0Device, outputFileSize, src0Host, outputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, inputFileSize, src1Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTColExpandDiv<T, dstRow, dstCol, src1Row, src1Col, highPrecision>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(outputFileSize);
    std::vector<float> devFinal(outputFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);
    auto resPrecision = highPrecision ? 0.0000001f : 0.001f;
    bool ret = ResultCmp(golden, devFinal, resPrecision);

    EXPECT_TRUE(ret);
}

TEST_F(TColExpandDivTest, case_fp32_32_64_1_64)
{
    test_tcolexpanddiv<float, 32, 64, 1, 64>();
}
TEST_F(TColExpandDivTest, case_fp32_8_32_1_32)
{
    test_tcolexpanddiv<float, 8, 32, 1, 32>();
}
TEST_F(TColExpandDivTest, case_fp16_16_64_1_64)
{
    test_tcolexpanddiv<aclFloat16, 16, 64, 1, 64>();
}
TEST_F(TColExpandDivTest, case_fp16_4_128_1_128)
{
    test_tcolexpanddiv<aclFloat16, 4, 128, 1, 128>();
}
TEST_F(TColExpandDivTest, case_fp32_40_32_1_32)
{
    test_tcolexpanddiv<float, 40, 32, 1, 32, true>();
}
TEST_F(TColExpandDivTest, case_fp16_16_128_1_128)
{
    test_tcolexpanddiv<aclFloat16, 16, 128, 1, 128, true>();
}
TEST_F(TColExpandDivTest, case_fp32_20_64_1_64)
{
    test_tcolexpanddiv<float, 20, 64, 1, 64, true>();
}
TEST_F(TColExpandDivTest, case_int32_16_32_1_32)
{
    test_tcolexpanddiv<int32_t, 16, 32, 1, 32>();
}
TEST_F(TColExpandDivTest, case_int16_16_64_1_64)
{
    test_tcolexpanddiv<int16_t, 16, 64, 1, 64>();
}
} // namespace TColExpandDivTest