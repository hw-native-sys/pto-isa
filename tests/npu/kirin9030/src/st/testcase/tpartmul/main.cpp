/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include "acl/acl.h"
#include "test_common.h"

using namespace std;
using namespace PtoTestCommon;

template <typename T, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC>
void LaunchTPartMul(T *out, T *src0, T *src1, void *stream);

class TPARTMULTest : public testing::Test {
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

template <typename T, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC>
void test_tpartmul()
{
    size_t src0FileSize = src0VR * src0VC * sizeof(T);
    size_t src1FileSize = src1VR * src1VC * sizeof(T);
    size_t dstFileSize = dstVR * dstVC * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&src0Host), src0FileSize);
    aclrtMallocHost((void **)(&src1Host), src1FileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", src0FileSize, src0Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", src1FileSize, src1Host, src1FileSize);

    aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTPartMul<T, dstVR, dstVC, src0VR, src0VC, src1VR, src1VC>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize);
    std::vector<T> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TPARTMULTest, case_float_64x64_64x64_64x64)
{
    test_tpartmul<float, 64, 64, 64, 64, 64, 64>();
}
TEST_F(TPARTMULTest, case_float_64x64_8x64_64x64)
{
    test_tpartmul<float, 64, 64, 8, 64, 64, 64>();
}
TEST_F(TPARTMULTest, case_float_64x64_64x8_64x64)
{
    test_tpartmul<float, 64, 64, 64, 8, 64, 64>();
}
TEST_F(TPARTMULTest, case_float_64x64_64x64_8x64)
{
    test_tpartmul<float, 64, 64, 64, 64, 8, 64>();
}
TEST_F(TPARTMULTest, case_float_64x64_64x64_64x8)
{
    test_tpartmul<float, 64, 64, 64, 64, 64, 8>();
}
TEST_F(TPARTMULTest, case_half_8x48_8x16_8x48)
{
    test_tpartmul<aclFloat16, 8, 48, 8, 16, 8, 48>();
}
TEST_F(TPARTMULTest, case_half_8x768_8x512_8x768)
{
    test_tpartmul<aclFloat16, 8, 768, 8, 512, 8, 768>();
}
TEST_F(TPARTMULTest, case_int16_8x48_8x48_8x16)
{
    test_tpartmul<int16_t, 8, 48, 8, 48, 8, 16>();
}
TEST_F(TPARTMULTest, case_int32_64x64_8x64_64x64)
{
    test_tpartmul<int32_t, 64, 64, 8, 64, 64, 64>();
}