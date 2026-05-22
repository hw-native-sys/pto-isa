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
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void launchTPOW_demo(uint8_t *out, uint8_t *src, void *stream);

class TPOWTest : public testing::Test {
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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_>
void LaunchTPow(T *out, T *src0, T *src1, void *stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_>
void test_tpow()
{
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);
    size_t readSize = 0;
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&src0Host), fileSize);
    aclrtMallocHost((void **)(&src1Host), fileSize);

    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", readSize, src0Host, fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input2.bin", readSize, src1Host, fileSize));

    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTPow<T, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(kGRows_ * kGCols_);
    std::vector<T> devFinal(kGRows_ * kGCols_);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", readSize, golden.data(), fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", readSize, devFinal.data(), fileSize));

    constexpr float eps = std::is_same_v<T, float> ? 0.0005f : 0.00005f;
    bool ret = ResultCmp<T>(golden, devFinal, eps);
    EXPECT_TRUE(ret);
}

TEST_F(TPOWTest, case_float_64x64_64x64_63x63)
{
    test_tpow<float, 64, 64, 64, 64, 63, 63>();
}
TEST_F(TPOWTest, case_int32_64x64_64x64_63x63)
{
    test_tpow<int32_t, 64, 64, 64, 64, 63, 63>();
}
TEST_F(TPOWTest, case_int16_64x64_64x64_63x63)
{
    test_tpow<int16_t, 64, 64, 64, 64, 63, 63>();
}
TEST_F(TPOWTest, case_half_16x256_16x256_16x256)
{
    test_tpow<aclFloat16, 16, 256, 16, 256, 16, 256>();
}
#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TPOWTest, case_bf16_16x256_16x256_16x256)
{
    test_tpow<bfloat16_t, 16, 256, 16, 256, 16, 256>();
}
#endif
