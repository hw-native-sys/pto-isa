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
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TGET_SCALE_ADDRTest : public testing::Test {
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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTGetScaleAddr(T *out, T *src, void *stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void test_tget_scale_addr()
{
    size_t srcFileSize = kGRows_ * kGCols_ * sizeof(T);
    size_t dstFileSize = kTRows_ * kTCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize));

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTGetScaleAddr<T, kGRows_, kGCols_, kTRows_, kTCols_>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize);
    std::vector<T> devFinal(dstFileSize);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TGET_SCALE_ADDRTest, case_float_64x64_64x64)
{
    test_tget_scale_addr<float, 64, 64, 64, 64>();
}
TEST_F(TGET_SCALE_ADDRTest, case_int32_64x64_64x64)
{
    test_tget_scale_addr<int32_t, 64, 64, 64, 64>();
}
TEST_F(TGET_SCALE_ADDRTest, case_int16_64x64_64x64)
{
    test_tget_scale_addr<int16_t, 64, 64, 64, 64>();
}
TEST_F(TGET_SCALE_ADDRTest, case_half_16x256_16x256)
{
    test_tget_scale_addr<aclFloat16, 16, 256, 16, 256>();
}
#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TGET_SCALE_ADDRTest, case_bf16_16x256_16x256)
{
    test_tget_scale_addr<bfloat16_t, 16, 256, 16, 256>();
}
#endif