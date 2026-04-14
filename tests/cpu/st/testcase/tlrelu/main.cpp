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
void launchTLrelu_demo(uint8_t *out, uint8_t *src, void *stream);

class TLRELUTest : public testing::Test {
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
void LaunchTLrelu(T *out, T *src, T *scalar, void *stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void test_tlrelu()
{
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);
    size_t scalarSize = sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost, *scalarHost;
    T *dstDevice, *srcDevice, *scalarDevice;
    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&srcHost), fileSize);
    aclrtMallocHost((void **)(&scalarHost), scalarSize);

    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&scalarDevice, scalarSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", fileSize, srcHost, fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/scalar.bin", scalarSize, scalarHost, scalarSize));
    aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(scalarDevice, scalarSize, scalarHost, scalarSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTLrelu<T, kGRows_, kGCols_, kTRows_, kTCols_>(dstDevice, srcDevice, scalarDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(scalarDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(scalarHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize);
    std::vector<T> devFinal(fileSize);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

const int NUM_16 = 16;
const int NUM_64 = 64;
const int NUM_256 = 256;
TEST_F(TLRELUTest, case_float_64x64_64x64_64x64)
{
    test_tlrelu<float, NUM_64, NUM_64, NUM_64, NUM_64>();
}
TEST_F(TLRELUTest, case_int32_64x64_64x64_64x64)
{
    test_tlrelu<int32_t, NUM_64, NUM_64, NUM_64, NUM_64>();
}
TEST_F(TLRELUTest, case_int16_64x64_64x64_64x64)
{
    test_tlrelu<int16_t, NUM_64, NUM_64, NUM_64, NUM_64>();
}
TEST_F(TLRELUTest, case_half_16x256_16x256_16x256)
{
    test_tlrelu<aclFloat16, NUM_16, NUM_256, NUM_16, NUM_256>();
}
#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TLRELUTest, case_bf16_16x256_16x256_16x256)
{
    test_tlrelu<bfloat16_t, NUM_16, NUM_256, NUM_16, NUM_256>();
}
#endif
