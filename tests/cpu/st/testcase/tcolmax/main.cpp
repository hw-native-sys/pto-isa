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
void launchTCOLMAX_demo(uint8_t *out, uint8_t *src, void *stream);

class TCOLMAXTest : public testing::Test {
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
void LaunchTCOLMAX(T *out, T *src, void *stream);

template <typename T, int kGSize_>
inline void init_dst(T *dstHost)
{
    for (size_t i = 0; i < kGSize_; i++) {
        dstHost[i] = 0;
    }
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void test_tcolmax()
{
    size_t inputSize = kGRows_ * kGCols_ * sizeof(T);
    size_t outputSize = kGCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), outputSize);
    aclrtMallocHost((void **)(&srcHost), inputSize);

    aclrtMalloc((void **)(&srcDevice), inputSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)(&dstDevice), outputSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", inputSize, srcHost, inputSize));
    init_dst<T, kGCols_>(dstHost);

    aclrtMemcpy(srcDevice, inputSize, srcHost, inputSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTCOLMAX<T, kGRows_, kGCols_, kTRows_, kTCols_>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputSize, dstDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(outputSize);
    std::vector<T> devFinal(outputSize);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", outputSize, golden.data(), outputSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", outputSize, devFinal.data(), outputSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TCOLMAXTest, case_float_64x64_64x64_64x64)
{
    test_tcolmax<float, 64, 64, 64, 64>();
}
TEST_F(TCOLMAXTest, case_half_16x256_16x256_16x256)
{
    test_tcolmax<aclFloat16, 16, 256, 16, 256>();
}
#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TCOLMAXTest, case_bf16_16x256_16x256_16x256)
{
    test_tcolmax<bfloat16_t, 16, 256, 16, 256>();
}
#endif
