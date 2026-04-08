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

class TRSQRTTest : public testing::Test {
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

template <typename T, int kDstRows_, int kDstCols_, int kSrcRows_, int kSrcCols_, int kValRows_, int kValCols_>
void LaunchTRSqrt(T *out, T *src, void *stream);

template <typename T, int kDstRows_, int kDstCols_, int kSrcRows_, int kSrcCols_, int kValRows_, int kValCols_>
void test_trsqrt()
{
    size_t fileSrcSize = kSrcRows_ * kSrcCols_ * sizeof(T);
    size_t fileDstSize = kDstRows_ * kDstCols_ * sizeof(T);
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), fileDstSize);
    aclrtMallocHost((void **)(&srcHost), fileSrcSize);

    aclrtMalloc((void **)&dstDevice, fileDstSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, fileSrcSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", fileSrcSize, srcHost, fileSrcSize);

    aclrtMemcpy(srcDevice, fileSrcSize, srcHost, fileSrcSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTRSqrt<T, kDstRows_, kDstCols_, kSrcRows_, kSrcCols_, kValRows_, kValCols_>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileDstSize, dstDevice, fileDstSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileDstSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileDstSize);
    std::vector<T> devFinal(fileDstSize);
    ReadFile(GetGoldenDir() + "/golden.bin", fileDstSize, golden.data(), fileDstSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileDstSize, devFinal.data(), fileDstSize);

    float eps = 0.0f;
    if constexpr (std::is_same_v<T, float>) {
        eps = 0.0001f;
    } else if constexpr (std::is_same_v<T, aclFloat16>) {
        eps = 0.001f;
    }
    bool ret = ResultCmp<T>(golden, devFinal, eps);

    EXPECT_TRUE(ret);
}

TEST_F(TRSQRTTest, case_float_64x64_64x64_64x64)
{
    test_trsqrt<float, 64, 64, 64, 64, 64, 64>();
}
TEST_F(TRSQRTTest, case_half_64x64_64x64_64x64)
{
    test_trsqrt<aclFloat16, 64, 64, 64, 64, 64, 64>();
}
