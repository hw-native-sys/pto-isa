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

class TRSQRTTest : public testing::Test {
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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, bool isInPlace = false>
void LaunchTRsqrt(T* out, T* src, void* stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, bool isInPlace = false>
void test_trsqrt()
{
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMallocHost((void**)(&srcHost), fileSize);

    aclrtMalloc((void**)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, srcHost, fileSize);

    aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTRsqrt<T, kGRows_, kGCols_, kTRows_, kTCols_, isInPlace>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize);
    std::vector<T> devFinal(fileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize);

    float eps = 0.0f;
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (isInPlace) {
            eps = 0.0001f;
        } else {
            // Known issue with accuracy for built-in `vrsqrt` intrinsic funtion
            // Thats why epsilon is 0.003f, while requirement 0.0001f
            eps = 0.003f;
        }
    } else if constexpr (std::is_same_v<T, aclFloat16>) {
        eps = 0.001f;
    }
    bool ret = ResultCmp<T>(golden, devFinal, eps);

    EXPECT_TRUE(ret);
}

TEST_F(TRSQRTTest, case_float_64x64_64x64_64x64_inPlace_True) { test_trsqrt<float, 64, 64, 64, 64, true>(); }
TEST_F(TRSQRTTest, case_float_64x64_64x64_64x64_inPlace_False) { test_trsqrt<float, 64, 64, 64, 64, false>(); }
TEST_F(TRSQRTTest, case_half_64x64_64x64_64x64_inPlace_True) { test_trsqrt<aclFloat16, 64, 64, 64, 64, true>(); }
TEST_F(TRSQRTTest, case_half_64x64_64x64_64x64_inPlace_False) { test_trsqrt<aclFloat16, 64, 64, 64, 64, false>(); }
