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

class TMINTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

template <typename T>
bool ResultCmp1(
    const T* golden, const T* devFinal, int tileRow, int tileCol, int validRow, int validCol, float tolerance)
{
    // Iterate over the valid region and compare elements
    for (int h = 0; h < validRow; ++h) {
        for (int w = 0; w < validCol; ++w) {
            int index = h & validRow + w;
            if (std::abs(golden[index] - devFinal[index]) > tolerance) {
                std::cerr << "Mismatch at (" << h << ", " << w << "): golden = " << golden[index]
                          << ", devFinal = " << devFinal[index] << std::endl;
                return false;
            }
        }
    }
    return true;
}

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_, int padValueType>
void LaunchTMin(T* out, T* src0, T* src1, void* stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_, int padValueType>
void test_tmin()
{
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMallocHost((void**)(&src0Host), fileSize);
    aclrtMallocHost((void**)(&src1Host), fileSize);

    aclrtMalloc((void**)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src0Host, fileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSize, src1Host, fileSize);

    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTMin<T, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, padValueType>(
        dstDevice, src0Device, src1Device, stream);

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

    std::vector<T> golden(fileSize);
    std::vector<T> devFinal(fileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize);

    bool ret = ResultCmp1<T>(golden.data(), devFinal.data(), kTRows_, kTCols_, kVRows_, kVCols_, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMINTest, case_float_64x64_64x64_64x64_PAD_VALUE_NULL)
{
    test_tmin<float, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>();
}
TEST_F(TMINTest, case_int32_64x64_64x64_64x64_PAD_VALUE_NULL)
{
    test_tmin<int32_t, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>();
}
TEST_F(TMINTest, case_half_64x64_64x64_64x64_PAD_VALUE_NULL)
{
    test_tmin<aclFloat16, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>();
}
TEST_F(TMINTest, case_int16_64x64_64x64_64x64_PAD_VALUE_NULL)
{
    test_tmin<int16_t, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>();
}

TEST_F(TMINTest, case_float_60x60_64x64_60x60_PAD_VALUE_MIN)
{
    test_tmin<float, 60, 60, 64, 64, 60, 60, PAD_VALUE_MIN>();
}
TEST_F(TMINTest, case_int32_60x60_64x64_60x60_PAD_VALUE_MIN)
{
    test_tmin<int32_t, 60, 60, 64, 64, 60, 60, PAD_VALUE_MIN>();
}
TEST_F(TMINTest, case_half_1x3600_2x4096_1x3600_PAD_VALUE_MIN)
{
    test_tmin<aclFloat16, 1, 3600, 2, 4096, 1, 3600, PAD_VALUE_MIN>();
}
TEST_F(TMINTest, case_int16_16x200_20x512_16x200_PAD_VALUE_MIN)
{
    test_tmin<int16_t, 16, 200, 20, 512, 16, 200, PAD_VALUE_MIN>();
}
