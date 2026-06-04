/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <type_traits>
#include <gtest/gtest.h>
#include "test_common.h"
#include "acl/acl.h"

using namespace std;
using namespace PtoTestCommon;

class TLOGTest : public testing::Test {
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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, bool isInPlace = false>
void LaunchTLog(T *out, T *src, void *stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, bool isInPlace = false>
void test_tlog()
{
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&srcHost), fileSize);

    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, srcHost, fileSize);

    aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTLog<T, kGRows_, kGCols_, kTRows_, kTCols_, isInPlace>(dstDevice, srcDevice, stream);

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
        eps = 0.0001f;
    } else if constexpr (std::is_same_v<T, aclFloat16>) {
        eps = 0.001f;
    }
    bool ret = ResultCmp<T>(golden, devFinal, eps);

    EXPECT_TRUE(ret);
}

TEST_F(TLOGTest, case_float_64x64_64x64_64x64_inPlace_True)
{
    test_tlog<float, 64, 64, 64, 64, true>();
}
TEST_F(TLOGTest, case_float_64x64_64x64_64x64_inPlace_False)
{
    test_tlog<float, 64, 64, 64, 64, false>();
}
TEST_F(TLOGTest, case_half_64x64_64x64_64x64_inPlace_True)
{
    test_tlog<aclFloat16, 64, 64, 64, 64, true>();
}
TEST_F(TLOGTest, case_half_64x64_64x64_64x64_inPlace_False)
{
    test_tlog<aclFloat16, 64, 64, 64, 64, false>();
}
TEST_F(TLOGTest, case_float_128x128_128x128_128x128_inPlace_False)
{
    test_tlog<float, 128, 128, 128, 128, false>();
}
TEST_F(TLOGTest, case_float_16x256_16x256_16x256_inPlace_False)
{
    test_tlog<float, 16, 256, 16, 256, false>();
}
TEST_F(TLOGTest, case_half_16x256_16x256_16x256_inPlace_False)
{
    test_tlog<aclFloat16, 16, 256, 16, 256, false>();
}
TEST_F(TLOGTest, case_float_128x128_128x128_128x128_inPlace_True)
{
    test_tlog<float, 128, 128, 128, 128, true>();
}
TEST_F(TLOGTest, case_half_16x256_16x256_16x256_inPlace_True)
{
    test_tlog<aclFloat16, 16, 256, 16, 256, true>();
}
TEST_F(TLOGTest, case_float_256x16_256x16_256x16_inPlace_False)
{
    test_tlog<float, 256, 16, 256, 16, false>();
}
TEST_F(TLOGTest, case_half_256x16_256x16_256x16_inPlace_False)
{
    test_tlog<aclFloat16, 256, 16, 256, 16, false>();
}
TEST_F(TLOGTest, case_float_32x256_32x256_32x256_inPlace_False)
{
    test_tlog<float, 32, 256, 32, 256, false>();
}
TEST_F(TLOGTest, case_half_32x256_32x256_32x256_inPlace_False)
{
    test_tlog<aclFloat16, 32, 256, 32, 256, false>();
}
TEST_F(TLOGTest, case_float_256x16_256x16_256x16_inPlace_True)
{
    test_tlog<float, 256, 16, 256, 16, true>();
}
TEST_F(TLOGTest, case_half_256x16_256x16_256x16_inPlace_True)
{
    test_tlog<aclFloat16, 256, 16, 256, 16, true>();
}