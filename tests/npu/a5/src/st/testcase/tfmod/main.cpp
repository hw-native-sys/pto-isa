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

class TFMODTest : public testing::Test {
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

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols, bool isHalf, bool highPrecision = false>
void LaunchTFMod(T *out, T *src0, T *src1, void *stream);

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols, bool isHalf, bool highPrecision = false>
void test_tfmod()
{
    size_t fileSize = kTRows_ * kTCols_ * sizeof(T);

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

    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src0Host, fileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSize, src1Host, fileSize);

    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTFMod<T, kTRows_, kTCols_, vRows, vCols, isHalf, highPrecision>(dstDevice, src0Device, src1Device, stream);

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

    float eps = 0.0005f;
    if constexpr (std::is_same_v<T, float>) {
        eps = 0.00005f;
    }
    eps = highPrecision ? 0.0000001f : eps;
    bool ret = ResultCmp<T>(golden, devFinal, eps);

    EXPECT_TRUE(ret);
}

TEST_F(TFMODTest, case1)
{
    test_tfmod<uint16_t, 64, 64, 64, 64, false>();
}

TEST_F(TFMODTest, case2)
{
    test_tfmod<uint16_t, 64, 64, 63, 63, false>();
}

TEST_F(TFMODTest, case3)
{
    test_tfmod<uint16_t, 1, 16384, 1, 16384, false>();
}

TEST_F(TFMODTest, case4)
{
    test_tfmod<uint16_t, 2048, 16, 2048, 16, false>();
}

TEST_F(TFMODTest, case5)
{
    test_tfmod<float, 32, 32, 32, 32, false, true>();
}

TEST_F(TFMODTest, case6)
{
    test_tfmod<uint32_t, 8, 8, 8, 8, false>();
}

TEST_F(TFMODTest, case7)
{
    test_tfmod<aclFloat16, 32, 32, 31, 31, true>();
}

TEST_F(TFMODTest, case8)
{
    test_tfmod<int16_t, 16, 16, 16, 16, false>();
}

TEST_F(TFMODTest, case9)
{
    test_tfmod<int32_t, 8, 8, 8, 8, false>();
}

TEST_F(TFMODTest, case10)
{
    test_tfmod<float, 64, 64, 64, 64, false, true>();
}

TEST_F(TFMODTest, case11)
{
    test_tfmod<float, 128, 128, 96, 96, false, true>();
}

TEST_F(TFMODTest, case12)
{
    test_tfmod<float, 128, 128, 96, 97, false, true>();
}