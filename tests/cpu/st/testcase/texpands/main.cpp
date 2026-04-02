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

class TEXPANDSTest : public testing::Test {
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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_, int padValueType>
void LaunchTExpandS(void *out, float scalar, void *stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kVRows_, int kVCols_, int padValueType>
void test_texpands()
{
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost;
    T *dstDevice;

    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    float scalar;
    std::string scalar_file = GetGoldenDir() + "/scalar.bin";
    std::ifstream file(scalar_file, std::ios::binary);
    file.read(reinterpret_cast<char *>(&scalar), 4);
    file.close();

    LaunchTExpandS<T, kGRows_, kGCols_, kTRows_, kTCols_, kVRows_, kVCols_, padValueType>(dstDevice, scalar, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);

    aclrtFreeHost(dstHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize);
    std::vector<T> devFinal(fileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TEXPANDSTest, case_float_64x64_64x64_64x64_PAD_VALUE_NULL)
{
    test_texpands<float, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>();
}
TEST_F(TEXPANDSTest, case_int32_64x64_64x64_64x64_PAD_VALUE_NULL)
{
    test_texpands<int32_t, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>();
}
TEST_F(TEXPANDSTest, case_half_64x64_64x64_64x64_PAD_VALUE_NULL)
{
    test_texpands<aclFloat16, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>();
}
#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TEXPANDSTest, case_bf16_64x64_64x64_64x64_PAD_VALUE_NULL)
{
    test_texpands<bfloat16_t, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>();
}
#endif
TEST_F(TEXPANDSTest, case_int16_64x64_64x64_64x64_PAD_VALUE_NULL)
{
    test_texpands<int16_t, 64, 64, 64, 64, 64, 64, PAD_VALUE_NULL>();
}

TEST_F(TEXPANDSTest, case_float_60x60_64x64_60x60_PAD_VALUE_MAX)
{
    test_texpands<float, 60, 60, 64, 64, 60, 60, PAD_VALUE_MAX>();
}
TEST_F(TEXPANDSTest, case_int32_60x60_64x64_60x60_PAD_VALUE_MAX)
{
    test_texpands<int32_t, 60, 60, 64, 64, 60, 60, PAD_VALUE_MAX>();
}
TEST_F(TEXPANDSTest, case_half_1x3600_2x4096_1x3600_PAD_VALUE_MAX)
{
    test_texpands<aclFloat16, 1, 3600, 2, 4096, 1, 3600, PAD_VALUE_MAX>();
}
#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TEXPANDSTest, case_bf16_1x3600_2x4096_1x3600_PAD_VALUE_MAX)
{
    test_texpands<bfloat16_t, 1, 3600, 2, 4096, 1, 3600, PAD_VALUE_MAX>();
}
#endif
TEST_F(TEXPANDSTest, case_int16_16x200_20x512_16x200_PAD_VALUE_MAX)
{
    test_texpands<int16_t, 16, 200, 20, 512, 16, 200, PAD_VALUE_MAX>();
}
