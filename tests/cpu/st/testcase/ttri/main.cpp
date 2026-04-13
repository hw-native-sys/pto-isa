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
#include "pto/pto-inst.hpp"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void launchTTRI_demo(uint8_t *out, uint8_t *src, void *stream);

class TTRITest : public testing::Test {
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

template <typename T, int isUpperOrLower, int diagonal, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTTRI(T *out, void *stream);

template <typename T, int isUpperOrLower, int diagonal, int kTRows_, int kTCols_, int vRows, int vCols>
void test_ttri()
{
    size_t fileSize = vRows * vCols * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost;
    T *dstDevice;

    aclrtMallocHost((void **)(&dstHost), fileSize);

    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    LaunchTTRI<T, isUpperOrLower, diagonal, kTRows_, kTCols_, vRows, vCols>(dstDevice, stream);

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
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TTRITest, case_float_4x8_4x4_1_0)
{
    test_ttri<float, 1, 0, 4, 8, 4, 4>();
}
TEST_F(TTRITest, case_float_64x64_64x64_1_0)
{
    test_ttri<float, 1, 0, 64, 64, 64, 64>();
}
TEST_F(TTRITest, case_int32_64x64_64x64_1_0)
{
    test_ttri<int32_t, 1, 0, 64, 64, 64, 64>();
}
TEST_F(TTRITest, case_int16_64x64_64x64_1_0)
{
    test_ttri<int16_t, 1, 0, 64, 64, 64, 64>();
}
TEST_F(TTRITest, case_half_16x256_16x256_1_0)
{
    test_ttri<aclFloat16, 1, 0, 16, 256, 16, 256>();
}
#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TTRITest, case_bf16_16x256_16x256_1_0)
{
    test_ttri<bfloat16_t, 1, 0, 16, 256, 16, 256>();
}
#endif
TEST_F(TTRITest, case_float_128x128_128x128_1_0)
{
    test_ttri<float, 1, 0, 128, 128, 128, 128>();
}
TEST_F(TTRITest, case_float_64x64_64x64_0_0)
{
    test_ttri<float, 0, 0, 64, 64, 64, 64>();
}
TEST_F(TTRITest, case_int32_64x64_64x64_0_0)
{
    test_ttri<int32_t, 0, 0, 64, 64, 64, 64>();
}
TEST_F(TTRITest, case_int16_64x64_64x64_0_0)
{
    test_ttri<int16_t, 0, 0, 64, 64, 64, 64>();
}
TEST_F(TTRITest, case_half_16x256_16x256_0_0)
{
    test_ttri<aclFloat16, 0, 0, 16, 256, 16, 256>();
}
#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TTRITest, case_bf16_16x256_16x256_0_0)
{
    test_ttri<bfloat16_t, 0, 0, 16, 256, 16, 256>();
}
#endif
TEST_F(TTRITest, case_float_128x128_128x128_0_0)
{
    test_ttri<float, 0, 0, 128, 128, 128, 128>();
}
TEST_F(TTRITest, case_float_128x128_128x125_0_0)
{
    test_ttri<float, 0, 0, 128, 128, 128, 125>();
}
TEST_F(TTRITest, case_uint32_64x64_64x64_1_0)
{
    test_ttri<uint32_t, 1, 0, 64, 64, 64, 64>();
}
TEST_F(TTRITest, case_uint32_64x64_64x64_0_0)
{
    test_ttri<uint32_t, 0, 0, 64, 64, 64, 64>();
}
TEST_F(TTRITest, case_float_128x128_128x111_0_2)
{
    test_ttri<float, 0, 2, 128, 128, 128, 111>();
}
TEST_F(TTRITest, case_float_128x128_128x111_0__2)
{
    test_ttri<float, 0, -2, 128, 128, 128, 111>();
}
TEST_F(TTRITest, case_float_128x128_128x111_1_2)
{
    test_ttri<float, 1, 2, 128, 128, 128, 111>();
}
TEST_F(TTRITest, case_float_128x128_128x111_1__2)
{
    test_ttri<float, 1, -2, 128, 128, 128, 111>();
}
TEST_F(TTRITest, case_float_128x128_128x31_0_444)
{
    test_ttri<float, 0, 444, 128, 128, 128, 31>();
}
TEST_F(TTRITest, case_float_128x128_128x31_1_444)
{
    test_ttri<float, 1, 444, 128, 128, 128, 31>();
}
TEST_F(TTRITest, case_float_128x128_128x31_0__444)
{
    test_ttri<float, 0, -444, 128, 128, 128, 31>();
}
TEST_F(TTRITest, case_float_128x128_128x31_1__444)
{
    test_ttri<float, 1, -444, 128, 128, 128, 31>();
}
