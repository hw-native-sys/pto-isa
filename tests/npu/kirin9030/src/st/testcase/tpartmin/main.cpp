/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include "acl/acl.h"
#include "test_common.h"

using namespace std;
using namespace PtoTestCommon;

namespace TPartMinTest {

template <typename T, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC, bool isHalf>
void LaunchTPartMin(T *out, T *src0, T *src1, void *stream);
template <typename T, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC, int dstTR, int dstTC,
          int src0TR, int src0TC, int src1TR, int src1TC, bool isHalf>
void LaunchTPartMin(T *out, T *src0, T *src1, void *stream);

class TPARTMINTest : public testing::Test {
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

template <typename T, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC, int dstTR, int dstTC,
          int src0TR, int src0TC, int src1TR, int src1TC, bool isHalf = false>
void test_tpartmin()
{
    size_t src0FileSize = src0VR * src0VC * sizeof(T);
    size_t src1FileSize = src1VR * src1VC * sizeof(T);
    size_t dstFileSize = dstVR * dstVC * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&src0Host), src0FileSize);
    aclrtMallocHost((void **)(&src1Host), src1FileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", src0FileSize, src0Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", src1FileSize, src1Host, src1FileSize);
    aclrtMemset(dstHost, dstFileSize, 0, dstFileSize);

    aclrtMemcpy(dstDevice, dstFileSize, dstHost, dstFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if constexpr (dstTR == 0 || dstTC == 0 || src0TR == 0 || src0TC == 0 || src1TR == 0 || src1TC == 0) {
        LaunchTPartMin<T, dstVR, dstVC, src0VR, src0VC, src1VR, src1VC, isHalf>(dstDevice, src0Device, src1Device,
                                                                                stream);
    } else {
        LaunchTPartMin<T, dstVR, dstVC, src0VR, src0VC, src1VR, src1VC, dstTR, dstTC, src0TR, src0TC, src1TR, src1TC,
                       isHalf>(dstDevice, src0Device, src1Device, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize);
    std::vector<T> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename T, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC, bool isHalf = false>
void test_tpartmin()
{
    test_tpartmin<T, dstVR, dstVC, src0VR, src0VC, src1VR, src1VC, 0, 0, 0, 0, 0, 0, isHalf>();
}

TEST_F(TPARTMINTest, case_fp32_64x64_64x64_64x64)
{
    test_tpartmin<float, 64, 64, 64, 64, 64, 64>();
}
TEST_F(TPARTMINTest, case_fp32_2x24_2x24_2x8)
{
    test_tpartmin<float, 2, 24, 2, 24, 2, 8>();
}
TEST_F(TPARTMINTest, case_fp32_128x64_128x64_96x64)
{
    test_tpartmin<float, 128, 64, 128, 64, 96, 64>();
}
TEST_F(TPARTMINTest, case_fp32_95x95_95x95_95x95)
{
    test_tpartmin<float, 95, 95, 95, 95, 95, 95>();
}
TEST_F(TPARTMINTest, case_fp32_122x123_104x123_122x110)
{
    test_tpartmin<float, 122, 123, 104, 123, 122, 110>();
}
TEST_F(TPARTMINTest, case_s16_122x123_104x123_122x110)
{
    test_tpartmin<int16_t, 122, 123, 104, 123, 122, 110>();
}
TEST_F(TPARTMINTest, case_s32_122x123_104x123_122x110)
{
    test_tpartmin<int32_t, 122, 123, 104, 123, 122, 110>();
}
TEST_F(TPARTMINTest, case_u16_122x123_104x123_122x110)
{
    test_tpartmin<uint16_t, 122, 123, 104, 123, 122, 110>();
}
TEST_F(TPARTMINTest, case_u32_122x123_104x123_122x110)
{
    test_tpartmin<uint32_t, 122, 123, 104, 123, 122, 110>();
}
TEST_F(TPARTMINTest, case_u8_122x123_104x123_122x110)
{
    test_tpartmin<uint8_t, 122, 123, 104, 123, 122, 110>();
}
TEST_F(TPARTMINTest, case_s8_122x123_104x123_122x110)
{
    test_tpartmin<int8_t, 122, 123, 104, 123, 122, 110>();
}
TEST_F(TPARTMINTest, case_fp16_122x123_104x123_122x110)
{
    test_tpartmin<aclFloat16, 122, 123, 104, 123, 122, 110, true>();
}
TEST_F(TPARTMINTest, case_fp16_5x33_5x33_5x33)
{
    test_tpartmin<aclFloat16, 5, 33, 5, 33, 5, 33, 6, 1520, 6, 1520, 6, 464, true>();
}
} // namespace TPartMinTest