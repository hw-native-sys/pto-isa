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
void LaunchTPartMin(T* out, T* src0, T* src1, void* stream);
template <
    typename T, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC, int dstTR, int dstTC, int src0TR,
    int src0TC, int src1TR, int src1TC, bool isHalf>
void LaunchTPartMin(T* out, T* src0, T* src1, void* stream);

class TPARTMINTest : public testing::Test {
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

template <typename T>
T* load_data(size_t srcRows, size_t srcCols, std::string filePath)
{
    T* srcHost;
    T* srcDevice;
    if (srcRows == 0 || srcCols == 0) {
        aclrtMalloc((void**)&srcDevice, 1, ACL_MEM_MALLOC_HUGE_FIRST);
    } else {
        size_t tileSize = srcRows * srcCols * sizeof(T);
        aclrtMallocHost((void**)(&srcHost), tileSize);
        aclrtMalloc((void**)&srcDevice, tileSize, ACL_MEM_MALLOC_HUGE_FIRST);
        ReadFile(filePath, tileSize, srcHost, tileSize);
        aclrtMemcpy(srcDevice, tileSize, srcHost, tileSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtFreeHost(srcHost);
    }
    return srcDevice;
}

template <
    typename T, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC, int dstTR, int dstTC, int src0TR,
    int src0TC, int src1TR, int src1TC, bool isHalf = false>
void test_tpartmin()
{
    size_t src0FileSize = src0VR * src0VC * sizeof(T);
    size_t src1FileSize = src1VR * src1VC * sizeof(T);
    size_t dstFileSize = dstVR * dstVC * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *dstDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(dstHost, dstFileSize, 0, dstFileSize);

    aclrtMemcpy(dstDevice, dstFileSize, dstHost, dstFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    T* src0Device = load_data<T>(src0VR, src0VC, GetGoldenDir() + "/input1.bin");
    T* src1Device = load_data<T>(src1VR, src1VC, GetGoldenDir() + "/input2.bin");
    if constexpr (dstTR == 0 || dstTC == 0 || src0TR == 0 || src0TC == 0 || src1TR == 0 || src1TC == 0) {
        LaunchTPartMin<T, dstVR, dstVC, src0VR, src0VC, src1VR, src1VC, isHalf>(
            dstDevice, src0Device, src1Device, stream);
    } else {
        LaunchTPartMin<
            T, dstVR, dstVC, src0VR, src0VC, src1VR, src1VC, dstTR, dstTC, src0TR, src0TC, src1TR, src1TC, isHalf>(
            dstDevice, src0Device, src1Device, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFreeHost(dstHost);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
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

TEST_F(TPARTMINTest, case_fp32_64x64_64x64_64x64) { test_tpartmin<float, 64, 64, 64, 64, 64, 64>(); }
TEST_F(TPARTMINTest, case_fp32_2x24_2x24_2x8) { test_tpartmin<float, 2, 24, 2, 24, 2, 8>(); }
TEST_F(TPARTMINTest, case_fp32_2x24_2x24_1x8) { test_tpartmin<float, 2, 24, 2, 24, 1, 8>(); }
TEST_F(TPARTMINTest, case_fp32_128x64_128x64_96x64) { test_tpartmin<float, 128, 64, 128, 64, 96, 64>(); }
TEST_F(TPARTMINTest, case_fp32_95x95_95x95_95x95) { test_tpartmin<float, 95, 95, 95, 95, 95, 95>(); }
TEST_F(TPARTMINTest, case_fp32_61x123_52x123_61x123) { test_tpartmin<float, 61, 123, 52, 123, 61, 123>(); }
TEST_F(TPARTMINTest, case_s16_61x123_52x123_61x123) { test_tpartmin<int16_t, 61, 123, 52, 123, 61, 123>(); }
TEST_F(TPARTMINTest, case_s32_61x123_52x123_61x123) { test_tpartmin<int32_t, 61, 123, 52, 123, 61, 123>(); }
TEST_F(TPARTMINTest, case_u16_61x123_52x123_61x123) { test_tpartmin<uint16_t, 61, 123, 52, 123, 61, 123>(); }
TEST_F(TPARTMINTest, case_u32_61x123_52x123_61x123) { test_tpartmin<uint32_t, 61, 123, 52, 123, 61, 123>(); }
TEST_F(TPARTMINTest, case_u8_61x123_52x123_61x123) { test_tpartmin<uint8_t, 61, 123, 52, 123, 61, 123>(); }
TEST_F(TPARTMINTest, case_s8_61x123_52x123_61x123) { test_tpartmin<int8_t, 61, 123, 52, 123, 61, 123>(); }
TEST_F(TPARTMINTest, case_fp16_61x123_52x123_61x123) { test_tpartmin<aclFloat16, 61, 123, 52, 123, 61, 123, true>(); }
TEST_F(TPARTMINTest, case_fp16_5x33_5x33_5x33)
{
    test_tpartmin<aclFloat16, 5, 33, 5, 33, 5, 33, 6, 1520, 6, 1520, 6, 464, true>();
}
TEST_F(TPARTMINTest, case_fp32_8x8_8x0_8x8) { test_tpartmin<float, 8, 8, 8, 0, 8, 8, 8, 8, 1, 8, 8, 8>(); }
TEST_F(TPARTMINTest, case_fp32_8x8_0x8_8x8) { test_tpartmin<float, 8, 8, 0, 8, 8, 8, 8, 8, 1, 8, 8, 8>(); }
TEST_F(TPARTMINTest, case_fp32_8x8_8x8_8x0) { test_tpartmin<float, 8, 8, 8, 8, 8, 0, 8, 8, 8, 8, 1, 8>(); }
TEST_F(TPARTMINTest, case_fp32_8x8_8x8_0x8) { test_tpartmin<float, 8, 8, 8, 8, 0, 8, 8, 8, 8, 8, 1, 8>(); }
} // namespace TPartMinTest
