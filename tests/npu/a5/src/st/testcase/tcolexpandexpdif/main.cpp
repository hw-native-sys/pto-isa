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
#include <type_traits>

using namespace std;
using namespace PtoTestCommon;

namespace TColExpandExpdifTest {
template <typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col>
void launchTColExpandExpdif(T* out, T* src0, T* src1, void* stream);

class TColExpandExpdifTest : public testing::Test {
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

template <typename T, uint32_t dstRow, uint32_t dstCol, uint32_t src1Row, uint32_t src1Col>
void test_tcolexpandexpdif()
{
    size_t inputFileSize = src1Row * src1Col * sizeof(T);
    size_t outputFileSize = dstRow * dstCol * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), outputFileSize);
    aclrtMallocHost((void**)(&src0Host), outputFileSize);
    aclrtMallocHost((void**)(&src1Host), inputFileSize);

    aclrtMalloc((void**)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", outputFileSize, src0Host, outputFileSize);
    ReadFile(GetGoldenDir() + "/input1.bin", inputFileSize, src1Host, inputFileSize);

    aclrtMemcpy(src0Device, outputFileSize, src0Host, outputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, inputFileSize, src1Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTColExpandExpdif<T, dstRow, dstCol, src1Row, src1Col>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    // Compare in the testcase's element type: reading f16 bytes as float
    // (the old vector<float> form) only ever passed because the golden was
    // bit-identical to the device output — any 1-ulp deviation flipped a
    // packed f16 pair's shared f32 exponent into huge bogus diffs.
    std::vector<T> golden(outputFileSize / sizeof(T));
    std::vector<T> devFinal(outputFileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);
    // fp16 tolerance: A6's fused vexpdif core deviates <= 1 ulp from the
    // correctly-rounded reference round16(exp32(x16)) on ~1-5% of elements
    // (same envelope as NVIDIA hexp). 1 f16 ulp relative is up to 2^-10
    // ≈ 0.00098, so eps=0.005 gives 5x margin. The fp32 path (exact) keeps
    // the tighter 0.001.
    float cmpEps = std::is_same<T, aclFloat16>::value ? 0.005f : 0.001f;
    bool ret = ResultCmp(golden, devFinal, cmpEps);

    EXPECT_TRUE(ret);
}

TEST_F(TColExpandExpdifTest, case_fp32_32_16_1_16) { test_tcolexpandexpdif<float, 32, 16, 1, 16>(); }
TEST_F(TColExpandExpdifTest, case_fp32_16_32_1_32) { test_tcolexpandexpdif<float, 16, 32, 1, 32>(); }
TEST_F(TColExpandExpdifTest, case_fp16_32_32_1_32) { test_tcolexpandexpdif<aclFloat16, 32, 32, 1, 32>(); }
TEST_F(TColExpandExpdifTest, case_fp16_16_128_1_128) { test_tcolexpandexpdif<aclFloat16, 16, 128, 1, 128>(); }
} // namespace TColExpandExpdifTest
