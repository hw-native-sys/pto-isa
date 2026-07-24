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
#include <climits>

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void launchTSELS_demo(uint8_t* out, uint8_t* src, void* stream);

class TSELSTest : public testing::Test {
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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTSelS(T* out, T scalar, uint32_t* src0, T* src1, void* stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void test_tsels()
{
    size_t totalElements = kGRows_ * kGCols_;
    size_t fileSize = totalElements * sizeof(T);

    // --- UPDATED: Calculate memory footprint for the 1-bit packed mask ---
    constexpr size_t bitsPerWord = sizeof(uint32_t) * CHAR_BIT; // 32 bits
    size_t totalMaskWords = (totalElements + bitsPerWord - 1) / bitsPerWord;
    size_t maskSize = totalMaskWords * sizeof(uint32_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src1Host;
    T *dstDevice, *src1Device;
    uint32_t *src0Host, *src0Device;

    T scalar;
    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMallocHost((void**)(&src0Host), maskSize);
    aclrtMallocHost((void**)(&src1Host), fileSize);

    aclrtMalloc((void**)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, maskSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t actualSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/mask.bin", actualSize, src0Host, maskSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", actualSize, src1Host, fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/scalar.bin", actualSize, &scalar, sizeof(scalar)));

    aclrtMemcpy(src0Device, maskSize, src0Host, maskSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTSelS<T, kGRows_, kGCols_, kTRows_, kTCols_>(dstDevice, scalar, src0Device, src1Device, stream);

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

    std::vector<T> golden(totalElements);
    std::vector<T> devFinal(totalElements);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TSELSTest, case_float_64x64_64x64_64x64) { test_tsels<float, 64, 64, 64, 64>(); }
TEST_F(TSELSTest, case_int32_64x64_64x64_64x64) { test_tsels<int32_t, 64, 64, 64, 64>(); }
TEST_F(TSELSTest, case_int16_64x64_64x64_64x64) { test_tsels<int16_t, 64, 64, 64, 64>(); }
TEST_F(TSELSTest, case_half_16x256_16x256_16x256) { test_tsels<aclFloat16, 16, 256, 16, 256>(); }
TEST_F(TSELSTest, case_uint16_64x64_64x64_64x64) { test_tsels<uint16_t, 64, 64, 64, 64>(); }
TEST_F(TSELSTest, case_uint32_64x64_64x64_64x64) { test_tsels<uint32_t, 64, 64, 64, 64>(); }
#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TSELSTest, case_bf16_16x256_16x256_16x256) { test_tsels<bfloat16_t, 16, 256, 16, 256>(); }
#endif
