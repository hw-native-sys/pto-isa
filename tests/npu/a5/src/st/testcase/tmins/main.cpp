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

class TMINSTest : public testing::Test {
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

template <typename T, int dstRow, int dstCol, int srcRow, int srcCol, int kVRows_, int kVCols_, int kPadValue_>
void LaunchTMins(T* out, T* src0, T* scalar, void* stream);

template <typename T, int dstRow, int dstCol, int srcRow, int srcCol, int kVRows_, int kVCols_, int kPadValue_>
void test_tmins()
{
    size_t srcfileSize = srcRow * srcCol * sizeof(T);
    size_t dstfileSize = dstRow * dstCol * sizeof(T);
    size_t scalarFileSize = sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), dstfileSize);
    aclrtMallocHost((void**)(&src0Host), srcfileSize);
    aclrtMallocHost((void**)(&src1Host), scalarFileSize);

    aclrtMalloc((void**)&dstDevice, dstfileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, srcfileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, scalarFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", srcfileSize, src0Host, srcfileSize);
    ReadFile(GetGoldenDir() + "/input_scalar.bin", scalarFileSize, src1Host, scalarFileSize);

    aclrtMemcpy(src0Device, srcfileSize, src0Host, srcfileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, scalarFileSize, src1Host, scalarFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTMins<T, dstRow, dstCol, srcRow, srcCol, kVRows_, kVCols_, kPadValue_>(
        dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstfileSize, dstDevice, dstfileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstfileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstfileSize);
    std::vector<T> devFinal(dstfileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstfileSize, golden.data(), dstfileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstfileSize, devFinal.data(), dstfileSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.0001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMINSTest, case_float_64x64_32x32_32x32) { test_tmins<float, 64, 64, 32, 32, 32, 32, PAD_VALUE_NULL>(); }
TEST_F(TMINSTest, case_float_128x128_64x64_64x64) { test_tmins<float, 128, 128, 64, 64, 64, 64, PAD_VALUE_NULL>(); }
TEST_F(TMINSTest, case_float_60x128_64x64_60x60) { test_tmins<float, 60, 128, 64, 64, 60, 60, PAD_VALUE_MAX>(); }
TEST_F(TMINSTest, case_float_16x200_20x512_16x200) { test_tmins<float, 16, 200, 20, 512, 16, 200, PAD_VALUE_MAX>(); }
TEST_F(TMINSTest, case_float_1x3600_2x4096_1x3600) { test_tmins<float, 1, 3600, 2, 4096, 1, 3600, PAD_VALUE_MAX>(); }
TEST_F(TMINSTest, case_int32_32x32_32x32_32x32) { test_tmins<int32_t, 32, 32, 32, 32, 32, 32, PAD_VALUE_NULL>(); }
TEST_F(TMINSTest, case_uint32_32x32_32x32_32x32) { test_tmins<uint32_t, 32, 32, 32, 32, 32, 32, PAD_VALUE_NULL>(); }
TEST_F(TMINSTest, case_int16_32x128_32x128_32x128) { test_tmins<int16_t, 32, 128, 32, 128, 32, 128, PAD_VALUE_NULL>(); }
TEST_F(TMINSTest, case_uint16_32x128_32x128_32x128)
{
    test_tmins<uint16_t, 32, 128, 32, 128, 32, 128, PAD_VALUE_NULL>();
}
TEST_F(TMINSTest, case_int8_32x128_32x128_32x128) { test_tmins<int8_t, 32, 128, 32, 128, 32, 128, PAD_VALUE_NULL>(); }
TEST_F(TMINSTest, case_uint8_32x128_32x128_32x128) { test_tmins<uint8_t, 32, 128, 32, 128, 32, 128, PAD_VALUE_NULL>(); }
TEST_F(TMINSTest, case_half_16x256_20x224_16x200)
{
    test_tmins<aclFloat16, 16, 256, 20, 224, 16, 200, PAD_VALUE_MAX>();
}
TEST_F(TMINSTest, case_half_1x256_1x224_1x200) { test_tmins<aclFloat16, 1, 256, 1, 224, 1, 200, PAD_VALUE_MAX>(); }