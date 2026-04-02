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
using namespace pto;

template <int32_t tilingKey>
void launchTCMPS_demo(uint8_t *out, uint8_t *src, void *stream);

class TCMPSTest : public testing::Test {
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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int cmpMode>
void LaunchTCmps(uint8_t *out, T *src0, T *src1, void *stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int cmpMode>
void test_tcmps()
{
    size_t fileSize = kTRows_ * kTCols_ * sizeof(T);
    size_t file_size_dst = kTRows_ * kTCols_ / 8;
    size_t scalarfileSize = sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *src0Host, *src0Device;
    T *src1Host, *src1Device;

    uint8_t *dstHost, *dstDevice;
    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&src0Host), fileSize);
    aclrtMallocHost((void **)(&src1Host), scalarfileSize);

    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src0Host, fileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", scalarfileSize, src1Host, scalarfileSize);

    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, scalarfileSize, src1Host, scalarfileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTCmps<T, kGRows_, kGCols_, kTRows_, kTCols_, cmpMode>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, file_size_dst, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, file_size_dst);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<uint8_t> golden(kGRows_ * kGCols_);
    std::vector<uint8_t> devFinal(kGRows_ * kGCols_);
    ReadFile(GetGoldenDir() + "/golden.bin", file_size_dst, golden.data(), file_size_dst);
    ReadFile(GetGoldenDir() + "/output.bin", file_size_dst, devFinal.data(), file_size_dst);

    bool ret = ResultCmp<uint8_t>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TCMPSTest, case_half_32x32_32x32_32x32)
{
    test_tcmps<aclFloat16, 32, 32, 32, 32, 5>();
}
#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TCMPSTest, case_bf16_32x32_32x32_32x32)
{
    test_tcmps<bfloat16_t, 32, 32, 32, 32, 5>();
}
#endif
TEST_F(TCMPSTest, case_float_1x64_1x64_1x64)
{
    test_tcmps<float, 1, 64, 1, 64, 0>();
}
TEST_F(TCMPSTest, case_float_8x64_8x64_8x64)
{
    test_tcmps<float, 8, 64, 8, 64, 4>();
}
TEST_F(TCMPSTest, case_float_4x64_4x64_4x64)
{
    test_tcmps<float, 4, 64, 4, 64, 1>();
}
TEST_F(TCMPSTest, case_int32_16x32_16x32_16x32)
{
    test_tcmps<int32_t, 16, 32, 16, 32, 0>();
}
TEST_F(TCMPSTest, case_float_128x128_128x128_128x128)
{
    test_tcmps<float, 128, 128, 128, 128, 3>();
}
TEST_F(TCMPSTest, case_int32_32x32_32x32_32x32)
{
    test_tcmps<int32_t, 32, 32, 32, 32, 0>();
}
