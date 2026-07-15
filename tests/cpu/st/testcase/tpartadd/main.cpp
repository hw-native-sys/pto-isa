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

using namespace PtoTestCommon;

template <int kRows, int kCols, int kValidRows1, int kValidCols1>
void LaunchTPARTADD(float* out, float* src0, float* src1, void* stream);

class TPARTADD_Test : public testing::Test {};

namespace {

constexpr int kDeviceId = 0;
constexpr float kEpsilon = 0.0f;
constexpr int kRows = 64;
constexpr int kCols = 64;
constexpr int kValidRows1 = 32;
constexpr int kValidCols1 = 32;

} // namespace

static std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

TEST_F(TPARTADD_Test, case_float_64x64_src1_32x32)
{
    const size_t fileSize = static_cast<size_t>(kRows) * static_cast<size_t>(kCols) * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(kDeviceId);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    float *dstHost, *src0Host, *src1Host;
    float *dstDevice, *src0Device, *src1Device;
    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMallocHost((void**)(&src0Host), fileSize);
    aclrtMallocHost((void**)(&src1Host), fileSize);
    aclrtMalloc((void**)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t readSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", readSize, src0Host, fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input2.bin", readSize, src1Host, fileSize));
    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTPARTADD<kRows, kCols, kValidRows1, kValidCols1>(dstDevice, src0Device, src1Device, stream);
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
    aclrtResetDevice(kDeviceId);
    aclFinalize();

    std::vector<float> golden(static_cast<size_t>(kRows) * static_cast<size_t>(kCols));
    std::vector<float> out(static_cast<size_t>(kRows) * static_cast<size_t>(kCols));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", readSize, golden.data(), fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", readSize, out.data(), fileSize));
    EXPECT_TRUE(ResultCmp<float>(golden, out.data(), kEpsilon));
}
