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
#include "partarg_test_utils.hpp"

using namespace PtoTestCommon;
using namespace pto;

template <int kRows, int kCols, int kValidRows1, int kValidCols1>
void LaunchTPARTARGMAX(
    float* outVal, float* src0Val, float* src1Val, uint32_t* outIdx, uint32_t* src0Idx, uint32_t* src1Idx,
    void* stream);

class TPARTARGMAX_Test : public testing::Test {};

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

TEST_F(TPARTARGMAX_Test, case_float_64x64_src1_32x32)
{
    const size_t valSize = static_cast<size_t>(kRows) * kCols * sizeof(float);
    const size_t idxSize = static_cast<size_t>(kRows) * kCols * sizeof(uint32_t);
    const std::string goldenDir = GetGoldenDir();

    float *dstValHost = nullptr, *src0ValHost = nullptr, *src1ValHost = nullptr;
    float *dstValDevice = nullptr, *src0ValDevice = nullptr, *src1ValDevice = nullptr;
    uint32_t *dstIdxHost = nullptr, *src0IdxHost = nullptr, *src1IdxHost = nullptr;
    uint32_t *dstIdxDevice = nullptr, *src0IdxDevice = nullptr, *src1IdxDevice = nullptr;

    aclInit(nullptr);
    aclrtSetDevice(kDeviceId);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    AllocateAndLoadData(src0ValHost, src0ValDevice, valSize, goldenDir + "/input0_val.bin");
    AllocateAndLoadData(src1ValHost, src1ValDevice, valSize, goldenDir + "/input1_val.bin");
    AllocateAndLoadData(src0IdxHost, src0IdxDevice, idxSize, goldenDir + "/input0_idx.bin");
    AllocateAndLoadData(src1IdxHost, src1IdxDevice, idxSize, goldenDir + "/input1_idx.bin");
    aclrtMallocHost((void**)(&dstValHost), valSize);
    aclrtMalloc((void**)&dstValDevice, valSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMallocHost((void**)(&dstIdxHost), idxSize);
    aclrtMalloc((void**)&dstIdxDevice, idxSize, ACL_MEM_MALLOC_HUGE_FIRST);

    RunPartArgKernelAndGetResults(
        LaunchTPARTARGMAX<kRows, kCols, kValidRows1, kValidCols1>, stream, dstValDevice, src0ValDevice, src1ValDevice,
        dstIdxDevice, src0IdxDevice, src1IdxDevice, dstValHost, dstIdxHost, valSize, idxSize);

    WriteFile(goldenDir + "/output_val.bin", dstValHost, valSize);
    WriteFile(goldenDir + "/output_idx.bin", dstIdxHost, idxSize);

    aclrtFree(dstValDevice);
    aclrtFree(src0ValDevice);
    aclrtFree(src1ValDevice);
    aclrtFree(dstIdxDevice);
    aclrtFree(src0IdxDevice);
    aclrtFree(src1IdxDevice);
    aclrtFreeHost(dstValHost);
    aclrtFreeHost(src0ValHost);
    aclrtFreeHost(src1ValHost);
    aclrtFreeHost(dstIdxHost);
    aclrtFreeHost(src0IdxHost);
    aclrtFreeHost(src1IdxHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(kDeviceId);
    aclFinalize();

    EXPECT_TRUE(VerifyPartArgResults(goldenDir, valSize, idxSize, kRows, kCols, kEpsilon));
}
