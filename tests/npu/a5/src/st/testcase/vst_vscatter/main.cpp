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
#include <gtest/gtest.h>
#include <acl/acl.h>

using namespace std;
using namespace PtoTestCommon;

template <uint32_t caseId>
void launchVstVscatterTestCase(void* out, void* payload, void* indexes, void* stream);

class VSTVScatterTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    const std::string suiteName = testInfo->test_suite_name();
    return "../" + suiteName + "." + caseName;
}

template <uint32_t caseId, int kRows, int kCols>
bool VstVscatterTestFramework()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    constexpr size_t numel = kRows * kCols;
    size_t dataSize = numel * sizeof(float);
    size_t idxSize = numel * sizeof(uint32_t);

    float* dstHost = nullptr;
    float* payloadHost = nullptr;
    uint32_t* idxHost = nullptr;
    float* dstDevice = nullptr;
    float* payloadDevice = nullptr;
    uint32_t* idxDevice = nullptr;

    aclrtMallocHost((void**)(&dstHost), dataSize);
    aclrtMallocHost((void**)(&payloadHost), dataSize);
    aclrtMallocHost((void**)(&idxHost), idxSize);
    aclrtMalloc((void**)&dstDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&payloadDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&idxDevice, idxSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", dataSize, payloadHost, dataSize);
    ReadFile(GetGoldenDir() + "/indexes.bin", idxSize, idxHost, idxSize);
    aclrtMemcpy(payloadDevice, dataSize, payloadHost, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(idxDevice, idxSize, idxHost, idxSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchVstVscatterTestCase<caseId>(dstDevice, payloadDevice, idxDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dataSize, dstDevice, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dataSize);

    aclrtFree(dstDevice);
    aclrtFree(payloadDevice);
    aclrtFree(idxDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(payloadHost);
    aclrtFreeHost(idxHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(numel);
    std::vector<float> devFinal(numel);
    ReadFile(GetGoldenDir() + "/golden.bin", dataSize, golden.data(), dataSize);
    ReadFile(GetGoldenDir() + "/output.bin", dataSize, devFinal.data(), dataSize);
    return ResultCmp<float>(golden, devFinal, 0.0f);
}

// 32x64 f32: VF1 vsts-clear + VF2 vscatter with cyclic index shift +128.
TEST_F(VSTVScatterTest, case_float_32x64_shift128) { EXPECT_TRUE((VstVscatterTestFramework<1, 32, 64>())); }

// 64x1024 f32 GM, processed as 16x1024 UB tiles; per-tile cyclic shift +2048.
TEST_F(VSTVScatterTest, case_float_64x1024_shift2048) { EXPECT_TRUE((VstVscatterTestFramework<2, 64, 1024>())); }
