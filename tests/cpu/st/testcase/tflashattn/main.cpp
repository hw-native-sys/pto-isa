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
#include <pto/pto-inst.hpp>

using namespace PtoTestCommon;

namespace {

constexpr int kSeqLen = 64;
constexpr int kHeadDim = 32;

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    return "../" + suiteName + "." + caseName;
}

} // namespace

void LaunchTFLASHATTN(float* out, float* q, float* k, float* v, void* stream);

class TFLASHATTNTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TFLASHATTNTest, case1)
{
    constexpr float kEps = 2e-4f;

    const std::size_t q_bytes = static_cast<std::size_t>(kSeqLen) * static_cast<std::size_t>(kHeadDim) * sizeof(float);
    const std::size_t out_bytes = q_bytes;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    float* outHost = nullptr;
    float* qHost = nullptr;
    float* kHost = nullptr;
    float* vHost = nullptr;

    float* outDevice = nullptr;
    float* qDevice = nullptr;
    float* kDevice = nullptr;
    float* vDevice = nullptr;

    aclrtMallocHost(reinterpret_cast<void**>(&outHost), out_bytes);
    aclrtMallocHost(reinterpret_cast<void**>(&qHost), q_bytes);
    aclrtMallocHost(reinterpret_cast<void**>(&kHost), q_bytes);
    aclrtMallocHost(reinterpret_cast<void**>(&vHost), q_bytes);

    aclrtMalloc(reinterpret_cast<void**>(&outDevice), out_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&qDevice), q_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&kDevice), q_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&vDevice), q_bytes, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t fileSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/x1_gm.bin", fileSize, qHost, q_bytes));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/x2_gm.bin", fileSize, kHost, q_bytes));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/x3_gm.bin", fileSize, vHost, q_bytes));

    aclrtMemcpy(qDevice, q_bytes, qHost, q_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(kDevice, q_bytes, kHost, q_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(vDevice, q_bytes, vHost, q_bytes, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTFLASHATTN(outDevice, qDevice, kDevice, vDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, out_bytes, outDevice, out_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RESULT_GTEST(WriteFile(GetGoldenDir() + "/output_z.bin", outHost, out_bytes));

    aclrtFree(outDevice);
    aclrtFree(qDevice);
    aclrtFree(kDevice);
    aclrtFree(vDevice);

    aclrtFreeHost(outHost);
    aclrtFreeHost(qHost);
    aclrtFreeHost(kHost);
    aclrtFreeHost(vHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    const std::size_t out_elems = static_cast<std::size_t>(kSeqLen) * static_cast<std::size_t>(kHeadDim);
    std::vector<float> golden(out_elems);
    std::vector<float> actual(out_elems);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), out_bytes));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output_z.bin", fileSize, actual.data(), out_bytes));

    const bool ok = ResultCmp(golden, actual, kEps);
    EXPECT_TRUE(ok);
}
