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

template <int descending, int kCols>
void LaunchTCI(int32_t* out, int32_t start, void* stream);

class TCI_Test : public testing::Test {};

namespace {

constexpr int32_t kStartS0 = 0;
constexpr int32_t kStartS100 = 100;
constexpr int kCols = 16;
constexpr int kDeviceId = 0;
constexpr float kEpsilon = 0.0f;

} // namespace

static std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

template <int descending>
static void run_case(int32_t start)
{
    const size_t outSize = static_cast<size_t>(kCols) * sizeof(int32_t);

    aclInit(nullptr);
    aclrtSetDevice(kDeviceId);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int32_t* dstHost;
    int32_t* dstDevice;
    aclrtMallocHost((void**)(&dstHost), outSize);
    aclrtMalloc((void**)&dstDevice, outSize, ACL_MEM_MALLOC_HUGE_FIRST);

    LaunchTCI<descending, kCols>(dstDevice, start, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outSize, dstDevice, outSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outSize);

    aclrtFree(dstDevice);
    aclrtFreeHost(dstHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(kDeviceId);
    aclFinalize();

    std::vector<int32_t> golden(kCols);
    std::vector<int32_t> out(kCols);
    size_t readSize = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", readSize, golden.data(), outSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", readSize, out.data(), outSize));
    EXPECT_TRUE(ResultCmp<int32_t>(golden, out.data(), kEpsilon));
}

TEST_F(TCI_Test, case_i32_asc_S0) { run_case<0>(kStartS0); }

TEST_F(TCI_Test, case_i32_desc_S100) { run_case<1>(kStartS100); }
