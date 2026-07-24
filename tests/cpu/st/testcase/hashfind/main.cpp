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

class HASHFINDTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    return "../" + suiteName + "." + caseName;
}

template <int kTileRows, int kTileCols, int kCap, int kMaxProbe>
void LaunchHashFind(int32_t* out, int32_t* table_keys, int32_t* table_vals, int32_t* queries, void* stream);

template <int kTileRows, int kTileCols, int kCap, int kMaxProbe>
void test_hashfind()
{
    const size_t tableBytes = kCap * sizeof(int32_t);
    const size_t queryBytes = kTileRows * kTileCols * sizeof(int32_t);
    const size_t outBytes = queryBytes;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int32_t *keysHost, *valsHost, *qHost, *outHost;
    int32_t *keysDevice, *valsDevice, *qDevice, *outDevice;

    aclrtMallocHost((void**)(&keysHost), tableBytes);
    aclrtMallocHost((void**)(&valsHost), tableBytes);
    aclrtMallocHost((void**)(&qHost), queryBytes);
    aclrtMallocHost((void**)(&outHost), outBytes);

    aclrtMalloc((void**)&keysDevice, tableBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&valsDevice, tableBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&qDevice, queryBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&outDevice, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t size = 0;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", size, keysHost, tableBytes));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input2.bin", size, valsHost, tableBytes));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input3.bin", size, qHost, queryBytes));

    aclrtMemcpy(keysDevice, tableBytes, keysHost, tableBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(valsDevice, tableBytes, valsHost, tableBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(qDevice, queryBytes, qHost, queryBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchHashFind<kTileRows, kTileCols, kCap, kMaxProbe>(outDevice, keysDevice, valsDevice, qDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, outBytes, outDevice, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", outHost, outBytes);

    aclrtFree(keysDevice);
    aclrtFree(valsDevice);
    aclrtFree(qDevice);
    aclrtFree(outDevice);
    aclrtFreeHost(keysHost);
    aclrtFreeHost(valsHost);
    aclrtFreeHost(qHost);
    aclrtFreeHost(outHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<int32_t> golden(outBytes / sizeof(int32_t));
    std::vector<int32_t> devFinal(outBytes / sizeof(int32_t));
    size_t outSize = outBytes;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", outSize, golden.data(), outBytes));
    outSize = outBytes;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", outSize, devFinal.data(), outBytes));
    EXPECT_TRUE(ResultCmp<int32_t>(golden, devFinal.data(), 0.0f));
}

TEST_F(HASHFINDTest, case_int32_16x16_cap512) { test_hashfind<16, 16, 512, 64>(); }
