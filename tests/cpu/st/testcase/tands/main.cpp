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
#include <pto/common/constants.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace pto;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void launchTAnds_demo(uint8_t* out, uint8_t* src, void* stream);

class TANDSTest : public testing::Test {
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
void LaunchTAnds(T* out, T* src, T* scalar, void* stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void test_tands()
{
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);
    size_t scalarSize = sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost, *scalarHost;
    T *dstDevice, *srcDevice, *scalarDevice;
    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMallocHost((void**)(&srcHost), fileSize);
    aclrtMallocHost((void**)(&scalarHost), scalarSize);

    aclrtMalloc((void**)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&scalarDevice, scalarSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", fileSize, srcHost, fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/scalar.bin", scalarSize, scalarHost, scalarSize));
    aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(scalarDevice, scalarSize, scalarHost, scalarSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTAnds<T, kGRows_, kGCols_, kTRows_, kTCols_>(dstDevice, srcDevice, scalarDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(scalarDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(scalarHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize);
    std::vector<T> devFinal(fileSize);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TANDSTest, case_int16_64x64_64x64_64x64) { test_tands<int16_t, NUM_64, NUM_64, NUM_64, NUM_64>(); }
TEST_F(TANDSTest, case_int32_64x64_64x64_64x64) { test_tands<int32_t, NUM_64, NUM_64, NUM_64, NUM_64>(); }
TEST_F(TANDSTest, case_int32_16x256_16x256_16x256) { test_tands<int32_t, NUM_16, NUM_256, NUM_16, NUM_256>(); }
TEST_F(TANDSTest, case_int32_77x96_77x96_77x96) { test_tands<int32_t, NUM_77, NUM_96, NUM_77, NUM_96>(); }
TEST_F(TANDSTest, case_int32_1x32_1x32_1x32) { test_tands<int32_t, NUM_1, NUM_32, NUM_1, NUM_32>(); }
TEST_F(TANDSTest, case_uint32_32x32_32x32_32x32) { test_tands<uint32_t, NUM_32, NUM_32, NUM_32, NUM_32>(); }
TEST_F(TANDSTest, case_uint32_8x64_8x64_8x64) { test_tands<uint32_t, NUM_8, NUM_64, NUM_8, NUM_64>(); }
TEST_F(TANDSTest, case_uint32_128x128_128x128_128x128) { test_tands<uint32_t, NUM_128, NUM_128, NUM_128, NUM_128>(); }
TEST_F(TANDSTest, case_uint32_16x32_16x32_16x32) { test_tands<uint32_t, NUM_16, NUM_32, NUM_16, NUM_32>(); }
