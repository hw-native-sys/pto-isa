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

template <int32_t tilingKey>
void launchTNOT_demo(uint8_t* out, uint8_t* src, void* stream);

class TNOTTest : public testing::Test {
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

template <typename T, int sTRows_, int sTCols_, int dTRows_, int dTCols_, int kGRows_, int kGCols_>
void LaunchTNot(T* out, T* src0, void* stream);

template <typename T, int sTRows_, int sTCols_, int dTRows_, int dTCols_, int kGRows_, int kGCols_>
void test_tnot()
{
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host;
    T *dstDevice, *src0Device;

    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMallocHost((void**)(&src0Host), fileSize);

    aclrtMalloc((void**)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src0Host, fileSize));

    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTNot<T, sTRows_, sTCols_, dTRows_, dTCols_, kGRows_, kGCols_>(dstDevice, src0Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
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
TEST_F(TNOTTest, case_0) { test_tnot<int32_t, 64, 64, 64, 64, 60, 55>(); }
TEST_F(TNOTTest, case_1) { test_tnot<int16_t, 64, 64, 64, 64, 60, 55>(); }
TEST_F(TNOTTest, case_2) { test_tnot<int32_t, 64, 64, 96, 96, 64, 60>(); }
TEST_F(TNOTTest, case_3) { test_tnot<int16_t, 64, 64, 96, 96, 64, 60>(); }

TEST_F(TNOTTest, case_4) { test_tnot<uint32_t, 64, 64, 64, 64, 60, 55>(); }
TEST_F(TNOTTest, case_5) { test_tnot<uint16_t, 64, 64, 64, 64, 60, 55>(); }
TEST_F(TNOTTest, case_6) { test_tnot<uint32_t, 96, 96, 96, 96, 64, 60>(); }
TEST_F(TNOTTest, case_7) { test_tnot<uint16_t, 96, 96, 64, 64, 64, 60>(); }
