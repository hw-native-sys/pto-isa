/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

class TINTERLEAVETest : public testing::Test {
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

template <typename T, int row, int col>
void LaunchTINTERLEAVE(T* dst1, T* dst0, T* src1, T* src0, void* stream);

template <typename T, int row, int col>
void test_TINTERLEAVE()
{
    size_t fileSize = row * col * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dst1Host, *dst0Host, *src1Host, *src0Host;
    T *dst1Device, *dst0Device, *src1Device, *src0Device;

    aclrtMallocHost((void**)(&dst1Host), fileSize);
    aclrtMallocHost((void**)(&dst0Host), fileSize);
    aclrtMallocHost((void**)(&src1Host), fileSize);
    aclrtMallocHost((void**)(&src0Host), fileSize);

    aclrtMalloc((void**)&dst1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dst0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input0.bin", fileSize, src0Host, fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src1Host, fileSize));

    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTINTERLEAVE<T, row, col>(dst1Device, dst0Device, src1Device, src0Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dst1Host, fileSize, dst1Device, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(dst0Host, fileSize, dst0Device, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    CHECK_RESULT_GTEST(WriteFile(GetGoldenDir() + "/output1.bin", dst1Host, fileSize));
    CHECK_RESULT_GTEST(WriteFile(GetGoldenDir() + "/output0.bin", dst0Host, fileSize));

    aclrtFree(dst1Device);
    aclrtFree(dst0Device);
    aclrtFree(src1Device);
    aclrtFree(src0Device);

    aclrtFreeHost(dst1Host);
    aclrtFreeHost(dst0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src0Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden0(row * col);
    std::vector<T> output0(row * col);

    std::vector<T> golden1(row * col);
    std::vector<T> output1(row * col);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/dst0.bin", fileSize, golden0.data(), fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output0.bin", fileSize, output0.data(), fileSize));
    bool ret0 = ResultCmp<T>(golden0, output0, 0.001f);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/dst1.bin", fileSize, golden1.data(), fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output1.bin", fileSize, output1.data(), fileSize));
    bool ret1 = ResultCmp<T>(golden1, output1, 0.001f);

    EXPECT_TRUE(ret0 && ret1);
}

TEST_F(TINTERLEAVETest, case_1) { test_TINTERLEAVE<float, 64, 64>(); }
TEST_F(TINTERLEAVETest, case_2) { test_TINTERLEAVE<int32_t, 16, 64>(); }
TEST_F(TINTERLEAVETest, case_3) { test_TINTERLEAVE<int16_t, 32, 128>(); }
TEST_F(TINTERLEAVETest, case_4) { test_TINTERLEAVE<aclFloat16, 20, 16>(); }
