/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include "acl/acl.h"
#include "test_common.h"

using namespace std;
using namespace PtoTestCommon;

template <typename T, int validRows, int validCols, int upperOrLower, int diagonal>
void LaunchTTri(T* out, void* stream);

class TTRITest : public testing::Test {
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

template <typename T, int validRows, int validCols, int upperOrLower, int diagonal>
void test_ttri()
{
    size_t fileSize = validRows * validCols * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T* dstHost;
    T* dstDevice;

    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMalloc((void**)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    LaunchTTri<T, validRows, validCols, upperOrLower, diagonal>(dstDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);

    aclrtFreeHost(dstHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize);
    std::vector<T> devFinal(fileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TTRITest, case_float_32x91_lower_diag_0) { test_ttri<float, 32, 91, 0, 0>(); }
TEST_F(TTRITest, case_float_128x128_lower_diag_0) { test_ttri<float, 128, 128, 0, 0>(); }
TEST_F(TTRITest, case_float_32x91_lower_diag_3) { test_ttri<float, 32, 91, 0, 3>(); }
TEST_F(TTRITest, case_float_128x128_lower_diag_3) { test_ttri<float, 128, 128, 0, 3>(); }
TEST_F(TTRITest, case_float_32x91_lower_diag_n3) { test_ttri<float, 32, 91, 0, -3>(); }
TEST_F(TTRITest, case_float_128x128_lower_diag_n3) { test_ttri<float, 128, 128, 0, -3>(); }
TEST_F(TTRITest, case_float_32x91_upper_diag_0) { test_ttri<float, 32, 91, 1, 0>(); }
TEST_F(TTRITest, case_float_128x128_upper_diag_0) { test_ttri<float, 128, 128, 1, 0>(); }
TEST_F(TTRITest, case_float_32x91_upper_diag_3) { test_ttri<float, 32, 91, 1, 3>(); }
TEST_F(TTRITest, case_float_128x128_upper_diag_3) { test_ttri<float, 128, 128, 1, 3>(); }
TEST_F(TTRITest, case_float_32x91_upper_diag_n3) { test_ttri<float, 32, 91, 1, -3>(); }
TEST_F(TTRITest, case_float_128x128_upper_diag_n3) { test_ttri<float, 128, 128, 1, -3>(); }
