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
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <typename T, int Rows, int Cols, int ValidRows, int ValidCols>
void LaunchTSel(T *out, uint8_t *mask, T *src0, T *src1, void *stream);

class TSELTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, int Rows, int Cols, int ValidRows, int ValidCols>
void test_tsel()
{
    size_t fileSize = Rows * Cols * sizeof(T);
    size_t maskFileSize = Rows * Cols * sizeof(uint8_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    uint8_t *maskHost;
    T *dstDevice, *src0Device, *src1Device;
    uint8_t *maskDevice;

    aclrtMallocHost((void **)(&dstHost), fileSize);
    aclrtMallocHost((void **)(&maskHost), maskFileSize);
    aclrtMallocHost((void **)(&src0Host), fileSize);
    aclrtMallocHost((void **)(&src1Host), fileSize);

    aclrtMalloc((void **)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&maskDevice, maskFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", fileSize, src0Host, fileSize);
    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src1Host, fileSize);
    ReadFile(GetGoldenDir() + "/mask.bin", maskFileSize, maskHost, maskFileSize);
    aclrtMemset(dstHost, fileSize, 0, fileSize);

    aclrtMemcpy(dstDevice, fileSize, dstHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(maskDevice, maskFileSize, maskHost, maskFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTSel<T, Rows, Cols, ValidRows, ValidCols>(dstDevice, maskDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(maskDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(maskHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
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

TEST_F(TSELTest, case1)
{
    test_tsel<float, 2, 128, 2, 128>();
}
TEST_F(TSELTest, case2)
{
    test_tsel<float, 2, 32, 2, 32>();
}
TEST_F(TSELTest, case3)
{
    test_tsel<float, 2, 160, 2, 160>();
}
TEST_F(TSELTest, case4)
{
    test_tsel<aclFloat16, 2, 128, 2, 128>();
}
TEST_F(TSELTest, case5)
{
    test_tsel<aclFloat16, 2, 32, 2, 32>();
}
TEST_F(TSELTest, case6)
{
    test_tsel<aclFloat16, 2, 160, 2, 160>();
}
TEST_F(TSELTest, case7)
{
    test_tsel<int8_t, 2, 128, 2, 128>();
}
TEST_F(TSELTest, case8)
{
    test_tsel<int8_t, 2, 32, 2, 32>();
}
TEST_F(TSELTest, case9)
{
    test_tsel<int8_t, 2, 160, 2, 160>();
}
TEST_F(TSELTest, case10)
{
    test_tsel<float, 2, 512, 2, 512>();
}