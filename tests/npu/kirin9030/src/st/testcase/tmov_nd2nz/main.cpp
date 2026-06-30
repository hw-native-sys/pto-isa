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

template <int kSrcRows, int kDstRows, int kCols>
void launchTMOV_nd2nz(aclFloat16 *out, aclFloat16 *src, void *stream);

class TMovNd2NzTest : public testing::Test {
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

template <int kSrcRows, int kDstRows, int kCols>
void test_tmov_nd2nz()
{
    size_t inputSize = kSrcRows * kCols * sizeof(aclFloat16);
    size_t outputSize = kDstRows * kCols * sizeof(aclFloat16);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    aclFloat16 *dstHost, *dstDevice;
    aclFloat16 *srcHost, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), outputSize);
    aclrtMallocHost((void **)(&srcHost), inputSize);
    aclrtMalloc((void **)(&dstDevice), outputSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)(&srcDevice), inputSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input_arr.bin", inputSize, srcHost, inputSize);

    aclrtMemcpy(srcDevice, inputSize, srcHost, inputSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMOV_nd2nz<kSrcRows, kDstRows, kCols>(dstDevice, srcDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputSize, dstDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, outputSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<uint8_t> golden(outputSize);
    std::vector<uint8_t> devFinal(outputSize);
    ReadFile(GetGoldenDir() + "/golden.bin", outputSize, golden.data(), outputSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", outputSize, devFinal.data(), outputSize);

    bool ret = ResultCmp<uint8_t>(golden, devFinal, 0.0f);
    EXPECT_TRUE(ret);
}

TEST_F(TMovNd2NzTest, case_half_1x128_1to16)
{
    test_tmov_nd2nz<1, 16, 128>();
}

TEST_F(TMovNd2NzTest, case_half_1x256_1to16)
{
    test_tmov_nd2nz<1, 16, 256>();
}

TEST_F(TMovNd2NzTest, case_half_16x256_16to16)
{
    test_tmov_nd2nz<16, 16, 256>();
}
