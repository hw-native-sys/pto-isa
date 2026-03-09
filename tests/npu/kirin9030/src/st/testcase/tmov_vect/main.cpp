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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void launchTMOV(T *out, T *src, void *stream);

template <int32_t testKey>
int gen_input_golden(uint8_t *input, uint8_t *golden);

class TMOVTest : public testing::Test {
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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void test_tmov()
{
    uint32_t N = kTRows_;
    uint32_t M = kTCols_;
    size_t dataSize = N * M * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *dstDevice;
    T *srcHost, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dataSize);
    aclrtMallocHost((void **)(&srcHost), dataSize);
    aclrtMalloc((void **)(&dstDevice), dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)(&srcDevice), dataSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input_arr.bin", dataSize, srcHost, dataSize);
    aclrtMemset(dstHost, dataSize, 0, dataSize);

    aclrtMemcpy(dstDevice, dataSize, dstHost, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice, dataSize, srcHost, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMOV<T, kGRows_, kGCols_, kTRows_, kTCols_>(dstDevice, srcDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dataSize, dstDevice, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, dataSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dataSize);
    std::vector<T> devFinal(dataSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dataSize, golden.data(), dataSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", dataSize, devFinal.data(), dataSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMOVTest, vect_copy_case1)
{
    test_tmov<float, 64, 64, 64, 64>();
}

TEST_F(TMOVTest, vect_copy_case2)
{
    test_tmov<float, 32, 32, 32, 32>();
}

TEST_F(TMOVTest, vect_copy_case3)
{
    test_tmov<float, 128, 128, 128, 128>();
}

TEST_F(TMOVTest, vect_copy_case4)
{
    test_tmov<float, 128, 32, 128, 32>();
}

TEST_F(TMOVTest, vect_copy_case5)
{
    test_tmov<float, 128, 64, 128, 64>();
}

TEST_F(TMOVTest, vect_copy_case6)
{
    test_tmov<aclFloat16, 64, 64, 64, 64>();
}

TEST_F(TMOVTest, vect_copy_case7)
{
    test_tmov<aclFloat16, 32, 32, 32, 32>();
}

TEST_F(TMOVTest, vect_copy_case8)
{
    test_tmov<aclFloat16, 128, 128, 128, 128>();
}

TEST_F(TMOVTest, vect_copy_case9)
{
    test_tmov<aclFloat16, 128, 32, 128, 32>();
}

TEST_F(TMOVTest, vect_copy_case10)
{
    test_tmov<aclFloat16, 128, 64, 128, 64>();
}

TEST_F(TMOVTest, vect_copy_case11)
{
    test_tmov<uint8_t, 64, 64, 64, 64>();
}

TEST_F(TMOVTest, vect_copy_case12)
{
    test_tmov<uint8_t, 32, 32, 32, 32>();
}

TEST_F(TMOVTest, vect_copy_case13)
{
    test_tmov<uint8_t, 128, 128, 128, 128>();
}

TEST_F(TMOVTest, vect_copy_case14)
{
    test_tmov<uint8_t, 128, 32, 128, 32>();
}

TEST_F(TMOVTest, vect_copy_case15)
{
    test_tmov<uint8_t, 128, 64, 128, 64>();
}
