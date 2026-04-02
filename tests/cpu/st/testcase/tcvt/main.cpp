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

using namespace std;
using namespace PtoTestCommon;

template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void launchTCVT(D *dst, S *src, void *stream);

class TCVTTest : public testing::Test {
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

template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void test_tcvt()
{
    uint32_t M = kGRows_;
    uint32_t N = kGCols_;

    size_t srcFileSize = M * N * sizeof(S);
    size_t dstFileSize = M * N * sizeof(D);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    D *dstHost, *dstDevice;
    S *srcHost, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/x1_gm.bin", srcFileSize, srcHost, srcFileSize));

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTCVT<D, S, kGRows_, kGCols_, kTRows_, kTCols_>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<D> golden(dstFileSize);
    std::vector<D> devFinal(dstFileSize);
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output_z.bin", dstFileSize, devFinal.data(), dstFileSize));

    bool ret = ResultCmp<D>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TCVTTest, case1)
{
    test_tcvt<int32_t, float, 128, 128, 128, 128>();
}

TEST_F(TCVTTest, case2)
{
    test_tcvt<float, int32_t, 256, 64, 256, 64>();
}

TEST_F(TCVTTest, case3)
{
    test_tcvt<int16_t, float, 16, 32, 16, 32>();
}

TEST_F(TCVTTest, case4)
{
    test_tcvt<int32_t, float, 32, 512, 32, 512>();
}

TEST_F(TCVTTest, case5)
{
    test_tcvt<int32_t, int16_t, 2, 512, 2, 512>();
}

TEST_F(TCVTTest, case6)
{
    test_tcvt<int32_t, float, 4, 4096, 4, 4096>();
}

TEST_F(TCVTTest, case7)
{
    test_tcvt<float, int16_t, 64, 64, 64, 64>();
}

TEST_F(TCVTTest, case8)
{
    test_tcvt<aclFloat16, float, 64, 64, 64, 64>();
}

TEST_F(TCVTTest, case9)
{
    test_tcvt<uint8_t, aclFloat16, 64, 64, 64, 64>();
}

#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TCVTTest, case10)
{
    test_tcvt<bfloat16_t, float, 64, 64, 64, 64>();
}

TEST_F(TCVTTest, case11)
{
    test_tcvt<float, bfloat16_t, 64, 64, 64, 64>();
}
#endif
