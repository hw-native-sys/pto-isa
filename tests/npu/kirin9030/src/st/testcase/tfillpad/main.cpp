/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <type_traits>
#include <gtest/gtest.h>
#include "test_common.h"
#include "acl/acl.h"

using namespace std;
using namespace PtoTestCommon;

class TFILLPADTest : public testing::Test {
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

template <int32_t testKey>
void launchTFILLPAD(uint8_t *out, uint8_t *src, void *stream);

template <typename T, int32_t srcRows, int32_t srcCols, int32_t dstRows, int32_t dstCols, int32_t testKey>
void test_tfillpad()
{
    size_t fileSizeSrc = srcRows * srcCols * sizeof(T);
    size_t fileSizeDst = dstRows * dstCols * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    void *dstHost, *srcHost;
    void *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&srcHost), fileSizeSrc);
    aclrtMallocHost((void **)(&dstHost), fileSizeDst);

    aclrtMalloc((void **)&dstDevice, fileSizeDst, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, fileSizeSrc, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", fileSizeSrc, srcHost, fileSizeSrc);
    aclrtMemset(dstHost, fileSizeDst, 0, fileSizeDst);

    aclrtMemcpy(dstDevice, fileSizeDst, dstHost, fileSizeDst, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice, fileSizeSrc, srcHost, fileSizeSrc, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTFILLPAD<testKey>((uint8_t *)dstDevice, (uint8_t *)srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSizeDst, dstDevice, fileSizeDst, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSizeDst);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstRows * dstCols);
    std::vector<T> devFinal(dstRows * dstCols);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSizeDst, golden.data(), fileSizeDst);
    ReadFile(GetGoldenDir() + "/output.bin", fileSizeDst, devFinal.data(), fileSizeDst);

    bool ret = ResultCmp(golden, devFinal, 0);

    EXPECT_TRUE(ret);
}

TEST_F(TFILLPADTest, case_float_GT_64_127_VT_64_128_BLK1_PADMAX)
{
    test_tfillpad<float, 64, 127, 64, 128, 1>();
}

TEST_F(TFILLPADTest, case_float_GT_64_127_VT_64_144_BLK1_PADMAX)
{
    test_tfillpad<float, 64, 127, 64, 144, 2>();
}

TEST_F(TFILLPADTest, case_float_GT_64_127_VT_64_160_BLK1_PADMAX)
{
    test_tfillpad<float, 64, 127, 64, 160, 3>();
}

TEST_F(TFILLPADTest, case_float_GT_260_7_VT_260_16_BLK1_PADMAX)
{
    test_tfillpad<float, 260, 7, 260, 16, 4>();
}

TEST_F(TFILLPADTest, case_float_GT_260_7_VT_260_16_BLK1_PADMAX_INPLACE)
{
    test_tfillpad<float, 260, 7, 260, 16, 5>();
}

TEST_F(TFILLPADTest, case_u16_GT_260_7_VT_260_32_BLK1_PADMAX)
{
    test_tfillpad<uint16_t, 260, 7, 260, 32, 6>();
}

TEST_F(TFILLPADTest, case_s8_GT_260_7_VT_260_64_BLK1_PADMAX)
{
    test_tfillpad<int8_t, 260, 7, 260, 64, 7>();
}

TEST_F(TFILLPADTest, case_u16_GT_259_7_VT_260_32_BLK1_PADMAX_EXPAND)
{
    test_tfillpad<uint16_t, 259, 7, 260, 32, 8>();
}

TEST_F(TFILLPADTest, case_s8_GT_259_7_VT_260_64_BLK1_PADMAX_EXPAND)
{
    test_tfillpad<int8_t, 259, 7, 260, 64, 9>();
}

TEST_F(TFILLPADTest, case_s16_GT_260_7_VT_260_32_BLK1_PADMIN)
{
    test_tfillpad<int16_t, 260, 7, 260, 32, 10>();
}

TEST_F(TFILLPADTest, case_s32_GT_260_7_VT_260_32_BLK1_PADMIN)
{
    test_tfillpad<int32_t, 260, 7, 260, 32, 11>();
}
