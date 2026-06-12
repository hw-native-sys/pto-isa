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

class MSCATTERTest : public testing::Test {
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
    return "../" + suiteName + "." + caseName;
}

template <int kTileRows, int kTileCols, int kDstLen>
void LaunchMScatter(float *out, float *srcTile, uint32_t *idx, void *stream);

template <int kTileRows, int kTileCols, int kTableRows, bool kExplicitRow>
void LaunchMScatterRow(float *out, float *srcTile, uint32_t *idx, void *stream);

template <int kTileRows, int kTileCols, int kDstLen, int kIdxCount>
void test_mscatter_common(void (*launch)(float *, float *, uint32_t *, void *))
{
    const size_t dstBytes = kDstLen * sizeof(float);
    const size_t srcBytes = kTileRows * kTileCols * sizeof(float);
    const size_t idxBytes = kIdxCount * sizeof(uint32_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    float *dstHost, *srcHost;
    uint32_t *idxHost;
    float *dstDevice, *srcDevice;
    uint32_t *idxDevice;

    aclrtMallocHost((void **)(&dstHost), dstBytes);
    aclrtMallocHost((void **)(&srcHost), srcBytes);
    aclrtMallocHost((void **)(&idxHost), idxBytes);

    aclrtMalloc((void **)&dstDevice, dstBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&idxDevice, idxBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t dstSize = dstBytes;
    size_t srcSize = srcBytes;
    size_t idxSize = idxBytes;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", dstSize, dstHost, dstBytes));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input2.bin", srcSize, srcHost, srcBytes));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input3.bin", idxSize, idxHost, idxBytes));

    aclrtMemcpy(dstDevice, dstBytes, dstHost, dstBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice, srcBytes, srcHost, srcBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(idxDevice, idxBytes, idxHost, idxBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    launch(dstDevice, srcDevice, idxDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstBytes, dstDevice, dstBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstBytes);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(idxDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(idxHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(dstBytes / sizeof(float));
    std::vector<float> devFinal(dstBytes / sizeof(float));
    dstSize = dstBytes;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", dstSize, golden.data(), dstBytes));
    dstSize = dstBytes;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", dstSize, devFinal.data(), dstBytes));
    EXPECT_TRUE(ResultCmp<float>(golden, devFinal, 0.001f));
}

TEST_F(MSCATTERTest, case_float_dst512_src16x16)
{
    test_mscatter_common<16, 16, 512, 16 * 16>(LaunchMScatter<16, 16, 512>);
}

TEST_F(MSCATTERTest, case_float_row_default_table32x16_src16x16)
{
    test_mscatter_common<16, 16, 32 * 16, 16>(LaunchMScatterRow<16, 16, 32, false>);
}

TEST_F(MSCATTERTest, case_float_row_explicit_table32x16_src16x16)
{
    test_mscatter_common<16, 16, 32 * 16, 16>(LaunchMScatterRow<16, 16, 32, true>);
}
