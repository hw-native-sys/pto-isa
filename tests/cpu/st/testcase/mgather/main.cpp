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

class MGATHERTest : public testing::Test {
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

template <int kTileRows, int kTileCols, int kSrcLen>
void LaunchMGather(float *out, float *src, uint32_t *idx, void *stream);

template <int kTileRows, int kTileCols, int kTableRows, bool kExplicitRow>
void LaunchMGatherRow(float *out, float *src, uint32_t *idx, void *stream);

template <int kTileRows, int kTileCols, int kSrcElems, int kIdxCount>
void test_mgather_common(void (*launch)(float *, float *, uint32_t *, void *))
{
    const size_t srcBytes = kSrcElems * sizeof(float);
    const size_t idxBytes = kIdxCount * sizeof(uint32_t);
    const size_t outBytes = kTileRows * kTileCols * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    float *srcHost, *outHost;
    uint32_t *idxHost;
    float *srcDevice, *outDevice;
    uint32_t *idxDevice;

    aclrtMallocHost((void **)(&srcHost), srcBytes);
    aclrtMallocHost((void **)(&outHost), outBytes);
    aclrtMallocHost((void **)(&idxHost), idxBytes);

    aclrtMalloc((void **)&srcDevice, srcBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&outDevice, outBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&idxDevice, idxBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t srcSize = srcBytes;
    size_t idxSize = idxBytes;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", srcSize, srcHost, srcBytes));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input2.bin", idxSize, idxHost, idxBytes));

    aclrtMemcpy(srcDevice, srcBytes, srcHost, srcBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(idxDevice, idxBytes, idxHost, idxBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    launch(outDevice, srcDevice, idxDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(outHost, outBytes, outDevice, outBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", outHost, outBytes);

    aclrtFree(srcDevice);
    aclrtFree(outDevice);
    aclrtFree(idxDevice);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(outHost);
    aclrtFreeHost(idxHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(outBytes / sizeof(float));
    std::vector<float> devFinal(outBytes / sizeof(float));
    size_t outSize = outBytes;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", outSize, golden.data(), outBytes));
    outSize = outBytes;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", outSize, devFinal.data(), outBytes));
    EXPECT_TRUE(ResultCmp<float>(golden, devFinal, 0.001f));
}

TEST_F(MGATHERTest, case_float_16x16_src512)
{
    test_mgather_common<16, 16, 512, 16 * 16>(LaunchMGather<16, 16, 512>);
}

TEST_F(MGATHERTest, case_float_row_default_table32x16_dst16x16)
{
    test_mgather_common<16, 16, 32 * 16, 16>(LaunchMGatherRow<16, 16, 32, false>);
}

TEST_F(MGATHERTest, case_float_row_explicit_table32x16_dst16x16)
{
    test_mgather_common<16, 16, 32 * 16, 16>(LaunchMGatherRow<16, 16, 32, true>);
}
