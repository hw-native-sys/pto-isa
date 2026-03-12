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
#include "acl/acl.h"
#include "runtime/rt.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void LaunchTPushPopMatmulAdd(uint8_t *ffts, uint8_t *out, uint8_t *srcA, uint8_t *srcB, uint8_t *bias, uint8_t *fifoMem,
                             void *stream);

class TPushPopCVTest : public testing::Test {
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

template <typename InT, typename OutT, int32_t key>
void TPushPopMatmulAddTestFunc(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(InT);
    size_t bFileSize = K * N * sizeof(InT);
    size_t biasFileSize = M * N * sizeof(OutT);
    size_t cFileSize = M * N * sizeof(OutT);
    size_t fifoFileSize = 2 * 16 * 32 * sizeof(OutT);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *srcAHost, *srcBHost, *biasHost;
    uint8_t *dstDevice, *srcADevice, *srcBDevice, *biasDevice, *fifoMemDevice;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&srcAHost), aFileSize);
    aclrtMallocHost((void **)(&srcBHost), bFileSize);
    aclrtMallocHost((void **)(&biasHost), biasFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&biasDevice, biasFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&fifoMemDevice, fifoFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, srcAHost, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, srcBHost, bFileSize);
    ReadFile(GetGoldenDir() + "/bias_gm.bin", biasFileSize, biasHost, biasFileSize);

    aclrtMemcpy(srcADevice, aFileSize, srcAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcBDevice, bFileSize, srcBHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(biasDevice, biasFileSize, biasHost, biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&ffts, &fftsLen);

    LaunchTPushPopMatmulAdd<key>((uint8_t *)ffts, dstDevice, srcADevice, srcBDevice, biasDevice, fifoMemDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcADevice);
    aclrtFree(srcBDevice);
    aclrtFree(biasDevice);
    aclrtFree(fifoMemDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcAHost);
    aclrtFreeHost(srcBHost);
    aclrtFreeHost(biasHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<OutT> golden(cFileSize);
    std::vector<OutT> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TPushPopCVTest, case1_half_single_tile)
{
    TPushPopMatmulAddTestFunc<aclFloat16, float, 1>(16, 32, 32);
}

TEST_F(TPushPopCVTest, case2_half_split_m)
{
    TPushPopMatmulAddTestFunc<aclFloat16, float, 2>(32, 32, 32);
}

TEST_F(TPushPopCVTest, case4_half_multi_tile_wrapping)
{
    TPushPopMatmulAddTestFunc<aclFloat16, float, 4>(64, 32, 32);
}

TEST_F(TPushPopCVTest, case3_float_single_tile)
{
    TPushPopMatmulAddTestFunc<float, float, 3>(16, 32, 32);
}