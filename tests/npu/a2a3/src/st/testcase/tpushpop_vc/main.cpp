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
void LaunchTPushPopVCMatmul(uint8_t *ffts, uint8_t *out, uint8_t *srcA, uint8_t *quantB, uint8_t *scale,
                            uint8_t *offset, uint8_t *fifoMem, void *stream);

class TPushPopVCTest : public testing::Test {
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

template <typename QuantT, typename InT, typename OutT, int32_t key>
void TPushPopVCMatmulTestFunc(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(InT);
    size_t quantBFileSize = K * N * sizeof(QuantT);
    size_t scaleFileSize = K * sizeof(OutT);
    size_t offsetFileSize = K * sizeof(OutT);
    size_t cFileSize = M * N * sizeof(OutT);
    size_t fifoFileSize = 2 * 64 * N * sizeof(OutT);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *srcAHost, *quantBHost, *scaleHost, *offsetHost;
    uint8_t *dstDevice, *srcADevice, *quantBDevice, *scaleDevice, *offsetDevice, *fifoMemDevice;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&srcAHost), aFileSize);
    aclrtMallocHost((void **)(&quantBHost), quantBFileSize);
    aclrtMallocHost((void **)(&scaleHost), scaleFileSize);
    aclrtMallocHost((void **)(&offsetHost), offsetFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&quantBDevice, quantBFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&scaleDevice, scaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&offsetDevice, offsetFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&fifoMemDevice, fifoFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, srcAHost, aFileSize);
    ReadFile(GetGoldenDir() + "/quant_b_gm.bin", quantBFileSize, quantBHost, quantBFileSize);
    ReadFile(GetGoldenDir() + "/scale_gm.bin", scaleFileSize, scaleHost, scaleFileSize);
    ReadFile(GetGoldenDir() + "/offset_gm.bin", offsetFileSize, offsetHost, offsetFileSize);

    aclrtMemcpy(srcADevice, aFileSize, srcAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(quantBDevice, quantBFileSize, quantBHost, quantBFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(scaleDevice, scaleFileSize, scaleHost, scaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(offsetDevice, offsetFileSize, offsetHost, offsetFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&ffts, &fftsLen);

    LaunchTPushPopVCMatmul<key>((uint8_t *)ffts, dstDevice, srcADevice, quantBDevice, scaleDevice, offsetDevice,
                                fifoMemDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcADevice);
    aclrtFree(quantBDevice);
    aclrtFree(scaleDevice);
    aclrtFree(offsetDevice);
    aclrtFree(fifoMemDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcAHost);
    aclrtFreeHost(quantBHost);
    aclrtFreeHost(scaleHost);
    aclrtFreeHost(offsetHost);
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

TEST_F(TPushPopVCTest, case1_int8_single_k_tile)
{
    TPushPopVCMatmulTestFunc<int8_t, float, float, 1>(16, 64, 32);
}

TEST_F(TPushPopVCTest, case2_int8_two_k_tiles)
{
    TPushPopVCMatmulTestFunc<int8_t, float, float, 2>(16, 128, 32);
}

TEST_F(TPushPopVCTest, case3_int8_four_k_tiles)
{
    TPushPopVCMatmulTestFunc<int8_t, float, float, 3>(16, 256, 32);
}

TEST_F(TPushPopVCTest, case4_int16_single_k_tile)
{
    TPushPopVCMatmulTestFunc<int16_t, float, float, 4>(16, 64, 32);
}

TEST_F(TPushPopVCTest, case5_int16_two_k_tiles)
{
    TPushPopVCMatmulTestFunc<int16_t, float, float, 5>(16, 128, 32);
}

TEST_F(TPushPopVCTest, case6_int16_four_k_tiles)
{
    TPushPopVCMatmulTestFunc<int16_t, float, float, 6>(16, 256, 32);
}