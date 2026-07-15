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

template <
    typename AT, typename BT, typename L0CT, typename BiasT, typename GMT, typename ScalingT, int M, int N, int K,
    int IsTransA, int IsTransB, int IsBias, int IsQuant, int ReluMode = 0, int Isdynamic = 0, int IsNd = 1>
void LaunchTMOV(GMT* out, AT* src0, BT* src1, BiasT* src2, ScalingT* src3, void* stream);

class TMOVTest : public testing::Test {
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

template <
    typename AT, typename BT, typename L0CT, typename BiasT, typename GMT, typename ScalingT, int M, int N, int K,
    int IsTransA, int IsTransB, int IsBias, int IsQuant, int ReluMode = 0, int Isdynamic = 0, int IsNd = 1>
void test_tmov()
{
    // The bias addr needs to be 64B aligned.
    uint32_t alignBiasN = (N * sizeof(BiasT) + 63) / 64 * 64 / sizeof(BiasT);
    // The Scaling addr needs to be 128B aligned.
    uint32_t alignFbN = (N * sizeof(ScalingT) + 127) / 128 * 128 / sizeof(ScalingT);
    size_t aFileSize = M * K * sizeof(AT); // uint16_t represent half
    size_t bFileSize = K * N * sizeof(BT); // uint16_t represent half
    size_t cFileSize = M * N * sizeof(GMT);
    size_t biasFileSize = alignBiasN * sizeof(BiasT);
    size_t fbFileSize = alignFbN * sizeof(ScalingT);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    GMT* dstHost;
    AT *src0Host, *src1Host;
    BiasT* src2Host;
    ScalingT* src3Host;
    GMT* dstDevice;
    AT *src0Device, *src1Device;
    BiasT* src2Device;
    ScalingT* src3Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);
    aclrtMallocHost((void**)(&src2Host), biasFileSize);
    aclrtMallocHost((void**)(&src3Host), fbFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src2Device, biasFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src3Device, fbFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/bias_gm.bin", biasFileSize, src2Host, biasFileSize);
    ReadFile(GetGoldenDir() + "/scaling_gm.bin", fbFileSize, src3Host, fbFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Device, biasFileSize, src2Host, biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src3Device, fbFileSize, src3Host, fbFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTMOV<
        AT, BT, L0CT, BiasT, GMT, ScalingT, M, N, K, IsTransA, IsTransB, IsBias, IsQuant, ReluMode, Isdynamic, IsNd>(
        dstDevice, src0Device, src1Device, src2Device, src3Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(src2Device);
    aclrtFree(src3Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src2Host);
    aclrtFreeHost(src3Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    std::vector<float> golden(cFileSize);
    std::vector<float> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

// template <typename AT, typename BT, typename L0CT, typename BiasT, typename GMT, typename ScalingT, int M, int N, int
// K, int IsTransA, int IsTransB, int IsBias, int IsQuant, int ReluMode=0, int Isdynamic=0, int IsNd=1>
TEST_F(TMOVTest, case1_bias_static_half_float_0_1_1_0_0_param)
{
    test_tmov<uint16_t, uint16_t, float, float, float, uint64_t, 64, 32, 80, 0, 1, 1, 0, 0, 0>();
}

TEST_F(TMOVTest, case2_bias_static_int8_int32_0_1_1_0_0_param)
{
    test_tmov<int8_t, int8_t, int32_t, int32_t, int32_t, uint64_t, 128, 64, 128, 0, 1, 1, 0, 0, 0>();
}

TEST_F(TMOVTest, case3_bias_static_float_float_0_1_1_0_0_param)
{
    test_tmov<float, float, float, float, float, uint64_t, 128, 48, 64, 0, 1, 1, 0, 0, 0>();
}

TEST_F(TMOVTest, case4_bias_dynamic_half_half_0_1_1_0_0_param)
{
    test_tmov<uint16_t, uint16_t, float, uint16_t, float, uint64_t, 64, 32, 80, 0, 1, 1, 0, 0, 1>();
}

TEST_F(TMOVTest, case5_bias_dynamic_float_half_0_1_1_0_0_param)
{
    test_tmov<float, float, float, uint16_t, float, uint64_t, 112, 48, 96, 0, 1, 1, 0, 0, 1>();
}

TEST_F(TMOVTest, case6_bias_static_float_half_0_1_1_0_0_param)
{
    test_tmov<float, float, float, uint16_t, float, uint64_t, 64, 128, 96, 0, 1, 1, 0, 0, 0>();
}

TEST_F(TMOVTest, case11_scaling_static_int32_int8_0_1_0_1_0_param)
{
    test_tmov<int8_t, int8_t, int32_t, int32_t, int8_t, uint64_t, 128, 112, 32, 0, 1, 0, 1, 0, 0>();
}

TEST_F(TMOVTest, case12_scaling_static_int32_half_0_1_0_1_0_param)
{
    test_tmov<int8_t, int8_t, int32_t, int32_t, uint16_t, uint64_t, 144, 80, 160, 0, 1, 0, 1, 0, 0>();
}

TEST_F(TMOVTest, case13_scaling_static_float_int8_0_1_0_1_0_param)
{
    test_tmov<uint16_t, uint16_t, float, float, int8_t, uint64_t, 64, 32, 80, 0, 1, 0, 1, 0, 0>();
}

TEST_F(TMOVTest, case14_scaling_dynamic_int32_int8_0_1_1_1_0_param)
{
    test_tmov<int8_t, int8_t, int32_t, int32_t, int8_t, uint64_t, 60, 17, 80, 0, 1, 1, 1, 0, 1>();
}

TEST_F(TMOVTest, case15_scaling_dynamic_int32_int8_0_1_1_1_0_param)
{
    test_tmov<int8_t, int8_t, int32_t, int32_t, int8_t, uint64_t, 15, 10, 30, 0, 1, 1, 1, 0, 1>();
}
