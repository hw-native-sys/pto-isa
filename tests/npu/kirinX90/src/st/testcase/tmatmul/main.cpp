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
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void LaunchTMATMUL(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int32_t tilingKey>
void LaunchTMATMULBIAS(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

class TMATMULTest : public testing::Test {
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

template <typename T, typename U, typename S, int32_t key>
void tmatmul_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(U);
    size_t bFileSize = K * N * sizeof(S);
    size_t cFileSize = M * N * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    aclrtMemset(dstHost, cFileSize, 0, cFileSize);

    aclrtMemcpy(dstDevice, cFileSize, dstHost, cFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTMATMUL<key>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(cFileSize);
    std::vector<T> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMATMULTest, case_norm_1)
{
    tmatmul_test<aclFloat16, aclFloat16, aclFloat16, 1>(40, 50, 60);
}

TEST_F(TMATMULTest, case_norm_2)
{
    tmatmul_test<int32_t, int8_t, int8_t, 2>(6, 7, 8);
}

TEST_F(TMATMULTest, case_norm_3)
{
    tmatmul_test<aclFloat16, aclFloat16, aclFloat16, 3>(1, 16, 512);
}

TEST_F(TMATMULTest, case_norm_4)
{
    tmatmul_test<int32_t, int8_t, int8_t, 4>(26, 15, 27);
}

TEST_F(TMATMULTest, case_norm_5)
{
    tmatmul_test<int32_t, int8_t, int8_t, 5>(101, 1, 99);
}

TEST_F(TMATMULTest, case_norm_6)
{
    tmatmul_test<aclFloat16, aclFloat16, aclFloat16, 6>(33, 16, 2);
}

TEST_F(TMATMULTest, case_norm_7)
{
    tmatmul_test<aclFloat16, aclFloat16, aclFloat16, 7>(17, 16, 2);
}

TEST_F(TMATMULTest, case_norm_8)
{
    tmatmul_test<int32_t, int8_t, int8_t, 8>(33, 15, 2);
}

template <typename T, typename U, typename S, typename B, int32_t key>
void tmatmul_bias_test(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(U);
    size_t bFileSize = K * N * sizeof(S);
    size_t cFileSize = M * N * sizeof(T);
    size_t biasFileSize = 1 * N * sizeof(B);
    if constexpr (std::is_same_v<T, aclFloat16>) {
        biasFileSize = 2 * N * sizeof(B);
    }

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *src2Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *src2Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);
    aclrtMallocHost((void **)(&src2Host), biasFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src2Device, biasFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/bias_gm.bin", biasFileSize, src2Host, biasFileSize);
    aclrtMemset(dstHost, cFileSize, 0, cFileSize);

    aclrtMemcpy(dstDevice, cFileSize, dstHost, cFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Device, biasFileSize, src2Host, biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTMATMULBIAS<key>(dstDevice, src0Device, src1Device, src2Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(src2Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src2Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(cFileSize);
    std::vector<T> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMATMULTest, case_bias_1)
{
    tmatmul_bias_test<int32_t, int8_t, int8_t, int32_t, 1>(8, 7, 6);
}

TEST_F(TMATMULTest, case_bias_2)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 2>(16, 15, 16);
}

TEST_F(TMATMULTest, case_bias_3)
{
    tmatmul_bias_test<int32_t, int8_t, int8_t, int32_t, 3>(66, 11, 1);
}

TEST_F(TMATMULTest, case_bias_4)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 4>(1, 16, 1);
}

TEST_F(TMATMULTest, case_bias_5)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 5>(29, 11, 41);
}

TEST_F(TMATMULTest, case_bias_6)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 6>(2, 16, 1);
}

TEST_F(TMATMULTest, case_bias_7)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 7>(4, 16, 1);
}

TEST_F(TMATMULTest, case_bias_8)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 8>(8, 16, 1);
}

TEST_F(TMATMULTest, case_bias_9)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 9>(4, 16, 2);
}

TEST_F(TMATMULTest, case_bias_10)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 10>(4, 16, 4);
}

TEST_F(TMATMULTest, case_bias_11)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 11>(4, 16, 8);
}

TEST_F(TMATMULTest, case_bias_12)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 12>(4, 1, 1);
}

TEST_F(TMATMULTest, case_bias_13)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 13>(4, 2, 1);
}

TEST_F(TMATMULTest, case_bias_14)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 14>(4, 4, 1);
}

TEST_F(TMATMULTest, case_bias_15)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 15>(4, 8, 1);
}

TEST_F(TMATMULTest, case_bias_16)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 16>(16, 16, 16);
}

TEST_F(TMATMULTest, case_bias_17)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 17>(2, 16, 3);
}

TEST_F(TMATMULTest, case_bias_18)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 18>(2, 16, 5);
}

TEST_F(TMATMULTest, case_bias_19)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 19>(2, 16, 12);
}

TEST_F(TMATMULTest, case_bias_20)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 20>(2, 16, 32);
}

TEST_F(TMATMULTest, case_bias_21)
{
    tmatmul_bias_test<int32_t, int8_t, int8_t, int32_t, 21>(4, 16, 2);
}

TEST_F(TMATMULTest, case_bias_22)
{
    tmatmul_bias_test<int32_t, int8_t, int8_t, int32_t, 22>(4, 16, 16);
}

TEST_F(TMATMULTest, case_bias_23)
{
    tmatmul_bias_test<int32_t, int8_t, int8_t, int32_t, 23>(4, 16, 32);
}

TEST_F(TMATMULTest, case_bias_24)
{
    tmatmul_bias_test<int32_t, int8_t, int8_t, int32_t, 24>(4, 16, 63);
}

TEST_F(TMATMULTest, case_bias_25)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 25>(2, 16, 33);
}

TEST_F(TMATMULTest, case_bias_26)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 26>(2, 16, 48);
}

TEST_F(TMATMULTest, case_bias_27)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 27>(2, 16, 63);
}

TEST_F(TMATMULTest, case_bias_28)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 28>(2, 16, 64);
}

TEST_F(TMATMULTest, case_bias_29)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 29>(29, 11, 2);
}

TEST_F(TMATMULTest, case_bias_30)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 30>(2, 16, 41);
}

TEST_F(TMATMULTest, case_bias_31)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 31>(17, 16, 2);
}

TEST_F(TMATMULTest, case_bias_32)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 32>(20, 16, 2);
}

TEST_F(TMATMULTest, case_bias_33)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 33>(32, 16, 2);
}

TEST_F(TMATMULTest, case_bias_34)
{
    tmatmul_bias_test<aclFloat16, aclFloat16, aclFloat16, aclFloat16, 34>(33, 16, 2);
}

TEST_F(TMATMULTest, case_bias_35)
{
    tmatmul_bias_test<int32_t, int8_t, int8_t, int32_t, 35>(33, 15, 2);
}
