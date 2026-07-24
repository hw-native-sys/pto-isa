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

template <int32_t tilingKey>
void LaunchTMATMUL(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

template <int32_t tilingKey>
void LaunchTMATMULBIAS(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, void* stream);

class TMATMULTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

class TMATMULBIASTest : public testing::Test {
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

template <typename T>
constexpr T CeilAlign(T num_1, T num_2)
{
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

template <typename T, typename U, typename S, int32_t key>
void TmatmulTest(uint32_t M, uint32_t K, uint32_t N)
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

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);

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

    bool ret = ResultCmp(golden, devFinal, 0.0001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMATMULTest, case1)
{
    uint32_t M = 31;
    uint32_t N = 58;
    uint32_t K = 120;

    TmatmulTest<float, uint16_t, uint16_t, 1>(M, K, N);
}

TEST_F(TMATMULTest, case2)
{
    uint32_t M = 65;
    uint32_t N = 89;
    uint32_t K = 90;

    TmatmulTest<int32_t, int8_t, int8_t, 2>(M, K, N);
}

TEST_F(TMATMULTest, case3)
{
    uint32_t M = 5;
    uint32_t N = 11;
    uint32_t K = 75;

    TmatmulTest<float, uint16_t, uint16_t, 3>(M, K, N);
}

TEST_F(TMATMULTest, case4)
{
    uint32_t M = 1;
    uint32_t N = 64;
    uint32_t K = 256;

    TmatmulTest<float, uint16_t, uint16_t, 4>(M, K, N);
}

TEST_F(TMATMULTest, case5)
{
    uint32_t M = 1;
    uint32_t N = 32;
    uint32_t K = 16;

    TmatmulTest<float, uint16_t, uint16_t, 5>(M, K, N);
}

TEST_F(TMATMULTest, case6)
{
    uint32_t M = 1;
    uint32_t N = 32;
    uint32_t K = 200;

    TmatmulTest<float, uint16_t, uint16_t, 6>(M, K, N);
}

TEST_F(TMATMULTest, case7) { TmatmulTest<float, float, float, 7>(16, 32, 64); }

TEST_F(TMATMULTest, case8) { TmatmulTest<float, float, float, 8>(5, 75, 11); }

template <typename T, typename U, typename S, typename biasType, int32_t key>
void TmatmulBiasTest(uint32_t M, uint32_t K, uint32_t N)
{
    size_t aFileSize = M * K * sizeof(U);
    size_t bFileSize = K * N * sizeof(S);
    size_t cFileSize = M * N * sizeof(T);
    size_t biasFileSize = 1 * N * sizeof(biasType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *src2Host, *goldenHost;
    uint8_t *dstDevice, *src0Device, *src1Device, *src2Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);
    aclrtMallocHost((void**)(&src2Host), biasFileSize);
    aclrtMallocHost((void**)(&goldenHost), cFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src2Device, biasFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/bias_gm.bin", biasFileSize, src2Host, biasFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, goldenHost, cFileSize);

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

TEST_F(TMATMULBIASTest, case1)
{
    uint32_t M = 26;
    uint32_t K = 100;
    uint32_t N = 94;

    TmatmulBiasTest<float, uint16_t, uint16_t, float, 1>(M, K, N);
}

TEST_F(TMATMULBIASTest, case2)
{
    uint32_t M = 101;
    uint32_t K = 288;
    uint32_t N = 67;

    TmatmulBiasTest<float, uint16_t, uint16_t, float, 2>(M, K, N);
}

TEST_F(TMATMULBIASTest, case3)
{
    uint32_t M = 15;
    uint32_t K = 16;
    uint32_t N = 15;

    TmatmulBiasTest<float, float, float, float, 3>(M, K, N);
}

TEST_F(TMATMULBIASTest, case4)
{
    uint32_t M = 55;
    uint32_t K = 127;
    uint32_t N = 29;

    TmatmulBiasTest<int32_t, int8_t, int8_t, int32_t, 4>(M, K, N);
}

TEST_F(TMATMULBIASTest, case5)
{
    uint32_t M = 11;
    uint32_t K = 402;
    uint32_t N = 30;

    TmatmulBiasTest<float, uint16_t, uint16_t, float, 5>(M, K, N);
}

TEST_F(TMATMULBIASTest, case6)
{
    uint32_t M = 150;
    uint32_t K = 89;
    uint32_t N = 50;

    TmatmulBiasTest<int32_t, int8_t, int8_t, int32_t, 6>(M, K, N);
}

TEST_F(TMATMULBIASTest, case7)
{
    uint32_t M = 135;
    uint32_t K = 64;
    uint32_t N = 88;

    TmatmulBiasTest<int32_t, int8_t, int8_t, int32_t, 7>(M, K, N);
}

TEST_F(TMATMULBIASTest, case8) { TmatmulBiasTest<float, uint16_t, uint16_t, float, 8>(1, 512, 32); }
