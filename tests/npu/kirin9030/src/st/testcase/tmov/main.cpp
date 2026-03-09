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
void launchTMovL12Bias(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template <int32_t tilingKey>
void launchTMovL12Fb(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *src2, void *stream);

template <int32_t tilingKey>
void launchTMovAcc2Vec(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

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

template <typename AType, typename BType, typename CType, typename BiasType, int32_t tilingKey>
void tMovL12Bias(uint32_t m, uint32_t n, uint32_t k)
{
    size_t aFileSize = m * k * sizeof(AType);
    size_t bFileSize = k * n * sizeof(BType);
    size_t cFileSize = m * n * sizeof(CType);
    size_t biasFileSize = n * sizeof(BiasType);

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
    launchTMovL12Bias<tilingKey>(dstDevice, src0Device, src1Device, src2Device, stream);

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

    std::vector<CType> golden(cFileSize);
    std::vector<CType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename AType, typename BType, typename CType, typename QuantType, int32_t tilingKey>
void tMovL12Fb(uint32_t m, uint32_t n, uint32_t k)
{
    size_t aFileSize = m * k * sizeof(AType);
    size_t bFileSize = k * n * sizeof(BType);
    size_t cFileSize = m * n * sizeof(CType);
    size_t quantFileSize = n * sizeof(QuantType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *src2Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *src2Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);
    aclrtMallocHost((void **)(&src2Host), quantFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src2Device, quantFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/quant_gm.bin", quantFileSize, src2Host, quantFileSize);
    aclrtMemset(dstHost, cFileSize, 0, cFileSize);

    aclrtMemcpy(dstDevice, cFileSize, dstHost, cFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Device, quantFileSize, src2Host, quantFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMovL12Fb<tilingKey>(dstDevice, src0Device, src1Device, src2Device, stream);

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

    std::vector<CType> golden(cFileSize);
    std::vector<CType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/quant_golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename AType, typename BType, typename CType, int32_t tilingKey>
void tMovAcc2Vec(uint32_t m, uint32_t n, uint32_t k)
{
    size_t aFileSize = m * k * sizeof(AType);
    size_t bFileSize = k * n * sizeof(BType);
    size_t cFileSize = m * n * sizeof(CType);

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
    launchTMovAcc2Vec<tilingKey>(dstDevice, src0Device, src1Device, stream);

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

    std::vector<CType> golden(cFileSize);
    std::vector<CType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TMOVTest, case_bias4)
{
    constexpr uint32_t M = 128;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 96;
    tMovL12Bias<int8_t, int8_t, int32_t, int32_t, 4>(M, N, K);
}

TEST_F(TMOVTest, case_bias5)
{
    constexpr uint32_t M = 31;
    constexpr uint32_t N = 63;
    constexpr uint32_t K = 32;
    tMovL12Bias<int8_t, int8_t, int32_t, int32_t, 5>(M, N, K);
}

TEST_F(TMOVTest, case_fixpipe1)
{
    constexpr uint32_t M = 32;
    constexpr uint32_t N = 128;
    constexpr uint32_t K = 32;
    tMovL12Fb<int8_t, int8_t, int8_t, uint64_t, 1>(M, N, K);
}

TEST_F(TMOVTest, case_fixpipe2)
{
    constexpr uint32_t M = 96;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 32;
    tMovL12Fb<int8_t, int8_t, uint16_t, uint64_t, 2>(M, N, K);
}

TEST_F(TMOVTest, case_acc2vec_Nz2Nd)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 64;
    tMovAcc2Vec<aclFloat16, aclFloat16, aclFloat16, 1>(M, N, K);
}

TEST_F(TMOVTest, case_acc2vec_Nz2Nz)
{
    constexpr uint32_t M = 64;
    constexpr uint32_t N = 64;
    constexpr uint32_t K = 64;
    tMovAcc2Vec<aclFloat16, aclFloat16, aclFloat16, 2>(M, N, K);
}
