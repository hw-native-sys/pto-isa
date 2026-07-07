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

template <int tilingKey>
void LaunchTStoreAcc2gmNz2nd(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int tilingKey>
void LaunchTStoreAcc2gmNz2nz(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int tilingKey>
void LaunchTStoreAcc2gmScalarNz2nd(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream, float scalarQuant);

template <int tilingKey>
void LaunchTStoreAcc2gmScalarNz2nz(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream, float scalarQuant);

template <int tilingKey>
void LaunchTStoreAcc2gmVectorNz2nd(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor, void *stream);

template <int tilingKey>
void LaunchTStoreAcc2gmVectorNz2nz(uint8_t *out, uint8_t *src0, uint8_t *src1, uint8_t *quantTensor, void *stream);

class TStoreAcc2gmTest : public testing::Test {
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

template <int tilingKey, typename dstDataType, typename srcDataType, int validM, int validN, int validK>
void test_tstore_acc2gm_nz2nd()
{
    size_t aFileSize = validM * validK * sizeof(srcDataType);
    size_t bFileSize = validK * validN * sizeof(srcDataType);
    size_t cFileSize = validM * validN * sizeof(dstDataType);

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
    aclrtMemset(dstDevice, cFileSize, 0, cFileSize);
    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTStoreAcc2gmNz2nd<tilingKey>(dstDevice, src0Device, src1Device, stream);

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

    std::vector<dstDataType> golden(cFileSize);
    std::vector<dstDataType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp<dstDataType>(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

template <int tilingKey, typename dstDataType, typename srcDataType, int validM, int validN, int validK>
void test_tstore_acc2gm_nz2nz()
{
    size_t aFileSize = validM * validK * sizeof(srcDataType);
    size_t bFileSize = validK * validN * sizeof(srcDataType);
    size_t cFileSize = validM * validN * sizeof(dstDataType);

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
    aclrtMemset(dstDevice, cFileSize, 0, cFileSize);
    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTStoreAcc2gmNz2nz<tilingKey>(dstDevice, src0Device, src1Device, stream);
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

    std::vector<dstDataType> golden(cFileSize);
    std::vector<dstDataType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp<dstDataType>(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

template <int tilingKey, typename dstDataType, typename srcDataType, int validM, int validN, int validK>
void test_tstore_acc2gm_scalar_nz2nd(float scalarQuant)
{
    size_t aFileSize = validM * validK * sizeof(srcDataType);
    size_t bFileSize = validK * validN * sizeof(srcDataType);
    size_t cFileSize = validM * validN * sizeof(dstDataType);

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
    aclrtMemset(dstDevice, cFileSize, 0, cFileSize);
    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTStoreAcc2gmScalarNz2nd<tilingKey>(dstDevice, src0Device, src1Device, stream, scalarQuant);
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

    std::vector<dstDataType> golden(cFileSize);
    std::vector<dstDataType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp<dstDataType>(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

template <int tilingKey, typename dstDataType, typename srcDataType, int validM, int validN, int validK>
void test_tstore_acc2gm_scalar_nz2nz(float scalarQuant)
{
    size_t aFileSize = validM * validK * sizeof(srcDataType);
    size_t bFileSize = validK * validN * sizeof(srcDataType);
    size_t cFileSize = validM * validN * sizeof(dstDataType);

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
    aclrtMemset(dstDevice, cFileSize, 0, cFileSize);
    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTStoreAcc2gmScalarNz2nz<tilingKey>(dstDevice, src0Device, src1Device, stream, scalarQuant);
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

    std::vector<dstDataType> golden(cFileSize);
    std::vector<dstDataType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp<dstDataType>(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

template <int tilingKey, typename dstDataType, typename srcDataType, int validM, int validN, int validK>
void test_tstore_acc2gm_vector_nz2nd()
{
    using ScalingT = uint64_t;
    constexpr int alignFbN = (validN * sizeof(ScalingT) + 127) / 128 * 128 / sizeof(ScalingT);
    size_t aFileSize = validM * validK * sizeof(srcDataType);
    size_t bFileSize = validK * validN * sizeof(srcDataType);
    size_t cFileSize = validM * validN * sizeof(dstDataType);
    size_t fbFileSize = alignFbN * sizeof(ScalingT);

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost;
    uint8_t *src0Host;
    uint8_t *src1Host;
    uint8_t *quantTensorHost;
    uint8_t *dstDevice;
    uint8_t *src0Device;
    uint8_t *src1Device;
    uint8_t *quantTensorDevice;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);
    aclrtMallocHost((void **)(&quantTensorHost), fbFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&quantTensorDevice, fbFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/quant_vector_gm.bin", fbFileSize, quantTensorHost, fbFileSize);
    aclrtMemset(dstDevice, cFileSize, 0, cFileSize);
    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(quantTensorDevice, fbFileSize, quantTensorHost, fbFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTStoreAcc2gmVectorNz2nd<tilingKey>(dstDevice, src0Device, src1Device, quantTensorDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(quantTensorDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(quantTensorHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<dstDataType> golden(cFileSize);
    std::vector<dstDataType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp<dstDataType>(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

template <int tilingKey, typename dstDataType, typename srcDataType, int validM, int validN, int validK>
void test_tstore_acc2gm_vector_nz2nz()
{
    using ScalingT = uint64_t;
    constexpr int alignFbN = (validN * sizeof(ScalingT) + 127) / 128 * 128 / sizeof(ScalingT);
    size_t aFileSize = validM * validK * sizeof(srcDataType);
    size_t bFileSize = validK * validN * sizeof(srcDataType);
    size_t cFileSize = validM * validN * sizeof(dstDataType);
    size_t fbFileSize = alignFbN * sizeof(ScalingT);

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost;
    uint8_t *src0Host;
    uint8_t *src1Host;
    uint8_t *quantTensorHost;
    uint8_t *dstDevice;
    uint8_t *src0Device;
    uint8_t *src1Device;
    uint8_t *quantTensorDevice;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);
    aclrtMallocHost((void **)(&quantTensorHost), fbFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&quantTensorDevice, fbFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile(GetGoldenDir() + "/quant_vector_gm.bin", fbFileSize, quantTensorHost, fbFileSize);
    aclrtMemset(dstDevice, cFileSize, 0, cFileSize);
    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(quantTensorDevice, fbFileSize, quantTensorHost, fbFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTStoreAcc2gmVectorNz2nz<tilingKey>(dstDevice, src0Device, src1Device, quantTensorDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(quantTensorDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(quantTensorHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<dstDataType> golden(cFileSize);
    std::vector<dstDataType> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp<dstDataType>(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TStoreAcc2gmTest, case1)
{
    test_tstore_acc2gm_nz2nd<1, uint16_t, uint16_t, 128, 128, 16>();
}

TEST_F(TStoreAcc2gmTest, case2)
{
    test_tstore_acc2gm_nz2nd<2, uint16_t, uint16_t, 31, 32, 15>();
}

TEST_F(TStoreAcc2gmTest, case3)
{
    test_tstore_acc2gm_nz2nd<3, uint16_t, uint16_t, 65, 128, 96>();
}

TEST_F(TStoreAcc2gmTest, case4)
{
    test_tstore_acc2gm_nz2nd<4, uint16_t, uint16_t, 73, 64, 32>();
}

TEST_F(TStoreAcc2gmTest, case5)
{
    test_tstore_acc2gm_nz2nz<1, uint16_t, uint16_t, 32, 64, 25>();
}

TEST_F(TStoreAcc2gmTest, case6)
{
    test_tstore_acc2gm_nz2nz<2, uint16_t, uint16_t, 48, 32, 45>();
}

TEST_F(TStoreAcc2gmTest, case7)
{
    test_tstore_acc2gm_nz2nz<3, uint16_t, uint16_t, 32, 64, 24>();
}

TEST_F(TStoreAcc2gmTest, case8)
{
    test_tstore_acc2gm_nz2nz<4, uint16_t, uint16_t, 96, 96, 23>();
}

TEST_F(TStoreAcc2gmTest, case9)
{
    test_tstore_acc2gm_nz2nd<7, int32_t, int8_t, 44, 128, 27>();
}

TEST_F(TStoreAcc2gmTest, case10)
{
    test_tstore_acc2gm_nz2nz<7, int32_t, int8_t, 64, 96, 30>();
}

TEST_F(TStoreAcc2gmTest, case11)
{
    test_tstore_acc2gm_scalar_nz2nd<1, uint16_t, int8_t, 64, 64, 64>(5);
}

TEST_F(TStoreAcc2gmTest, case12)
{
    test_tstore_acc2gm_scalar_nz2nd<2, int8_t, int8_t, 31, 32, 26>(2);
}

TEST_F(TStoreAcc2gmTest, case13)
{
    test_tstore_acc2gm_scalar_nz2nd<3, uint8_t, int8_t, 16, 32, 17>(2);
}

TEST_F(TStoreAcc2gmTest, case14)
{
    test_tstore_acc2gm_scalar_nz2nz<1, uint16_t, int8_t, 64, 32, 64>(5);
}

TEST_F(TStoreAcc2gmTest, case15)
{
    test_tstore_acc2gm_scalar_nz2nz<2, int8_t, int8_t, 32, 32, 32>(2);
}

TEST_F(TStoreAcc2gmTest, case16)
{
    test_tstore_acc2gm_scalar_nz2nz<3, uint8_t, int8_t, 160, 64, 17>(2);
}

TEST_F(TStoreAcc2gmTest, case17)
{
    test_tstore_acc2gm_scalar_nz2nd<5, int8_t, uint16_t, 25, 35, 32>(2);
}

TEST_F(TStoreAcc2gmTest, case18)
{
    test_tstore_acc2gm_scalar_nz2nd<6, uint8_t, uint16_t, 16, 20, 25>(1);
}

TEST_F(TStoreAcc2gmTest, case20)
{
    test_tstore_acc2gm_scalar_nz2nz<5, int8_t, uint16_t, 16, 64, 15>(2);
}

TEST_F(TStoreAcc2gmTest, case22)
{
    test_tstore_acc2gm_vector_nz2nd<1, uint16_t, int8_t, 55, 88, 32>();
}

TEST_F(TStoreAcc2gmTest, case23)
{
    test_tstore_acc2gm_vector_nz2nd<2, int8_t, int8_t, 34, 85, 19>();
}

TEST_F(TStoreAcc2gmTest, case24)
{
    test_tstore_acc2gm_vector_nz2nd<3, uint8_t, int8_t, 31, 32, 29>();
}

TEST_F(TStoreAcc2gmTest, case26)
{
    test_tstore_acc2gm_vector_nz2nd<7, int8_t, uint16_t, 33, 65, 25>();
}

TEST_F(TStoreAcc2gmTest, case27)
{
    test_tstore_acc2gm_vector_nz2nd<8, uint8_t, uint16_t, 19, 32, 23>();
}

TEST_F(TStoreAcc2gmTest, case28)
{
    test_tstore_acc2gm_vector_nz2nz<1, uint16_t, int8_t, 64, 128, 64>();
}

TEST_F(TStoreAcc2gmTest, case29)
{
    test_tstore_acc2gm_vector_nz2nz<2, int8_t, int8_t, 32, 32, 31>();
}

TEST_F(TStoreAcc2gmTest, case30)
{
    test_tstore_acc2gm_vector_nz2nz<3, uint8_t, int8_t, 48, 32, 23>();
}

TEST_F(TStoreAcc2gmTest, case32)
{
    test_tstore_acc2gm_vector_nz2nz<7, int8_t, uint16_t, 32, 64, 25>();
}

TEST_F(TStoreAcc2gmTest, case33)
{
    test_tstore_acc2gm_vector_nz2nz<8, uint8_t, uint16_t, 16, 32, 23>();
}

TEST_F(TStoreAcc2gmTest, case_relu_1)
{
    test_tstore_acc2gm_nz2nd<21, uint16_t, uint16_t, 117, 97, 71>();
}

TEST_F(TStoreAcc2gmTest, case_relu_11)
{
    test_tstore_acc2gm_nz2nz<21, uint16_t, uint16_t, 160, 80, 51>();
}

TEST_F(TStoreAcc2gmTest, case_relu_21)
{
    test_tstore_acc2gm_scalar_nz2nd<21, int8_t, uint16_t, 77, 34, 81>(2);
}

TEST_F(TStoreAcc2gmTest, case_relu_31)
{
    test_tstore_acc2gm_scalar_nz2nz<21, int8_t, int8_t, 96, 32, 159>(2);
}

TEST_F(TStoreAcc2gmTest, case_relu_41)
{
    test_tstore_acc2gm_vector_nz2nd<21, int8_t, uint16_t, 85, 77, 66>();
}

TEST_F(TStoreAcc2gmTest, case_relu_51)
{
    test_tstore_acc2gm_vector_nz2nz<21, int8_t, int8_t, 128, 128, 123>();
}
