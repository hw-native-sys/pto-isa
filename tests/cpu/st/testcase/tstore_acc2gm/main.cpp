/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include "test_common.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <int tilingKey>
void LaunchTStoreAcc2gmNz2nd(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <int tilingKey>
void LaunchTStoreAcc2gmScalarNz2nd(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream, float scalarQuant);

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

TEST_F(TStoreAcc2gmTest, case1)
{
    test_tstore_acc2gm_nz2nd<1, float, float, 128, 128, 16>();
}

TEST_F(TStoreAcc2gmTest, case2)
{
    test_tstore_acc2gm_nz2nd<2, float, float, 31, 32, 15>();
}

TEST_F(TStoreAcc2gmTest, case3)
{
    test_tstore_acc2gm_nz2nd<3, float, uint16_t, 65, 128, 96>();
}

TEST_F(TStoreAcc2gmTest, case4)
{
    test_tstore_acc2gm_nz2nd<4, uint16_t, uint16_t, 73, 64, 32>();
}

TEST_F(TStoreAcc2gmTest, case5)
{
    test_tstore_acc2gm_nz2nd<5, float, uint16_t, 13, 32, 25>();
}

TEST_F(TStoreAcc2gmTest, case6)
{
    test_tstore_acc2gm_nz2nd<6, uint16_t, uint16_t, 100, 222, 60>();
}

TEST_F(TStoreAcc2gmTest, case13)
{
    test_tstore_acc2gm_nz2nd<7, int32_t, int8_t, 44, 128, 27>();
}

TEST_F(TStoreAcc2gmTest, case16)
{
    test_tstore_acc2gm_scalar_nz2nd<1, uint16_t, int8_t, 64, 64, 64>(5);
}

TEST_F(TStoreAcc2gmTest, case17)
{
    test_tstore_acc2gm_scalar_nz2nd<2, int8_t, int8_t, 31, 32, 26>(2);
}

TEST_F(TStoreAcc2gmTest, case18)
{
    test_tstore_acc2gm_scalar_nz2nd<3, uint8_t, int8_t, 16, 32, 17>(2);
}

TEST_F(TStoreAcc2gmTest, case24)
{
    test_tstore_acc2gm_scalar_nz2nd<5, uint8_t, uint16_t, 25, 35, 32>(2);
}

TEST_F(TStoreAcc2gmTest, case25)
{
    test_tstore_acc2gm_scalar_nz2nd<6, uint8_t, float, 16, 20, 25>(1);
}

TEST_F(TStoreAcc2gmTest, case26)
{
    test_tstore_acc2gm_scalar_nz2nd<7, uint16_t, uint16_t, 49, 65, 37>(3);
}

TEST_F(TStoreAcc2gmTest, case27)
{
    test_tstore_acc2gm_scalar_nz2nd<8, uint16_t, uint16_t, 160, 79, 51>(3);
}

TEST_F(TStoreAcc2gmTest, case_relu_1)
{
    test_tstore_acc2gm_nz2nd<21, float, float, 117, 97, 71>();
}

TEST_F(TStoreAcc2gmTest, case_relu_21)
{
    test_tstore_acc2gm_scalar_nz2nd<21, uint8_t, uint16_t, 77, 34, 81>(2);
}