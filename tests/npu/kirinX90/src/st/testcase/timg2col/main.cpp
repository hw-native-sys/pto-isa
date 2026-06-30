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
void launchTIMG2COL(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

class TIMG2COLTest : public testing::Test {
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

template <int32_t key, typename T, typename U>
void timg2col_test(uint32_t FMN, uint32_t FMC1, uint32_t FMH, uint32_t FMW, uint32_t FMC0, uint32_t FTC1, uint32_t FTH,
                   uint32_t FTW, uint32_t FTN, uint32_t FTC0, uint8_t dilationH = 1, uint8_t dilationW = 1,
                   uint8_t strideH = 1, uint8_t strideW = 1, uint8_t padTop = 1, uint8_t padBottom = 1,
                   uint8_t padLeft = 1, uint8_t padRight = 1)
{
    uint32_t heightOut = 0;
    uint32_t widthOut = 0;
    if (strideH != 0 && strideW != 0) {
        heightOut = (FMH + padTop + padBottom - dilationH * (FTH - 1) - 1) / strideH + 1;
        widthOut = (FMW + padLeft + padRight - dilationW * (FTW - 1) - 1) / strideW + 1;
    }
    uint32_t M = FMN * heightOut * widthOut;
    uint32_t N = FTN;
    size_t aFileSize = FMN * FMC1 * FMH * FMW * FMC0 * sizeof(U);
    size_t bFileSize = FTC1 * FTH * FTW * FTN * FTC0 * sizeof(U);
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

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTIMG2COL<key>(dstDevice, src0Device, src1Device, stream);

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
    if ((key == 4) || (key == 8)) {
        ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    } else {
        ReadFile(GetGoldenDir() + "/golden_NC1HWC0.bin", cFileSize, golden.data(), cFileSize);
    }
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}
template <int32_t key, typename T, typename U>
void timg2col_test_fractal4d(uint32_t FMN, uint32_t FMC1, uint32_t FMH, uint32_t FMW, uint32_t FMC0, uint32_t FTDIM3,
                             uint32_t FTDIM2, uint32_t FTDIM1, uint32_t FTDIM0, uint32_t FTH, uint32_t FTW,
                             uint8_t dilationH = 1, uint8_t dilationW = 1, uint8_t strideH = 1, uint8_t strideW = 1,
                             uint8_t padTop = 1, uint8_t padBottom = 1, uint8_t padLeft = 1, uint8_t padRight = 1)
{
    uint32_t widthOut = 0;
    uint32_t heightOut = 0;
    if (strideH != 0 && strideW != 0) {
        heightOut = (FMH + padTop + padBottom - dilationH * (FTH - 1) - 1) / strideH + 1;
        widthOut = (FMW + padLeft + padRight - dilationW * (FTW - 1) - 1) / strideW + 1;
    }
    uint32_t M = FMN * heightOut * widthOut;
    uint32_t N = FTDIM2 * FTDIM1;
    size_t aFileSize = FMN * FMC1 * FMH * FMW * FMC0 * sizeof(U);
    size_t bFileSize = FTDIM3 * FTDIM2 * FTDIM1 * FTDIM0 * sizeof(U);
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

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTIMG2COL<key>(dstDevice, src0Device, src1Device, stream);

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
    if (key == 12) {
        ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    } else {
        ReadFile(GetGoldenDir() + "/golden_NC1HWC0.bin", cFileSize, golden.data(), cFileSize);
    }
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}
TEST_F(TIMG2COLTest, case4_int8)
{
    timg2col_test<4, int32_t, int8_t>(1, 1, 8, 16, 32, 1, 3, 3, 16, 32);
}
TEST_F(TIMG2COLTest, case8_int8_splitk)
{
    timg2col_test<8, int32_t, int8_t>(1, 2, 29, 60, 32, 2, 2, 2, 64, 32, 2, 2, 2, 2, 1, 1, 1, 0);
}
TEST_F(TIMG2COLTest, case12_int8_fractalZ4d)
{
    timg2col_test_fractal4d<12, int32_t, int8_t>(1, 2, 29, 60, 32, 8, 4, 16, 32, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0);
}