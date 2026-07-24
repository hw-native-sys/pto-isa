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
#include <cstdint>

using namespace std;
using namespace PtoTestCommon;

template <
    typename T, int format, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4, int WN5,
    int BASEM, int BASEK>
void launchTLOADSCALE(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

class TLOADSCALETest : public testing::Test {
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
    typename T, int format, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4, int WN5,
    int BASEM, int BASEK>
void TLOADSCALEFUNC()
{
    size_t aFileSize = WN1 * WN2 * WN3 * WN4 * WN5 * sizeof(T);
    size_t bFileSize = N4 * N5 * sizeof(T);
    size_t cFileSize = N1 * N2 * N3 * N4 * N5 * sizeof(T);

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
    launchTLOADSCALE<T, format, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
        dstDevice, src0Device, src1Device, stream);

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

// shape[0] == 1, L1Size = [validRow, validCol]
TEST_F(TLOADSCALETest, 1_1_2_16_2_1_2_3_16_2_16_4_scale_ZZ2ZZ)
{
    TLOADSCALEFUNC<uint8_t, 11, 1, 1, 2, 16, 2, 1, 2, 3, 16, 2, 16, 4>();
}
TEST_F(TLOADSCALETest, 1_2_1_16_2_1_3_2_16_2_2_32_scale_NN2NN)
{
    TLOADSCALEFUNC<uint8_t, 12, 1, 2, 1, 16, 2, 1, 3, 2, 16, 2, 2, 32>();
}
// shape[0] == 1, L1Size > [validRow, validCol]
TEST_F(TLOADSCALETest, 1_2_2_16_2_1_2_3_16_2_48_10_scale_ZZ2ZZ)
{
    TLOADSCALEFUNC<uint8_t, 11, 1, 2, 2, 16, 2, 1, 2, 3, 16, 2, 48, 10>();
}
TEST_F(TLOADSCALETest, 1_2_2_16_2_1_3_2_16_2_8_64_scale_NN2NN)
{
    TLOADSCALEFUNC<uint8_t, 12, 1, 2, 2, 16, 2, 1, 3, 2, 16, 2, 8, 64>();
}
TEST_F(TLOADSCALETest, 1_5_33_16_2_1_11_40_16_2_128_96_scale_ZZ2ZZ)
{
    TLOADSCALEFUNC<uint8_t, 11, 1, 5, 33, 16, 2, 1, 11, 40, 16, 2, 128, 96>();
}
TEST_F(TLOADSCALETest, 1_64_29_16_2_1_65_59_16_2_58_1088_scale_NN2NN)
{
    TLOADSCALEFUNC<uint8_t, 12, 1, 64, 29, 16, 2, 1, 65, 59, 16, 2, 58, 1088>();
}
// shape[0] > 1, L1Size = [validRow, validCol]
TEST_F(TLOADSCALETest, 3_1_2_16_2_3_2_3_16_2_48_4_scale_ZZ2ZZ)
{
    TLOADSCALEFUNC<uint8_t, 11, 3, 1, 2, 16, 2, 3, 2, 3, 16, 2, 48, 4>();
}
TEST_F(TLOADSCALETest, 4_2_1_16_2_4_3_2_16_2_2_128_scale_NN2NN)
{
    TLOADSCALEFUNC<uint8_t, 12, 4, 2, 1, 16, 2, 4, 3, 2, 16, 2, 2, 128>();
}

// shape[0] > 1, L1Size > [validRow, validCol]
TEST_F(TLOADSCALETest, 4_3_3_16_2_4_10_5_16_2_192_10_scale_ZZ2ZZ)
{
    TLOADSCALEFUNC<uint8_t, 11, 4, 3, 3, 16, 2, 4, 10, 5, 16, 2, 192, 10>();
}
TEST_F(TLOADSCALETest, 7_5_3_16_2_7_7_11_16_2_12_560_scale_NN2NN)
{
    TLOADSCALEFUNC<uint8_t, 12, 7, 5, 3, 16, 2, 7, 7, 11, 16, 2, 12, 560>();
}
