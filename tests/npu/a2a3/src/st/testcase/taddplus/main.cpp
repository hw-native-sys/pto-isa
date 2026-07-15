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

class TADDPLUSTest : public testing::Test {
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
    typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
    int vCols>
void LaunchTAdd(T* out, T* src0, T* src1, void* stream);

template <
    typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
    int vCols>
void test_tadd()
{
    size_t dstFileSize = dstTileH * dstTileW * sizeof(T);
    size_t src0FileSize = src0TileH * src0TileW * sizeof(T);
    size_t src1FileSize = src1TileH * src1TileW * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&src0Host), src0FileSize);
    aclrtMallocHost((void**)(&src1Host), src1FileSize);
    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", src0FileSize, src0Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", src1FileSize, src1Host, src1FileSize);

    aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTAdd<T, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols>(
        dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);
    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize);
    std::vector<T> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TADDPLUSTest, case_half_16x64_16x128_16x128_16x64) { test_tadd<aclFloat16, 16, 64, 16, 128, 16, 128, 16, 64>(); }

TEST_F(TADDPLUSTest, case_float_16x32_16x64_16x32_16x32) { test_tadd<float, 16, 32, 16, 64, 16, 32, 16, 32>(); }

TEST_F(TADDPLUSTest, case_int16_32x128_32x128_32x256_32x128)
{
    test_tadd<int16_t, 32, 128, 32, 128, 32, 256, 32, 128>();
}

TEST_F(TADDPLUSTest, case_int32_16x32_16x64_16x32_16x32) { test_tadd<int32_t, 16, 32, 16, 64, 16, 32, 16, 32>(); }

TEST_F(TADDPLUSTest, case_half_16x64_16x128_16x128_16x63) { test_tadd<aclFloat16, 16, 64, 16, 128, 16, 128, 16, 63>(); }

TEST_F(TADDPLUSTest, case_float_16x32_16x64_16x32_16x31) { test_tadd<float, 16, 32, 16, 64, 16, 32, 16, 31>(); }

TEST_F(TADDPLUSTest, case_int16_32x128_32x128_32x256_32x127)
{
    test_tadd<int16_t, 32, 128, 32, 128, 32, 256, 32, 127>();
}

TEST_F(TADDPLUSTest, case_int32_16x32_16x64_16x32_16x31) { test_tadd<int32_t, 16, 32, 16, 64, 16, 32, 16, 31>(); }