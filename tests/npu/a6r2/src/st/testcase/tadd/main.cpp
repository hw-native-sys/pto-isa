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

class TADDTest : public testing::Test {
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
template <typename T, int tileH, int tileW, int vRows, int vCols>
void LaunchTAddInplace(T* out, T* src1, void* stream);

template <int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows, int vCols>
void LaunchTAddHalf(aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);

template <typename T>
void CheckTAddResult(size_t fileSizeDst)
{
    std::vector<T> golden(fileSizeDst / sizeof(T));
    std::vector<T> devFinal(fileSizeDst / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", fileSizeDst, golden.data(), fileSizeDst);
    ReadFile(GetGoldenDir() + "/output.bin", fileSizeDst, devFinal.data(), fileSizeDst);
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)
        EXPECT_TRUE(ResultCmpExact(golden, devFinal.data()));
    else
        EXPECT_TRUE(ResultCmp<T>(golden, devFinal, 0.001f));
}

template <
    typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
    int vCols, bool isHalf = false>
void test_tadd()
{
    size_t fileSizeDst = dstTileH * dstTileW * sizeof(T);
    size_t fileSizeSrc0 = src0TileH * src0TileW * sizeof(T);
    size_t fileSizeSrc1 = src1TileH * src1TileW * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), fileSizeDst);
    aclrtMallocHost((void**)(&src0Host), fileSizeSrc0);
    aclrtMallocHost((void**)(&src1Host), fileSizeSrc1);

    aclrtMalloc((void**)&dstDevice, fileSizeDst, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, fileSizeSrc0, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, fileSizeSrc1, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(dstDevice, fileSizeDst, 0, fileSizeDst);

    memset(dstHost, 0, fileSizeDst);
    ReadFile(GetGoldenDir() + "/input1.bin", fileSizeSrc0, src0Host, fileSizeSrc0);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSizeSrc1, src1Host, fileSizeSrc1);

    aclrtMemcpy(dstDevice, fileSizeDst, dstHost, fileSizeDst, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src0Device, fileSizeSrc0, src0Host, fileSizeSrc0, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSizeSrc1, src1Host, fileSizeSrc1, ACL_MEMCPY_HOST_TO_DEVICE);
    // At present compiler cannot distinguish between aclFloat16 and uint16_t
    if constexpr (isHalf) {
        LaunchTAddHalf<dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols>(
            dstDevice, src0Device, src1Device, stream);
    } else {
        LaunchTAdd<T, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols>(
            dstDevice, src0Device, src1Device, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSizeDst, dstDevice, fileSizeDst, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSizeDst);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    CheckTAddResult<T>(fileSizeDst);
}

TEST_F(TADDTest, case_float_64x64_64x64_64x64_64x64) { test_tadd<float, 64, 64, 64, 64, 64, 64, 64, 64>(); }
TEST_F(TADDTest, case_float_64x128_64x128_64x128_64x128) { test_tadd<float, 64, 128, 64, 128, 64, 128, 64, 128>(); }
TEST_F(TADDTest, case_int32_64x64_64x64_64x64_64x64) { test_tadd<int32_t, 64, 64, 64, 64, 64, 64, 64, 64>(); }
TEST_F(TADDTest, case_int64_4x16_4x16_4x16_4x15) { test_tadd<int64_t, 4, 16, 4, 16, 4, 16, 4, 15>(); }
TEST_F(TADDTest, case_uint64_4x16_4x16_4x16_4x15) { test_tadd<uint64_t, 4, 16, 4, 16, 4, 16, 4, 15>(); }
TEST_F(TADDTest, case_int64_32x32_32x32_32x32_32x32) { test_tadd<int64_t, 32, 32, 32, 32, 32, 32, 32, 32>(); }
TEST_F(TADDTest, case_int64_1x1024_1x1024_1x1024_1x1024) { test_tadd<int64_t, 1, 1024, 1, 1024, 1, 1024, 1, 1024>(); }
TEST_F(TADDTest, case_int16_64x64_64x64_64x64_64x64) { test_tadd<int16_t, 64, 64, 64, 64, 64, 64, 64, 64>(); }
TEST_F(TADDTest, case_half_16x256_16x256_16x256_16x256)
{
    test_tadd<aclFloat16, 16, 256, 16, 256, 16, 256, 16, 256, true>();
}
TEST_F(TADDTest, case_half_16x64_16x128_16x128_16x64)
{
    test_tadd<aclFloat16, 16, 64, 16, 128, 16, 128, 16, 64, true>();
}
TEST_F(TADDTest, case_float_16x32_16x64_16x32_16x32) { test_tadd<float, 16, 32, 16, 64, 16, 32, 16, 32>(); }
TEST_F(TADDTest, case_int16_32x128_32x128_32x256_32x128) { test_tadd<int16_t, 32, 128, 32, 128, 32, 256, 32, 128>(); }
TEST_F(TADDTest, case_int32_16x32_16x64_16x32_16x32) { test_tadd<int32_t, 16, 32, 16, 64, 16, 32, 16, 32>(); }
TEST_F(TADDTest, case_half_16x64_16x128_16x128_16x63)
{
    test_tadd<aclFloat16, 16, 64, 16, 128, 16, 128, 16, 63, true>();
}
TEST_F(TADDTest, case_float_16x32_16x64_16x32_16x31) { test_tadd<float, 16, 32, 16, 64, 16, 32, 16, 31>(); }
TEST_F(TADDTest, case_int16_32x128_32x128_32x256_32x127) { test_tadd<int16_t, 32, 128, 32, 128, 32, 256, 32, 127>(); }
TEST_F(TADDTest, case_int32_16x32_16x64_16x32_16x31) { test_tadd<int32_t, 16, 32, 16, 64, 16, 32, 16, 31>(); }
TEST_F(TADDTest, case_half_2x128_2x128_2x128_1x106) { test_tadd<aclFloat16, 2, 128, 2, 128, 2, 128, 1, 106, true>(); }

template <typename T, int tileH, int tileW, int vRows, int vCols>
void test_tadd_inplace()
{
    size_t fileSize = tileH * tileW * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src1Host;
    T *dstDevice, *src1Device;

    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMallocHost((void**)(&src1Host), fileSize);

    aclrtMalloc((void**)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, dstHost, fileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSize, src1Host, fileSize);

    aclrtMemcpy(dstDevice, fileSize, dstHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTAddInplace<T, tileH, tileW, vRows, vCols>(dstDevice, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    CheckTAddResult<T>(fileSize);
}

TEST_F(TADDTest, case_int64_4x32_inplace) { test_tadd_inplace<int64_t, 4, 32, 4, 32>(); }
TEST_F(TADDTest, case_uint64_4x32_inplace) { test_tadd_inplace<uint64_t, 4, 32, 4, 32>(); }
TEST_F(TADDTest, case_int64_1x1024_inplace) { test_tadd_inplace<int64_t, 1, 1024, 1, 1024>(); }
TEST_F(TADDTest, case_int64_4x64_40_inplace) { test_tadd_inplace<int64_t, 4, 64, 4, 40>(); }
TEST_F(TADDTest, case_int64_1x2048_2045_inplace) { test_tadd_inplace<int64_t, 1, 2048, 1, 2045>(); }
