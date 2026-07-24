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

class TCONCATTest : public testing::Test {
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
    int vCols0, int vCols1>
void LaunchTConcat(T* out, T* src0, T* src1, void* stream);

template <
    int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows, int vCols0,
    int vCols1>
void LaunchTConcatHalf(aclFloat16* out, aclFloat16* src0, aclFloat16* src1, void* stream);

template <
    typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
    int vCols0, int vCols1>
void test_tconcat()
{
    size_t dstSize = dstTileH * dstTileW * sizeof(T);
    size_t src0Size = src0TileH * src0TileW * sizeof(T);
    size_t src1Size = src1TileH * src1TileW * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&src0Host), src0Size);
    aclrtMallocHost((void**)(&src1Host), src1Size);
    aclrtMallocHost((void**)(&dstHost), dstSize);

    aclrtMalloc((void**)&src0Device, src0Size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, src1Size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dstDevice, dstSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", src0Size, src0Host, src0Size);
    ReadFile(GetGoldenDir() + "/input2.bin", src1Size, src1Host, src1Size);
    aclrtMemset(dstHost, dstSize, 0, dstSize);

    aclrtMemcpy(src0Device, src0Size, src0Host, src0Size, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1Size, src1Host, src1Size, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dstDevice, dstSize, dstHost, dstSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if constexpr (std::is_same<T, aclFloat16>::value) {
        LaunchTConcatHalf<dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols0, vCols1>(
            dstDevice, src0Device, src1Device, stream);
    } else {
        LaunchTConcat<T, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols0, vCols1>(
            dstDevice, src0Device, src1Device, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstSize, dstDevice, dstSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstSize);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(dstHost);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(dstDevice);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> devFinal(dstTileH * dstTileW);
    std::vector<T> golden(dstTileH * dstTileW);
    ReadFile(GetGoldenDir() + "/golden.bin", dstSize, golden.data(), dstSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstSize, devFinal.data(), dstSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);
    ASSERT_TRUE(ret);
}

TEST_F(TCONCATTest, case_float_64x128_64x64_64x64_64x64_64x64)
{
    test_tconcat<float, 64, 128, 64, 64, 64, 64, 64, 64, 64>();
}

TEST_F(TCONCATTest, case_int32_64x128_64x64_64x64_64x64_64x64)
{
    test_tconcat<int32_t, 64, 128, 64, 64, 64, 64, 64, 64, 64>();
}

TEST_F(TCONCATTest, case_half_16x256_16x128_16x128_16x128_16x128)
{
    test_tconcat<aclFloat16, 16, 256, 16, 128, 16, 128, 16, 128, 128>();
}

TEST_F(TCONCATTest, case_float_16x64_16x32_16x32_16x32_16x32)
{
    test_tconcat<float, 16, 64, 16, 32, 16, 32, 16, 32, 32>();
}

TEST_F(TCONCATTest, case_int16_32x256_32x128_32x128_32x128_32x128)
{
    test_tconcat<int16_t, 32, 256, 32, 128, 32, 128, 32, 128, 128>();
}

TEST_F(TCONCATTest, case_half_16x128_16x64_16x64_16x63_16x64)
{
    test_tconcat<aclFloat16, 16, 128, 16, 64, 16, 64, 16, 63, 64>();
}

TEST_F(TCONCATTest, case_float_16x64_16x32_16x32_16x31_16x32)
{
    test_tconcat<float, 16, 64, 16, 32, 16, 32, 16, 31, 32>();
}

TEST_F(TCONCATTest, case_int16_32x256_32x128_32x128_32x127_32x128)
{
    test_tconcat<int16_t, 32, 256, 32, 128, 32, 128, 32, 127, 128>();
}
