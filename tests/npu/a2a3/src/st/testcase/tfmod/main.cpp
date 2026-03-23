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

class TFMODTest : public testing::Test {
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

template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
          int vCols>
void LaunchTFMOD(T *out, T *src0, T *src1, void *stream);

template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
          int vCols>
void test_tfmod()
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

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&src0Host), src0FileSize);
    aclrtMallocHost((void **)(&src1Host), src1FileSize);
    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", src0FileSize, src0Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", src1FileSize, src1Host, src1FileSize);

    aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTFMOD<T, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols>(dstDevice, src0Device,
                                                                                                 src1Device, stream);

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

    using U = std::conditional_t<std::is_same_v<T, aclFloat16>, _Float16, T>;

    std::vector<U> golden(dstFileSize);
    std::vector<U> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<U>(golden, devFinal, 0.001f, 0, 1000, false, true);
    EXPECT_TRUE(ret);
}

TEST_F(TFMODTest, case_float_16x64_16x128_16x128_16x64)
{
    test_tfmod<float, 16, 64, 16, 128, 16, 128, 16, 64>();
}

TEST_F(TFMODTest, case_float_16x32_16x64_16x32_16x32)
{
    test_tfmod<float, 16, 32, 16, 64, 16, 32, 16, 32>();
}

TEST_F(TFMODTest, case_float_16x64_16x128_16x128_16x63)
{
    test_tfmod<float, 16, 64, 16, 128, 16, 128, 16, 63>();
}

TEST_F(TFMODTest, case_float_2x32_2x64_2x32_2x31)
{
    test_tfmod<float, 2, 32, 2, 64, 2, 32, 2, 31>();
}

TEST_F(TFMODTest, case_float_1x8192_1x8192_1x8192_1x8192)
{
    test_tfmod<float, 1, 8192, 1, 8192, 1, 8192, 1, 8192>();
}