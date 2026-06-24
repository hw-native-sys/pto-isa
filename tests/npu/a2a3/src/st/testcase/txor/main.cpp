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

class TXORTest : public testing::Test {
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
void LaunchTXor(T *out, T *src0, T *src1, void *stream);

template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
          int vCols>
void test_txor()
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

    aclrtMallocHost((void **)(&dstHost), fileSizeDst);
    aclrtMallocHost((void **)(&src0Host), fileSizeSrc0);
    aclrtMallocHost((void **)(&src1Host), fileSizeSrc1);

    aclrtMalloc((void **)&dstDevice, fileSizeDst, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, fileSizeSrc0, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, fileSizeSrc1, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", fileSizeSrc0, src0Host, fileSizeSrc0);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSizeSrc1, src1Host, fileSizeSrc1);

    aclrtMemcpy(src0Device, fileSizeSrc0, src0Host, fileSizeSrc0, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSizeSrc1, src1Host, fileSizeSrc1, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTXor<T, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols>(dstDevice, src0Device,
                                                                                                src1Device, stream);

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

    std::vector<T> golden(fileSizeDst);
    std::vector<T> devFinal(fileSizeDst);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSizeDst, golden.data(), fileSizeDst);
    ReadFile(GetGoldenDir() + "/output.bin", fileSizeDst, devFinal.data(), fileSizeDst);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TXORTest, case_int16_64x64_64x64_64x64_64x64)
{
    test_txor<int16_t, 64, 64, 64, 64, 64, 64, 64, 64>();
}
TEST_F(TXORTest, case_int16_32x128_32x128_32x256_32x128)
{
    test_txor<int16_t, 32, 128, 32, 128, 32, 256, 32, 128>();
}
TEST_F(TXORTest, case_int16_32x128_32x128_32x256_32x127)
{
    test_txor<int16_t, 32, 128, 32, 128, 32, 256, 32, 127>();
}
TEST_F(TXORTest, case_uint16_64x64_64x64_64x64_64x64)
{
    test_txor<uint16_t, 64, 64, 64, 64, 64, 64, 64, 64>();
}
TEST_F(TXORTest, case_uint16_32x128_32x128_32x256_32x128)
{
    test_txor<uint16_t, 32, 128, 32, 128, 32, 256, 32, 128>();
}
TEST_F(TXORTest, case_uint16_32x128_32x128_32x256_32x127)
{
    test_txor<uint16_t, 32, 128, 32, 128, 32, 256, 32, 127>();
}
TEST_F(TXORTest, case_int8_32x128_32x128_32x256_32x127)
{
    test_txor<int8_t, 32, 128, 32, 128, 32, 256, 32, 127>();
}
TEST_F(TXORTest, case_uint8_32x128_32x128_32x256_32x127)
{
    test_txor<uint8_t, 32, 128, 32, 128, 32, 256, 32, 127>();
}
TEST_F(TXORTest, case_int32_32x128_32x128_32x256_32x127)
{
    test_txor<int32_t, 32, 128, 32, 128, 32, 256, 32, 127>();
}
TEST_F(TXORTest, case_uint32_32x128_32x128_32x256_32x127)
{
    test_txor<uint32_t, 32, 128, 32, 128, 32, 256, 32, 127>();
}
