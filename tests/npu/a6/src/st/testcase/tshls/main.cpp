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

class TSHLSTest : public testing::Test {
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

template <typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int vRows, int vCols>
void LaunchTShlS(T* out, T* src, T scalar, void* stream);

template <typename T, int tileRow, int tileCol, int validRow, int validCol>
void LaunchTShlSInplace(T* out, T* src, T scalar, void* stream);

template <typename T, int dstTileH, int dstTileW, int srcTileH, int srcTileW, int vRows, int vCols>
void test_tshls()
{
    size_t fileSizeDst = dstTileH * dstTileW * sizeof(T);
    size_t fileSizeSrc0 = srcTileH * srcTileW * sizeof(T);
    size_t fileSizeSrc1 = sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host;
    T *dstDevice, *src0Device;
    T scalar;

    aclrtMallocHost((void**)(&dstHost), fileSizeDst);
    aclrtMallocHost((void**)(&src0Host), fileSizeSrc0);

    aclrtMalloc((void**)&dstDevice, fileSizeDst, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, fileSizeSrc0, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(dstDevice, fileSizeDst, 0, fileSizeDst);

    ReadFile(GetGoldenDir() + "/input1.bin", fileSizeSrc0, src0Host, fileSizeSrc0);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSizeSrc1, (void*)&scalar, sizeof(T));

    aclrtMemcpy(src0Device, fileSizeSrc0, src0Host, fileSizeSrc0, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTShlS<T, dstTileH, dstTileW, srcTileH, srcTileW, vRows, vCols>(dstDevice, src0Device, scalar, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSizeDst, dstDevice, fileSizeDst, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSizeDst);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSizeDst / sizeof(T));
    std::vector<T> devFinal(fileSizeDst / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", fileSizeDst, golden.data(), fileSizeDst);
    ReadFile(GetGoldenDir() + "/output.bin", fileSizeDst, devFinal.data(), fileSizeDst);

    bool ret;
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        ret = ResultCmpExact(golden, devFinal.data());
    } else {
        ret = ResultCmp<T>(golden, devFinal, 0.001f);
    }

    EXPECT_TRUE(ret);
}

template <typename T, int tileRow, int tileCol, int validRow, int validCol>
void test_tshls_inplace()
{
    size_t fileSize = tileRow * tileCol * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *dstDevice;
    T scalar;

    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMalloc((void**)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", fileSize, dstHost, fileSize);
    size_t scalarSize = sizeof(T);
    ReadFile(GetGoldenDir() + "/divider.bin", scalarSize, &scalar, sizeof(T));

    aclrtMemcpy(dstDevice, fileSize, dstHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTShlSInplace<T, tileRow, tileCol, validRow, validCol>(dstDevice, dstDevice, scalar, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFreeHost(dstHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize / sizeof(T));
    std::vector<T> devFinal(fileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize);

    bool ret;
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        ret = ResultCmpExact(golden, devFinal.data());
    } else {
        ret = ResultCmp<T>(golden, devFinal, 0.001f);
    }
    EXPECT_TRUE(ret);
}

TEST_F(TSHLSTest, case_int16_64x64_64x64_64x64) { test_tshls<int16_t, 64, 64, 64, 64, 64, 64>(); }
TEST_F(TSHLSTest, case_int16_32x128_32x128_32x128) { test_tshls<int16_t, 32, 128, 32, 128, 32, 128>(); }
TEST_F(TSHLSTest, case_int16_32x112_32x128_32x111) { test_tshls<int16_t, 32, 112, 32, 128, 32, 111>(); }
TEST_F(TSHLSTest, case_uint16_64x64_64x64_64x64) { test_tshls<uint16_t, 64, 64, 64, 64, 64, 64>(); }
TEST_F(TSHLSTest, case_uint16_32x128_32x128_32x128) { test_tshls<uint16_t, 32, 128, 32, 128, 32, 128>(); }
TEST_F(TSHLSTest, case_uint16_32x112_32x128_32x111) { test_tshls<uint16_t, 32, 112, 32, 128, 32, 111>(); }
TEST_F(TSHLSTest, case_uint16_1x112_1x128_1x111) { test_tshls<uint16_t, 1, 112, 1, 128, 1, 111>(); }
TEST_F(TSHLSTest, case_int64_4x16_4x16_4x16) { test_tshls<int64_t, 4, 16, 4, 16, 4, 16>(); }
TEST_F(TSHLSTest, case_uint64_4x16_4x16_4x16) { test_tshls<uint64_t, 4, 16, 4, 16, 4, 16>(); }
TEST_F(TSHLSTest, case_int64_4x16_shift_0) { test_tshls<int64_t, 4, 16, 4, 16, 4, 16>(); }
TEST_F(TSHLSTest, case_int64_4x16_shift_1) { test_tshls<int64_t, 4, 16, 4, 16, 4, 16>(); }
TEST_F(TSHLSTest, case_int64_4x16_shift_31) { test_tshls<int64_t, 4, 16, 4, 16, 4, 16>(); }
TEST_F(TSHLSTest, case_int64_4x16_shift_32) { test_tshls<int64_t, 4, 16, 4, 16, 4, 16>(); }
TEST_F(TSHLSTest, case_int64_4x16_shift_33) { test_tshls<int64_t, 4, 16, 4, 16, 4, 16>(); }
TEST_F(TSHLSTest, case_int64_4x16_shift_63) { test_tshls<int64_t, 4, 16, 4, 16, 4, 16>(); }
TEST_F(TSHLSTest, case_int64_4x16_shift_64) { test_tshls<int64_t, 4, 16, 4, 16, 4, 16>(); }
TEST_F(TSHLSTest, case_int64_4x16_shift_65) { test_tshls<int64_t, 4, 16, 4, 16, 4, 16>(); }
TEST_F(TSHLSTest, case_uint64_4x16_shift_0) { test_tshls<uint64_t, 4, 16, 4, 16, 4, 16>(); }
TEST_F(TSHLSTest, case_uint64_4x16_shift_1) { test_tshls<uint64_t, 4, 16, 4, 16, 4, 16>(); }
TEST_F(TSHLSTest, case_uint64_4x16_shift_31) { test_tshls<uint64_t, 4, 16, 4, 16, 4, 16>(); }
TEST_F(TSHLSTest, case_uint64_4x16_shift_32) { test_tshls<uint64_t, 4, 16, 4, 16, 4, 16>(); }
TEST_F(TSHLSTest, case_uint64_4x16_shift_33) { test_tshls<uint64_t, 4, 16, 4, 16, 4, 16>(); }
TEST_F(TSHLSTest, case_uint64_4x16_shift_63) { test_tshls<uint64_t, 4, 16, 4, 16, 4, 16>(); }
TEST_F(TSHLSTest, case_uint64_4x16_shift_64) { test_tshls<uint64_t, 4, 16, 4, 16, 4, 16>(); }
TEST_F(TSHLSTest, case_uint64_4x16_shift_65) { test_tshls<uint64_t, 4, 16, 4, 16, 4, 16>(); }
TEST_F(TSHLSTest, case_int64_1x16364_1x16364_1x16364) { test_tshls<int64_t, 1, 16364, 1, 16364, 1, 16364>(); }
TEST_F(TSHLSTest, case_uint64_1x16364_1x16364_1x16364) { test_tshls<uint64_t, 1, 16364, 1, 16364, 1, 16364>(); }
TEST_F(TSHLSTest, case_int64_1x16368_1x16368_1x16368) { test_tshls<int64_t, 1, 16368, 1, 16368, 1, 16368>(); }
TEST_F(TSHLSTest, case_uint64_1x16368_1x16368_1x16368) { test_tshls<uint64_t, 1, 16368, 1, 16368, 1, 16368>(); }
TEST_F(TSHLSTest, case_int64_4x32_inplace) { test_tshls_inplace<int64_t, 4, 32, 4, 32>(); }
TEST_F(TSHLSTest, case_uint64_4x32_inplace) { test_tshls_inplace<uint64_t, 4, 32, 4, 32>(); }
TEST_F(TSHLSTest, case_int64_1x1024_inplace) { test_tshls_inplace<int64_t, 1, 1024, 1, 1024>(); }
TEST_F(TSHLSTest, case_int64_4x64_40_inplace) { test_tshls_inplace<int64_t, 4, 64, 4, 40>(); }
TEST_F(TSHLSTest, case_int64_1x2048_2045_inplace) { test_tshls_inplace<int64_t, 1, 2048, 1, 2045>(); }
