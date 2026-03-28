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

template <typename dataType, typename idxType, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH,
          int src1TileW, int vRows, int vCols0, int vCols1>
void LaunchTConcat(dataType *out, dataType *src0, dataType *src1, idxType *src0Idx, idxType *src1Idx, void *stream);

template <typename idxType, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW,
          int vRows, int vCols0, int vCols1>
void LaunchTConcatHalf(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, idxType *src0Idx, idxType *src1Idx,
                       void *stream);

template <typename dataType, typename idxType, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH,
          int src1TileW, int vRows, int vCols0, int vCols1>
void test_tconcat()
{
    size_t fileSizeDst = dstTileH * dstTileW * sizeof(dataType);
    size_t fileSizeSrc0 = src0TileH * src0TileW * sizeof(dataType);
    size_t fileSizeSrc1 = src1TileH * src1TileW * sizeof(dataType);
    size_t fileSizeSrc0Idx = src0TileH * src0TileW * sizeof(idxType);
    size_t fileSizeSrc1Idx = src1TileH * src1TileW * sizeof(idxType);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    dataType *dstHost, *src0Host, *src1Host;
    idxType *src0IdxHost, *src1IdxHost;
    dataType *dstDevice, *src0Device, *src1Device;
    idxType *src0IdxDevice, *src1IdxDevice;

    aclrtMallocHost((void **)(&dstHost), fileSizeDst);
    aclrtMallocHost((void **)(&src0Host), fileSizeSrc0);
    aclrtMallocHost((void **)(&src1Host), fileSizeSrc1);
    aclrtMallocHost((void **)(&src0IdxHost), fileSizeSrc0Idx);
    aclrtMallocHost((void **)(&src1IdxHost), fileSizeSrc1Idx);

    aclrtMalloc((void **)&dstDevice, fileSizeDst, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, fileSizeSrc0, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, fileSizeSrc1, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0IdxDevice, fileSizeSrc0Idx, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1IdxDevice, fileSizeSrc1Idx, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", fileSizeSrc0, src0Host, fileSizeSrc0);
    ReadFile(GetGoldenDir() + "/input1.bin", fileSizeSrc1, src1Host, fileSizeSrc1);
    ReadFile(GetGoldenDir() + "/src0_idx.bin", fileSizeSrc0Idx, src0IdxHost, fileSizeSrc0Idx);
    ReadFile(GetGoldenDir() + "/src1_idx.bin", fileSizeSrc1Idx, src1IdxHost, fileSizeSrc1Idx);

    aclrtMemcpy(src0Device, fileSizeSrc0, src0Host, fileSizeSrc0, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSizeSrc1, src1Host, fileSizeSrc1, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src0IdxDevice, fileSizeSrc0Idx, src0IdxHost, fileSizeSrc0Idx, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1IdxDevice, fileSizeSrc1Idx, src1IdxHost, fileSizeSrc1Idx, ACL_MEMCPY_HOST_TO_DEVICE);
    if constexpr (std::is_same<dataType, aclFloat16>::value) {
        LaunchTConcatHalf<idxType, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols0,
                          vCols1>(dstDevice, src0Device, src1Device, src0IdxDevice, src1IdxDevice, stream);
    } else {
        LaunchTConcat<dataType, idxType, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols0,
                      vCols1>(dstDevice, src0Device, src1Device, src0IdxDevice, src1IdxDevice, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSizeDst, dstDevice, fileSizeDst, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSizeDst);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(src0IdxDevice);
    aclrtFree(src1IdxDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src0IdxHost);
    aclrtFreeHost(src1IdxHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<dataType> golden(fileSizeDst);
    std::vector<dataType> devFinal(fileSizeDst);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSizeDst, golden.data(), fileSizeDst);
    ReadFile(GetGoldenDir() + "/output.bin", fileSizeDst, devFinal.data(), fileSizeDst);

    bool ret = ResultCmp<dataType>(golden, devFinal, 0.001f);
    ASSERT_TRUE(ret);
}

TEST_F(TCONCATTest, case_int16_16x32_16x16_16x16_8x16_8x16)
{
    test_tconcat<int16_t, int16_t, 16, 32, 16, 16, 16, 16, 8, 16, 16>();
}

TEST_F(TCONCATTest, case_int32_64x128_64x64_64x64_64x64_64x64)
{
    test_tconcat<int32_t, int16_t, 64, 128, 64, 64, 64, 64, 64, 64, 64>();
}

TEST_F(TCONCATTest, case_half_16x256_16x128_16x128_16x128_16x128)
{
    test_tconcat<aclFloat16, int32_t, 16, 256, 16, 128, 16, 128, 16, 128, 128>();
}

TEST_F(TCONCATTest, case_float_16x64_16x32_16x32_16x32_16x32)
{
    test_tconcat<float, int16_t, 16, 64, 16, 32, 16, 32, 16, 32, 32>();
}

TEST_F(TCONCATTest, case_int16_32x256_32x128_32x128_32x128_32x128)
{
    test_tconcat<int16_t, int16_t, 32, 256, 32, 128, 32, 128, 32, 128, 128>();
}