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

class TPAIRREDUCESUMTest : public testing::Test {
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
void LaunchTPairReduceSum(T* out, T* src0, void* stream);

template <int dstTileH, int dstTileW, int src0TileH, int src0TileW, int vRows, int vCols>
void LaunchTPairReduceSumHalf(aclFloat16* out, aclFloat16* src0, void* stream);

template <
    typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int vRows, int vCols, bool isHalf = false>
void test_tpairreducesum()
{
    size_t fileSizeDst = dstTileH * dstTileW * sizeof(T);
    size_t fileSizeSrc0 = src0TileH * src0TileW * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host;
    T *dstDevice, *src0Device;

    aclrtMallocHost((void**)(&dstHost), fileSizeDst);
    aclrtMallocHost((void**)(&src0Host), fileSizeSrc0);

    aclrtMalloc((void**)&dstDevice, fileSizeDst, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, fileSizeSrc0, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemset(dstHost, fileSizeDst, 0, fileSizeDst);
    ReadFile(GetGoldenDir() + "/input1.bin", fileSizeSrc0, src0Host, fileSizeSrc0);

    aclrtMemcpy(dstDevice, fileSizeDst, dstHost, fileSizeDst, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src0Device, fileSizeSrc0, src0Host, fileSizeSrc0, ACL_MEMCPY_HOST_TO_DEVICE);
    if constexpr (isHalf) {
        LaunchTPairReduceSumHalf<dstTileH, dstTileW, src0TileH, src0TileW, vRows, vCols>(dstDevice, src0Device, stream);
    } else {
        LaunchTPairReduceSum<T, dstTileH, dstTileW, src0TileH, src0TileW, vRows, vCols>(dstDevice, src0Device, stream);
    }

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

    std::vector<T> golden(fileSizeDst);
    std::vector<T> devFinal(fileSizeDst);
    ReadFile(GetGoldenDir() + "/golden.bin", fileSizeDst, golden.data(), fileSizeDst);
    ReadFile(GetGoldenDir() + "/output.bin", fileSizeDst, devFinal.data(), fileSizeDst);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TPAIRREDUCESUMTest, case_float_64x64_64x64_64x64) { test_tpairreducesum<float, 64, 64, 64, 64, 64, 64>(); }
TEST_F(TPAIRREDUCESUMTest, case_float_64x128_64x128_64x128) { test_tpairreducesum<float, 64, 128, 64, 128, 64, 128>(); }
TEST_F(TPAIRREDUCESUMTest, case_half_16x256_16x256_16x256)
{
    test_tpairreducesum<aclFloat16, 16, 256, 16, 256, 16, 256, true>();
}
TEST_F(TPAIRREDUCESUMTest, case_half_16x64_16x128_16x64)
{
    test_tpairreducesum<aclFloat16, 16, 64, 16, 128, 16, 64, true>();
}
TEST_F(TPAIRREDUCESUMTest, case_float_16x32_16x64_16x32) { test_tpairreducesum<float, 16, 32, 16, 64, 16, 32>(); }
TEST_F(TPAIRREDUCESUMTest, case_half_16x64_16x128_16x63)
{
    test_tpairreducesum<aclFloat16, 16, 64, 16, 128, 16, 63, true>();
}
TEST_F(TPAIRREDUCESUMTest, case_float_16x32_16x64_16x31) { test_tpairreducesum<float, 16, 32, 16, 64, 16, 31>(); }
TEST_F(TPAIRREDUCESUMTest, case_half_2x128_2x128_1x106)
{
    test_tpairreducesum<aclFloat16, 2, 128, 2, 128, 1, 106, true>();
}
