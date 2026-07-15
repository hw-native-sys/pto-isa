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

TEST_F(TPAIRREDUCESUMTest, case_float_32x64_32x64_32x64) { test_tpairreducesum<float, 32, 64, 32, 64, 32, 64>(); }
TEST_F(TPAIRREDUCESUMTest, case_float_32x128_32x128_32x128) { test_tpairreducesum<float, 32, 128, 32, 128, 32, 128>(); }
TEST_F(TPAIRREDUCESUMTest, case_half_8x256_8x256_8x256)
{
    test_tpairreducesum<aclFloat16, 8, 256, 8, 256, 8, 256, true>();
}
TEST_F(TPAIRREDUCESUMTest, case_half_8x64_8x128_8x64) { test_tpairreducesum<aclFloat16, 8, 64, 8, 128, 8, 64, true>(); }
TEST_F(TPAIRREDUCESUMTest, case_float_8x32_8x64_8x32) { test_tpairreducesum<float, 8, 32, 8, 64, 8, 32>(); }
TEST_F(TPAIRREDUCESUMTest, case_half_8x64_8x128_8x63) { test_tpairreducesum<aclFloat16, 8, 64, 8, 128, 8, 63, true>(); }
TEST_F(TPAIRREDUCESUMTest, case_float_8x32_8x64_8x31) { test_tpairreducesum<float, 8, 32, 8, 64, 8, 31>(); }
TEST_F(TPAIRREDUCESUMTest, case_half_4x128_4x128_2x106)
{
    test_tpairreducesum<aclFloat16, 4, 128, 4, 128, 2, 106, true>();
}
TEST_F(TPAIRREDUCESUMTest, case_float_8x128_8x128_8x127) { test_tpairreducesum<float, 8, 128, 8, 128, 8, 127>(); }
TEST_F(TPAIRREDUCESUMTest, case_half_8x256_8x256_8x255)
{
    test_tpairreducesum<aclFloat16, 8, 256, 8, 256, 8, 255, true>();
}
