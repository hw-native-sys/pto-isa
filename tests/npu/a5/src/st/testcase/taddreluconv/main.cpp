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
#include <gtest/gtest.h>
#include <acl/acl.h>
#include <securec.h>

using namespace std;
using namespace PtoTestCommon;

template <uint32_t caseId>
void launchTADDRELUCONVTestCase(void* out, void* src0, void* src1, aclrtStream stream);

class TADDRELUCONVTest : public testing::Test {
public:
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
    uint32_t caseId, typename DstT, typename SrcT, int row, int col, int toleranceX1000 = 1, int validRow = row,
    int validCol = col>
bool TADDRELUCONVTestFramework()
{
    float tolerance = static_cast<float>(toleranceX1000) / 1000.0f;

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t srcByteSize = row * col * sizeof(SrcT);
    size_t dstByteSize = row * col * sizeof(DstT);
    SrcT* src0Host;
    SrcT* src1Host;
    DstT* dstHost;
    SrcT* src0Device;
    SrcT* src1Device;
    DstT* dstDevice;

    aclrtMallocHost((void**)(&src0Host), srcByteSize);
    aclrtMallocHost((void**)(&src1Host), srcByteSize);
    aclrtMallocHost((void**)(&dstHost), dstByteSize);

    aclrtMalloc((void**)&src0Device, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    if constexpr (validRow < row || validCol < col) {
        DstT* zeroHost;
        aclrtMallocHost((void**)(&zeroHost), dstByteSize);
        memset_s(zeroHost, dstByteSize, 0, dstByteSize);
        aclrtMemcpy(dstDevice, dstByteSize, zeroHost, dstByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtFreeHost(zeroHost);
    }

    ReadFile(GetGoldenDir() + "/input0.bin", srcByteSize, src0Host, srcByteSize);
    ReadFile(GetGoldenDir() + "/input1.bin", srcByteSize, src1Host, srcByteSize);

    aclrtMemcpy(src0Device, srcByteSize, src0Host, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, srcByteSize, src1Host, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTADDRELUCONVTestCase<caseId>(dstDevice, src0Device, src1Device, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(dstDevice);

    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(dstHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<DstT> golden(row * col);
    std::vector<DstT> devFinal(row * col);
    ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);

    return ResultCmp<DstT>(golden, devFinal, tolerance);
}

TEST_F(TADDRELUCONVTest, case1)
{
    bool ret = TADDRELUCONVTestFramework<1, aclFloat16, float, 32, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case2)
{
    bool ret = TADDRELUCONVTestFramework<2, aclFloat16, float, 16, 128>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case3)
{
    bool ret = TADDRELUCONVTestFramework<3, aclFloat16, float, 31, 96>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case4)
{
    bool ret = TADDRELUCONVTestFramework<4, aclFloat16, float, 7, 192>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case5)
{
    bool ret = TADDRELUCONVTestFramework<5, aclFloat16, float, 64, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case6)
{
    bool ret = TADDRELUCONVTestFramework<6, aclFloat16, float, 13, 48>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case7)
{
    bool ret = TADDRELUCONVTestFramework<7, aclFloat16, float, 16, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case8)
{
    bool ret = TADDRELUCONVTestFramework<8, aclFloat16, float, 8, 128>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case9)
{
    bool ret = TADDRELUCONVTestFramework<9, aclFloat16, float, 4, 256>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case10)
{
    bool ret = TADDRELUCONVTestFramework<10, aclFloat16, float, 16, 32>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case11)
{
    bool ret = TADDRELUCONVTestFramework<11, int8_t, aclFloat16, 16, 128, 1000>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case12)
{
    bool ret = TADDRELUCONVTestFramework<12, int8_t, aclFloat16, 8, 64, 1000>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case13)
{
    bool ret = TADDRELUCONVTestFramework<13, int8_t, aclFloat16, 8, 128, 1000>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case14)
{
    bool ret = TADDRELUCONVTestFramework<14, int8_t, aclFloat16, 8, 64, 1000>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case15)
{
    bool ret = TADDRELUCONVTestFramework<15, int8_t, int16_t, 16, 128, 0>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case16)
{
    bool ret = TADDRELUCONVTestFramework<16, int8_t, int16_t, 8, 64, 0>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case17)
{
    bool ret = TADDRELUCONVTestFramework<17, int8_t, int16_t, 8, 128, 0>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case18)
{
    bool ret = TADDRELUCONVTestFramework<18, int8_t, int16_t, 8, 64, 0>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case19_fp32_partial_32x128_16x96)
{
    bool ret = TADDRELUCONVTestFramework<19, aclFloat16, float, 32, 128, 1, 16, 96>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case20_fp32_partial_col_16x128_16x65)
{
    bool ret = TADDRELUCONVTestFramework<20, aclFloat16, float, 16, 128, 1, 16, 65>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case21_fp32_partial_row_32x64_10x64)
{
    bool ret = TADDRELUCONVTestFramework<21, aclFloat16, float, 32, 64, 1, 10, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case22_fp32_single_row_1x256)
{
    bool ret = TADDRELUCONVTestFramework<22, aclFloat16, float, 1, 256>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case23_fp32_many_rows_128x32)
{
    bool ret = TADDRELUCONVTestFramework<23, aclFloat16, float, 128, 32>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case24_fp32_zero_src0_16x128)
{
    bool ret = TADDRELUCONVTestFramework<24, aclFloat16, float, 16, 128>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case25_fp32_negate_src1_16x64)
{
    bool ret = TADDRELUCONVTestFramework<25, aclFloat16, float, 16, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case26_fp32_large_values_16x128)
{
    bool ret = TADDRELUCONVTestFramework<26, aclFloat16, float, 16, 128, 3>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case27_fp16_partial_16x256_8x192)
{
    bool ret = TADDRELUCONVTestFramework<27, int8_t, aclFloat16, 16, 256, 1000, 8, 192>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case28_fp16_partial_col_8x256_8x129)
{
    bool ret = TADDRELUCONVTestFramework<28, int8_t, aclFloat16, 8, 256, 1000, 8, 129>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case29_fp16_single_row_1x128)
{
    bool ret = TADDRELUCONVTestFramework<29, int8_t, aclFloat16, 1, 128, 1000>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case30_fp16_zero_src0_8x128)
{
    bool ret = TADDRELUCONVTestFramework<30, int8_t, aclFloat16, 8, 128, 1000>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case31_fp16_negate_src1_8x128)
{
    bool ret = TADDRELUCONVTestFramework<31, int8_t, aclFloat16, 8, 128, 0>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case32_fp16_near_saturation_16x128)
{
    bool ret = TADDRELUCONVTestFramework<32, int8_t, aclFloat16, 16, 128, 1000>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case33_s16_partial_16x256_10x192)
{
    bool ret = TADDRELUCONVTestFramework<33, int8_t, int16_t, 16, 256, 0, 10, 192>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case34_s16_partial_col_8x256_8x129)
{
    bool ret = TADDRELUCONVTestFramework<34, int8_t, int16_t, 8, 256, 0, 8, 129>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case35_s16_single_row_1x128)
{
    bool ret = TADDRELUCONVTestFramework<35, int8_t, int16_t, 1, 128, 0>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case36_s16_zero_src0_8x128)
{
    bool ret = TADDRELUCONVTestFramework<36, int8_t, int16_t, 8, 128, 0>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case37_s16_negate_src1_8x128)
{
    bool ret = TADDRELUCONVTestFramework<37, int8_t, int16_t, 8, 128, 0>();
    EXPECT_TRUE(ret);
}

TEST_F(TADDRELUCONVTest, case38_s16_near_saturation_8x128)
{
    bool ret = TADDRELUCONVTestFramework<38, int8_t, int16_t, 8, 128, 0>();
    EXPECT_TRUE(ret);
}
