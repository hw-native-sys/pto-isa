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
#include <gtest/gtest.h>
#include <acl/acl.h>

using namespace std;
using namespace PtoTestCommon;

template <typename T, int DstRow, int DstCol, int SrcRow, int SrcCol, pto::MaskPattern mask>
void launchTScatterMaskTestCase(void* out, void* src, void* stream);

class TCOLSCATTERTest : public testing::Test {
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

template <typename T, pto::MaskPattern PATTERN, uint32_t DST_ROW, uint32_t DST_COL, uint32_t SRC_ROW, uint32_t SRC_COL>
void test_scatter_mask()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t srcSize = SRC_ROW * SRC_COL * sizeof(T);
    size_t dstSize = DST_ROW * DST_COL * sizeof(T);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstSize);
    aclrtMallocHost((void**)(&srcHost), srcSize);
    aclrtMalloc((void**)&dstDevice, dstSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcSize, srcHost, srcSize);
    aclrtMemcpy(srcDevice, srcSize, srcHost, srcSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTScatterMaskTestCase<T, DST_ROW, DST_COL, SRC_ROW, SRC_COL, PATTERN>(dstDevice, srcDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstSize, dstDevice, dstSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(DST_ROW * DST_COL);
    std::vector<T> devFinal(DST_ROW * DST_COL);
    ReadFile(GetGoldenDir() + "/golden.bin", dstSize, golden.data(), dstSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstSize, devFinal.data(), dstSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.0f);
    EXPECT_TRUE(ret);
}

TEST_F(TCOLSCATTERTest, case_mask_half_16x64_16x64_P1111)
{
    test_scatter_mask<uint16_t, pto::MaskPattern::P1111, 16, 64, 16, 64>();
}

TEST_F(TCOLSCATTERTest, case_mask_float_16x64_16x64_P1111)
{
    test_scatter_mask<float, pto::MaskPattern::P1111, 16, 64, 16, 64>();
}

TEST_F(TCOLSCATTERTest, case_mask_int32_16x64_16x64_P1111)
{
    test_scatter_mask<int32_t, pto::MaskPattern::P1111, 16, 64, 16, 64>();
}

TEST_F(TCOLSCATTERTest, case_mask_half_16x64_32x128_P1010)
{
    test_scatter_mask<uint16_t, pto::MaskPattern::P1010, 32, 128, 16, 64>();
}

TEST_F(TCOLSCATTERTest, case_mask_half_16x64_32x128_P0101)
{
    test_scatter_mask<uint16_t, pto::MaskPattern::P0101, 32, 128, 16, 64>();
}

TEST_F(TCOLSCATTERTest, case_mask_float_16x64_32x128_P1010)
{
    test_scatter_mask<float, pto::MaskPattern::P1010, 32, 128, 16, 64>();
}

TEST_F(TCOLSCATTERTest, case_mask_float_16x64_32x128_P0101)
{
    test_scatter_mask<float, pto::MaskPattern::P0101, 32, 128, 16, 64>();
}

TEST_F(TCOLSCATTERTest, case_mask_int32_16x64_32x128_P1010)
{
    test_scatter_mask<int32_t, pto::MaskPattern::P1010, 32, 128, 16, 64>();
}

TEST_F(TCOLSCATTERTest, case_mask_int32_16x64_32x128_P0101)
{
    test_scatter_mask<int32_t, pto::MaskPattern::P0101, 32, 128, 16, 64>();
}

TEST_F(TCOLSCATTERTest, case_mask_half_4x64_16x256_P1000)
{
    test_scatter_mask<uint16_t, pto::MaskPattern::P1000, 16, 256, 4, 64>();
}

TEST_F(TCOLSCATTERTest, case_mask_half_4x64_16x256_P0100)
{
    test_scatter_mask<uint16_t, pto::MaskPattern::P0100, 16, 256, 4, 64>();
}

TEST_F(TCOLSCATTERTest, case_mask_half_4x64_16x256_P0010)
{
    test_scatter_mask<uint16_t, pto::MaskPattern::P0010, 16, 256, 4, 64>();
}

TEST_F(TCOLSCATTERTest, case_mask_half_4x64_16x256_P0001)
{
    test_scatter_mask<uint16_t, pto::MaskPattern::P0001, 16, 256, 4, 64>();
}

TEST_F(TCOLSCATTERTest, case_mask_float_4x64_16x256_P1000)
{
    test_scatter_mask<float, pto::MaskPattern::P1000, 16, 256, 4, 64>();
}

TEST_F(TCOLSCATTERTest, case_mask_float_4x64_16x256_P0100)
{
    test_scatter_mask<float, pto::MaskPattern::P0100, 16, 256, 4, 64>();
}

TEST_F(TCOLSCATTERTest, case_mask_float_4x64_16x256_P0010)
{
    test_scatter_mask<float, pto::MaskPattern::P0010, 16, 256, 4, 64>();
}

TEST_F(TCOLSCATTERTest, case_mask_float_4x64_16x256_P0001)
{
    test_scatter_mask<float, pto::MaskPattern::P0001, 16, 256, 4, 64>();
}

TEST_F(TCOLSCATTERTest, case_mask_int32_4x64_16x256_P1000)
{
    test_scatter_mask<int32_t, pto::MaskPattern::P1000, 16, 256, 4, 64>();
}

TEST_F(TCOLSCATTERTest, case_mask_int32_4x64_16x256_P0100)
{
    test_scatter_mask<int32_t, pto::MaskPattern::P0100, 16, 256, 4, 64>();
}

TEST_F(TCOLSCATTERTest, case_mask_int32_4x64_16x256_P0010)
{
    test_scatter_mask<int32_t, pto::MaskPattern::P0010, 16, 256, 4, 64>();
}

TEST_F(TCOLSCATTERTest, case_mask_int32_4x64_16x256_P0001)
{
    test_scatter_mask<int32_t, pto::MaskPattern::P0001, 16, 256, 4, 64>();
}
