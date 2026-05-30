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

template <uint32_t caseId>
void launchTScatterTestCase(void *out, void *src, void *indexes, aclrtStream stream);

template <typename T, int DstRow, int DstCol, int SrcRow, int SrcCol, pto::MaskPattern mask>
void launchTScatterMaskTestCase(void *out, void *src, void *stream);

class TSCATTERTest : public testing::Test {
public:
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

template <uint32_t caseId, typename T, typename TI, uint32_t Src0Row, uint32_t Src0Col, uint32_t Src1Row,
          uint32_t Src1Col>
bool TScatterTestFramework()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t dataSize = Src0Row * Src0Col * sizeof(T);
    size_t idxSize = Src1Row * Src1Col * sizeof(TI);
    T *dstHost;
    T *srcHost;
    TI *indHost;
    T *dstDevice;
    T *srcDevice;
    TI *indDevice;

    aclrtMallocHost((void **)(&dstHost), dataSize);
    aclrtMallocHost((void **)(&srcHost), dataSize);
    aclrtMallocHost((void **)(&indHost), idxSize);

    aclrtMalloc((void **)&dstDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&indDevice, idxSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", dataSize, srcHost, dataSize);
    ReadFile(GetGoldenDir() + "/indexes.bin", idxSize, indHost, idxSize);
    aclrtMemcpy(srcDevice, dataSize, srcHost, dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(indDevice, idxSize, indHost, idxSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTScatterTestCase<caseId>(dstDevice, srcDevice, indDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dataSize, dstDevice, dataSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dataSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(indDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(indHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(Src0Row * Src0Col);
    std::vector<T> devFinal(Src0Row * Src0Col);
    ReadFile(GetGoldenDir() + "/golden.bin", dataSize, golden.data(), dataSize);
    ReadFile(GetGoldenDir() + "/output.bin", dataSize, devFinal.data(), dataSize);

    return ResultCmp<T>(golden, devFinal, 0.0f);
}

TEST_F(TSCATTERTest, case_int16_uint16_2x32_1x32)
{
    bool ret = TScatterTestFramework<1, int16_t, uint16_t, 2, 32, 1, 32>();
    EXPECT_TRUE(ret);
}

TEST_F(TSCATTERTest, case_half_uint16_63x64_63x64)
{
    bool ret = TScatterTestFramework<2, aclFloat16, uint16_t, 63, 64, 63, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TSCATTERTest, case_int32_uint32_31x128_31x128)
{
    bool ret = TScatterTestFramework<3, int32_t, uint32_t, 31, 128, 31, 128>();
    EXPECT_TRUE(ret);
}

TEST_F(TSCATTERTest, case_int16_int16_15x192_15x192)
{
    bool ret = TScatterTestFramework<4, int16_t, int16_t, 15, 192, 15, 192>();
    EXPECT_TRUE(ret);
}

TEST_F(TSCATTERTest, case_float_int32_7x448_7x448)
{
    bool ret = TScatterTestFramework<5, float, int32_t, 7, 448, 7, 448>();
    EXPECT_TRUE(ret);
}

TEST_F(TSCATTERTest, case_int8_uint16_256x32_256x32)
{
    bool ret = TScatterTestFramework<6, int8_t, uint16_t, 256, 32, 256, 32>();
    EXPECT_TRUE(ret);
}
TEST_F(TSCATTERTest, case_float_uint32_32x64_32x64)
{
    bool ret = TScatterTestFramework<7, float, uint32_t, 32, 64, 32, 64>();
    EXPECT_TRUE(ret);
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

    aclrtMallocHost((void **)(&dstHost), dstSize);
    aclrtMallocHost((void **)(&srcHost), srcSize);
    aclrtMalloc((void **)&dstDevice, dstSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcSize, ACL_MEM_MALLOC_HUGE_FIRST);

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

TEST_F(TSCATTERTest, case_mask_half_16x64_16x64_P1111)
{
    test_scatter_mask<uint16_t, pto::MaskPattern::P1111, 16, 64, 16, 64>();
}

TEST_F(TSCATTERTest, case_mask_float_16x64_16x64_P1111)
{
    test_scatter_mask<float, pto::MaskPattern::P1111, 16, 64, 16, 64>();
}

TEST_F(TSCATTERTest, case_mask_int32_16x64_16x64_P1111)
{
    test_scatter_mask<int32_t, pto::MaskPattern::P1111, 16, 64, 16, 64>();
}

TEST_F(TSCATTERTest, case_mask_half_16x64_16x128_P1010)
{
    test_scatter_mask<uint16_t, pto::MaskPattern::P1010, 16, 128, 16, 64>();
}

TEST_F(TSCATTERTest, case_mask_half_16x64_16x128_P0101)
{
    test_scatter_mask<uint16_t, pto::MaskPattern::P0101, 16, 128, 16, 64>();
}

TEST_F(TSCATTERTest, case_mask_float_16x64_16x128_P1010)
{
    test_scatter_mask<float, pto::MaskPattern::P1010, 16, 128, 16, 64>();
}

TEST_F(TSCATTERTest, case_mask_float_16x64_16x128_P0101)
{
    test_scatter_mask<float, pto::MaskPattern::P0101, 16, 128, 16, 64>();
}

TEST_F(TSCATTERTest, case_mask_int32_16x64_16x128_P1010)
{
    test_scatter_mask<int32_t, pto::MaskPattern::P1010, 16, 128, 16, 64>();
}

TEST_F(TSCATTERTest, case_mask_int32_16x64_16x128_P0101)
{
    test_scatter_mask<int32_t, pto::MaskPattern::P0101, 16, 128, 16, 64>();
}

TEST_F(TSCATTERTest, case_mask_half_16x64_16x256_P1000)
{
    test_scatter_mask<uint16_t, pto::MaskPattern::P1000, 16, 256, 16, 64>();
}

TEST_F(TSCATTERTest, case_mask_half_16x64_16x256_P0100)
{
    test_scatter_mask<uint16_t, pto::MaskPattern::P0100, 16, 256, 16, 64>();
}

TEST_F(TSCATTERTest, case_mask_half_16x64_16x256_P0010)
{
    test_scatter_mask<uint16_t, pto::MaskPattern::P0010, 16, 256, 16, 64>();
}

TEST_F(TSCATTERTest, case_mask_half_16x64_16x256_P0001)
{
    test_scatter_mask<uint16_t, pto::MaskPattern::P0001, 16, 256, 16, 64>();
}

TEST_F(TSCATTERTest, case_mask_float_16x64_16x256_P1000)
{
    test_scatter_mask<float, pto::MaskPattern::P1000, 16, 256, 16, 64>();
}

TEST_F(TSCATTERTest, case_mask_float_16x64_16x256_P0100)
{
    test_scatter_mask<float, pto::MaskPattern::P0100, 16, 256, 16, 64>();
}

TEST_F(TSCATTERTest, case_mask_float_16x64_16x256_P0010)
{
    test_scatter_mask<float, pto::MaskPattern::P0010, 16, 256, 16, 64>();
}

TEST_F(TSCATTERTest, case_mask_float_16x64_16x256_P0001)
{
    test_scatter_mask<float, pto::MaskPattern::P0001, 16, 256, 16, 64>();
}

TEST_F(TSCATTERTest, case_mask_int32_16x64_16x256_P1000)
{
    test_scatter_mask<int32_t, pto::MaskPattern::P1000, 16, 256, 16, 64>();
}

TEST_F(TSCATTERTest, case_mask_int32_16x64_16x256_P0100)
{
    test_scatter_mask<int32_t, pto::MaskPattern::P0100, 16, 256, 16, 64>();
}

TEST_F(TSCATTERTest, case_mask_int32_16x64_16x256_P0010)
{
    test_scatter_mask<int32_t, pto::MaskPattern::P0010, 16, 256, 16, 64>();
}

TEST_F(TSCATTERTest, case_mask_int32_16x64_16x256_P0001)
{
    test_scatter_mask<int32_t, pto::MaskPattern::P0001, 16, 256, 16, 64>();
}
