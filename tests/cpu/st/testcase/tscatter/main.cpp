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
#include "tscatter_common.h"
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TSCATTERTest : public testing::Test {
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
    return "../" + suiteName + "." + caseName;
}

template <int kTRows_, int kTCols_>
void LaunchTScatter(float *out, float *src, uint16_t *idx, void *stream);

template <int kTRows_, int kTCols_>
void test_tscatter()
{
    const size_t tileBytes = kTRows_ * kTCols_ * sizeof(float);
    const size_t idxBytes = kTRows_ * kTCols_ * sizeof(uint16_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    float *dstHost, *srcHost;
    uint16_t *idxHost;
    float *dstDevice, *srcDevice;
    uint16_t *idxDevice;

    aclrtMallocHost((void **)(&dstHost), tileBytes);
    aclrtMallocHost((void **)(&srcHost), tileBytes);
    aclrtMallocHost((void **)(&idxHost), idxBytes);

    aclrtMalloc((void **)&dstDevice, tileBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, tileBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&idxDevice, idxBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t tileSize = tileBytes;
    size_t idxSize = idxBytes;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", tileSize, srcHost, tileBytes));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input2.bin", idxSize, idxHost, idxBytes));

    aclrtMemcpy(srcDevice, tileBytes, srcHost, tileBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(idxDevice, idxBytes, idxHost, idxBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTScatter<kTRows_, kTCols_>(dstDevice, srcDevice, idxDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, tileBytes, dstDevice, tileBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, tileBytes);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(idxDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(idxHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(tileBytes / sizeof(float));
    std::vector<float> devFinal(tileBytes / sizeof(float));
    tileSize = tileBytes;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", tileSize, golden.data(), tileBytes));
    tileSize = tileBytes;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", tileSize, devFinal.data(), tileBytes));
    EXPECT_TRUE(ResultCmp<float>(golden, devFinal, 0.001f));
}

TEST_F(TSCATTERTest, case_float_16x16_16x16_16x16)
{
    test_tscatter<16, 16>();
}

// --- Mask-pattern TSCATTER tests ---

template <int32_t tilingKey>
void launchTSCATTER_masked(uint8_t *out, uint8_t *src, void *stream);

template <typename T, uint8_t PATTERN, uint32_t ROW, uint32_t DST_COL, uint32_t MASK_DIVISOR>
void test_scatter_masked()
{
    constexpr uint32_t SRC_COL = DST_COL / MASK_DIVISOR;
    size_t srcSize = ROW * SRC_COL * sizeof(T);
    size_t dstSize = ROW * DST_COL * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *srcHost;
    uint8_t *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstSize);
    aclrtMallocHost((void **)(&srcHost), srcSize);
    aclrtMalloc((void **)&dstDevice, dstSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t readSize = srcSize;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/x1_gm.bin", readSize, srcHost, srcSize));

    aclrtMemcpy(srcDevice, srcSize, srcHost, srcSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTSCATTER_masked<PATTERN>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstSize, dstDevice, dstSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, dstSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    constexpr size_t numElements = ROW * DST_COL;
    std::vector<T> golden(numElements);
    std::vector<T> devFinal(numElements);
    readSize = dstSize;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", readSize, golden.data(), dstSize));
    readSize = dstSize;
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output_z.bin", readSize, devFinal.data(), dstSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

// float
TEST_F(TSCATTERTest, case_masked_float_P0101)
{
    test_scatter_masked<float, FP0101, FLOAT_P0101_ROW, FLOAT_P0101_COL, 2>();
}

TEST_F(TSCATTERTest, case_masked_float_P1010)
{
    test_scatter_masked<float, FP1010, FLOAT_P1010_ROW, FLOAT_P1010_COL, 2>();
}

TEST_F(TSCATTERTest, case_masked_float_P0001)
{
    test_scatter_masked<float, FP0001, FLOAT_P0001_ROW, FLOAT_P0001_COL, 4>();
}

TEST_F(TSCATTERTest, case_masked_float_P0010)
{
    test_scatter_masked<float, FP0010, FLOAT_P0010_ROW, FLOAT_P0010_COL, 4>();
}

TEST_F(TSCATTERTest, case_masked_float_P0100)
{
    test_scatter_masked<float, FP0100, FLOAT_P0100_ROW, FLOAT_P0100_COL, 4>();
}

TEST_F(TSCATTERTest, case_masked_float_P1000)
{
    test_scatter_masked<float, FP1000, FLOAT_P1000_ROW, FLOAT_P1000_COL, 4>();
}

TEST_F(TSCATTERTest, case_masked_float_P1111)
{
    test_scatter_masked<float, FP1111, FLOAT_P1111_ROW, FLOAT_P1111_COL, 1>();
}

// half
TEST_F(TSCATTERTest, case_masked_half_P0101)
{
    test_scatter_masked<half, HP0101, HALF_P0101_ROW, HALF_P0101_COL, 2>();
}

TEST_F(TSCATTERTest, case_masked_half_P1010)
{
    test_scatter_masked<half, HP1010, HALF_P1010_ROW, HALF_P1010_COL, 2>();
}

TEST_F(TSCATTERTest, case_masked_half_P0001)
{
    test_scatter_masked<half, HP0001, HALF_P0001_ROW, HALF_P0001_COL, 4>();
}

TEST_F(TSCATTERTest, case_masked_half_P0100)
{
    test_scatter_masked<half, HP0100, HALF_P0100_ROW, HALF_P0100_COL, 4>();
}

TEST_F(TSCATTERTest, case_masked_half_P1000)
{
    test_scatter_masked<half, HP1000, HALF_P1000_ROW, HALF_P1000_COL, 4>();
}

// uint16 / int16
TEST_F(TSCATTERTest, case_masked_U16_P0101)
{
    test_scatter_masked<uint16_t, U16P0101, HALF_P0101_ROW, HALF_P0101_COL, 2>();
}

TEST_F(TSCATTERTest, case_masked_U16_P1010)
{
    test_scatter_masked<uint16_t, U16P1010, HALF_P1010_ROW, HALF_P1010_COL, 2>();
}

TEST_F(TSCATTERTest, case_masked_I16_P0001)
{
    test_scatter_masked<int16_t, I16P0001, HALF_P0001_ROW, HALF_P0001_COL, 4>();
}

TEST_F(TSCATTERTest, case_masked_I16_P0010)
{
    test_scatter_masked<int16_t, I16P0010, HALF_P0010_ROW, HALF_P0010_COL, 4>();
}

// uint32 / int32
TEST_F(TSCATTERTest, case_masked_U32_P0100)
{
    test_scatter_masked<uint32_t, U32P0100, FLOAT_P0100_ROW, FLOAT_P0100_COL, 4>();
}

TEST_F(TSCATTERTest, case_masked_I32_P1000)
{
    test_scatter_masked<int32_t, I32P1000, FLOAT_P1000_ROW, FLOAT_P1000_COL, 4>();
}

TEST_F(TSCATTERTest, case_masked_I32_P1111)
{
    test_scatter_masked<int32_t, I32P1111, FLOAT_P1111_ROW, FLOAT_P1111_COL, 1>();
}
