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
#include <cstdint>
#include <gtest/gtest.h>
#include "tgather_common.h"
#include <pto/pto-inst.hpp>

using namespace std;
using namespace PtoTestCommon;

inline constexpr int GATHER_ROW_DIR = 0;
inline constexpr int GATHER_COL_DIR = 1;

template <int32_t tilingKey>
void launchTGATHER_demo(uint8_t *out, uint8_t *src, void *stream);

constexpr int HALF_SIZE = 2;
constexpr int QUARTER_SIZE = 4;
class TGATHERTest : public testing::Test {
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
    std::cout << fullPath << std::endl;
    return fullPath;
}

template <typename T, uint8_t PATTERN, uint32_t ROW, uint32_t COL, uint32_t OUT_DIM, int DIRECTION>
void execute_gather_test(const std::string &goldenDir)
{
    constexpr size_t srcRows = ROW;
    constexpr size_t srcCols = COL;
    constexpr size_t dstRows = (DIRECTION == GATHER_COL_DIR) ? OUT_DIM : ROW;
    constexpr size_t dstCols = (DIRECTION == GATHER_COL_DIR) ? COL : OUT_DIM;

    constexpr size_t srcSize = srcRows * srcCols * sizeof(T);
    constexpr size_t dstSize = dstRows * dstCols * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    uint8_t *dstHost, *src0Host;
    uint8_t *dstDevice, *src0Device;

    aclrtMallocHost((void **)(&dstHost), dstSize);
    aclrtMallocHost((void **)(&src0Host), srcSize);
    aclrtMalloc((void **)&dstDevice, srcSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, srcSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t srcSizeVar = srcSize;
    ReadFile(goldenDir + "/x1_gm.bin", srcSizeVar, src0Host, srcSize);
    aclrtMemcpy(src0Device, srcSize, src0Host, srcSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTGATHER_demo<PATTERN>(dstDevice, src0Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstSize, dstDevice, dstSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(goldenDir + "/output_z.bin", dstHost, dstSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(dstSize / sizeof(float));
    std::vector<float> devFinal(dstSize / sizeof(float));
    size_t readSize = dstSize;
    ReadFile(goldenDir + "/golden.bin", readSize, golden.data(), dstSize);
    readSize = dstSize;
    ReadFile(goldenDir + "/output_z.bin", readSize, devFinal.data(), dstSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

template <typename T, uint8_t PATTERN, uint32_t ROW, uint32_t COL>
void test_gather()
{
    constexpr uint32_t dstCols = []() constexpr {
        if constexpr (PATTERN == HP1111 || PATTERN == FP1111 || PATTERN == I32P1111) {
            return COL;
        } else if constexpr (PATTERN == HP0101 || PATTERN == HP1010 || PATTERN == FP0101 || PATTERN == FP1010 ||
                             PATTERN == U16P0101 || PATTERN == U16P1010) {
            return COL / 2;
        } else {
            return COL / 4;
        }
    }();

    execute_gather_test<T, PATTERN, ROW, COL, dstCols, GATHER_ROW_DIR>(GetGoldenDir());
}

template <typename T, uint8_t PATTERN, uint32_t ROW, uint32_t COL, uint32_t OUT_ROW>
void test_gather_col()
{
    execute_gather_test<T, PATTERN, ROW, COL, OUT_ROW, GATHER_COL_DIR>(GetGoldenDir());
}

TEST_F(TGATHERTest, case1_float_P0101)
{
    test_gather<float, FP0101, FLOAT_P0101_ROW, FLOAT_P0101_COL>();
}

TEST_F(TGATHERTest, case1_float_P1010)
{
    test_gather<float, FP1010, FLOAT_P1010_ROW, FLOAT_P1010_COL>();
}

TEST_F(TGATHERTest, case1_float_P0001)
{
    test_gather<float, FP0001, FLOAT_P0001_ROW, FLOAT_P0001_COL>();
}

TEST_F(TGATHERTest, case1_float_P0010)
{
    test_gather<float, FP0010, FLOAT_P0010_ROW, FLOAT_P0010_COL>();
}

TEST_F(TGATHERTest, case1_float_P0100)
{
    test_gather<float, FP0100, FLOAT_P0100_ROW, FLOAT_P0100_COL>();
}

TEST_F(TGATHERTest, case1_float_P1000)
{
    test_gather<float, FP1000, FLOAT_P1000_ROW, FLOAT_P1000_COL>();
}

TEST_F(TGATHERTest, case1_float_P1111)
{
    test_gather<float, FP1111, FLOAT_P1111_ROW, FLOAT_P1111_COL>();
}

TEST_F(TGATHERTest, case1_half_P0101)
{
    test_gather<uint16_t, HP0101, HALF_P0101_ROW, HALF_P0101_COL>();
}

TEST_F(TGATHERTest, case1_half_P1010)
{
    test_gather<uint16_t, HP1010, HALF_P1010_ROW, HALF_P1010_COL>();
}

TEST_F(TGATHERTest, case1_half_P0001)
{
    test_gather<uint16_t, HP0001, HALF_P0001_ROW, HALF_P0001_COL>();
}

TEST_F(TGATHERTest, case1_half_P0100)
{
    test_gather<uint16_t, HP0100, HALF_P0100_ROW, HALF_P0100_COL>();
}

TEST_F(TGATHERTest, case1_half_P1000)
{
    test_gather<uint16_t, HP1000, HALF_P1000_ROW, HALF_P1000_COL>();
}

TEST_F(TGATHERTest, case1_U16_P0101)
{
    test_gather<uint16_t, U16P0101, HALF_P0101_ROW, HALF_P0101_COL>();
}

TEST_F(TGATHERTest, case1_U16_P1010)
{
    test_gather<uint16_t, U16P1010, HALF_P1010_ROW, HALF_P1010_COL>();
}

TEST_F(TGATHERTest, case1_I16_P0001)
{
    test_gather<uint16_t, I16P0001, HALF_P0001_ROW, HALF_P0001_COL>();
}

TEST_F(TGATHERTest, case1_I16_P0010)
{
    test_gather<uint16_t, I16P0010, HALF_P0010_ROW, HALF_P0010_COL>();
}

TEST_F(TGATHERTest, case1_U32_P0100)
{
    test_gather<uint32_t, U32P0100, FLOAT_P0100_ROW, FLOAT_P0100_COL>();
}

TEST_F(TGATHERTest, case1_I32_P1000)
{
    test_gather<int32_t, I32P1000, FLOAT_P1000_ROW, FLOAT_P1000_COL>();
}

TEST_F(TGATHERTest, case1_I32_P1111)
{
    test_gather<int32_t, I32P1111, FLOAT_P1111_ROW, FLOAT_P1111_COL>();
}

TEST_F(TGATHERTest, case_col_float_P0101)
{
    test_gather_col<float, COL_FP0101, COL_HALF_P0101_ROW, COL_HALF_P0101_COL, COL_HALF_P0101_ROW / 2>();
}

TEST_F(TGATHERTest, case_col_float_P1010)
{
    test_gather_col<float, COL_FP1010, COL_HALF_P1010_ROW, COL_HALF_P1010_COL, COL_HALF_P1010_ROW / 2>();
}

TEST_F(TGATHERTest, case_col_float_P0001)
{
    test_gather_col<float, COL_FP0001, COL_HALF_P0001_ROW, COL_HALF_P0001_COL, COL_HALF_P0001_ROW / 4>();
}

TEST_F(TGATHERTest, case_col_float_P0010)
{
    test_gather_col<float, COL_FP0010, COL_HALF_P0010_ROW, COL_HALF_P0010_COL, COL_HALF_P0010_ROW / 4>();
}

TEST_F(TGATHERTest, case_col_float_P0100)
{
    test_gather_col<float, COL_FP0100, COL_HALF_P0100_ROW, COL_HALF_P0100_COL, COL_HALF_P0100_ROW / 4>();
}

TEST_F(TGATHERTest, case_col_float_P1000)
{
    test_gather_col<float, COL_FP1000, COL_HALF_P1000_ROW, COL_HALF_P1000_COL, COL_HALF_P1000_ROW / 4>();
}

TEST_F(TGATHERTest, case_col_float_P1111)
{
    test_gather_col<float, COL_FP1111, COL_HALF_P1111_ROW, COL_HALF_P1111_COL, COL_HALF_P1111_ROW>();
}

TEST_F(TGATHERTest, case_col_half_P0101)
{
    test_gather_col<uint16_t, COL_HP0101, COL_HALF_P0101_ROW, COL_HALF_P0101_COL, COL_HALF_P0101_ROW / 2>();
}

TEST_F(TGATHERTest, case_col_half_P1010)
{
    test_gather_col<uint16_t, COL_HP1010, COL_HALF_P1010_ROW, COL_HALF_P1010_COL, COL_HALF_P1010_ROW / 2>();
}

TEST_F(TGATHERTest, case_col_half_P0001)
{
    test_gather_col<uint16_t, COL_HP0001, COL_HALF_P0001_ROW, COL_HALF_P0001_COL, COL_HALF_P0001_ROW / 4>();
}

TEST_F(TGATHERTest, case_col_half_P0010)
{
    test_gather_col<uint16_t, COL_HP0010, COL_HALF_P0010_ROW, COL_HALF_P0010_COL, COL_HALF_P0010_ROW / 4>();
}

TEST_F(TGATHERTest, case_col_half_P0100)
{
    test_gather_col<uint16_t, COL_HP0100, COL_HALF_P0100_ROW, COL_HALF_P0100_COL, COL_HALF_P0100_ROW / 4>();
}

TEST_F(TGATHERTest, case_col_half_P1000)
{
    test_gather_col<uint16_t, COL_HP1000, COL_HALF_P1000_ROW, COL_HALF_P1000_COL, COL_HALF_P1000_ROW / 4>();
}

TEST_F(TGATHERTest, case_col_half_P1111)
{
    test_gather_col<uint16_t, COL_HP1111, COL_HALF_P1111_ROW, COL_HALF_P1111_COL, COL_HALF_P1111_ROW>();
}

// Gather 1D tests
void launchTGATHER1D_demo_float(float *out, float *src0, int32_t *src1, aclrtStream stream);
void launchTGATHER1D_demo_int32(int32_t *out, int32_t *src0, int32_t *src1, aclrtStream stream);
void launchTGATHER1D_demo_half(int16_t *out, int16_t *src0, int32_t *src1, aclrtStream stream);
void launchTGATHER1D_demo_int16(int16_t *out, int16_t *src0, int32_t *src1, aclrtStream stream);

TEST_F(TGATHERTest, case_1D_float_32x1024_16x64)
{
    size_t src0FileSize = 32 * 1024 * sizeof(float);
    size_t src1FileSize = 16 * 64 * sizeof(int32_t);
    size_t dstFileSize = 16 * 64 * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    float *dstHost, *src0Host;
    int32_t *src1Host;
    float *dstDevice, *src0Device;
    int32_t *src1Device;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&src0Host), src0FileSize);
    aclrtMallocHost((void **)(&src1Host), src1FileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/src0.bin", src0FileSize, src0Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/src1.bin", src1FileSize, src1Host, src1FileSize);

    aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTGATHER1D_demo_float(dstDevice, src0Device, src1Device, stream);

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

    std::vector<float> golden(dstFileSize);
    std::vector<float> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<float>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TGATHERTest, case_1D_int32_32x512_16x256)
{
    size_t src0FileSize = 32 * 512 * sizeof(int32_t);
    size_t src1FileSize = 16 * 256 * sizeof(int32_t);
    size_t dstFileSize = 16 * 256 * sizeof(int32_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int32_t *dstHost, *src0Host;
    int32_t *src1Host;
    int32_t *dstDevice, *src0Device;
    int32_t *src1Device;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&src0Host), src0FileSize);
    aclrtMallocHost((void **)(&src1Host), src1FileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/src0.bin", src0FileSize, src0Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/src1.bin", src1FileSize, src1Host, src1FileSize);

    aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTGATHER1D_demo_int32(dstDevice, src0Device, src1Device, stream);

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

    std::vector<int32_t> golden(dstFileSize);
    std::vector<int32_t> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<int32_t>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TGATHERTest, case_1D_half_16x1024_16x128)
{
    size_t src0FileSize = 16 * 1024 * sizeof(int16_t);
    size_t src1FileSize = 16 * 128 * sizeof(int32_t);
    size_t dstFileSize = 16 * 128 * sizeof(int16_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int16_t *dstHost, *src0Host;
    int32_t *src1Host;
    int16_t *dstDevice, *src0Device;
    int32_t *src1Device;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&src0Host), src0FileSize);
    aclrtMallocHost((void **)(&src1Host), src1FileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/src0.bin", src0FileSize, src0Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/src1.bin", src1FileSize, src1Host, src1FileSize);

    aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTGATHER1D_demo_half(dstDevice, src0Device, src1Device, stream);

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

    std::vector<aclFloat16> golden(dstFileSize);
    std::vector<aclFloat16> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<aclFloat16>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TGATHERTest, case_1D_int16_32x256_32x64)
{
    size_t src0FileSize = 32 * 256 * sizeof(int16_t);
    size_t src1FileSize = 32 * 64 * sizeof(int32_t);
    size_t dstFileSize = 32 * 64 * sizeof(int16_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int16_t *dstHost, *src0Host;
    int32_t *src1Host;
    int16_t *dstDevice, *src0Device;
    int32_t *src1Device;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&src0Host), src0FileSize);
    aclrtMallocHost((void **)(&src1Host), src1FileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/src0.bin", src0FileSize, src0Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/src1.bin", src1FileSize, src1Host, src1FileSize);

    aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTGATHER1D_demo_int16(dstDevice, src0Device, src1Device, stream);

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

    std::vector<int16_t> golden(dstFileSize);
    std::vector<int16_t> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<int16_t>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}
