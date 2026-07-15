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
#include "tgather_common.h"

using namespace std;
using namespace PtoTestCommon;

void launchTGATHER_demo_float(float* out, float* src0, int32_t* src1, aclrtStream stream);
void launchTGATHER_demo_int32(int32_t* out, int32_t* src0, int32_t* src1, aclrtStream stream);
void launchTGATHER_demo_half(int16_t* out, int16_t* src0, int16_t* src1, aclrtStream stream);
void launchTGATHER_demo_int16(int16_t* out, int16_t* src0, int16_t* src1, aclrtStream stream);

constexpr int HALF_SIZE = 2;
constexpr int QUARTER_SIZE = 4;

template <
    typename srcT, typename dstT, int kGRows_, int kGCols_, int kTRows_, int kTCols_, pto::MaskPattern maskPattern>
void LaunchTGATHER(dstT* out, srcT* src, void* stream);

class TGATHERTest : public testing::Test {
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

TEST_F(TGATHERTest, case1_float)
{
    size_t src0FileSize = 32 * 864 * sizeof(float);
    size_t src1FileSize = 16 * 64 * sizeof(int32_t);
    size_t dstFileSize = 16 * 64 * sizeof(float);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    float *dstHost, *src0Host;
    int32_t* src1Host;
    float *dstDevice, *src0Device;
    int32_t* src1Device;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&src0Host), src0FileSize);
    aclrtMallocHost((void**)(&src1Host), src1FileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/src0.bin", src0FileSize, src0Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/src1.bin", src1FileSize, src1Host, src1FileSize);
    aclrtMemset(dstHost, dstFileSize, 0, dstFileSize);

    aclrtMemcpy(dstDevice, dstFileSize, dstHost, dstFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTGATHER_demo_float(dstDevice, src0Device, src1Device, stream);

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

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TGATHERTest, case2_int32)
{
    size_t src0FileSize = 32 * 512 * sizeof(int32_t);
    size_t src1FileSize = 16 * 256 * sizeof(int32_t);
    size_t dstFileSize = 16 * 256 * sizeof(int32_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int32_t *dstHost, *src0Host;
    int32_t* src1Host;
    int32_t *dstDevice, *src0Device;
    int32_t* src1Device;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&src0Host), src0FileSize);
    aclrtMallocHost((void**)(&src1Host), src1FileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/src0.bin", src0FileSize, src0Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/src1.bin", src1FileSize, src1Host, src1FileSize);

    aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTGATHER_demo_int32(dstDevice, src0Device, src1Device, stream);

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

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TGATHERTest, case3_half)
{
    size_t src0FileSize = 16 * 1024 * sizeof(int16_t);
    size_t src1FileSize = 16 * 128 * sizeof(int16_t);
    size_t dstFileSize = 16 * 128 * sizeof(int16_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int16_t *dstHost, *src0Host;
    int16_t* src1Host;
    int16_t *dstDevice, *src0Device;
    int16_t* src1Device;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&src0Host), src0FileSize);
    aclrtMallocHost((void**)(&src1Host), src1FileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/src0.bin", src0FileSize, src0Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/src1.bin", src1FileSize, src1Host, src1FileSize);

    aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTGATHER_demo_half(dstDevice, src0Device, src1Device, stream);

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

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TGATHERTest, case4_int16)
{
    size_t src0FileSize = 32 * 256 * sizeof(int16_t);
    size_t src1FileSize = 32 * 64 * sizeof(int16_t);
    size_t dstFileSize = 32 * 64 * sizeof(int16_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    int16_t *dstHost, *src0Host;
    int16_t* src1Host;
    int16_t *dstDevice, *src0Device;
    int16_t* src1Device;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&src0Host), src0FileSize);
    aclrtMallocHost((void**)(&src1Host), src1FileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/src0.bin", src0FileSize, src0Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/src1.bin", src1FileSize, src1Host, src1FileSize);

    aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTGATHER_demo_int16(dstDevice, src0Device, src1Device, stream);

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

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename T, pto::MaskPattern PATTERN, uint32_t ROW, uint32_t COL, typename dstT = T>
void test_gather()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t size = ROW * COL * sizeof(T);
    size_t dstsize = 0;
    if constexpr (PATTERN == pto::MaskPattern::P1111) {
        dstsize = size;
    } else if constexpr (PATTERN == pto::MaskPattern::P0101 || PATTERN == pto::MaskPattern::P1010) {
        dstsize = size / HALF_SIZE;
    } else {
        dstsize = size / QUARTER_SIZE;
    }
    T *src0Host, *src0Device;
    dstT *dstHost, *dstDevice;

    aclrtMallocHost((void**)(&dstHost), dstsize);
    aclrtMallocHost((void**)(&src0Host), size);
    aclrtMalloc((void**)&dstDevice, size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, size, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", size, src0Host, size);

    aclrtMemcpy(src0Device, size, src0Host, size, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTGATHER<T, dstT, ROW, COL, ROW, COL, PATTERN>(dstDevice, src0Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstsize, dstDevice, dstsize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, dstsize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstsize);
    std::vector<T> devFinal(dstsize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstsize, golden.data(), dstsize);
    ReadFile(GetGoldenDir() + "/output_z.bin", dstsize, devFinal.data(), dstsize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TGATHERTest, case1_float_P0101)
{
    test_gather<float, pto::MaskPattern::P0101, FLOAT_P0101_ROW, FLOAT_P0101_COL>();
}

TEST_F(TGATHERTest, case1_float_P1010)
{
    test_gather<float, pto::MaskPattern::P1010, FLOAT_P1010_ROW, FLOAT_P1010_COL>();
}

TEST_F(TGATHERTest, case1_float_P0001)
{
    test_gather<float, pto::MaskPattern::P0001, FLOAT_P0001_ROW, FLOAT_P0001_COL>();
}

TEST_F(TGATHERTest, case1_float_P0010)
{
    test_gather<float, pto::MaskPattern::P0010, FLOAT_P0010_ROW, FLOAT_P0010_COL>();
}

TEST_F(TGATHERTest, case1_float_P0100)
{
    test_gather<float, pto::MaskPattern::P0100, FLOAT_P0100_ROW, FLOAT_P0100_COL>();
}

TEST_F(TGATHERTest, case1_float_P1000)
{
    test_gather<float, pto::MaskPattern::P1000, FLOAT_P1000_ROW, FLOAT_P1000_COL>();
}

TEST_F(TGATHERTest, case1_float_P1111)
{
    test_gather<float, pto::MaskPattern::P1111, FLOAT_P1111_ROW, FLOAT_P1111_COL>();
}

TEST_F(TGATHERTest, case1_float_int_P1010)
{
    test_gather<float, pto::MaskPattern::P1010, FLOAT_P1010_ROW, FLOAT_P1010_COL, int32_t>();
}

TEST_F(TGATHERTest, case1_half_P0101)
{
    test_gather<uint16_t, pto::MaskPattern::P0101, HALF_P0101_ROW, HALF_P0101_COL>();
}

TEST_F(TGATHERTest, case1_half_P1010)
{
    test_gather<uint16_t, pto::MaskPattern::P1010, HALF_P1010_ROW, HALF_P1010_COL>();
}

TEST_F(TGATHERTest, case1_half_P0001)
{
    test_gather<uint16_t, pto::MaskPattern::P0001, HALF_P0001_ROW, HALF_P0001_COL>();
}

TEST_F(TGATHERTest, case1_half_P0010)
{
    test_gather<uint16_t, pto::MaskPattern::P0010, HALF_P0010_ROW, HALF_P0010_COL>();
}

TEST_F(TGATHERTest, case1_half_P0100)
{
    test_gather<uint16_t, pto::MaskPattern::P0100, HALF_P0100_ROW, HALF_P0100_COL>();
}

TEST_F(TGATHERTest, case1_half_P1000)
{
    test_gather<uint16_t, pto::MaskPattern::P1000, HALF_P1000_ROW, HALF_P1000_COL>();
}

TEST_F(TGATHERTest, case1_half_P1111)
{
    test_gather<uint16_t, pto::MaskPattern::P1111, HALF_P1111_ROW, HALF_P1111_COL>();
}

TEST_F(TGATHERTest, case1_U16_P0101)
{
    test_gather<uint16_t, pto::MaskPattern::P0101, HALF_P0101_ROW, HALF_P0101_COL>();
}

TEST_F(TGATHERTest, case1_U16_P1010)
{
    test_gather<uint16_t, pto::MaskPattern::P1010, HALF_P1010_ROW, HALF_P1010_COL>();
}

TEST_F(TGATHERTest, case1_I16_P0001)
{
    test_gather<int16_t, pto::MaskPattern::P0001, HALF_P0001_ROW, HALF_P0001_COL>();
}

TEST_F(TGATHERTest, case1_I16_P0010)
{
    test_gather<int16_t, pto::MaskPattern::P0010, HALF_P0010_ROW, HALF_P0010_COL>();
}

TEST_F(TGATHERTest, case1_U32_P0100)
{
    test_gather<uint32_t, pto::MaskPattern::P0100, FLOAT_P0100_ROW, FLOAT_P0100_COL>();
}

TEST_F(TGATHERTest, case1_I32_P1000)
{
    test_gather<int32_t, pto::MaskPattern::P1000, FLOAT_P1000_ROW, FLOAT_P1000_COL>();
}

TEST_F(TGATHERTest, case1_I32_P1111)
{
    test_gather<int32_t, pto::MaskPattern::P1111, FLOAT_P1111_ROW, FLOAT_P1111_COL>();
}

TEST_F(TGATHERTest, case1_b8_P0101) { test_gather<int8_t, pto::MaskPattern::P0101, HALF_P0101_ROW, HALF_P0101_COL>(); }

TEST_F(TGATHERTest, case1_b8_P1010) { test_gather<uint8_t, pto::MaskPattern::P1010, HALF_P1010_ROW, HALF_P1010_COL>(); }

TEST_F(TGATHERTest, case1_b8_P0001) { test_gather<int8_t, pto::MaskPattern::P0001, HALF_P0001_ROW, HALF_P0001_COL>(); }

TEST_F(TGATHERTest, case1_b8_P0010) { test_gather<uint8_t, pto::MaskPattern::P0010, HALF_P0010_ROW, HALF_P0010_COL>(); }

TEST_F(TGATHERTest, case1_b8_P0100) { test_gather<int8_t, pto::MaskPattern::P0100, HALF_P0100_ROW, HALF_P0100_COL>(); }

TEST_F(TGATHERTest, case1_b8_P1000) { test_gather<uint8_t, pto::MaskPattern::P1000, HALF_P1000_ROW, HALF_P1000_COL>(); }

TEST_F(TGATHERTest, case1_b8_P1111) { test_gather<int8_t, pto::MaskPattern::P1111, HALF_P1111_ROW, HALF_P1111_COL>(); }
