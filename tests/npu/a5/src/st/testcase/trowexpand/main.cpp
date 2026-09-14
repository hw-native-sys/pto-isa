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

namespace TRowExpandTest {
template <typename T, uint32_t rows, uint32_t srcCols, uint32_t dstValidCols, uint32_t dstCols>
void launchTROWEXPAND(T* out, T* src, void* stream);
class TROWEXPANDTest : public testing::Test {
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

template <typename T, uint32_t rows, uint32_t srcCols, uint32_t dstValidCols, uint32_t dstCols>
void test_trowexpand()
{
    size_t inputFileSize = rows * srcCols * sizeof(T);
    size_t outputFileSize = rows * dstCols * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host;
    T *dstDevice, *src0Device;

    aclrtMallocHost((void**)(&dstHost), outputFileSize);
    aclrtMallocHost((void**)(&src0Host), inputFileSize);

    aclrtMalloc((void**)&dstDevice, outputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, inputFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", inputFileSize, src0Host, inputFileSize);

    aclrtMemcpy(src0Device, inputFileSize, src0Host, inputFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTROWEXPAND<T, rows, srcCols, dstValidCols, dstCols>(dstDevice, src0Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputFileSize, dstDevice, outputFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, outputFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(outputFileSize / sizeof(T));
    std::vector<T> devFinal(outputFileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", outputFileSize, golden.data(), outputFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", outputFileSize, devFinal.data(), outputFileSize);
    bool ret;
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        ret = ResultCmpExact(golden, devFinal.data());
    } else {
        ret = ResultCmp(golden, devFinal, 0.001f);
    }

    EXPECT_TRUE(ret);
}

TEST_F(TROWEXPANDTest, case0_half_16_16_16_512) { test_trowexpand<aclFloat16, 16, 16, 512, 512>(); }
TEST_F(TROWEXPANDTest, case1_int8_16_32_16_256) { test_trowexpand<int8_t, 16, 32, 256, 256>(); }
TEST_F(TROWEXPANDTest, case2_float_16_8_16_128) { test_trowexpand<float, 16, 8, 128, 128>(); }
TEST_F(TROWEXPANDTest, case3_half_16_16_16_511) { test_trowexpand<aclFloat16, 16, 16, 511, 512>(); }
TEST_F(TROWEXPANDTest, case4_int8_16_32_16_255) { test_trowexpand<int8_t, 16, 32, 255, 256>(); }
TEST_F(TROWEXPANDTest, case5_float_16_8_16_127) { test_trowexpand<float, 16, 8, 127, 128>(); }
TEST_F(TROWEXPANDTest, case6_int64_4_16_4_16) { test_trowexpand<int64_t, 4, 16, 16, 16>(); }
TEST_F(TROWEXPANDTest, case7_uint64_4_16_4_16) { test_trowexpand<uint64_t, 4, 16, 16, 16>(); }
TEST_F(TROWEXPANDTest, case8_int64_4_16_4_64) { test_trowexpand<int64_t, 4, 16, 64, 64>(); }
TEST_F(TROWEXPANDTest, case9_uint64_4_16_4_64) { test_trowexpand<uint64_t, 4, 16, 64, 64>(); }
TEST_F(TROWEXPANDTest, case10_int64_1_1_1_16368) { test_trowexpand<int64_t, 1, 1, 16368, 16368>(); }
TEST_F(TROWEXPANDTest, case11_uint64_1_1_1_16368) { test_trowexpand<uint64_t, 1, 1, 16368, 16368>(); }

template <typename T, bool compact>
void launchTROWEXPANDGuard(T* out, T* src, void* stream);

template <typename T, bool compact>
void test_trowexpand_guard()
{
    constexpr int srcCols = compact ? 1 : 4;
    constexpr size_t inputBytes = 8 * srcCols * sizeof(T);
    constexpr size_t outputBytes = 96 * sizeof(T);
    std::vector<T> input(8 * srcCols);
    std::vector<T> expected(96, static_cast<T>(0x5a5a5a5a5a5a5a5aULL));
    std::vector<T> actual(96);
    for (int row = 0; row < 8; ++row) {
        T value = static_cast<T>((1LL << 54) + (static_cast<int64_t>(row + 1) << 32) + 19 * row + 3);
        if constexpr (std::is_same_v<T, int64_t>) {
            if (row % 3 == 0) {
                value = -value;
            }
        } else {
            value += 1ULL << 63;
        }
        for (int col = 0; col < srcCols; ++col) {
            input[row * srcCols + col] = value + col * 1000003;
        }
        if (row < 5) {
            for (int col = 0; col < 3; ++col) {
                expected[row * 4 + col] = value;
            }
        }
    }

    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
    ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    aclrtStream stream;
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS);
    T *srcDevice, *dstDevice;
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&srcDevice), inputBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&dstDevice), outputBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    EXPECT_EQ(aclrtMemcpy(srcDevice, inputBytes, input.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    EXPECT_EQ(aclrtMemset(dstDevice, outputBytes, 0x5a, outputBytes), ACL_SUCCESS);
    launchTROWEXPANDGuard<T, compact>(dstDevice, srcDevice, stream);
    EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    EXPECT_EQ(aclrtMemcpy(actual.data(), outputBytes, dstDevice, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    EXPECT_EQ(aclrtFree(srcDevice), ACL_SUCCESS);
    EXPECT_EQ(aclrtFree(dstDevice), ACL_SUCCESS);
    EXPECT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(0), ACL_SUCCESS);
    EXPECT_EQ(aclFinalize(), ACL_SUCCESS);

    // Include physical padding, inactive rows, and the 512-byte guard in the exact comparison.
    EXPECT_TRUE(ResultCmpExact(expected, actual.data()));
}

TEST_F(TROWEXPANDTest, guard_int64_dn8x1_v5x1_to8x4_v5x3) { test_trowexpand_guard<int64_t, true>(); }

TEST_F(TROWEXPANDTest, guard_int64_nd8x4_v5x1_to8x4_v5x3) { test_trowexpand_guard<int64_t, false>(); }

TEST_F(TROWEXPANDTest, guard_uint64_dn8x1_v5x1_to8x4_v5x3) { test_trowexpand_guard<uint64_t, true>(); }

TEST_F(TROWEXPANDTest, guard_uint64_nd8x4_v5x1_to8x4_v5x3) { test_trowexpand_guard<uint64_t, false>(); }
} // namespace TRowExpandTest
