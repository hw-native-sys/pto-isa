/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include <climits>
#include <cstring>
#include "acl/acl.h"
#include "test_common.h"

using namespace std;
using namespace PtoTestCommon;

template <typename T, int validRows, int validCols, int upperOrLower>
void LaunchTTri(T* out, int diagonal, void* stream);
template <typename T, int staticRows, int staticCols, int validRows, int validCols, int upperOrLower>
void LaunchTTriDyn(T* out, int diagonal, void* stream);

class TTRITest : public testing::Test {
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

template <typename T, int validRows, int validCols, int upperOrLower>
void test_ttri(int diagonal)
{
    size_t fileSize = validRows * validCols * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T* dstHost;
    T* dstDevice;

    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMalloc((void**)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    LaunchTTri<T, validRows, validCols, upperOrLower>(dstDevice, diagonal, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);

    aclrtFreeHost(dstHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize / sizeof(T));
    std::vector<T> devFinal(fileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize);

    bool ret;
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        ret = ResultCmpExact(golden, devFinal.data());
    } else {
        ret = ResultCmp<T>(golden, devFinal, 0.001f);
    }

    EXPECT_TRUE(ret);
}

TEST_F(TTRITest, case_fp16_20x32_lower_diag_0) { test_ttri<aclFloat16, 20, 32, 0>(0); }
TEST_F(TTRITest, case_uint8_20x32_lower_diag_0) { test_ttri<uint8_t, 20, 32, 0>(0); }
TEST_F(TTRITest, case_float_32x91_lower_diag_0) { test_ttri<float, 32, 91, 0>(0); }
TEST_F(TTRITest, case_float_128x128_lower_diag_0) { test_ttri<float, 128, 128, 0>(0); }
TEST_F(TTRITest, case_float_32x91_lower_diag_3) { test_ttri<float, 32, 91, 0>(3); }
TEST_F(TTRITest, case_float_128x128_lower_diag_3) { test_ttri<float, 128, 128, 0>(3); }
TEST_F(TTRITest, case_float_32x91_lower_diag_n3) { test_ttri<float, 32, 91, 0>(-3); }
TEST_F(TTRITest, case_float_128x128_lower_diag_n3) { test_ttri<float, 128, 128, 0>(-3); }
TEST_F(TTRITest, case_float_32x91_upper_diag_0) { test_ttri<float, 32, 91, 1>(0); }
TEST_F(TTRITest, case_float_128x128_upper_diag_0) { test_ttri<float, 128, 128, 1>(0); }
TEST_F(TTRITest, case_float_32x91_upper_diag_3) { test_ttri<float, 32, 91, 1>(3); }
TEST_F(TTRITest, case_float_128x128_upper_diag_3) { test_ttri<float, 128, 128, 1>(3); }
TEST_F(TTRITest, case_float_32x91_upper_diag_n3) { test_ttri<float, 32, 91, 1>(-3); }
TEST_F(TTRITest, case_float_128x128_upper_diag_n3) { test_ttri<float, 128, 128, 1>(-3); }
TEST_F(TTRITest, case_float_763x32_lower_diag_n41) { test_ttri<float, 763, 32, 0>(-41); }
TEST_F(TTRITest, case_float_763x32_upper_diag_n41) { test_ttri<float, 763, 32, 1>(-41); }
TEST_F(TTRITest, case_int64_4x15_upper_diag_0) { test_ttri<int64_t, 4, 15, 1>(0); }
TEST_F(TTRITest, case_uint64_4x15_lower_diag_n1) { test_ttri<uint64_t, 4, 15, 0>(-1); }
TEST_F(TTRITest, case_int64_4x64_upper_diag_0) { test_ttri<int64_t, 4, 64, 1>(0); }
TEST_F(TTRITest, case_uint64_4x64_lower_diag_n1) { test_ttri<uint64_t, 4, 64, 0>(-1); }

// --- Dynamic (static != valid) test cases ---

template <typename T, int staticRows, int staticCols, int validRows, int validCols, int upperOrLower>
void test_ttri_dyn(int diagonal)
{
    size_t fileSize = validRows * validCols * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T* dstHost;
    T* dstDevice;

    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMalloc((void**)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    LaunchTTriDyn<T, staticRows, staticCols, validRows, validCols, upperOrLower>(dstDevice, diagonal, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);

    aclrtFreeHost(dstHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize / sizeof(T));
    std::vector<T> devFinal(fileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize);

    bool ret;
    if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
        ret = ResultCmpExact(golden, devFinal.data());
    } else {
        ret = ResultCmp<T>(golden, devFinal, 0.001f);
    }

    EXPECT_TRUE(ret);
}

TEST_F(TTRITest, case_fp16_s30x208_v30x208_upper_diag_0) { test_ttri_dyn<aclFloat16, 30, 208, 30, 208, 1>(0); }
TEST_F(TTRITest, case_fp16_s30x208_v30x176_upper_diag_0) { test_ttri_dyn<aclFloat16, 30, 208, 30, 176, 1>(0); }
TEST_F(TTRITest, case_fp16_s293x16_v269x16_lower_diag_n41) { test_ttri_dyn<aclFloat16, 293, 16, 269, 16, 0>(-41); }
TEST_F(TTRITest, case_fp16_s293x16_v293x16_lower_diag_n41) { test_ttri_dyn<aclFloat16, 293, 16, 293, 16, 0>(-41); }
TEST_F(TTRITest, case_fp16_s293x16_v287x16_lower_diag_n41) { test_ttri_dyn<aclFloat16, 293, 16, 287, 16, 0>(-41); }
TEST_F(TTRITest, case_int8_s32x128_v32x128_lower_diag_0) { test_ttri_dyn<int8_t, 32, 128, 32, 128, 0>(0); }
TEST_F(TTRITest, case_int8_s32x128_v24x112_lower_diag_0) { test_ttri_dyn<int8_t, 32, 128, 24, 112, 0>(0); }
TEST_F(TTRITest, case_fp16_s293x16_v1x16_lower_diag_0) { test_ttri_dyn<aclFloat16, 293, 16, 1, 16, 0>(0); }
TEST_F(TTRITest, case_fp16_s293x16_v2x16_lower_diag_0) { test_ttri_dyn<aclFloat16, 293, 16, 2, 16, 0>(0); }

template <typename T, int kind, int rows, int cols, int validRows, int validCols, int upperOrLower, int diagonal>
void LaunchTTriGuard(T* out, void* stream);

template <typename T, int kind, int rows, int cols, int validRows, int validCols, int upperOrLower, int diagonal>
void test_ttri_guard()
{
    constexpr int guardRows = (512 + cols * sizeof(T) - 1) / (cols * sizeof(T));
    constexpr size_t fileSize = (rows + guardRows) * cols * sizeof(T);
    std::vector<uint8_t> expected(fileSize, 0x5a);
    std::vector<uint8_t> actual(fileSize);
    for (int row = 0; row < validRows; ++row) {
        for (int col = 0; col < validCols; ++col) {
            const int64_t boundary = static_cast<int64_t>(row) + diagonal;
            const bool selected = upperOrLower ? col >= boundary : col <= boundary;
            T value = static_cast<T>(selected);
            if constexpr (kind == 1) {
                value = static_cast<T>(selected ? 0x3c00 : 0);
            } else if constexpr (kind == 2) {
                value = static_cast<T>(selected ? 0x3f80 : 0);
            }
            std::memcpy(expected.data() + (row * cols + col) * sizeof(T), &value, sizeof(T));
        }
    }

    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
    ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    aclrtStream stream;
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS);
    T* dstDevice;
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&dstDevice), fileSize, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    EXPECT_EQ(aclrtMemset(dstDevice, fileSize, 0x5a, fileSize), ACL_SUCCESS);
    LaunchTTriGuard<T, kind, rows, cols, validRows, validCols, upperOrLower, diagonal>(dstDevice, stream);
    EXPECT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    EXPECT_EQ(aclrtMemcpy(actual.data(), fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    EXPECT_EQ(aclrtFree(dstDevice), ACL_SUCCESS);
    EXPECT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS);
    EXPECT_EQ(aclrtResetDevice(0), ACL_SUCCESS);
    EXPECT_EQ(aclFinalize(), ACL_SUCCESS);

    // Check valid elements, row padding, invalid rows, and the trailing guard byte for byte.
    for (size_t byte = 0; byte < fileSize; ++byte) {
        ASSERT_EQ(actual[byte], expected[byte]) << "byte offset " << byte;
    }
}

#define PTO_TTRI_GUARD_CASE(name, type, kind, rows, cols, validRows, validCols, upper, diagonal) \
    TEST_F(TTRITest, name) { test_ttri_guard<type, kind, rows, cols, validRows, validCols, upper, diagonal>(); }
#include "ttri_guard_cases.inc"
#undef PTO_TTRI_GUARD_CASE
