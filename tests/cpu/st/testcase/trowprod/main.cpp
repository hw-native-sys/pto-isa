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
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>
#include <array>
#include <bit>

using namespace std;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void launchTROWPROD_demo(uint8_t* out, uint8_t* src, void* stream);

class TROWPRODTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override { pto::NPU_MEMORY_INIT(pto::NPUArch::A2A3); }
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTROWPROD(T* out, T* src, void* stream);

template <typename T, int kGSize_>
inline void init_dst(T* dstHost)
{
    for (size_t i = 0; i < kGSize_; i++) {
        dstHost[i] = 0;
    }
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void test_trowprod()
{
    size_t fileSize = kGRows_ * kGCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMallocHost((void**)(&srcHost), fileSize);

    aclrtMalloc((void**)(&dstDevice), fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)(&srcDevice), fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(dstDevice, fileSize, 0, fileSize);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input.bin", fileSize, srcHost, fileSize));
    init_dst<T, kGRows_ * kGCols_>(dstHost);

    aclrtMemcpy(srcDevice, fileSize, srcHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTROWPROD<T, kGRows_, kGCols_, kTRows_, kTCols_>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize / sizeof(T));
    std::vector<T> devFinal(fileSize / sizeof(T));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TROWPRODTest, case_float_64x64_64x64_64x64) { test_trowprod<float, 64, 64, 64, 64>(); }
TEST_F(TROWPRODTest, case_half_16x256_16x256_16x256) { test_trowprod<aclFloat16, 16, 256, 16, 256>(); }

template <typename T, int Rows, int Cols>
void ExpectRowProdFp32(const std::array<T, Cols>& rowValues, int validCols, T expected)
{
    using Src = pto::Tile<pto::TileType::Vec, T, Rows, Cols, pto::BLayout::RowMajor, Rows, -1>;
    using Dst = pto::Tile<pto::TileType::Vec, T, Rows, Cols, pto::BLayout::RowMajor, Rows, -1>;
    pto::NPU_MEMORY_INIT(pto::NPUArch::A5);
    Src src(validCols);
    Src tmp(validCols);
    Dst dst(validCols);
    pto::TASSIGN(src, 0);
    for (int row = 0; row < Rows; ++row) {
        std::copy(rowValues.begin(), rowValues.end(), src.data() + row * Cols);
    }
    const T sentinel = static_cast<T>(123);
    pto::TASSIGN(dst, Src::GetSizeInBytes());
    std::fill_n(dst.data(), Dst::Numel, sentinel);
    pto::TROWPROD(dst, src, tmp);
    for (int row = 0; row < Rows; ++row) {
        ASSERT_EQ(std::bit_cast<uint16_t>(dst.data()[row * Cols]), std::bit_cast<uint16_t>(expected)) << "row=" << row;
        for (int col = 1; col < Cols; ++col) {
            EXPECT_EQ(dst.data()[row * Cols + col], sentinel);
        }
    }
}

TEST_F(TROWPRODTest, case_half_fp32_accumulation)
{
    // a = b = 1 + 2^-10, c = 1.25. Stepwise half rounding gives 0x3d02
    // (1.251953125); fp32 accumulation rounds the full product a*b*c once to
    // 0x3d03 (1.2529296875).
    const half a = std::bit_cast<half>(static_cast<uint16_t>(0x3c01));
    const half c = std::bit_cast<half>(static_cast<uint16_t>(0x3d00));
    const half one = std::bit_cast<half>(static_cast<uint16_t>(0x3c00));
    const half expected = std::bit_cast<half>(static_cast<uint16_t>(0x3d03));
    const std::array<half, 16> rowValues = {a, a, c, one, one, one, one, one, one, one, one, one, one, one, one, one};
    ExpectRowProdFp32<half, 4, 16>(rowValues, 16, expected);
}
