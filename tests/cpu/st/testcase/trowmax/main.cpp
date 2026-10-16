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
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>
#include <algorithm>

using namespace std;
using namespace PtoTestCommon;

class TROWMAXTest : public testing::Test {
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

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTROWMAX(T* out, T* src, void* stream);

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void test_trowmax()
{
    size_t dstFileSize = kTRows_ * sizeof(T);
    size_t srcFileSize = kTRows_ * kTCols_ * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&srcHost), srcFileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(dstDevice, dstFileSize, 0, dstFileSize);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", srcFileSize, srcHost, srcFileSize));

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTROWMAX<T, kGRows_, kGCols_, kTRows_, kTCols_>(dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize / sizeof(T));
    std::vector<T> devFinal(dstFileSize / sizeof(T));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize));

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TROWMAXTest, case_float_64x64_64x64_64x64) { test_trowmax<float, 64, 64, 64, 64>(); }
TEST_F(TROWMAXTest, case_half_64x64_64x64_64x64) { test_trowmax<aclFloat16, 64, 64, 64, 64>(); }
#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TROWMAXTest, case_bf16_64x64_64x64_64x64) { test_trowmax<bfloat16_t, 64, 64, 64, 64>(); }
#endif
TEST_F(TROWMAXTest, case_half_161x161_32x32_161x161) { test_trowmax<aclFloat16, 161, 161, 32, 32>(); }
TEST_F(TROWMAXTest, case_float_77x81_32x16_77x81) { test_trowmax<float, 77, 81, 32, 16>(); }
TEST_F(TROWMAXTest, case_float_32x32_32x16_32x32) { test_trowmax<float, 32, 32, 32, 16>(); }
TEST_F(TROWMAXTest, case_int8_64x64_64x64_64x64) { test_trowmax<int8_t, 64, 64, 64, 64>(); }
TEST_F(TROWMAXTest, case_uint8_64x64_64x64_64x64) { test_trowmax<uint8_t, 64, 64, 64, 64>(); }
TEST_F(TROWMAXTest, case_int64_64x64_64x64_64x64) { test_trowmax<int64_t, 64, 64, 64, 64>(); }
TEST_F(TROWMAXTest, case_uint64_64x64_64x64_64x64) { test_trowmax<uint64_t, 64, 64, 64, 64>(); }

using RowMaxValidSrc = pto::Tile<pto::TileType::Vec, float, 16, 16, pto::BLayout::RowMajor, -1, -1>;
using RowMaxValidDst = pto::Tile<pto::TileType::Vec, float, 8, 8, pto::BLayout::RowMajor, -1, 1>;

template <typename Src, typename Dst>
void InvokeTROWMAXForValidation(Src& src, Dst& dst)
{
    Src tmp;
    pto::TASSIGN(src, 0);
    pto::TASSIGN(tmp, Src::GetSizeInBytes());
    pto::TASSIGN(dst, Src::GetSizeInBytes() * 2);
    std::fill_n(src.data(), Src::Numel, static_cast<typename Src::DType>(1));
    std::fill_n(dst.data(), Dst::Numel, static_cast<typename Dst::DType>(123));
    pto::TROWMAX(dst, src, tmp);
}

TEST_F(TROWMAXTest, case_rejects_zero_valid_rows)
{
    RowMaxValidSrc emptyRows(0, 16);
    RowMaxValidDst dst(2);
    EXPECT_DEATH(InvokeTROWMAXForValidation(emptyRows, dst), "must be non-empty");
}

TEST_F(TROWMAXTest, case_rejects_zero_valid_cols)
{
    RowMaxValidSrc emptyCols(4, 0);
    RowMaxValidDst dst(4);
    EXPECT_DEATH(InvokeTROWMAXForValidation(emptyCols, dst), "must be non-empty");
}

TEST_F(TROWMAXTest, case_rejects_mismatched_valid_rows)
{
    RowMaxValidSrc src(4, 16);
    RowMaxValidDst dst(3);
    EXPECT_DEATH(InvokeTROWMAXForValidation(src, dst), "preserves row count");
}

TEST_F(TROWMAXTest, case_accepts_non_empty_input)
{
    RowMaxValidSrc src(2, 16);
    RowMaxValidDst dst(2);
    src.data()[0] = 5.0f;
    src.data()[1] = 9.0f;
    src.data()[2] = 3.0f;
    InvokeTROWMAXForValidation(src, dst);
    EXPECT_EQ(dst.data()[0], 9.0f);
}
