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
#include <pto/pto-inst.hpp>

using namespace std;
using namespace pto;
using namespace PtoTestCommon;

template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_, pto::SaturationMode saturation>
void launchTCVT(D* dst, S* src, void* stream);
template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_, pto::SaturationMode saturation>
void launchTCVTWithTmp(D* dst, S* src, void* stream);
template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void launchTCVTWithTmpDefault(D* dst, S* src, void* stream);

class TCVTTest : public testing::Test {
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

template <typename T>
constexpr size_t StorageElements(size_t elements)
{
    if constexpr (IsTwinType<T>()) {
        return (elements + 1) / 2;
    } else {
        return elements;
    }
}

template <
    typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_,
    pto::SaturationMode saturation = pto::SaturationMode::OFF, bool withTmp = false, bool explicitSaturation = true>
void test_tcvt(const std::string& goldenDir = GetGoldenDir())
{
    uint32_t M = kGRows_;
    uint32_t N = kGCols_;

    size_t numElements = M * N;
    size_t srcFileSize = StorageElements<S>(numElements) * sizeof(S);
    size_t dstFileSize = StorageElements<D>(numElements) * sizeof(D);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    D *dstHost, *dstDevice;
    S *srcHost, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&srcHost), srcFileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(goldenDir + "/x1_gm.bin", srcFileSize, srcHost, srcFileSize));

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if constexpr (!withTmp) {
        launchTCVT<D, S, kGRows_, kGCols_, kTRows_, kTCols_, saturation>(dstDevice, srcDevice, stream);
    } else if constexpr (explicitSaturation) {
        launchTCVTWithTmp<D, S, kGRows_, kGCols_, kTRows_, kTCols_, saturation>(dstDevice, srcDevice, stream);
    } else {
        launchTCVTWithTmpDefault<D, S, kGRows_, kGCols_, kTRows_, kTCols_>(dstDevice, srcDevice, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(goldenDir + "/output_z.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<D> golden(dstFileSize / sizeof(D));
    std::vector<D> devFinal(dstFileSize / sizeof(D));
    CHECK_RESULT_GTEST(ReadFile(goldenDir + "/golden.bin", dstFileSize, golden.data(), dstFileSize));
    CHECK_RESULT_GTEST(ReadFile(goldenDir + "/output_z.bin", dstFileSize, devFinal.data(), dstFileSize));

    bool ret = ResultCmp<D>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TCVTTest, case1) { test_tcvt<int32_t, float, 128, 128, 128, 128>(); }

TEST_F(TCVTTest, case2) { test_tcvt<float, int32_t, 256, 64, 256, 64>(); }

TEST_F(TCVTTest, case3) { test_tcvt<int16_t, float, 16, 32, 16, 32>(); }

TEST_F(TCVTTest, case4) { test_tcvt<int32_t, float, 32, 512, 32, 512>(); }

TEST_F(TCVTTest, case5) { test_tcvt<int32_t, int16_t, 2, 512, 2, 512>(); }

TEST_F(TCVTTest, case6) { test_tcvt<int32_t, float, 4, 4096, 4, 4096>(); }

TEST_F(TCVTTest, case7) { test_tcvt<float, int16_t, 64, 64, 64, 64>(); }

TEST_F(TCVTTest, case8) { test_tcvt<aclFloat16, float, 64, 64, 64, 64>(); }

TEST_F(TCVTTest, case9) { test_tcvt<uint8_t, aclFloat16, 64, 64, 64, 64>(); }

TEST_F(TCVTTest, case10) { test_tcvt<float, int32_t, 64, 64, 64, 64, pto::SaturationMode::ON>(); }

TEST_F(TCVTTest, case11) { test_tcvt<float, int8_t, 128, 128, 128, 128, pto::SaturationMode::ON>(); }

TEST_F(TCVTTest, case12) { test_tcvt<uint8_t, float, 64, 64, 64, 64, pto::SaturationMode::ON>(); }

TEST_F(TCVTTest, case13) { test_tcvt<int16_t, int32_t, 64, 64, 64, 64, pto::SaturationMode::ON>(); }

TEST_F(TCVTTest, case14) { test_tcvt<int8_t, aclFloat16, 32, 32, 32, 32, pto::SaturationMode::ON>(); }

TEST_F(TCVTTest, case15) { test_tcvt<uint8_t, aclFloat16, 64, 64, 64, 64, pto::SaturationMode::ON>(); }

TEST_F(TCVTTest, case16) { test_tcvt<uint16_t, float, 64, 64, 64, 64>(); }

TEST_F(TCVTTest, case17) { test_tcvt<float, uint16_t, 64, 64, 64, 64>(); }

TEST_F(TCVTTest, case18) { test_tcvt<uint16_t, int32_t, 64, 64, 64, 64, pto::SaturationMode::ON>(); }

TEST_F(TCVTTest, case19) { test_tcvt<int32_t, uint16_t, 64, 64, 64, 64, pto::SaturationMode::ON>(); }

TEST_F(TCVTTest, case20) { test_tcvt<int4b_t, float, 64, 64, 64, 64>(); }

TEST_F(TCVTTest, case21) { test_tcvt<float, int4b_t, 64, 64, 64, 64>(); }

TEST_F(TCVTTest, case22) { test_tcvt<int4b_t, float, 64, 64, 64, 64, pto::SaturationMode::ON>(); }

TEST_F(TCVTTest, case23) { test_tcvt<float, int4b_t, 64, 64, 64, 64, pto::SaturationMode::ON>(); }

TEST_F(TCVTTest, case24) { test_tcvt<float4_e2m1x2_t, float, 64, 64, 64, 64>(); }

TEST_F(TCVTTest, case25) { test_tcvt<float, float4_e2m1x2_t, 64, 64, 64, 64>(); }

TEST_F(TCVTTest, case27) { test_tcvt<float, float4_e2m1x2_t, 64, 64, 64, 64, pto::SaturationMode::ON>(); }

TEST_F(TCVTTest, case28) { test_tcvt<float4_e1m2x2_t, float, 64, 64, 64, 64>(); }

TEST_F(TCVTTest, case29) { test_tcvt<float, float4_e1m2x2_t, 64, 64, 64, 64>(); }

TEST_F(TCVTTest, case31) { test_tcvt<float, float4_e1m2x2_t, 64, 64, 64, 64, pto::SaturationMode::ON>(); }

TEST_F(TCVTTest, case14_with_tmp_explicit_saturation)
{
    test_tcvt<int8_t, aclFloat16, 32, 32, 32, 32, pto::SaturationMode::ON, true, true>("../TCVTTest.case14");
}

TEST_F(TCVTTest, case9_with_tmp_default_saturation)
{
    test_tcvt<uint8_t, aclFloat16, 64, 64, 64, 64, pto::SaturationMode::OFF, true, false>("../TCVTTest.case9");
}

template <typename D, typename S>
void testFp4Expansion()
{
    constexpr unsigned rows = 2;
    constexpr unsigned cols = 512;
    constexpr double e2m1[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
    constexpr double e1m2[] = {0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75};
    Tile<TileType::Vec, S, rows, cols> src;
    Tile<TileType::Vec, D, rows, cols> dst;
    TASSIGN(src, 0);
    TASSIGN(dst, 8192);
    auto* packed = reinterpret_cast<uint8_t*>(src.data());
    for (unsigned r = 0; r < rows; ++r) {
        for (unsigned b = 0; b < cols / 2; ++b) {
            packed[r * cols / 2 + b] = static_cast<uint8_t>(b + r);
        }
    }
    for (RoundMode mode :
         {RoundMode::CAST_NONE, RoundMode::CAST_RINT, RoundMode::CAST_ROUND, RoundMode::CAST_FLOOR,
          RoundMode::CAST_CEIL, RoundMode::CAST_TRUNC, RoundMode::CAST_ODD}) {
        for (SaturationMode saturation : {SaturationMode::OFF, SaturationMode::ON}) {
            TCVT(dst, src, mode, saturation);
            for (unsigned r = 0; r < rows; ++r) {
                for (unsigned c = 0; c < cols; ++c) {
                    const unsigned code = (packed[r * cols / 2 + c / 2] >> (4 * (c % 2))) & 15;
                    const double magnitude = std::is_same_v<S, float4_e2m1x2_t> ? e2m1[code & 7] : e1m2[code & 7];
                    const double expected = code & 8 ? -magnitude : magnitude;
                    EXPECT_EQ(static_cast<double>(dst.GetElement(r, c)), expected)
                        << "row=" << r << " col=" << c << " code=" << code;
                }
            }
        }
    }
}

TEST_F(TCVTTest, fp4_to_floating_preserves_fractional_values)
{
    testFp4Expansion<bfloat16_t, float4_e2m1x2_t>();
    testFp4Expansion<half, float4_e2m1x2_t>();
    testFp4Expansion<float, float4_e2m1x2_t>();
    testFp4Expansion<double, float4_e2m1x2_t>();
    testFp4Expansion<bfloat16_t, float4_e1m2x2_t>();
    testFp4Expansion<half, float4_e1m2x2_t>();
    testFp4Expansion<float, float4_e1m2x2_t>();
    testFp4Expansion<double, float4_e1m2x2_t>();
}

TEST_F(TCVTTest, float_to_integer_retains_rounding)
{
    Tile<TileType::Vec, float, 1, 8> src;
    Tile<TileType::Vec, int32_t, 1, 8> dst;
    TASSIGN(src, 0);
    TASSIGN(dst, 8192);
    const float values[] = {-2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 1.25f, -1.25f};
    const int32_t rint[] = {-2, -2, 0, 0, 2, 2, 1, -1};
    const int32_t round[] = {-3, -2, -1, 1, 2, 3, 1, -1};
    for (unsigned c = 0; c < 8; ++c) {
        src.SetElement(0, c, values[c]);
    }
    for (RoundMode mode : {RoundMode::CAST_RINT, RoundMode::CAST_ROUND}) {
        TCVT(dst, src, mode);
        for (unsigned c = 0; c < 8; ++c) {
            EXPECT_EQ(dst.GetElement(0, c), mode == RoundMode::CAST_RINT ? rint[c] : round[c]);
        }
    }
}
