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

template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kValidRows_ = kTRows_,
          int kValidCols_ = kTCols_>
void launchTCVT(D *dst, S *src, void *stream);

// Saturation mode test launcher
template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void launchTCVTSaturationTest(D *dstSaturated, D *dstTruncated, D *dstDefault, S *src, void *stream);

// NonSatTorch test launcher (with explicit tmp tile)
template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kValidRows_ = kTRows_,
          int kValidCols_ = kTCols_>
void launchTCVTNonSatTorch(D *dst, S *src, void *stream);

// Int4b_t (S4) conversion launchers
// kGCols_ is the number of fp16 elements (= number of int4 elements)
template <int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void launchTCVT_fp16_to_s4(uint8_t *dst, aclFloat16 *src, void *stream);

template <int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void launchTCVT_s4_to_fp16(aclFloat16 *dst, uint8_t *src, void *stream);

class TCVTTest : public testing::Test {
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

template <typename D, typename S>
struct TcvtTestResources {
    D *dstHost;
    D *dstDevice;
    S *srcHost;
    S *srcDevice;
    size_t srcFileSize;
    size_t dstFileSize;
    aclrtStream stream;
};

template <typename D, typename S, int kGRows_, int kGCols_>
TcvtTestResources<D, S> SetupTcvtTest()
{
    TcvtTestResources<D, S> res;
    res.srcFileSize = static_cast<size_t>(kGRows_) * kGCols_ * sizeof(S);
    res.dstFileSize = static_cast<size_t>(kGRows_) * kGCols_ * sizeof(D);
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtCreateStream(&res.stream);
    aclrtMallocHost((void **)(&res.dstHost), res.dstFileSize);
    aclrtMallocHost((void **)(&res.srcHost), res.srcFileSize);
    aclrtMalloc((void **)&res.dstDevice, res.dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&res.srcDevice, res.srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ReadFile(GetGoldenDir() + "/x1_gm.bin", res.srcFileSize, res.srcHost, res.srcFileSize);
    aclrtMemcpy(res.srcDevice, res.srcFileSize, res.srcHost, res.srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    return res;
}

template <typename D, typename S>
void CleanupTcvtTest(TcvtTestResources<D, S> &res)
{
    aclrtFree(res.dstDevice);
    aclrtFree(res.srcDevice);
    aclrtFreeHost(res.dstHost);
    aclrtFreeHost(res.srcHost);
    aclrtDestroyStream(res.stream);
    aclrtResetDevice(0);
    aclFinalize();
}

template <typename D, int kValidRows_, int kTRows_, int kValidCols_, int kTCols_>
bool CompareResults(const std::vector<D> &golden, const std::vector<D> &devFinal, uint32_t N)
{
    constexpr bool isPartialTile = (kValidRows_ != kTRows_) || (kValidCols_ != kTCols_);
    if constexpr (isPartialTile) {
        for (uint32_t r = 0; r < kValidRows_; r++) {
            std::vector<D> goldenRow(golden.data() + r * N, golden.data() + r * N + kValidCols_);
            std::vector<D> devRow(devFinal.data() + r * N, devFinal.data() + r * N + kValidCols_);
            if (!ResultCmp<D>(goldenRow, devRow, 0.001f)) {
                return false;
            }
        }
        return true;
    } else {
        return ResultCmp<D>(golden, devFinal, 0.001f);
    }
}

template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kValidRows_ = kTRows_,
          int kValidCols_ = kTCols_>
void test_tcvt()
{
    auto res = SetupTcvtTest<D, S, kGRows_, kGCols_>();
    launchTCVT<D, S, kGRows_, kGCols_, kTRows_, kTCols_, kValidRows_, kValidCols_>(res.dstDevice, res.srcDevice,
                                                                                   res.stream);
    aclrtSynchronizeStream(res.stream);
    aclrtMemcpy(res.dstHost, res.dstFileSize, res.dstDevice, res.dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output_z.bin", res.dstHost, res.dstFileSize);
    CleanupTcvtTest(res);

    std::vector<D> golden(res.dstFileSize);
    std::vector<D> devFinal(res.dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", res.dstFileSize, golden.data(), res.dstFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", res.dstFileSize, devFinal.data(), res.dstFileSize);

    bool ret = CompareResults<D, kValidRows_, kTRows_, kValidCols_, kTCols_>(golden, devFinal, kGCols_);
    EXPECT_TRUE(ret);
}

// Macro to generate test cases for all shapes for a given type pair
#define GENERATE_TCVT_TESTS(dst_type, src_type, type_name)       \
    TEST_F(TCVTTest, case_##type_name##_1x32)                    \
    {                                                            \
        test_tcvt<dst_type, src_type, 1, 32, 1, 32>();           \
    }                                                            \
    TEST_F(TCVTTest, case_##type_name##_2x64)                    \
    {                                                            \
        test_tcvt<dst_type, src_type, 2, 64, 2, 64>();           \
    }                                                            \
    TEST_F(TCVTTest, case_##type_name##_4x32)                    \
    {                                                            \
        test_tcvt<dst_type, src_type, 4, 32, 4, 32>();           \
    }                                                            \
    TEST_F(TCVTTest, case_##type_name##_8x64)                    \
    {                                                            \
        test_tcvt<dst_type, src_type, 8, 64, 8, 64>();           \
    }                                                            \
    TEST_F(TCVTTest, case_##type_name##_1x256)                   \
    {                                                            \
        test_tcvt<dst_type, src_type, 1, 256, 1, 256>();         \
    }                                                            \
    TEST_F(TCVTTest, case_##type_name##_8x128)                   \
    {                                                            \
        test_tcvt<dst_type, src_type, 8, 128, 8, 128>();         \
    }                                                            \
    TEST_F(TCVTTest, case_##type_name##_4x128_4x65)              \
    {                                                            \
        test_tcvt<dst_type, src_type, 4, 128, 4, 128, 4, 65>();  \
    }                                                            \
    TEST_F(TCVTTest, case_##type_name##_4x256_4x200)             \
    {                                                            \
        test_tcvt<dst_type, src_type, 4, 256, 4, 256, 4, 200>(); \
    }                                                            \
    TEST_F(TCVTTest, case_##type_name##_1x256_1x129)             \
    {                                                            \
        test_tcvt<dst_type, src_type, 1, 256, 1, 256, 1, 129>(); \
    }                                                            \
    TEST_F(TCVTTest, case_##type_name##_2x32_2x16)               \
    {                                                            \
        test_tcvt<dst_type, src_type, 2, 32, 2, 32, 2, 16>();    \
    }

// FP32 Source → fp16, int16, int32, int64
GENERATE_TCVT_TESTS(aclFloat16, float, fp32_fp16)
GENERATE_TCVT_TESTS(int16_t, float, fp32_int16)
GENERATE_TCVT_TESTS(int32_t, float, fp32_int32)
GENERATE_TCVT_TESTS(int64_t, float, fp32_int64)
GENERATE_TCVT_TESTS(float, float, fp32_fp32)

// FP16 Source → fp32, int32, int16, int8, uint8
GENERATE_TCVT_TESTS(float, aclFloat16, fp16_fp32)
GENERATE_TCVT_TESTS(int32_t, aclFloat16, fp16_int32)
GENERATE_TCVT_TESTS(int16_t, aclFloat16, fp16_int16)
GENERATE_TCVT_TESTS(int8_t, aclFloat16, fp16_int8)
GENERATE_TCVT_TESTS(uint8_t, aclFloat16, fp16_uint8)

// I8 Source → fp16
GENERATE_TCVT_TESTS(aclFloat16, int8_t, int8_fp16)

// U8 Source → fp16
GENERATE_TCVT_TESTS(aclFloat16, uint8_t, uint8_fp16)

// I16 Source → fp16, fp32
GENERATE_TCVT_TESTS(aclFloat16, int16_t, int16_fp16)
GENERATE_TCVT_TESTS(float, int16_t, int16_fp32)

// I32 Source → fp32, fp16, int16, int64
GENERATE_TCVT_TESTS(float, int32_t, int32_fp32)
GENERATE_TCVT_TESTS(aclFloat16, int32_t, int32_fp16)
GENERATE_TCVT_TESTS(int16_t, int32_t, int32_int16)
GENERATE_TCVT_TESTS(int64_t, int32_t, int32_int64)

// I64 Source → fp32, int32
GENERATE_TCVT_TESTS(float, int64_t, int64_fp32)
GENERATE_TCVT_TESTS(int32_t, int64_t, int64_int32)

// ============================================================================
// Int4b_t (S4) Conversion Tests
// ============================================================================
// kGCols_ = number of fp16/int4 elements. Packed byte count = kGCols_ / 2.

// FP16 → S4 test: src is fp16, dst is packed uint8_t (2 int4 elements per byte)
template <int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void test_tcvt_fp16_to_s4()
{
    size_t numElements = static_cast<size_t>(kGRows_) * kGCols_;
    size_t srcFileSize = numElements * sizeof(aclFloat16);
    size_t dstFileSize = numElements / 2; // packed: 2 elements per byte

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    aclFloat16 *srcHost;
    uint8_t *dstHost;
    aclFloat16 *srcDevice;
    uint8_t *dstDevice;

    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", srcFileSize, srcHost, srcFileSize);
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTCVT_fp16_to_s4<kGRows_, kGCols_, kTRows_, kTCols_>(dstDevice, srcDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<uint8_t> golden(dstFileSize);
    std::vector<uint8_t> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", dstFileSize, devFinal.data(), dstFileSize);
    EXPECT_TRUE(ResultCmp<uint8_t>(golden, devFinal, 0.001f));
}

// S4 → FP16 test: src is packed uint8_t, dst is fp16
template <int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void test_tcvt_s4_to_fp16()
{
    size_t numElements = static_cast<size_t>(kGRows_) * kGCols_;
    size_t srcFileSize = numElements / 2; // packed: 2 elements per byte
    size_t dstFileSize = numElements * sizeof(aclFloat16);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *srcHost;
    aclFloat16 *dstHost;
    uint8_t *srcDevice;
    aclFloat16 *dstDevice;

    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", srcFileSize, srcHost, srcFileSize);
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTCVT_s4_to_fp16<kGRows_, kGCols_, kTRows_, kTCols_>(dstDevice, srcDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, dstFileSize);

    aclrtFree(srcDevice);
    aclrtFree(dstDevice);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(dstHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<aclFloat16> golden(dstFileSize);
    std::vector<aclFloat16> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", dstFileSize, devFinal.data(), dstFileSize);
    EXPECT_TRUE(ResultCmp<aclFloat16>(golden, devFinal, 0.001f));
}

#define GENERATE_TCVT_FP16_TO_S4_TESTS(type_name, gR, gC, tR, tC) \
    TEST_F(TCVTTest, case_##type_name##_##gR##x##gC)              \
    {                                                             \
        test_tcvt_fp16_to_s4<gR, gC, tR, tC>();                   \
    }

#define GENERATE_TCVT_S4_TO_FP16_TESTS(type_name, gR, gC, tR, tC) \
    TEST_F(TCVTTest, case_##type_name##_##gR##x##gC)              \
    {                                                             \
        test_tcvt_s4_to_fp16<gR, gC, tR, tC>();                   \
    }

// FP16 → S4
GENERATE_TCVT_FP16_TO_S4_TESTS(fp16_s4, 1, 64, 1, 64)
GENERATE_TCVT_FP16_TO_S4_TESTS(fp16_s4, 1, 128, 1, 128)
GENERATE_TCVT_FP16_TO_S4_TESTS(fp16_s4, 1, 256, 1, 256)
GENERATE_TCVT_FP16_TO_S4_TESTS(fp16_s4, 2, 128, 2, 128)
GENERATE_TCVT_FP16_TO_S4_TESTS(fp16_s4, 4, 128, 4, 128)
GENERATE_TCVT_FP16_TO_S4_TESTS(fp16_s4, 8, 128, 8, 128)

// S4 → FP16
GENERATE_TCVT_S4_TO_FP16_TESTS(s4_fp16, 1, 64, 1, 64)
GENERATE_TCVT_S4_TO_FP16_TESTS(s4_fp16, 1, 128, 1, 128)
GENERATE_TCVT_S4_TO_FP16_TESTS(s4_fp16, 1, 256, 1, 256)
GENERATE_TCVT_S4_TO_FP16_TESTS(s4_fp16, 2, 128, 2, 128)
GENERATE_TCVT_S4_TO_FP16_TESTS(s4_fp16, 4, 128, 4, 128)
GENERATE_TCVT_S4_TO_FP16_TESTS(s4_fp16, 8, 128, 8, 128)

// ============================================================================
// Saturation Mode Tests
// ============================================================================

template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void test_tcvt_saturation()
{
    uint32_t M = kGRows_;
    uint32_t N = kGCols_;

    size_t srcFileSize = M * N * sizeof(S);
    size_t dstFileSize = M * N * sizeof(D);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    D *dstSatHost, *dstTruncHost, *dstDefaultHost, *dstSatDevice, *dstTruncDevice, *dstDefaultDevice;
    S *srcHost, *srcDevice;

    aclrtMallocHost((void **)(&dstSatHost), dstFileSize);
    aclrtMallocHost((void **)(&dstTruncHost), dstFileSize);
    aclrtMallocHost((void **)(&dstDefaultHost), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);

    aclrtMalloc((void **)&dstSatDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstTruncDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstDefaultDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", srcFileSize, srcHost, srcFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    // Run saturation test - produces saturated, truncated, and default outputs
    launchTCVTSaturationTest<D, S, kGRows_, kGCols_, kTRows_, kTCols_>(dstSatDevice, dstTruncDevice, dstDefaultDevice,
                                                                       srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstSatHost, dstFileSize, dstSatDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(dstTruncHost, dstFileSize, dstTruncDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(dstDefaultHost, dstFileSize, dstDefaultDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    // Write output files IMMEDIATELY after getting results - ensures fresh data every run
    WriteFile(GetGoldenDir() + "/output_saturated.bin", dstSatHost, dstFileSize);
    WriteFile(GetGoldenDir() + "/output_truncated.bin", dstTruncHost, dstFileSize);
    WriteFile(GetGoldenDir() + "/output_default.bin", dstDefaultHost, dstFileSize);

    // Compare truncated output (PyTorch only provides TRUNC mode golden data)
    std::vector<D> goldenTrunc(dstFileSize);
    std::vector<D> devTrunc(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden_truncated.bin", dstFileSize, goldenTrunc.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output_truncated.bin", dstFileSize, devTrunc.data(), dstFileSize);
    bool truncOk = ResultCmp<D>(goldenTrunc, devTrunc, 0.001f);

    // Compare default output
    // PyTorch only provides truncated mode golden data, so we compare against that
    std::string goldenDefaultFile = GetGoldenDir() + "/golden_truncated.bin";

    std::vector<D> goldenDefault(dstFileSize);
    std::vector<D> devDefault(dstFileSize);
    ReadFile(goldenDefaultFile, dstFileSize, goldenDefault.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output_default.bin", dstFileSize, devDefault.data(), dstFileSize);
    bool defaultOk = ResultCmp<D>(goldenDefault, devDefault, 0.001f);

    aclrtFree(dstSatDevice);
    aclrtFree(dstTruncDevice);
    aclrtFree(dstDefaultDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstSatHost);
    aclrtFreeHost(dstTruncHost);
    aclrtFreeHost(dstDefaultHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    EXPECT_TRUE(truncOk) << "Saturation mode OFF (TRUNC) output mismatch";
    EXPECT_TRUE(defaultOk) << "Default mode output mismatch (compared against PyTorch TRUNC golden)";
}

// Saturation mode test cases (only for supported conversions on A2A3)
// Minimal saturation mode tests (fp32→int8 is NOT supported on A2A3 hardware)
// Disabled at compile time by default - define ENABLE_SATURATION_TESTS to enable
#ifdef ENABLE_SATURATION_TESTS
TEST_F(TCVTTest, saturation_fp16_int8_1x32)
{
    test_tcvt_saturation<int8_t, aclFloat16, 1, 32, 1, 32>();
}

TEST_F(TCVTTest, saturation_fp32_int16_1x32)
{
    test_tcvt_saturation<int16_t, float, 1, 32, 1, 32>();
}

TEST_F(TCVTTest, saturation_fp16_int16_1x32)
{
    test_tcvt_saturation<int16_t, aclFloat16, 1, 32, 1, 32>();
}

TEST_F(TCVTTest, saturation_fp16_uint8_1x32)
{
    test_tcvt_saturation<uint8_t, aclFloat16, 1, 32, 1, 32>();
}

TEST_F(TCVTTest, saturation_int64_int32_1x32)
{
    test_tcvt_saturation<int32_t, int64_t, 1, 32, 1, 32>();
}

TEST_F(TCVTTest, saturation_int32_int16_1x32)
{
    test_tcvt_saturation<int16_t, int32_t, 1, 32, 1, 32>();
}
#endif // ENABLE_SATURATION_TESTS

// ============================================================================
// NonSatTorch Tests (with explicit tmp tile)
// ============================================================================

template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kValidRows_ = kTRows_,
          int kValidCols_ = kTCols_>
void test_tcvt_nonsattorch()
{
    auto res = SetupTcvtTest<D, S, kGRows_, kGCols_>();
    launchTCVTNonSatTorch<D, S, kGRows_, kGCols_, kTRows_, kTCols_, kValidRows_, kValidCols_>(
        res.dstDevice, res.srcDevice, res.stream);
    aclrtSynchronizeStream(res.stream);
    aclrtMemcpy(res.dstHost, res.dstFileSize, res.dstDevice, res.dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output_truncated.bin", res.dstHost, res.dstFileSize);

    std::vector<D> golden(res.dstFileSize);
    std::vector<D> devFinal(res.dstFileSize);
    ReadFile(GetGoldenDir() + "/golden_truncated.bin", res.dstFileSize, golden.data(), res.dstFileSize);
    ReadFile(GetGoldenDir() + "/output_truncated.bin", res.dstFileSize, devFinal.data(), res.dstFileSize);
    CleanupTcvtTest(res);

    bool ret = CompareResults<D, kValidRows_, kTRows_, kValidCols_, kTCols_>(golden, devFinal, kGCols_);
    EXPECT_TRUE(ret) << "NonSatTorch output mismatch";
}

TEST_F(TCVTTest, nonsattorch_fp16_int8_1x32)
{
    test_tcvt_nonsattorch<int8_t, aclFloat16, 1, 32, 1, 32>();
}

TEST_F(TCVTTest, nonsattorch_fp16_int8_2x64)
{
    test_tcvt_nonsattorch<int8_t, aclFloat16, 2, 64, 2, 64>();
}

TEST_F(TCVTTest, nonsattorch_fp16_int8_8x128)
{
    test_tcvt_nonsattorch<int8_t, aclFloat16, 8, 128, 8, 128>();
}

TEST_F(TCVTTest, nonsattorch_fp16_int16_1x32)
{
    test_tcvt_nonsattorch<int16_t, aclFloat16, 1, 32, 1, 32>();
}

TEST_F(TCVTTest, nonsattorch_fp32_int16_1x32)
{
    test_tcvt_nonsattorch<int16_t, float, 1, 32, 1, 32>();
}

TEST_F(TCVTTest, nonsattorch_fp16_int8_4x128_4x65)
{
    test_tcvt_nonsattorch<int8_t, aclFloat16, 4, 128, 4, 128, 4, 65>();
}

TEST_F(TCVTTest, nonsattorch_fp16_int8_2x32_2x16)
{
    test_tcvt_nonsattorch<int8_t, aclFloat16, 2, 32, 2, 32, 2, 16>();
}

TEST_F(TCVTTest, nonsattorch_fp16_int16_4x128_4x65)
{
    test_tcvt_nonsattorch<int16_t, aclFloat16, 4, 128, 4, 128, 4, 65>();
}

TEST_F(TCVTTest, nonsattorch_fp32_int16_4x128_4x65)
{
    test_tcvt_nonsattorch<int16_t, float, 4, 128, 4, 128, 4, 65>();
}
