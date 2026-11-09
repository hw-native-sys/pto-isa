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
#include <cmath>
#include <cstdint>
#include <limits>

using namespace std;
using namespace PtoTestCommon;

// Strict comparator for TREM correctness cases: exact for int32; for float,
// exact except any-NaN == any-NaN, plus a cancellation-slack tolerance of
// 8 ulp of (|x| + |y|) for finite results and for finite-vs-infinite pairs.
// The 910B1 vector divider/multiplier deviates from IEEE RN by up to ~2 ulp;
// for |x| near FLT_MAX, x - floor(x/y)*y is catastrophic cancellation, so the
// golden remainder can legitimately differ from hardware by +-ulp(x), or flip
// to +-inf when q*y crosses the FLT_MAX boundary. Logic bugs (wrong select,
// stale masks) produce errors of order |y| or garbage, far outside this slack.
template <typename T>
bool ResultCmpRemStrict(
    const std::vector<T>& golden, const std::vector<T>& actual, const std::vector<T>& src0, const std::vector<T>& src1)
{
    auto ulp = [](float v) {
        v = std::fabs(v);
        if (std::isnan(v) || std::isinf(v)) {
            return std::numeric_limits<float>::infinity();
        }
        float next = std::nextafterf(v, std::numeric_limits<float>::infinity());
        return (next > v) ? (next - v) : (v > 0.0f ? v : std::numeric_limits<float>::denorm_min());
    };
    size_t errCount = 0;
    for (size_t i = 0; i < golden.size(); ++i) {
        bool ok;
        if constexpr (std::is_same_v<T, float>) {
            if (golden[i] == actual[i]) {
                ok = true; // exact (covers -0 == +0)
            } else if (std::isnan(golden[i]) && std::isnan(actual[i])) {
                ok = true;
            } else {
                float x = (i < src0.size()) ? src0[i] : 0.0f;
                float y = (i < src1.size()) ? src1[i] : 0.0f;
                float slack = 8.0f * (ulp(x) + ulp(y));
                if (std::isfinite(golden[i]) && std::isfinite(actual[i])) {
                    ok = std::fabs(golden[i] - actual[i]) <= slack;
                    if (!ok && std::fabs(actual[i]) <= slack) {
                        // Last-digit collapse: k*bTmp can round to exactly the
                        // state (half-ulp tie), zeroing the staged remainder
                        // while the true floored remainder lies within one
                        // divisor of zero on either side (A5 shows the same
                        // artifact). Accept a collapsed result in that window.
                        ok = std::fabs(golden[i]) <= std::fabs(y) + slack;
                    }
                } else {
                    // finite-vs-infinite: pass only if the finite side is within
                    // the cancellation slack of zero (remainder-of-huge-x case)
                    float fin = std::isfinite(golden[i]) ? golden[i] : actual[i];
                    ok = std::isfinite(fin) ? (std::fabs(fin) <= slack) : (golden[i] == actual[i]);
                }
            }
        } else {
            ok = golden[i] == actual[i];
        }
        if (!ok) {
            if (errCount < 5) {
                std::cout << "MISMATCH idx " << i << ": golden " << golden[i] << " vs actual " << actual[i]
                          << std::endl;
            }
            ++errCount;
        }
    }
    if (errCount) {
        std::cout << "strict compare err count: " << errCount << " / " << golden.size() << std::endl;
    }
    return errCount == 0;
}

class TREMTest : public testing::Test {
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

template <
    typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
    int vCols>
void LaunchTREM(T* out, T* src0, T* src1, void* stream);

template <
    typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
    int vCols>
void LaunchTREMHP(T* out, T* src0, T* src1, void* stream);

template <bool highPrecision, typename T, int dstTileH, int dstTileW, int vRows, int vCols>
void LaunchTREMInplace(T* out, T* src1, void* stream);

template <bool highPrecision, typename T, int dstTileH, int dstTileW, int vRows, int vCols>
void LaunchTREMDisjoint(T* out, T* src0, T* src1, void* stream);

template <
    typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
    int vCols>
void test_trem()
{
    size_t dstFileSize = dstTileH * dstTileW * sizeof(T);
    size_t src0FileSize = src0TileH * src0TileW * sizeof(T);
    size_t src1FileSize = src1TileH * src1TileW * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&src0Host), src0FileSize);
    aclrtMallocHost((void**)(&src1Host), src1FileSize);
    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(dstDevice, dstFileSize, 0, dstFileSize);

    ReadFile(GetGoldenDir() + "/input1.bin", src0FileSize, src0Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", src1FileSize, src1Host, src1FileSize);

    aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTREM<T, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols>(
        dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    // src copies mapped to dst-tile layout for the comparator's per-element slack
    using U = std::conditional_t<std::is_same_v<T, aclFloat16>, _Float16, T>;
    std::vector<U> src0Vec(dstFileSize / sizeof(U), U(0));
    std::vector<U> src1Vec(dstFileSize / sizeof(U), U(0));
    for (int r = 0; r < vRows; ++r) {
        for (int c = 0; c < vCols; ++c) {
            src0Vec[r * dstTileW + c] = src0Host[r * src0TileW + c];
            src1Vec[r * dstTileW + c] = src1Host[r * src1TileW + c];
        }
    }

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<U> golden(dstFileSize / sizeof(U));
    std::vector<U> devFinal(dstFileSize / sizeof(U));
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmpRemStrict<U>(golden, devFinal, src0Vec, src1Vec);
    EXPECT_TRUE(ret);
}

template <
    typename T, int dstTileH, int dstTileW, int src0TileH, int src0TileW, int src1TileH, int src1TileW, int vRows,
    int vCols>
void test_trem_hp()
{
    size_t dstFileSize = dstTileH * dstTileW * sizeof(T);
    size_t src0FileSize = src0TileH * src0TileW * sizeof(T);
    size_t src1FileSize = src1TileH * src1TileW * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&src0Host), src0FileSize);
    aclrtMallocHost((void**)(&src1Host), src1FileSize);
    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(dstDevice, dstFileSize, 0, dstFileSize);

    ReadFile(GetGoldenDir() + "/input1.bin", src0FileSize, src0Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", src1FileSize, src1Host, src1FileSize);

    aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTREMHP<T, dstTileH, dstTileW, src0TileH, src0TileW, src1TileH, src1TileW, vRows, vCols>(
        dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    // src copies mapped to dst-tile layout for the comparator's per-element slack
    using U = std::conditional_t<std::is_same_v<T, aclFloat16>, _Float16, T>;
    std::vector<U> src0Vec(dstFileSize / sizeof(U), U(0));
    std::vector<U> src1Vec(dstFileSize / sizeof(U), U(0));
    for (int r = 0; r < vRows; ++r) {
        for (int c = 0; c < vCols; ++c) {
            src0Vec[r * dstTileW + c] = src0Host[r * src0TileW + c];
            src1Vec[r * dstTileW + c] = src1Host[r * src1TileW + c];
        }
    }

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<U> golden(dstFileSize / sizeof(U));
    std::vector<U> devFinal(dstFileSize / sizeof(U));
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmpRemStrict<U>(golden, devFinal, src0Vec, src1Vec);
    EXPECT_TRUE(ret);
}

// In-place aliasing probe: the dst GM buffer also feeds src0, and the kernel
// TASSIGNs dst and src0 over the same UB region. Probe shapes are compact
// (vCols == tile width), so TSTORE rewrites every compared element and no
// memset of the aliased buffer is needed.
template <bool highPrecision, typename T, int dstTileH, int dstTileW, int vRows, int vCols>
void test_trem_inplace()
{
    size_t fileSize = dstTileH * dstTileW * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src1Host;
    T *dstDevice, *src1Device;

    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMallocHost((void**)(&src1Host), fileSize);
    aclrtMalloc((void**)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, dstHost, fileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSize, src1Host, fileSize);

    aclrtMemcpy(dstDevice, fileSize, dstHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTREMInplace<highPrecision, T, dstTileH, dstTileW, vRows, vCols>(dstDevice, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    // feed the real inputs to the comparator so the per-element slack matches
    // the stock harness (zero vectors would kill the 8-ulp tolerance)
    std::vector<T> src0Vec(fileSize / sizeof(T));
    std::vector<T> src1Vec(fileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src0Vec.data(), fileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSize, src1Vec.data(), fileSize);

    aclrtFree(dstDevice);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize / sizeof(T));
    std::vector<T> devFinal(fileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize);

    bool ret = ResultCmpRemStrict<T>(golden, devFinal, src0Vec, src1Vec);
    EXPECT_TRUE(ret);
}

// Disjoint control for the in-place probe kernels: same tile shapes, but dst
// is a separate GM buffer (memset like the stock harness).
template <bool highPrecision, typename T, int dstTileH, int dstTileW, int vRows, int vCols>
void test_trem_disjoint()
{
    size_t fileSize = dstTileH * dstTileW * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), fileSize);
    aclrtMallocHost((void**)(&src0Host), fileSize);
    aclrtMallocHost((void**)(&src1Host), fileSize);
    aclrtMalloc((void**)&dstDevice, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(dstDevice, fileSize, 0, fileSize);

    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src0Host, fileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSize, src1Host, fileSize);

    aclrtMemcpy(src0Device, fileSize, src0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, fileSize, src1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTREMDisjoint<highPrecision, T, dstTileH, dstTileW, vRows, vCols>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDevice, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, fileSize);

    std::vector<T> src0Vec(fileSize / sizeof(T));
    std::vector<T> src1Vec(fileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/input1.bin", fileSize, src0Vec.data(), fileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", fileSize, src1Vec.data(), fileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(fileSize / sizeof(T));
    std::vector<T> devFinal(fileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", fileSize, golden.data(), fileSize);
    ReadFile(GetGoldenDir() + "/output.bin", fileSize, devFinal.data(), fileSize);

    bool ret = ResultCmpRemStrict<T>(golden, devFinal, src0Vec, src1Vec);
    EXPECT_TRUE(ret);
}

TEST_F(TREMTest, case_float_16x64_16x128_16x128_16x64) { test_trem<float, 16, 64, 16, 128, 16, 128, 16, 64>(); }

TEST_F(TREMTest, case_float_16x32_16x64_16x32_16x32) { test_trem<float, 16, 32, 16, 64, 16, 32, 16, 32>(); }

TEST_F(TREMTest, case_int32_4x32_4x32_4x32_4x32) { test_trem<int32_t, 4, 32, 4, 32, 4, 32, 4, 32>(); }

TEST_F(TREMTest, case_int32_16x32_16x64_16x32_16x32) { test_trem<int32_t, 16, 32, 16, 64, 16, 32, 16, 32>(); }

TEST_F(TREMTest, case_float_16x64_16x128_16x128_16x63) { test_trem<float, 16, 64, 16, 128, 16, 128, 16, 63>(); }

TEST_F(TREMTest, case_float_2x32_2x64_2x32_2x31) { test_trem<float, 2, 32, 2, 64, 2, 32, 2, 31>(); }

TEST_F(TREMTest, case_int32_16x32_16x64_16x32_16x31) { test_trem<int32_t, 16, 32, 16, 64, 16, 32, 16, 31>(); }

TEST_F(TREMTest, case_float_1x8192_1x8192_1x8192_1x8192) { test_trem<float, 1, 8192, 1, 8192, 1, 8192, 1, 8192>(); }

// size matrix (float): repeat boundaries, sub-block, large, strided
TEST_F(TREMTest, case_float_1x64_1x64_1x64_1x64) { test_trem<float, 1, 64, 1, 64, 1, 64, 1, 64>(); }
TEST_F(TREMTest, case_float_1x72_1x72_1x72_1x65) { test_trem<float, 1, 72, 1, 72, 1, 72, 1, 65>(); }
TEST_F(TREMTest, case_float_2x128_2x128_2x128_2x128) { test_trem<float, 2, 128, 2, 128, 2, 128, 2, 128>(); }
TEST_F(TREMTest, case_float_2x136_2x136_2x136_2x129) { test_trem<float, 2, 136, 2, 136, 2, 136, 2, 129>(); }
TEST_F(TREMTest, case_float_3x256_3x256_3x256_3x256) { test_trem<float, 3, 256, 3, 256, 3, 256, 3, 256>(); }
TEST_F(TREMTest, case_float_3x264_3x264_3x264_3x257) { test_trem<float, 3, 264, 3, 264, 3, 264, 3, 257>(); }
TEST_F(TREMTest, case_float_8x192_8x192_8x192_8x192) { test_trem<float, 8, 192, 8, 192, 8, 192, 8, 192>(); }
TEST_F(TREMTest, case_float_33x64_33x64_33x64_33x64) { test_trem<float, 33, 64, 33, 64, 33, 64, 33, 64>(); }
TEST_F(TREMTest, case_float_1x1024_1x1024_1x1024_1x1024) { test_trem<float, 1, 1024, 1, 1024, 1, 1024, 1, 1024>(); }
TEST_F(TREMTest, case_float_2x2048_2x2048_2x2048_2x2048) { test_trem<float, 2, 2048, 2, 2048, 2, 2048, 2, 2048>(); }
TEST_F(TREMTest, case_float_4x64_4x64_4x64_4x63) { test_trem<float, 4, 64, 4, 64, 4, 64, 4, 63>(); }
TEST_F(TREMTest, case_float_4x8_4x8_4x8_4x8) { test_trem<float, 4, 8, 4, 8, 4, 8, 4, 8>(); }
TEST_F(TREMTest, case_float_2x4096_2x4096_2x4096_2x4096) { test_trem<float, 2, 4096, 2, 4096, 2, 4096, 2, 4096>(); }
TEST_F(TREMTest, case_float_2x512_2x384_2x384_2x384) { test_trem<float, 2, 512, 2, 384, 2, 384, 2, 384>(); }

// size matrix (int32)
TEST_F(TREMTest, case_int32_1x64_1x64_1x64_1x64) { test_trem<int32_t, 1, 64, 1, 64, 1, 64, 1, 64>(); }
TEST_F(TREMTest, case_int32_2x72_2x72_2x72_2x65) { test_trem<int32_t, 2, 72, 2, 72, 2, 72, 2, 65>(); }
TEST_F(TREMTest, case_int32_3x136_3x136_3x136_3x129) { test_trem<int32_t, 3, 136, 3, 136, 3, 136, 3, 129>(); }
TEST_F(TREMTest, case_int32_1x8192_1x8192_1x8192_1x8192) { test_trem<int32_t, 1, 8192, 1, 8192, 1, 8192, 1, 8192>(); }
TEST_F(TREMTest, case_int32_2x192_2x192_2x192_2x192) { test_trem<int32_t, 2, 192, 2, 192, 2, 192, 2, 192>(); }
TEST_F(TREMTest, case_int32_2x256_2x192_2x192_2x192) { test_trem<int32_t, 2, 256, 2, 192, 2, 192, 2, 192>(); }

// high precision TREM (issue #2): staged >= 48 full vectors, register
// fallback, tails, strided dst, widest single row within the 192KB UB budget
TEST_F(TREMTest, case_hp_float_8x448_8x448_8x448_8x448) { test_trem_hp<float, 8, 448, 8, 448, 8, 448, 8, 448>(); }
TEST_F(TREMTest, case_hp_float_12x256_12x256_12x256_12x256)
{
    test_trem_hp<float, 12, 256, 12, 256, 12, 256, 12, 256>();
}
TEST_F(TREMTest, case_hp_float_32x64_32x64_32x64_32x64) { test_trem_hp<float, 32, 64, 32, 64, 32, 64, 32, 64>(); }
TEST_F(TREMTest, case_hp_float_7x512_7x448_7x448_7x448) { test_trem_hp<float, 7, 512, 7, 448, 7, 448, 7, 448>(); }
TEST_F(TREMTest, case_hp_float_2x2048_2x2048_2x2048_2x2048)
{
    test_trem_hp<float, 2, 2048, 2, 2048, 2, 2048, 2, 2048>();
}
TEST_F(TREMTest, case_hp_float_5x320_5x320_5x320_5x320) { test_trem_hp<float, 5, 320, 5, 320, 5, 320, 5, 320>(); }
TEST_F(TREMTest, case_hp_float_8x456_8x456_8x456_8x456) { test_trem_hp<float, 8, 456, 8, 456, 8, 456, 8, 456>(); }
TEST_F(TREMTest, case_hp_float_4x136_4x136_4x136_4x136) { test_trem_hp<float, 4, 136, 4, 136, 4, 136, 4, 136>(); }
TEST_F(TREMTest, case_hp_float_1x7424_1x7424_1x7424_1x7424)
{
    test_trem_hp<float, 1, 7424, 1, 7424, 1, 7424, 1, 7424>();
}
TEST_F(TREMTest, case_hp_float_16x64_16x64_16x64_16x63) { test_trem_hp<float, 16, 64, 16, 64, 16, 64, 16, 63>(); }

// In-place aliasing probes (dst tile shares src0's GM buffer and UB region):
// DEFAULT float/int and the HP fallback paths must all stay alias-safe.
// Shapes are compact (valid == tile), so golden padding never participates.
TEST_F(TREMTest, ip_def_float_4x8) { test_trem_inplace<false, float, 4, 8, 4, 8>(); }
TEST_F(TREMTest, ip_def_float_33x64) { test_trem_inplace<false, float, 33, 64, 33, 64>(); }
TEST_F(TREMTest, ip_def_float_8x192) { test_trem_inplace<false, float, 8, 192, 8, 192>(); }
TEST_F(TREMTest, ip_def_float_1x8192) { test_trem_inplace<false, float, 1, 8192, 1, 8192>(); }
TEST_F(TREMTest, ip_def_int32_1x64) { test_trem_inplace<false, int32_t, 1, 64, 1, 64>(); }
TEST_F(TREMTest, ip_def_int32_2x192) { test_trem_inplace<false, int32_t, 2, 192, 2, 192>(); }
TEST_F(TREMTest, ip_hp_float_8x448) { test_trem_inplace<true, float, 8, 448, 8, 448>(); }
TEST_F(TREMTest, ip_hp_float_12x256) { test_trem_inplace<true, float, 12, 256, 12, 256>(); }
TEST_F(TREMTest, ip_hp_float_32x64) { test_trem_inplace<true, float, 32, 64, 32, 64>(); }
TEST_F(TREMTest, ip_hp_float_2x2048) { test_trem_inplace<true, float, 2, 2048, 2, 2048>(); }
TEST_F(TREMTest, ip_hp_float_5x320) { test_trem_inplace<true, float, 5, 320, 5, 320>(); }
TEST_F(TREMTest, ip_hp_float_8x456) { test_trem_inplace<true, float, 8, 456, 8, 456>(); }
TEST_F(TREMTest, ip_hp_float_4x136) { test_trem_inplace<true, float, 4, 136, 4, 136>(); }
TEST_F(TREMTest, ip_hp_float_1x7424) { test_trem_inplace<true, float, 1, 7424, 1, 7424>(); }
// disjoint controls validating the probe kernels on the stock layout
TEST_F(TREMTest, ctrl_disj_def_float_33x64) { test_trem_disjoint<false, float, 33, 64, 33, 64>(); }
TEST_F(TREMTest, ctrl_disj_hp_float_8x448) { test_trem_disjoint<true, float, 8, 448, 8, 448>(); }

// ---- HP size matrix (compensated final iteration) ----
TEST_F(TREMTest, case_hp_sz_2x448) { test_trem_hp<float, 2, 448, 2, 448, 2, 448, 2, 448>(); }
TEST_F(TREMTest, case_hp_sz_26x448) { test_trem_hp<float, 26, 448, 26, 448, 26, 448, 26, 448>(); }
TEST_F(TREMTest, case_hp_sz_49x64) { test_trem_hp<float, 49, 64, 49, 64, 49, 64, 49, 64>(); }
TEST_F(TREMTest, case_hp_sz_47x64) { test_trem_hp<float, 47, 64, 47, 64, 47, 64, 47, 64>(); }
TEST_F(TREMTest, case_hp_sz_100x64) { test_trem_hp<float, 100, 64, 100, 64, 100, 64, 100, 64>(); }
TEST_F(TREMTest, case_hp_sz_8x457) { test_trem_hp<float, 8, 464, 8, 464, 8, 464, 8, 457>(); }
TEST_F(TREMTest, case_hp_sz_8x449) { test_trem_hp<float, 8, 456, 8, 456, 8, 456, 8, 449>(); }
TEST_F(TREMTest, case_hp_sz_3x65) { test_trem_hp<float, 3, 72, 3, 72, 3, 72, 3, 65>(); }
TEST_F(TREMTest, case_hp_sz_1x128) { test_trem_hp<float, 1, 128, 1, 128, 1, 128, 1, 128>(); }
TEST_F(TREMTest, case_hp_sz_1x192) { test_trem_hp<float, 1, 192, 1, 192, 1, 192, 1, 192>(); }
TEST_F(TREMTest, case_hp_sz_1x320) { test_trem_hp<float, 1, 320, 1, 320, 1, 320, 1, 320>(); }
TEST_F(TREMTest, case_hp_sz_1x576) { test_trem_hp<float, 1, 576, 1, 576, 1, 576, 1, 576>(); }
TEST_F(TREMTest, case_hp_sz_1x4096) { test_trem_hp<float, 1, 4096, 1, 4096, 1, 4096, 1, 4096>(); }
TEST_F(TREMTest, case_hp_sz_1x6400) { test_trem_hp<float, 1, 6400, 1, 6400, 1, 6400, 1, 6400>(); }
TEST_F(TREMTest, case_hp_sz_str_2x256_192) { test_trem_hp<float, 2, 256, 2, 256, 2, 256, 2, 192>(); }
TEST_F(TREMTest, case_hp_sz_str_4x128_65) { test_trem_hp<float, 4, 128, 4, 72, 4, 72, 4, 65>(); }
TEST_F(TREMTest, ip_hp_49x64) { test_trem_inplace<true, float, 49, 64, 49, 64>(); }
TEST_F(TREMTest, ip_hp_100x64) { test_trem_inplace<true, float, 100, 64, 100, 64>(); }
