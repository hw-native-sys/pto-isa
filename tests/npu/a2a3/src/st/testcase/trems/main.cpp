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
#include <acl/acl.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>

using namespace std;
using namespace PtoTestCommon;

// Strict comparator for TREMS correctness cases:
// - int32: exact
// - float: exact bits, except any-NaN == any-NaN (payloads may differ), plus
//   a cancellation-slack tolerance of 8 ulp of (|x| + |s|) for finite results
//   and finite-vs-infinite pairs. The 910B1 vector divider/multiplier deviates
//   from IEEE RN by up to ~2 ulp; for |x| near FLT_MAX, x - floor(x/s)*s is
//   catastrophic cancellation, so the golden remainder can legitimately differ
//   from hardware by +-ulp(x), or flip to +-inf when q*s crosses the FLT_MAX
//   boundary. Logic bugs (wrong select, stale masks) produce errors of order
//   |s| or garbage, far outside this slack.
// The previous ResultCmp(eps=0.001) tolerated up to 0.1% wrong elements,
// which hides row/repeat-boundary mask bugs these cases target.
template <typename T>
bool ResultCmpRemStrict(
    const std::vector<T>& golden, const std::vector<T>& actual, const std::vector<T>& src, float scalarVal)
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
                float x = (i < src.size()) ? src[i] : 0.0f;
                float slack = 8.0f * (ulp(x) + ulp(scalarVal));
                if (std::isfinite(golden[i]) && std::isfinite(actual[i])) {
                    ok = std::fabs(golden[i] - actual[i]) <= slack;
                    if (!ok && std::fabs(actual[i]) <= slack) {
                        // Last-digit collapse, see the trem comparator comment.
                        ok = std::fabs(golden[i]) <= std::fabs(scalarVal) + slack;
                    }
                } else {
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

template <uint32_t caseId>
void launchTREMSTestCase(void* out, void* src, float scalar, aclrtStream stream);

template <uint32_t caseId>
void launchTREMSIpTestCase(void* out, float scalar, aclrtStream stream);

template <uint32_t caseId>
void launchTREMSDisjTestCase(void* out, void* src, float scalar, aclrtStream stream);

class TREMSTest : public testing::Test {
public:
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

template <uint32_t caseId, typename T, int dstTileRow, int dstTileCol, int row, int validRow, int col, int srcValidCol>
void TREMSTestFramework()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t dstByteSize = dstTileRow * dstTileCol * sizeof(T);
    size_t srcByteSize = row * col * sizeof(T);
    T* dstHost;
    T* srcHost;
    T* dstDevice;
    T* srcDevice;
    float scalar;

    aclrtMallocHost((void**)(&dstHost), dstByteSize);
    aclrtMallocHost((void**)(&srcHost), srcByteSize);

    aclrtMalloc((void**)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(dstDevice, dstByteSize, 0, dstByteSize);

    ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
    std::string scalar_file = GetGoldenDir() + "/scalar.bin";
    std::ifstream file(scalar_file, std::ios::binary);

    file.read(reinterpret_cast<char*>(&scalar), 4);
    file.close();
    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTREMSTestCase<caseId>(dstDevice, srcDevice, scalar, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

    // src copy mapped to dst-tile layout for the comparator's per-element slack
    std::vector<T> srcVec(dstByteSize / sizeof(T), T(0));
    for (int r = 0; r < (int)validRow; ++r) {
        for (int c = 0; c < srcValidCol; ++c) {
            srcVec[r * dstTileCol + c] = srcHost[r * col + c];
        }
    }

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstByteSize / sizeof(T));
    std::vector<T> devFinal(dstByteSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);

    bool ret = ResultCmpRemStrict<T>(golden, devFinal, srcVec, scalar);
    EXPECT_TRUE(ret);
}

// In-place aliasing probe: the dst GM buffer also feeds src, and the kernel
// TASSIGNs dst and src over the same UB region. Probe shapes are compact
// (valid == tile), so TSTORE rewrites every compared element and no memset of
// the aliased buffer is needed.
template <uint32_t caseId, typename T, int dstTileRow, int dstTileCol, int row, int validRow, int col, int srcValidCol>
void TREMSTestIpFramework()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t dstByteSize = dstTileRow * dstTileCol * sizeof(T);
    size_t srcByteSize = row * col * sizeof(T);
    T* dstHost;
    T* dstDevice;
    float scalar;

    aclrtMallocHost((void**)(&dstHost), dstByteSize);
    aclrtMalloc((void**)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, dstHost, srcByteSize);
    std::string scalar_file = GetGoldenDir() + "/scalar.bin";
    std::ifstream file(scalar_file, std::ios::binary);

    file.read(reinterpret_cast<char*>(&scalar), 4);
    file.close();
    aclrtMemcpy(dstDevice, srcByteSize, dstHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTREMSIpTestCase<caseId>(dstDevice, scalar, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

    // feed the real input to the comparator so the per-element slack matches
    // the stock harness (zero vectors would kill the 8-ulp tolerance)
    std::vector<T> srcVec(dstByteSize / sizeof(T), T(0));
    for (int r = 0; r < (int)validRow; ++r) {
        for (int c = 0; c < srcValidCol; ++c) {
            srcVec[r * dstTileCol + c] = dstHost[r * col + c];
        }
    }

    aclrtFree(dstDevice);
    aclrtFreeHost(dstHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstByteSize / sizeof(T));
    std::vector<T> devFinal(dstByteSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);

    bool ret = ResultCmpRemStrict<T>(golden, devFinal, srcVec, scalar);
    EXPECT_TRUE(ret);
}

// Disjoint control for the in-place probe kernels: same shapes, dst in its own
// GM buffer (memset like the stock harness).
template <uint32_t caseId, typename T, int dstTileRow, int dstTileCol, int row, int validRow, int col, int srcValidCol>
void TREMSTestDisjFramework()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t dstByteSize = dstTileRow * dstTileCol * sizeof(T);
    size_t srcByteSize = row * col * sizeof(T);
    T* dstHost;
    T* srcHost;
    T* dstDevice;
    T* srcDevice;
    float scalar;

    aclrtMallocHost((void**)(&dstHost), dstByteSize);
    aclrtMallocHost((void**)(&srcHost), srcByteSize);

    aclrtMalloc((void**)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(dstDevice, dstByteSize, 0, dstByteSize);

    ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);
    std::string scalar_file = GetGoldenDir() + "/scalar.bin";
    std::ifstream file(scalar_file, std::ios::binary);

    file.read(reinterpret_cast<char*>(&scalar), 4);
    file.close();
    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTREMSDisjTestCase<caseId>(dstDevice, srcDevice, scalar, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

    std::vector<T> srcVec(dstByteSize / sizeof(T), T(0));
    for (int r = 0; r < (int)validRow; ++r) {
        for (int c = 0; c < srcValidCol; ++c) {
            srcVec[r * dstTileCol + c] = srcHost[r * col + c];
        }
    }

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstByteSize / sizeof(T));
    std::vector<T> devFinal(dstByteSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);

    bool ret = ResultCmpRemStrict<T>(golden, devFinal, srcVec, scalar);
    EXPECT_TRUE(ret);
}

TEST_F(TREMSTest, case1) { TREMSTestFramework<1, float, 32, 64, 32, 32, 64, 64>(); }

TEST_F(TREMSTest, case3) { TREMSTestFramework<3, int32_t, 31, 128, 31, 31, 128, 128>(); }

TEST_F(TREMSTest, case5) { TREMSTestFramework<5, float, 7, 448, 7, 7, 448, 448>(); }

TEST_F(TREMSTest, case6) { TREMSTestFramework<6, float, 256, 16, 256, 256, 16, 16>(); }

TEST_F(TREMSTest, case7) { TREMSTestFramework<7, float, 32, 128, 32, 32, 64, 64>(); }

TEST_F(TREMSTest, case9) { TREMSTestFramework<9, int32_t, 31, 256, 31, 31, 128, 128>(); }

TEST_F(TREMSTest, case11) { TREMSTestFramework<11, float, 7, 512, 7, 7, 448, 448>(); }

TEST_F(TREMSTest, case12) { TREMSTestFramework<12, float, 256, 32, 256, 256, 16, 16>(); }

TEST_F(TREMSTest, case15) { TREMSTestFramework<15, int32_t, 1, 8192, 1, 1, 8192, 8192>(); }

TEST_F(TREMSTest, case16) { TREMSTestFramework<16, float, 1, 8192, 1, 1, 8192, 8192>(); }

// size matrix (float): repeat-boundary crossing and extremes
TEST_F(TREMSTest, case20) { TREMSTestFramework<20, float, 1, 64, 1, 1, 64, 64>(); }
TEST_F(TREMSTest, case21) { TREMSTestFramework<21, float, 1, 72, 1, 1, 72, 65>(); }
TEST_F(TREMSTest, case22) { TREMSTestFramework<22, float, 2, 128, 2, 2, 128, 127>(); }
TEST_F(TREMSTest, case23) { TREMSTestFramework<23, float, 2, 128, 2, 2, 128, 128>(); }
TEST_F(TREMSTest, case24) { TREMSTestFramework<24, float, 2, 136, 2, 2, 136, 129>(); }
TEST_F(TREMSTest, case25) { TREMSTestFramework<25, float, 3, 256, 3, 3, 256, 256>(); }
TEST_F(TREMSTest, case26) { TREMSTestFramework<26, float, 3, 264, 3, 3, 264, 257>(); }
TEST_F(TREMSTest, case27) { TREMSTestFramework<27, float, 8, 192, 8, 8, 192, 192>(); }
TEST_F(TREMSTest, case28) { TREMSTestFramework<28, float, 33, 64, 33, 33, 64, 64>(); }
TEST_F(TREMSTest, case29) { TREMSTestFramework<29, float, 1, 1024, 1, 1, 1024, 1024>(); }
TEST_F(TREMSTest, case30) { TREMSTestFramework<30, float, 2, 2048, 2, 2, 2048, 2048>(); }
TEST_F(TREMSTest, case31) { TREMSTestFramework<31, float, 4, 64, 4, 4, 64, 63>(); }
TEST_F(TREMSTest, case32) { TREMSTestFramework<32, float, 4, 8, 4, 4, 8, 8>(); }
TEST_F(TREMSTest, case33) { TREMSTestFramework<33, float, 2, 4096, 2, 2, 4096, 4096>(); }
TEST_F(TREMSTest, case34) { TREMSTestFramework<34, float, 2, 512, 2, 2, 384, 384>(); }
TEST_F(TREMSTest, case35) { TREMSTestFramework<35, float, 5, 128, 5, 5, 128, 128>(); }

// size matrix (int32)
TEST_F(TREMSTest, case40) { TREMSTestFramework<40, int32_t, 1, 64, 1, 1, 64, 64>(); }
TEST_F(TREMSTest, case41) { TREMSTestFramework<41, int32_t, 2, 72, 2, 2, 72, 65>(); }
TEST_F(TREMSTest, case42) { TREMSTestFramework<42, int32_t, 3, 136, 3, 3, 136, 129>(); }
TEST_F(TREMSTest, case43) { TREMSTestFramework<43, int32_t, 1, 8192, 1, 1, 8192, 8192>(); }
TEST_F(TREMSTest, case44) { TREMSTestFramework<44, int32_t, 2, 192, 2, 2, 192, 192>(); }
TEST_F(TREMSTest, case45) { TREMSTestFramework<45, int32_t, 2, 256, 2, 2, 192, 192>(); }

// high precision remainder (issue #2): staged >= 48 full vectors, register
// fallback, tails, strided dst (goldens are the true floored remainder %)
TEST_F(TREMSTest, case50) { TREMSTestFramework<50, float, 8, 448, 8, 8, 448, 448>(); }
TEST_F(TREMSTest, case51) { TREMSTestFramework<51, float, 12, 256, 12, 12, 256, 256>(); }
TEST_F(TREMSTest, case52) { TREMSTestFramework<52, float, 32, 64, 32, 32, 64, 64>(); }
TEST_F(TREMSTest, case53) { TREMSTestFramework<53, float, 7, 512, 7, 7, 448, 448>(); }
TEST_F(TREMSTest, case54) { TREMSTestFramework<54, float, 2, 2048, 2, 2, 2048, 2048>(); }
TEST_F(TREMSTest, case55) { TREMSTestFramework<55, float, 5, 320, 5, 5, 320, 320>(); }
TEST_F(TREMSTest, case56) { TREMSTestFramework<56, float, 8, 456, 8, 8, 456, 456>(); }
TEST_F(TREMSTest, case57) { TREMSTestFramework<57, float, 4, 136, 4, 4, 136, 136>(); }
TEST_F(TREMSTest, case58) { TREMSTestFramework<58, float, 1, 8192, 1, 1, 8192, 8192>(); }
TEST_F(TREMSTest, case59) { TREMSTestFramework<59, float, 16, 64, 16, 16, 64, 63>(); }

// In-place aliasing probes (dst tile shares src's GM buffer and UB region):
// DEFAULT float/int and the HP fallback must all stay alias-safe. Shapes are
// compact (valid == tile), so golden padding never participates.
TEST_F(TREMSTest, ip_def_float_33x64) { TREMSTestIpFramework<60, float, 33, 64, 33, 33, 64, 64>(); }
TEST_F(TREMSTest, ip_def_float_4x8) { TREMSTestIpFramework<61, float, 4, 8, 4, 4, 8, 8>(); }
TEST_F(TREMSTest, ip_def_float_1x8192) { TREMSTestIpFramework<62, float, 1, 8192, 1, 1, 8192, 8192>(); }
TEST_F(TREMSTest, ip_def_int32_2x192) { TREMSTestIpFramework<63, int32_t, 2, 192, 2, 2, 192, 192>(); }
TEST_F(TREMSTest, ip_hp_float_8x448) { TREMSTestIpFramework<64, float, 8, 448, 8, 8, 448, 448>(); }
TEST_F(TREMSTest, ip_hp_float_32x64) { TREMSTestIpFramework<65, float, 32, 64, 32, 32, 64, 64>(); }
// disjoint control validating the probe kernel on the stock layout
TEST_F(TREMSTest, ctrl_disj_def_float_33x64) { TREMSTestDisjFramework<66, float, 33, 64, 33, 33, 64, 64>(); }

// ===== HP exactness repro: 4580025 % -7.25 (returned -3.25 before the
// compensated final iteration). The HP path must be bit-exact here - both
// operands, the quotient and the true remainder are exactly representable,
// and A5's fused multiply-add delivers the exact value. =====
extern "C" void LaunchTREMHpExactRepro(float* out, float* src0, float* src1, float scalar, void* stream);

TEST_F(TREMSTest, hp_exact_repro)
{
    const float scalar = -7.25f;
    // expected HP results verified against A5 (fused vmula) and numpy
    const float expectedHp[8] = {-3.0f, -4.25f, 0.0f, -1.0f, -6.25f, -1.75f, -7.0f, -6.25f};
    std::vector<float> src = {4580025.0f, -4580025.0f, 4580028.0f, 100.5f, -100.5f, 16777216.0f, 4580021.0f, 1.0f};
    src.resize(64, 0.0f);
    std::vector<float> src1(64, scalar);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t fileSize = 64 * sizeof(float);
    float *hostOut, *devSrc0, *devSrc1, *devOut;
    aclrtMallocHost((void**)&hostOut, fileSize * 4);
    aclrtMalloc((void**)&devSrc0, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&devSrc1, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&devOut, fileSize * 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(devSrc0, fileSize, src.data(), fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devSrc1, fileSize, src1.data(), fileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTREMHpExactRepro(devOut, devSrc0, devSrc1, scalar, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(hostOut, fileSize * 4, devOut, fileSize * 4, ACL_MEMCPY_DEVICE_TO_HOST);

    // rows: 0 TREMS DEFAULT | 1 TREMS HP | 2 TREM DEFAULT | 3 TREM HP
    bool ok = true;
    for (int c = 0; c < 8; ++c) {
        if (hostOut[64 + c] != expectedHp[c] || hostOut[192 + c] != expectedHp[c]) {
            printf(
                "HP exactness MISMATCH col %d: TREMS %.9g TREM %.9g expected %.9g\n", c, hostOut[64 + c],
                hostOut[192 + c], expectedHp[c]);
            ok = false;
        }
    }

    aclrtFree(devSrc0);
    aclrtFree(devSrc1);
    aclrtFree(devOut);
    aclrtFreeHost(hostOut);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    EXPECT_TRUE(ok);
}
// ===== end HP exactness repro =====

// ===== HP exhaustive exactness matrix (value classes for the compensated
// final iteration): every hazard class of HpFusedSub as 64-lane vectors,
// asserted against a double-precision oracle. Lanes with quotient < 2^24 are
// asserted EXACTLY (the HP contract there is exactness, matching A5's fused
// vmula); wider quotients fall back to the 8-ulp cancellation slack. =====
extern "C" void LaunchTREMHpExactRepro(float* out, float* src0, float* src1, float scalar, void* stream);

static float HpOracleRem(float x, float y)
{
    if (std::isnan(x) || std::isnan(y) || std::isinf(x)) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (std::isinf(y)) {
        return x; // fmod semantics: x % +-inf = x
    }
    double xd = (double)x;
    double yd = (double)y;
    double q = std::floor(xd / yd);
    return (float)(xd - q * yd);
}

static float HpUlp(float v)
{
    v = std::fabs(v);
    if (std::isnan(v) || std::isinf(v)) {
        return std::numeric_limits<float>::infinity();
    }
    float next = std::nextafterf(v, std::numeric_limits<float>::infinity());
    return next > v ? next - v : (v > 0.0f ? v : std::numeric_limits<float>::denorm_min());
}

// Runs the 1x64 repro kernel and asserts both HP outputs (rows 1 and 3).
static void HpExactMatrixRun(const std::vector<float>& xv, const std::vector<float>& yv, float scalar)
{
    ASSERT_EQ(xv.size(), 64u);
    ASSERT_EQ(yv.size(), 64u);
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    size_t fileSize = 64 * sizeof(float);
    float *hostOut, *devSrc0, *devSrc1, *devOut;
    aclrtMallocHost((void**)&hostOut, fileSize * 4);
    aclrtMalloc((void**)&devSrc0, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&devSrc1, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&devOut, fileSize * 4, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(devSrc0, fileSize, xv.data(), fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devSrc1, fileSize, yv.data(), fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTREMHpExactRepro(devOut, devSrc0, devSrc1, scalar, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(hostOut, fileSize * 4, devOut, fileSize * 4, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtFree(devSrc0);
    aclrtFree(devSrc1);
    aclrtFree(devOut);
    aclrtFreeHost(hostOut);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    bool ok = true;
    for (unsigned i = 0; i < 64; ++i) {
        const float x = xv[i];
        const float y = yv[i];
        if (y == 0.0f || std::isnan(y)) {
            continue; // outside the HP contract
        }
        const float exp = HpOracleRem(x, y);
        // TREMS row only counts where the lane divisor equals the scalar
        const float rows[2] = {hostOut[64 + i], hostOut[192 + i]}; // {S-HP, V-HP}
        for (int op = 0; op < 2; ++op) {
            if (op == 0 && y != scalar) {
                continue;
            }
            const float act = rows[op];
            const char* opName = op == 0 ? "TREMS" : "TREM";
            bool laneOk;
            if (std::isnan(exp)) {
                laneOk = std::isnan(act);
            } else if (std::fabs((double)x) < std::fabs((double)y) * 16777216.0) {
                laneOk = (act == exp); // exact single-digit contract
            } else {
                float slack = 8.0f * (HpUlp(x) + HpUlp(y));
                laneOk = std::isfinite(act) ? std::fabs(act - exp) <= slack : std::fabs(exp) <= slack;
            }
            if (!laneOk) {
                printf("%s lane %u: x=%.9g y=%.9g expected %.9g actual %.9g\n", opName, i, x, y, exp, act);
                ok = false;
            }
        }
    }
    EXPECT_TRUE(ok);
}

static std::vector<float> FillLanes(std::initializer_list<float> vals)
{
    std::vector<float> v(vals);
    v.resize(64, vals.size() ? *(vals.begin()) : 0.0f);
    return v;
}

// lane i: x = k[i]*b + r[i] built exactly in double
static void BuildX(std::vector<float>& xv, const std::vector<double>& k, const std::vector<double>& r, double b)
{
    xv.clear();
    for (unsigned i = 0; i < 64; ++i) {
        xv.push_back((float)(k[i % k.size()] * b + r[i % r.size()]));
    }
}

TEST_F(TREMSTest, hp_val_k0_below) // |x| < |y|: k = 0
{
    float b = 7.25f;
    auto xv = FillLanes({0.0f, -0.0f, 3.625f, -7.2499f, 7.25e-40f, -1.1754944e-38f, 123.456f, -0.5f});
    auto yv = FillLanes({b});
    HpExactMatrixRun(xv, yv, b);
}

TEST_F(TREMSTest, hp_val_k_small_seam) // k = 1..4095 and the 4096/4097 split seams
{
    float b = 7.25f;
    std::vector<double> k = {1, 2, 3, 63, 64, 4094, 4095, 4096, 4097, 8191, 8192, 8193, 12287, 12288, 12289};
    std::vector<double> r = {0.25, 3.5, 7.0, 0.0};
    std::vector<float> xv;
    BuildX(xv, k, r, b);
    auto yv = FillLanes({b});
    HpExactMatrixRun(xv, yv, b);
}

TEST_F(TREMSTest, hp_val_k_pow2_rich_b) // power-of-two k with a full-mantissa b
{
    float b = 1.0000001f; // 24-bit significand
    std::vector<double> k = {65536, 1048576, 4194304, 8388607, 8388608, 8388609, 16777214, 16777215};
    std::vector<double> r = {0.25, 0.5, 0.75, 0.0625};
    std::vector<float> xv;
    BuildX(xv, k, r, b);
    auto yv = FillLanes({b});
    HpExactMatrixRun(xv, yv, b);
}

TEST_F(TREMSTest, hp_val_exact_multiples) // x = k*b exactly: r must be +-0
{
    float b = 7.25f;
    std::vector<double> k = {1, 13, 631727, 631728, 8388607, 8388608, 16777215, 1000000};
    std::vector<float> xv;
    BuildX(xv, k, {0.0}, b);
    auto yv = FillLanes({b});
    HpExactMatrixRun(xv, yv, b);
}

TEST_F(TREMSTest, hp_val_exact_multiples_rich_b)
{
    float b = 1.0000001f;
    std::vector<double> k = {1, 631727, 8388608, 16777215, 4194304};
    std::vector<float> xv;
    BuildX(xv, k, {0.0}, b);
    auto yv = FillLanes({b});
    HpExactMatrixRun(xv, yv, b);
}

TEST_F(TREMSTest, hp_val_multiple_pm_ulp) // one ulp around exact multiples
{
    float b = 7.25f;
    std::vector<double> k = {631727, 631728, 8388608, 16777215};
    std::vector<float> xv;
    BuildX(xv, k, {0.0}, b);
    // interleave +-ulp variants around the exact multiples
    std::vector<float> xk = {4580021.0f, 4580028.0f}; // 631727*b, 631728*b
    std::vector<float> built;
    for (unsigned i = 0; i < 64; ++i) {
        float base = xk[i % xk.size()];
        built.push_back(
            i % 4 < 2 ? std::nextafterf(base, std::numeric_limits<float>::infinity()) : std::nextafterf(base, 0.0f));
    }
    auto yv = FillLanes({b});
    HpExactMatrixRun(built, yv, b);
}

TEST_F(TREMSTest, hp_val_tie_both_directions) // half-ulp ties rounding up and down
{
    float b = 7.25f;
    // 631727*7.25 = 4580020.75 (tie, even side up); 631729*7.25 = 4580035.25
    // 3355443*7.25 = 24326971.75 (26-bit product); mix odd/even k for both tie sides
    std::vector<double> k = {631727, 631729, 3355443, 3355445, 999999, 1000001};
    std::vector<double> r = {0.0, 0.25, 1.5};
    std::vector<float> xv;
    BuildX(xv, k, r, b);
    auto yv = FillLanes({b});
    HpExactMatrixRun(xv, yv, b);
}

TEST_F(TREMSTest, hp_val_tiny_remainders) // collapse zone: r near 0 and near b
{
    float b = 7.25f;
    std::vector<double> k = {631727, 8388608};
    std::vector<double> r = {0.25, 0.0625, 0.001953125, 0.000244140625, 7.2497559, 7.125};
    std::vector<float> xv;
    BuildX(xv, k, r, b);
    auto yv = FillLanes({b});
    HpExactMatrixRun(xv, yv, b);
}

TEST_F(TREMSTest, hp_val_b_powers_of_two)
{
    auto xv = FillLanes({0.75f, 3.25f, 1e6f + 0.5f, 8388608.5f, 16777215.0f, 0.0f});
    auto yv = FillLanes({0.5f, 1.0f, 2.0f, 4.0f, 1024.0f, 1048576.0f});
    HpExactMatrixRun(xv, yv, 1.0f); // S asserts only the y==1.0 lanes
}

TEST_F(TREMSTest, hp_val_b_rich_mantissas)
{
    float b = 7.2499995f; // dense significand
    std::vector<double> k = {631727, 12345, 16777215};
    std::vector<double> r = {0.5, 3.75, 0.0};
    std::vector<float> xv;
    BuildX(xv, k, r, b);
    auto yv = FillLanes({b});
    HpExactMatrixRun(xv, yv, b);
}

TEST_F(TREMSTest, hp_val_guard_threshold) // |b| around FLT_MAX/4097 both sides
{
    float bSafe = std::numeric_limits<float>::max() / 4097.0f;
    float below = std::nextafterf(bSafe, 0.0f);
    float above = std::nextafterf(bSafe, std::numeric_limits<float>::infinity());
    auto yv = FillLanes({below, bSafe, above, 1e35f, 1e36f, std::numeric_limits<float>::max()});
    auto xv = FillLanes({1.0f, 2.5f, yv[0], yv[3] * 0.5f, 1e30f, 4095.5f});
    HpExactMatrixRun(xv, yv, below); // S asserts only the uniform==below lanes (none vary)
}

TEST_F(TREMSTest, hp_val_subnormal_divisor) // lift regime and its boundary
{
    float minNormal = 1.1754944e-38f;
    auto yv = FillLanes({1.4012985e-45f, 5e-40f, 2.9387359e-39f, minNormal, std::nextafterf(minNormal, 1e30f)});
    auto xv = FillLanes({1e-38f, 7e-39f, 2.5e-44f, minNormal * 3, 1e-30f, 0.0f});
    HpExactMatrixRun(xv, yv, minNormal);
}

TEST_F(TREMSTest, hp_val_denormal_remainder) // normal x, b with subnormal r
{
    float minNormal = 1.1754944e-38f;
    float b = minNormal; // r granularity 2^-149
    std::vector<double> k = {1, 3, 8191};
    std::vector<double> r = {1.4012985e-45, 7.0064923e-46, 2.8025969e-45}; // 1, 0.5, 2 quanta
    std::vector<float> xv;
    BuildX(xv, k, r, b);
    auto yv = FillLanes({b});
    HpExactMatrixRun(xv, yv, b);
}

TEST_F(TREMSTest, hp_val_sign_combos) // all four sign combos at the repro scale
{
    float b = 7.25f;
    std::vector<double> k = {631727, 631728};
    std::vector<double> r = {0.25, 4.25, 0.0};
    std::vector<float> xv;
    BuildX(xv, k, r, b);
    for (unsigned i = 0; i < 64; ++i) {
        xv[i] = (i / 8) % 2 ? -xv[i] : xv[i];
    }
    auto yv = FillLanes({b});
    for (unsigned i = 0; i < 64; ++i) {
        yv[i] = (i / 4) % 2 ? -b : b;
    }
    // V covers every lane; S is asserted on the +b half via a second run
    HpExactMatrixRun(xv, yv, b);
    for (auto& v : yv) {
        v = -b;
    }
    HpExactMatrixRun(xv, yv, -b);
}

TEST_F(TREMSTest, hp_val_special_x) // inf/NaN/0 dividends
{
    float b = 7.25f;
    float nan = std::numeric_limits<float>::quiet_NaN();
    float inf = std::numeric_limits<float>::infinity();
    auto xv = FillLanes({inf, -inf, nan, 0.0f, -0.0f, 4580025.0f});
    auto yv = FillLanes({b});
    HpExactMatrixRun(xv, yv, b);
}

TEST_F(TREMSTest, hp_val_inf_divisor)
{
    float inf = std::numeric_limits<float>::infinity();
    auto xv = FillLanes({4580025.0f, -3.0f, inf, -inf, 0.0f, 1e38f});
    auto yv = FillLanes({inf});
    HpExactMatrixRun(xv, yv, inf);
}

TEST_F(TREMSTest, hp_val_repro_family) // the original bug's neighborhood
{
    float b = 7.25f;
    std::vector<float> xv;
    for (unsigned i = 0; i < 64; ++i) {
        // k = 631727 or 631728, x off by whole units and fractions around 4580025
        double k = (i % 2) ? 631728 : 631727;
        double frac[8] = {0, 0.25, 0.5, 0.75, 1.0, 3.0, 4.25, -0.5};
        xv.push_back((float)(k * 7.25 + frac[(i / 2) % 8]));
    }
    auto yv = FillLanes({b});
    HpExactMatrixRun(xv, yv, b);
}

TEST_F(TREMSTest, hp_val_mixed_lanes) // one of each class in one vector (V-heavy)
{
    float bSafe = std::numeric_limits<float>::max() / 4097.0f;
    float minNormal = 1.1754944e-38f;
    float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> xv = {0.0f,        3.625f, 10.5f,  4580025.0f, -4580025.0f, 4580021.0f, 4580028.0f, 8388608.0f,
                             16777215.0f, 1e30f,  1e-38f, 7e-39f,     2.5e-44f,    1e35f,      4095.5f,    nan};
    std::vector<float> yv = {7.25f, 7.25f, 3.5f,  7.25f,  7.25f,          7.25f,     7.25f, 7.25f,
                             7.25f, 7.25f, 7.25f, 5e-40f, 1.4012985e-45f, minNormal, bSafe, 7.25f};
    xv.resize(64, 4580025.0f);
    yv.resize(64, 7.25f);
    HpExactMatrixRun(xv, yv, 7.25f);
}
// ===== end HP exhaustive exactness matrix =====

// ---- HP size matrix (compensated final iteration): staged row counts, the
// 48-vector gate +-1, tails, capVecs seams, strided dst, in-place fallback,
// thin-many-rows, max width ----
TEST_F(TREMSTest, hp_sz_2x448) { TREMSTestFramework<67, float, 2, 448, 2, 2, 448, 448>(); }
TEST_F(TREMSTest, hp_sz_3x448) { TREMSTestFramework<68, float, 3, 448, 3, 3, 448, 448>(); }
TEST_F(TREMSTest, hp_sz_5x448) { TREMSTestFramework<69, float, 5, 448, 5, 5, 448, 448>(); }
TEST_F(TREMSTest, hp_sz_13x448) { TREMSTestFramework<70, float, 13, 448, 13, 13, 448, 448>(); }
TEST_F(TREMSTest, hp_sz_26x448) { TREMSTestFramework<71, float, 26, 448, 26, 26, 448, 448>(); }
TEST_F(TREMSTest, hp_sz_49x64) { TREMSTestFramework<72, float, 49, 64, 49, 49, 64, 64>(); }
TEST_F(TREMSTest, hp_sz_47x64) { TREMSTestFramework<73, float, 47, 64, 47, 47, 64, 64>(); }
TEST_F(TREMSTest, hp_sz_100x64) { TREMSTestFramework<74, float, 100, 64, 100, 100, 64, 64>(); }
TEST_F(TREMSTest, hp_sz_255x64) { TREMSTestFramework<75, float, 255, 64, 255, 255, 64, 64>(); }
TEST_F(TREMSTest, hp_sz_8x457) { TREMSTestFramework<76, float, 8, 464, 8, 8, 464, 457>(); }
TEST_F(TREMSTest, hp_sz_8x449) { TREMSTestFramework<77, float, 8, 456, 8, 8, 456, 449>(); }
TEST_F(TREMSTest, hp_sz_3x65) { TREMSTestFramework<78, float, 3, 72, 3, 3, 72, 65>(); }
TEST_F(TREMSTest, hp_sz_1x128) { TREMSTestFramework<79, float, 1, 128, 1, 1, 128, 128>(); }
TEST_F(TREMSTest, hp_sz_1x192) { TREMSTestFramework<80, float, 1, 192, 1, 1, 192, 192>(); }
TEST_F(TREMSTest, hp_sz_1x320) { TREMSTestFramework<81, float, 1, 320, 1, 1, 320, 320>(); }
TEST_F(TREMSTest, hp_sz_1x576) { TREMSTestFramework<82, float, 1, 576, 1, 1, 576, 576>(); }
TEST_F(TREMSTest, hp_sz_1x4096) { TREMSTestFramework<83, float, 1, 4096, 1, 1, 4096, 4096>(); }
TEST_F(TREMSTest, hp_sz_1x6400) { TREMSTestFramework<84, float, 1, 6400, 1, 1, 6400, 6400>(); }
TEST_F(TREMSTest, hp_sz_1x11136) { TREMSTestFramework<85, float, 1, 11136, 1, 1, 11136, 11136>(); }
TEST_F(TREMSTest, hp_sz_str_2x256_192) { TREMSTestFramework<86, float, 2, 256, 2, 2, 192, 192>(); }
TEST_F(TREMSTest, hp_sz_str_4x128_65) { TREMSTestFramework<87, float, 4, 128, 4, 4, 72, 65>(); }
TEST_F(TREMSTest, ip_hp_2x2048) { TREMSTestIpFramework<88, float, 2, 2048, 2, 2, 2048, 2048>(); }
TEST_F(TREMSTest, ip_hp_49x64) { TREMSTestIpFramework<89, float, 49, 64, 49, 49, 64, 64>(); }
TEST_F(TREMSTest, hp_sz_6x640) { TREMSTestFramework<90, float, 6, 640, 6, 6, 640, 640>(); }
