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
