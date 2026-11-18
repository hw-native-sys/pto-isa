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
#include <gtest/gtest.h>
#include <acl/acl.h>
#include <cmath>
#include <vector>
#include <cstdint>

using namespace PtoTestCommon;

extern "C" {
void LaunchTMulaSign_f32_ff(float*, float*, float*, float*, void*);
void LaunchTMulaSign_f32_tf(float*, float*, float*, float*, void*);
void LaunchTMulaSign_f32_tt(float*, float*, float*, float*, void*);
void LaunchTMulaSign_f32_ft(float*, float*, float*, float*, void*);
void LaunchTMulaSign_f16_ff(aclFloat16*, aclFloat16*, aclFloat16*, aclFloat16*, void*);
void LaunchTMulaSign_f16_tf(aclFloat16*, aclFloat16*, aclFloat16*, aclFloat16*, void*);
void LaunchTMulaSign_f16_tt(aclFloat16*, aclFloat16*, aclFloat16*, aclFloat16*, void*);
void LaunchTMulaSign_f16_ft(aclFloat16*, aclFloat16*, aclFloat16*, aclFloat16*, void*);
void LaunchTMulaSign_f32_tf_1x16(float*, float*, float*, float*, void*);
void LaunchTMulaSign_f32_tf_8x32(float*, float*, float*, float*, void*);
void LaunchTMulaSign_f32_tf_1x4096(float*, float*, float*, float*, void*);
void LaunchTMulaSign_f32_tf_v4x20(float*, float*, float*, float*, void*);
void LaunchTMulaSign_f32_tt_v5x12(float*, float*, float*, float*, void*);
void LaunchTMulaSign_f16_tf_v4x37(aclFloat16*, aclFloat16*, aclFloat16*, aclFloat16*, void*);
void LaunchTMulaSign_f32_tf_extreme(float*, float*, float*, float*, void*);
void LaunchTMulaSign_f32_tf_64x64(float*, float*, float*, float*, void*);
void LaunchTMulaSign_f32_tt_16x512(float*, float*, float*, float*, void*);
void LaunchTMulaSign_f32_ff_2728x8(float*, float*, float*, float*, void*);
void LaunchTMulaSign_f16_tf_64x128(aclFloat16*, aclFloat16*, aclFloat16*, aclFloat16*, void*);
void LaunchTMulaSign_f32_tf_v3x200(float*, float*, float*, float*, void*);
void LaunchTMulaSign_f32_ff_zeros(float*, float*, float*, float*, void*);
void LaunchTMulaSign_f32_tf_inf(float*, float*, float*, float*, void*);
void LaunchTMulaSign_f32_tt_denorm(float*, float*, float*, float*, void*);
void LaunchTMulaSign_f32_ft_overflow(float*, float*, float*, float*, void*);
}

namespace {

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    return "../" + std::string(testInfo->test_suite_name()) + "." + testInfo->name();
}

// aclFloat16 is a raw uint16_t typedef on the NPU host build: a plain static_cast
// decodes the bits as an integer. Route it through aclFloat16ToFloat instead.
template <typename T>
float ToFloat(T v)
{
    return static_cast<float>(v);
}
float ToFloat(aclFloat16 v)
{
#if defined(__CPU_SIM) || defined(__COSTMODEL)
    return static_cast<float>(v);
#else
    return aclFloat16ToFloat(v);
#endif
}

template <typename T>
void testTMulaSign(int sign, const char* goldenDir)
{
    const int total = 64;
    size_t fileSize = total * sizeof(T);
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    T *dstHost, *s0Host, *s1Host, *accHost;
    T *dstDev, *s0Dev, *s1Dev, *accDev;
    aclrtMallocHost((void**)&dstHost, fileSize);
    aclrtMallocHost((void**)&s0Host, fileSize);
    aclrtMallocHost((void**)&s1Host, fileSize);
    aclrtMallocHost((void**)&accHost, fileSize);
    aclrtMalloc((void**)&dstDev, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&s0Dev, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&s1Dev, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&accDev, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ReadFile(std::string(goldenDir) + "/input0.bin", fileSize, s0Host, fileSize);
    ReadFile(std::string(goldenDir) + "/input1.bin", fileSize, s1Host, fileSize);
    ReadFile(std::string(goldenDir) + "/input_dst.bin", fileSize, accHost, fileSize);
    aclrtMemcpy(s0Dev, fileSize, s0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(s1Dev, fileSize, s1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(accDev, fileSize, accHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if constexpr (std::is_same_v<T, float>) {
        typedef void (*Fn)(float*, float*, float*, float*, void*);
        Fn fns[] = {LaunchTMulaSign_f32_ff, LaunchTMulaSign_f32_tf, LaunchTMulaSign_f32_tt, LaunchTMulaSign_f32_ft};
        fns[sign](dstDev, s0Dev, s1Dev, accDev, stream);
    } else {
        typedef void (*Fn)(aclFloat16*, aclFloat16*, aclFloat16*, aclFloat16*, void*);
        Fn fns[] = {LaunchTMulaSign_f16_ff, LaunchTMulaSign_f16_tf, LaunchTMulaSign_f16_tt, LaunchTMulaSign_f16_ft};
        fns[sign](dstDev, s0Dev, s1Dev, accDev, stream);
    }
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDev, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    std::vector<T> golden(total);
    ReadFile(std::string(goldenDir) + "/golden.bin", fileSize, golden.data(), fileSize);
    for (int i = 0; i < total; ++i) {
        float expected = ToFloat(golden[i]);
        float actual = ToFloat(dstHost[i]);
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(actual)) << "idx=" << i;
            continue;
        }
        if (std::isinf(expected)) {
            EXPECT_FLOAT_EQ(expected, actual) << "idx=" << i;
            continue;
        }
        float diff = std::abs(expected - actual);
        float ratio = diff / std::max(std::abs(expected), 1e-6f);
        EXPECT_LE(ratio, 0.001f) << "idx=" << i << " sign=" << sign << " expected=" << expected << " actual=" << actual;
    }
    aclrtFree(dstDev);
    aclrtFree(s0Dev);
    aclrtFree(s1Dev);
    aclrtFree(accDev);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(s0Host);
    aclrtFreeHost(s1Host);
    aclrtFreeHost(accHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

template <typename T>
void testSimple(void (*fn)(T*, T*, T*, T*, void*), int total, const char* dir)
{
    size_t fileSize = total * sizeof(T);
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    T *dstHost, *s0Host, *s1Host, *accHost;
    T *dstDev, *s0Dev, *s1Dev, *accDev;
    aclrtMallocHost((void**)&dstHost, fileSize);
    aclrtMallocHost((void**)&s0Host, fileSize);
    aclrtMallocHost((void**)&s1Host, fileSize);
    aclrtMallocHost((void**)&accHost, fileSize);
    aclrtMalloc((void**)&dstDev, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&s0Dev, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&s1Dev, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&accDev, fileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    // bounded cases only store the valid region; keep the rest deterministic
    aclrtMemset(dstDev, fileSize, 0, fileSize);
    ReadFile(std::string(dir) + "/input0.bin", fileSize, s0Host, fileSize);
    ReadFile(std::string(dir) + "/input1.bin", fileSize, s1Host, fileSize);
    ReadFile(std::string(dir) + "/input_dst.bin", fileSize, accHost, fileSize);
    aclrtMemcpy(s0Dev, fileSize, s0Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(s1Dev, fileSize, s1Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(accDev, fileSize, accHost, fileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    fn(dstDev, s0Dev, s1Dev, accDev, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, fileSize, dstDev, fileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    std::vector<T> golden(total);
    ReadFile(std::string(dir) + "/golden.bin", fileSize, golden.data(), fileSize);
    for (int i = 0; i < total; ++i) {
        float e = ToFloat(golden[i]);
        float a = ToFloat(dstHost[i]);
        if (std::isnan(e)) {
            EXPECT_TRUE(std::isnan(a)) << "idx=" << i;
            continue;
        }
        if (std::isinf(e)) {
            EXPECT_FLOAT_EQ(e, a) << "idx=" << i;
            continue;
        }
        float d = std::abs(e - a);
        float r = d / std::max(std::abs(e), 1e-6f);
        EXPECT_LE(r, 0.001f) << "idx=" << i << " exp=" << e << " act=" << a;
    }
    aclrtFree(dstDev);
    aclrtFree(s0Dev);
    aclrtFree(s1Dev);
    aclrtFree(accDev);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(s0Host);
    aclrtFreeHost(s1Host);
    aclrtFreeHost(accHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

} // namespace

TEST(TMULASignTest, f32_ff) { testTMulaSign<float>(0, GetGoldenDir().c_str()); }
TEST(TMULASignTest, f32_tf) { testTMulaSign<float>(1, GetGoldenDir().c_str()); }
TEST(TMULASignTest, f32_tt) { testTMulaSign<float>(2, GetGoldenDir().c_str()); }
TEST(TMULASignTest, f32_ft) { testTMulaSign<float>(3, GetGoldenDir().c_str()); }
TEST(TMULASignTest, f16_ff) { testTMulaSign<aclFloat16>(0, GetGoldenDir().c_str()); }
TEST(TMULASignTest, f16_tf) { testTMulaSign<aclFloat16>(1, GetGoldenDir().c_str()); }
TEST(TMULASignTest, f16_tt) { testTMulaSign<aclFloat16>(2, GetGoldenDir().c_str()); }
TEST(TMULASignTest, f16_ft) { testTMulaSign<aclFloat16>(3, GetGoldenDir().c_str()); }
TEST(TMULASignTest, f32_tf_1x16) { testSimple<float>(LaunchTMulaSign_f32_tf_1x16, 16, GetGoldenDir().c_str()); }
TEST(TMULASignTest, f32_tf_8x32) { testSimple<float>(LaunchTMulaSign_f32_tf_8x32, 256, GetGoldenDir().c_str()); }
TEST(TMULASignTest, f32_tf_1x4096) { testSimple<float>(LaunchTMulaSign_f32_tf_1x4096, 4096, GetGoldenDir().c_str()); }
TEST(TMULASignTest, f32_tf_v4x20) { testSimple<float>(LaunchTMulaSign_f32_tf_v4x20, 128, GetGoldenDir().c_str()); }
TEST(TMULASignTest, f32_tt_v5x12) { testSimple<float>(LaunchTMulaSign_f32_tt_v5x12, 128, GetGoldenDir().c_str()); }
TEST(TMULASignTest, f16_tf_v4x37) { testSimple<aclFloat16>(LaunchTMulaSign_f16_tf_v4x37, 256, GetGoldenDir().c_str()); }
TEST(TMULASignTest, f32_tf_extreme) { testSimple<float>(LaunchTMulaSign_f32_tf_extreme, 64, GetGoldenDir().c_str()); }
// multi-VL / large tile stress
TEST(TMULASignTest, f32_tf_64x64) { testSimple<float>(LaunchTMulaSign_f32_tf_64x64, 4096, GetGoldenDir().c_str()); }
TEST(TMULASignTest, f32_tt_16x512) { testSimple<float>(LaunchTMulaSign_f32_tt_16x512, 8192, GetGoldenDir().c_str()); }
TEST(TMULASignTest, f32_ff_2728x8) { testSimple<float>(LaunchTMulaSign_f32_ff_2728x8, 21824, GetGoldenDir().c_str()); }
TEST(TMULASignTest, f16_tf_64x128)
{
    testSimple<aclFloat16>(LaunchTMulaSign_f16_tf_64x128, 8192, GetGoldenDir().c_str());
}
TEST(TMULASignTest, f32_tf_v3x200) { testSimple<float>(LaunchTMulaSign_f32_tf_v3x200, 1024, GetGoldenDir().c_str()); }
// dedicated extreme-value categories
TEST(TMULASignTest, f32_ff_zeros) { testSimple<float>(LaunchTMulaSign_f32_ff_zeros, 64, GetGoldenDir().c_str()); }
TEST(TMULASignTest, f32_tf_inf) { testSimple<float>(LaunchTMulaSign_f32_tf_inf, 64, GetGoldenDir().c_str()); }
TEST(TMULASignTest, f32_tt_denorm) { testSimple<float>(LaunchTMulaSign_f32_tt_denorm, 64, GetGoldenDir().c_str()); }
TEST(TMULASignTest, f32_ft_overflow) { testSimple<float>(LaunchTMulaSign_f32_ft_overflow, 64, GetGoldenDir().c_str()); }
