/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <fstream>

#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

// FP8 wrappers for testing
struct fp8_e4m3_wrapper {
    int8_t value;
    operator int8_t() const { return value; }
    operator float() const { return static_cast<float>(value); }
};
struct fp8_e5m2_wrapper {
    int8_t value;
    operator int8_t() const { return value; }
    operator float() const { return static_cast<float>(value); }
};
struct hifloat8_wrapper {
    int8_t value;
    operator int8_t() const { return value; }
    operator float() const { return static_cast<float>(value); }
};
struct fp4_e1m2x2_wrapper {
    uint8_t value;
    operator uint8_t() const { return value; }
};
struct fp4_e2m1x2_wrapper {
    uint8_t value;
    operator uint8_t() const { return value; }
};
struct bf16_wrapper {
    uint16_t value;
    operator uint16_t() const { return value; }
};

template <
    typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kValidRows_ = kTRows_,
    int kValidCols_ = kTCols_>
void launchTCVT(D* dst, S* src, void* stream);

template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void launchTCVTSaturationTest(D* dstSaturated, D* dstTruncated, D* dstDefault, S* src, void* stream);

namespace {
bool checkAcl(aclError status, const char* operation, bool& healthy)
{
    if (status != ACL_SUCCESS) {
        healthy = false;
        ADD_FAILURE() << operation << ": ACL error " << status << "; stopping device work";
        return false;
    }
    return true;
}

bool checkFile(bool success, const char* operation, const std::string& path, bool& healthy)
{
    if (!success) {
        healthy = false;
        ADD_FAILURE() << operation << ": " << path << "; stopping device work";
    }
    return success;
}

bool readFileExact(const std::string& path, size_t bytes, void* buffer, bool& healthy)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!checkFile(file.is_open(), "open for reading", path, healthy)) {
        return false;
    }
    bool success = checkFile(file.tellg() == static_cast<std::streamoff>(bytes), "read size", path, healthy);
    if (success) {
        file.seekg(0, std::ios::beg);
        file.read(static_cast<char*>(buffer), static_cast<std::streamsize>(bytes));
        success = checkFile(file.good() && file.gcount() == static_cast<std::streamsize>(bytes), "read", path, healthy);
    }
    file.clear();
    file.close();
    bool closed = checkFile(!file.fail(), "close after reading", path, healthy);
    return success && closed;
}

bool writeFileExact(const std::string& path, const void* buffer, size_t bytes, bool& healthy)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!checkFile(file.is_open(), "open for writing", path, healthy)) {
        return false;
    }
    file.write(static_cast<const char*>(buffer), static_cast<std::streamsize>(bytes));
    bool success = checkFile(file.good(), "write", path, healthy);
    file.clear();
    file.close();
    bool closed = checkFile(!file.fail(), "close after writing", path, healthy);
    return success && closed;
}

template <typename T, bool Device>
class AclBuffer {
public:
    explicit AclBuffer(bool& healthy) : healthy_(healthy) {}
    AclBuffer(const AclBuffer&) = delete;
    AclBuffer& operator=(const AclBuffer&) = delete;
    ~AclBuffer()
    {
        if (data != nullptr) {
            if constexpr (Device) {
                checkAcl(aclrtFree(data), "aclrtFree", healthy_);
            } else {
                checkAcl(aclrtFreeHost(data), "aclrtFreeHost", healthy_);
            }
        }
    }
    bool allocate(size_t bytes)
    {
        void* acquired = nullptr;
        aclError status;
        if constexpr (Device) {
            status = aclrtMalloc(&acquired, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        } else {
            status = aclrtMallocHost(&acquired, bytes);
        }
        if (!checkAcl(status, Device ? "aclrtMalloc" : "aclrtMallocHost", healthy_)) {
            return false;
        }
        data = static_cast<T*>(acquired);
        return true;
    }
    T* data = nullptr;

private:
    bool& healthy_;
};
} // namespace

class TCVTTest : public testing::Test {
protected:
    static void SetUpTestSuite()
    {
        ASSERT_TRUE(checkAcl(aclInit(nullptr), "aclInit", healthy));
        initialized = true;
        ASSERT_TRUE(checkAcl(aclrtSetDevice(0), "aclrtSetDevice", healthy));
        deviceOpened = true;
        aclrtContext acquired = nullptr;
        ASSERT_TRUE(checkAcl(aclrtCreateContext(&acquired, 0), "aclrtCreateContext", healthy));
        context = acquired;
        healthy = true;
    }
    static void TearDownTestSuite()
    {
        if (context != nullptr) {
            checkAcl(aclrtDestroyContext(context), "aclrtDestroyContext", healthy);
            context = nullptr;
        }
        if (deviceOpened) {
            checkAcl(aclrtResetDevice(0), "aclrtResetDevice", healthy);
            deviceOpened = false;
        }
        if (initialized) {
            checkAcl(aclFinalize(), "aclFinalize", healthy);
            initialized = false;
        }
        healthy = false;
    }
    void SetUp() override
    {
        ASSERT_TRUE(healthy) << "Device work stopped after an infrastructure error";
        ASSERT_TRUE(checkAcl(aclrtSetCurrentContext(context), "aclrtSetCurrentContext", healthy));
        aclrtStream acquired = nullptr;
        ASSERT_TRUE(checkAcl(aclrtCreateStream(&acquired), "aclrtCreateStream", healthy));
        stream = acquired;
    }
    void TearDown() override
    {
        if (stream != nullptr) {
            checkAcl(aclrtDestroyStream(stream), "aclrtDestroyStream", healthy);
            stream = nullptr;
        }
    }
    template <
        typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kValidRows_ = kTRows_,
        int kValidCols_ = kTCols_>
    void test_tcvt();
    template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
    void test_tcvt_saturation();

    aclrtStream stream = nullptr;
    inline static aclrtContext context = nullptr;
    inline static bool initialized = false;
    inline static bool deviceOpened = false;
    inline static bool healthy = false;
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_, int kValidRows_, int kValidCols_>
void TCVTTest::test_tcvt()
{
    uint32_t M = kGRows_;
    uint32_t N = kGCols_;

    constexpr bool isFp4Dst = std::is_same_v<D, fp4_e1m2x2_wrapper> || std::is_same_v<D, fp4_e2m1x2_wrapper>;
    constexpr bool isFp4Src = std::is_same_v<S, fp4_e1m2x2_wrapper> || std::is_same_v<S, fp4_e2m1x2_wrapper>;
    size_t srcCols = isFp4Src ? ((N + 1) / 2) : N;
    size_t dstCols = isFp4Dst ? ((N + 1) / 2) : N;

    size_t srcFileSize = M * srcCols * sizeof(S);
    size_t dstFileSize = M * dstCols * sizeof(D);

    AclBuffer<D, false> dstHost(healthy);
    AclBuffer<S, false> srcHost(healthy);
    AclBuffer<D, true> dstDevice(healthy);
    AclBuffer<S, true> srcDevice(healthy);
    ASSERT_TRUE(dstHost.allocate(dstFileSize));
    ASSERT_TRUE(srcHost.allocate(srcFileSize));
    ASSERT_TRUE(readFileExact(GetGoldenDir() + "/x1_gm.bin", srcFileSize, srcHost.data, healthy));
    ASSERT_TRUE(dstDevice.allocate(dstFileSize));
    // TSTORE writes only the valid region; the existing golden expects zeros elsewhere.
    ASSERT_TRUE(checkAcl(aclrtMemset(dstDevice.data, dstFileSize, 0, dstFileSize), "aclrtMemset dst", healthy));
    ASSERT_TRUE(srcDevice.allocate(srcFileSize));

    ASSERT_TRUE(checkAcl(
        aclrtMemcpy(srcDevice.data, srcFileSize, srcHost.data, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE), "src H2D",
        healthy));
    launchTCVT<D, S, kGRows_, kGCols_, kTRows_, kTCols_, kValidRows_, kValidCols_>(
        dstDevice.data, srcDevice.data, stream);
    ASSERT_TRUE(checkAcl(aclrtGetLastError(ACL_RT_THREAD_LEVEL), "kernel launch", healthy));
    ASSERT_TRUE(checkAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream", healthy));
    ASSERT_TRUE(checkAcl(
        aclrtMemcpy(dstHost.data, dstFileSize, dstDevice.data, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST), "dst D2H",
        healthy));
    ASSERT_TRUE(writeFileExact(GetGoldenDir() + "/output_z.bin", dstHost.data, dstFileSize, healthy));

    std::vector<D> golden(dstFileSize / sizeof(D));
    std::vector<D> devFinal(dstFileSize / sizeof(D));
    size_t goldenFileSize = dstFileSize;
    size_t outputFileSize = dstFileSize;
    ASSERT_TRUE(readFileExact(GetGoldenDir() + "/golden.bin", goldenFileSize, golden.data(), healthy));
    ASSERT_TRUE(readFileExact(GetGoldenDir() + "/output_z.bin", outputFileSize, devFinal.data(), healthy));

    bool ret = ResultCmp<D>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

#define GENERATE_TCVT_TESTS(dst_type, src_type, type_name)                                                        \
    TEST_F(TCVTTest, case_##type_name##_1x128) { test_tcvt<dst_type, src_type, 1, 128, 1, 128>(); }               \
    TEST_F(TCVTTest, case_##type_name##_2x64) { test_tcvt<dst_type, src_type, 2, 64, 2, 64>(); }                  \
    TEST_F(TCVTTest, case_##type_name##_4x32) { test_tcvt<dst_type, src_type, 4, 32, 4, 32>(); }                  \
    TEST_F(TCVTTest, case_##type_name##_2x128) { test_tcvt<dst_type, src_type, 2, 128, 2, 128>(); }               \
    TEST_F(TCVTTest, case_##type_name##_4x128_4x65) { test_tcvt<dst_type, src_type, 4, 128, 4, 128, 4, 65>(); }   \
    TEST_F(TCVTTest, case_##type_name##_4x256_4x200) { test_tcvt<dst_type, src_type, 4, 256, 4, 256, 4, 200>(); } \
    TEST_F(TCVTTest, case_##type_name##_1x256_1x129) { test_tcvt<dst_type, src_type, 1, 256, 1, 256, 1, 129>(); }

// fp4 packs 2 nibbles/byte: tile Cols are nibble-counted, so a [4,32] fp4 tile
// has 16B rows — below the 32B vec row alignment (enforced by the pto_tile
// static_assert; the CPU harness never used sub-64-nibble fp4 shapes either).
// fp4 keeps every other shape (all >= 64 nibbles = 32B rows).
#define GENERATE_TCVT_TESTS_FP4(dst_type, src_type, type_name)                                                    \
    TEST_F(TCVTTest, case_##type_name##_1x128) { test_tcvt<dst_type, src_type, 1, 128, 1, 128>(); }               \
    TEST_F(TCVTTest, case_##type_name##_2x64) { test_tcvt<dst_type, src_type, 2, 64, 2, 64>(); }                  \
    TEST_F(TCVTTest, case_##type_name##_2x128) { test_tcvt<dst_type, src_type, 2, 128, 2, 128>(); }               \
    TEST_F(TCVTTest, case_##type_name##_4x128_4x65) { test_tcvt<dst_type, src_type, 4, 128, 4, 128, 4, 65>(); }   \
    TEST_F(TCVTTest, case_##type_name##_4x256_4x200) { test_tcvt<dst_type, src_type, 4, 256, 4, 256, 4, 200>(); } \
    TEST_F(TCVTTest, case_##type_name##_1x256_1x129) { test_tcvt<dst_type, src_type, 1, 256, 1, 256, 1, 129>(); }

// FP32 Source → fp16, bf16, int16, int32, int64, fp8 variants
GENERATE_TCVT_TESTS(aclFloat16, float, fp32_fp16)
GENERATE_TCVT_TESTS(aclFloat16, float, fp32_bf16)
GENERATE_TCVT_TESTS(int16_t, float, fp32_int16)
GENERATE_TCVT_TESTS(int32_t, float, fp32_int32)
GENERATE_TCVT_TESTS(int64_t, float, fp32_int64)
GENERATE_TCVT_TESTS(fp8_e4m3_wrapper, float, fp32_fp8_e4m3)
GENERATE_TCVT_TESTS(fp8_e5m2_wrapper, float, fp32_fp8_e5m2)
GENERATE_TCVT_TESTS(hifloat8_wrapper, float, fp32_h8)
GENERATE_TCVT_TESTS(float, float, fp32_fp32)

// FP16 Source → fp32, int32, int16, int8, uint8, h8
GENERATE_TCVT_TESTS(float, aclFloat16, fp16_fp32)
GENERATE_TCVT_TESTS(int32_t, aclFloat16, fp16_int32)
GENERATE_TCVT_TESTS(int16_t, aclFloat16, fp16_int16)
GENERATE_TCVT_TESTS(int8_t, aclFloat16, fp16_int8)
GENERATE_TCVT_TESTS(uint8_t, aclFloat16, fp16_uint8)
GENERATE_TCVT_TESTS(hifloat8_wrapper, aclFloat16, fp16_h8)

// BF16 Source → fp32, int32, half, fp4
GENERATE_TCVT_TESTS(float, aclFloat16, bf16_fp32)
GENERATE_TCVT_TESTS(int32_t, aclFloat16, bf16_int32)
// GENERATE_TCVT_TESTS(aclFloat16, bfloat16_t, bf16_fp16)
GENERATE_TCVT_TESTS_FP4(fp4_e1m2x2_wrapper, bf16_wrapper, bf16_fp4_e1m2x2)
GENERATE_TCVT_TESTS_FP4(fp4_e2m1x2_wrapper, bf16_wrapper, bf16_fp4_e2m1x2)

// FP4 Source → bf16
GENERATE_TCVT_TESTS_FP4(bf16_wrapper, fp4_e1m2x2_wrapper, fp4_e1m2x2_bf16)
GENERATE_TCVT_TESTS_FP4(bf16_wrapper, fp4_e2m1x2_wrapper, fp4_e2m1x2_bf16)

// U8 Source → half, uint16
GENERATE_TCVT_TESTS(aclFloat16, uint8_t, uint8_fp16)
// GENERATE_TCVT_TESTS(uint16_t, uint8_t, uint8_uint16)

// I8 Source → half, int16, int32
GENERATE_TCVT_TESTS(aclFloat16, int8_t, int8_fp16)
GENERATE_TCVT_TESTS(int16_t, int8_t, int8_int16)
GENERATE_TCVT_TESTS(int32_t, int8_t, int8_int32)

// I16 Source → uint8, half, float, uint32, int32
GENERATE_TCVT_TESTS(uint8_t, int16_t, int16_uint8)
GENERATE_TCVT_TESTS(aclFloat16, int16_t, int16_fp16)
GENERATE_TCVT_TESTS(float, int16_t, int16_fp32)
GENERATE_TCVT_TESTS(uint32_t, int16_t, int16_uint32)
GENERATE_TCVT_TESTS(int32_t, int16_t, int16_int32)

// I32 Source → float, int16, uint16, int64, uint8
GENERATE_TCVT_TESTS(float, int32_t, int32_fp32)
GENERATE_TCVT_TESTS(int16_t, int32_t, int32_int16)
// GENERATE_TCVT_TESTS(uint16_t, int32_t, int32_uint16)
GENERATE_TCVT_TESTS(int64_t, int32_t, int32_int64)
GENERATE_TCVT_TESTS(uint8_t, int32_t, int32_uint8)

// U32 Source → uint8, uint16, int16
GENERATE_TCVT_TESTS(uint8_t, uint32_t, uint32_uint8)
// GENERATE_TCVT_TESTS(uint16_t, uint32_t, uint32_uint16)
GENERATE_TCVT_TESTS(int16_t, uint32_t, uint32_int16)

// I64 Source → float, int32
GENERATE_TCVT_TESTS(float, int64_t, int64_fp32)
GENERATE_TCVT_TESTS(int32_t, int64_t, int64_int32)

// FP8 Source → float
GENERATE_TCVT_TESTS(float, fp8_e4m3_wrapper, fp8_e4m3_fp32)
GENERATE_TCVT_TESTS(float, fp8_e5m2_wrapper, fp8_e5m2_fp32)
GENERATE_TCVT_TESTS(float, hifloat8_wrapper, h8_fp32)

// ============================================================================
// Saturation Mode Tests
// ============================================================================

template <typename D, typename S, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void TCVTTest::test_tcvt_saturation()
{
    uint32_t M = kGRows_;
    uint32_t N = kGCols_;

    size_t srcFileSize = M * N * sizeof(S);
    size_t dstFileSize = M * N * sizeof(D);

    AclBuffer<D, false> dstSatHost(healthy), dstTruncHost(healthy), dstDefaultHost(healthy);
    AclBuffer<S, false> srcHost(healthy);
    AclBuffer<D, true> dstSatDevice(healthy), dstTruncDevice(healthy), dstDefaultDevice(healthy);
    AclBuffer<S, true> srcDevice(healthy);
    ASSERT_TRUE(dstSatHost.allocate(dstFileSize));
    ASSERT_TRUE(dstTruncHost.allocate(dstFileSize));
    ASSERT_TRUE(dstDefaultHost.allocate(dstFileSize));
    ASSERT_TRUE(srcHost.allocate(srcFileSize));
    ASSERT_TRUE(readFileExact(GetGoldenDir() + "/x1_gm.bin", srcFileSize, srcHost.data, healthy));
    ASSERT_TRUE(dstSatDevice.allocate(dstFileSize));
    ASSERT_TRUE(checkAcl(aclrtMemset(dstSatDevice.data, dstFileSize, 0, dstFileSize), "aclrtMemset dstSat", healthy));
    ASSERT_TRUE(dstTruncDevice.allocate(dstFileSize));
    ASSERT_TRUE(
        checkAcl(aclrtMemset(dstTruncDevice.data, dstFileSize, 0, dstFileSize), "aclrtMemset dstTrunc", healthy));
    ASSERT_TRUE(dstDefaultDevice.allocate(dstFileSize));
    ASSERT_TRUE(
        checkAcl(aclrtMemset(dstDefaultDevice.data, dstFileSize, 0, dstFileSize), "aclrtMemset dstDefault", healthy));
    ASSERT_TRUE(srcDevice.allocate(srcFileSize));

    ASSERT_TRUE(checkAcl(
        aclrtMemcpy(srcDevice.data, srcFileSize, srcHost.data, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE), "src H2D",
        healthy));

    // Run saturation test - produces saturated, truncated, and default outputs
    launchTCVTSaturationTest<D, S, kGRows_, kGCols_, kTRows_, kTCols_>(
        dstSatDevice.data, dstTruncDevice.data, dstDefaultDevice.data, srcDevice.data, stream);
    ASSERT_TRUE(checkAcl(aclrtGetLastError(ACL_RT_THREAD_LEVEL), "kernel launch", healthy));
    ASSERT_TRUE(checkAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream", healthy));
    ASSERT_TRUE(checkAcl(
        aclrtMemcpy(dstSatHost.data, dstFileSize, dstSatDevice.data, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST),
        "saturated D2H", healthy));
    ASSERT_TRUE(checkAcl(
        aclrtMemcpy(dstTruncHost.data, dstFileSize, dstTruncDevice.data, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST),
        "truncated D2H", healthy));
    ASSERT_TRUE(checkAcl(
        aclrtMemcpy(dstDefaultHost.data, dstFileSize, dstDefaultDevice.data, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST),
        "default D2H", healthy));

    ASSERT_TRUE(writeFileExact(GetGoldenDir() + "/output_saturated.bin", dstSatHost.data, dstFileSize, healthy));
    ASSERT_TRUE(writeFileExact(GetGoldenDir() + "/output_truncated.bin", dstTruncHost.data, dstFileSize, healthy));
    ASSERT_TRUE(writeFileExact(GetGoldenDir() + "/output_default.bin", dstDefaultHost.data, dstFileSize, healthy));

    std::vector<D> goldenTrunc(dstFileSize / sizeof(D));
    std::vector<D> devTrunc(dstFileSize / sizeof(D));
    ASSERT_TRUE(readFileExact(GetGoldenDir() + "/golden_truncated.bin", dstFileSize, goldenTrunc.data(), healthy));
    ASSERT_TRUE(readFileExact(GetGoldenDir() + "/output_truncated.bin", dstFileSize, devTrunc.data(), healthy));
    bool truncOk = ResultCmp<D>(goldenTrunc, devTrunc, 0.001f);

    std::string goldenDefaultFile = GetGoldenDir() + "/golden_truncated.bin";

    std::vector<D> goldenDefault(dstFileSize / sizeof(D));
    std::vector<D> devDefault(dstFileSize / sizeof(D));
    ASSERT_TRUE(readFileExact(goldenDefaultFile, dstFileSize, goldenDefault.data(), healthy));
    ASSERT_TRUE(readFileExact(GetGoldenDir() + "/output_default.bin", dstFileSize, devDefault.data(), healthy));
    bool defaultOk = ResultCmp<D>(goldenDefault, devDefault, 0.001f);

    EXPECT_TRUE(truncOk) << "Saturation mode OFF (TRUNC) output mismatch";
    EXPECT_TRUE(defaultOk) << "Default mode output mismatch (compared against PyTorch TRUNC golden)";
}

// Saturation mode test cases (only for supported conversions on A5)
// Minimal saturation mode tests (fp32→int8 is NOT supported on A5 hardware)
// Disabled at compile time by default - define ENABLE_SATURATION_TESTS to enable
#ifdef ENABLE_SATURATION_TESTS
TEST_F(TCVTTest, saturation_fp16_int8_1x32) { test_tcvt_saturation<int8_t, aclFloat16, 1, 32, 1, 32>(); }

TEST_F(TCVTTest, saturation_fp32_int16_1x32) { test_tcvt_saturation<int16_t, float, 1, 32, 1, 32>(); }

TEST_F(TCVTTest, saturation_fp16_int16_1x32) { test_tcvt_saturation<int16_t, aclFloat16, 1, 32, 1, 32>(); }

TEST_F(TCVTTest, saturation_fp16_uint8_1x32) { test_tcvt_saturation<uint8_t, aclFloat16, 1, 32, 1, 32>(); }

TEST_F(TCVTTest, saturation_int64_int32_1x32) { test_tcvt_saturation<int32_t, int64_t, 1, 32, 1, 32>(); }

TEST_F(TCVTTest, saturation_int32_int16_1x32) { test_tcvt_saturation<int16_t, int32_t, 1, 32, 1, 32>(); }
#endif // ENABLE_SATURATION_TESTS
