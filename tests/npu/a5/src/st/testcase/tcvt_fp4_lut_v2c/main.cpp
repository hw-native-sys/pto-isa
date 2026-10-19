/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include "acl/acl.h"

#include "cases.h"

namespace {
constexpr size_t kGuardBytes = 32;
constexpr uint8_t kCanary = 0xA5;

bool checkAcl(aclError status, const char* operation, bool& healthy)
{
    if (status != ACL_SUCCESS) {
        healthy = false;
        ADD_FAILURE() << operation << ": ACL error " << status << "; stopping device work";
        return false;
    }
    return true;
}

// E2M1 decoded arithmetically in units of 1/2, independently of the device LUT.
int decodeFp4HalfUnits(uint8_t code)
{
    int exponent = (code >> 1) & 3;
    int mantissa = code & 1;
    int magnitude = exponent == 0 ? mantissa : (2 + mantissa) << (exponent - 1);
    return (code & 8) ? -magnitude : magnitude;
}

int aEighthUnits(int m, int k, int tileK)
{
    int magnitude = 1 + (m * 5 + k * 3 + (k / tileK) * 7) % 7;
    return ((m + k / 3) & 1) ? -magnitude : magnitude;
}

uint8_t encodeE4M3(float value)
{
    uint8_t sign = std::signbit(value) ? 0x80 : 0;
    int exponent;
    float fraction = std::frexp(std::fabs(value), &exponent);
    int mantissa = static_cast<int>(std::lround((fraction * 2.0f - 1.0f) * 8.0f));
    return sign | static_cast<uint8_t>(((exponent + 6) << 3) | mantissa);
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(bool& healthy) : healthy_(healthy) {}
    ~DeviceBuffer()
    {
        if (data != nullptr) {
            checkAcl(aclrtFree(data), "aclrtFree", healthy_);
        }
    }
    bool allocate(size_t bytes)
    {
        return checkAcl(
            aclrtMalloc(reinterpret_cast<void**>(&data), bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc", healthy_);
    }
    uint8_t* data = nullptr;

private:
    bool& healthy_;
};

class Fp4LutV2cTest : public testing::Test {
protected:
    static void SetUpTestSuite()
    {
        ASSERT_TRUE(checkAcl(aclInit(nullptr), "aclInit", healthy));
        initialized = true;
        ASSERT_TRUE(checkAcl(aclrtSetDevice(0), "aclrtSetDevice", healthy));
        deviceOpened = true;
        ASSERT_TRUE(checkAcl(aclrtCreateContext(&context, 0), "aclrtCreateContext", healthy));
        healthy = true;
    }
    static void TearDownTestSuite()
    {
        if (context != nullptr) {
            checkAcl(aclrtDestroyContext(context), "aclrtDestroyContext", healthy);
        }
        if (deviceOpened) {
            checkAcl(aclrtResetDevice(0), "aclrtResetDevice", healthy);
        }
        if (initialized) {
            checkAcl(aclFinalize(), "aclFinalize", healthy);
        }
    }
    void SetUp() override
    {
        ASSERT_TRUE(healthy) << "Device work stopped after an ACL infrastructure error";
        ASSERT_TRUE(checkAcl(aclrtSetCurrentContext(context), "aclrtSetCurrentContext", healthy));
        ASSERT_TRUE(checkAcl(aclrtCreateStream(&stream), "aclrtCreateStream", healthy));
    }
    void TearDown() override
    {
        if (stream != nullptr) {
            checkAcl(aclrtDestroyStream(stream), "aclrtDestroyStream", healthy);
        }
    }
    void runCase(int id)
    {
        const auto& test = kFp4LutV2cCases[id];
        SCOPED_TRACE(test.name);
        const int sourceRows = test.nk ? test.n : test.totalK;
        const int packedColumns = (test.nk ? test.totalK : test.n) / 2;
        std::vector<uint8_t> packedB(static_cast<size_t>(sourceRows) * packedColumns);
        std::array<bool, 256> seen{};
        for (int row = 0; row < sourceRows; ++row) {
            for (int col = 0; col < packedColumns; ++col) {
                int kIteration = test.nk ? (2 * col) / test.tileK : row / test.tileK;
                uint8_t packed = static_cast<uint8_t>(row * 73 + col * 37 + kIteration * 19);
                packedB[static_cast<size_t>(row) * packedColumns + col] = packed;
                seen[packed] = true;
            }
        }
        for (int packed = 0; packed < 256; ++packed) {
            ASSERT_TRUE(seen[packed]) << "Missing packed byte " << packed;
        }

        std::vector<uint8_t> a(static_cast<size_t>(test.m) * test.totalK);
        for (int m = 0; m < test.m; ++m) {
            for (int k = 0; k < test.totalK; ++k) {
                a[static_cast<size_t>(m) * test.totalK + k] = encodeE4M3(aEighthUnits(m, k, test.tileK) / 8.0f);
            }
        }
        std::vector<float> expected(static_cast<size_t>(test.m) * test.n);
        for (int m = 0; m < test.m; ++m) {
            for (int n = 0; n < test.n; ++n) {
                int64_t sumSixteenths = 0;
                for (int k = 0; k < test.totalK; ++k) {
                    // KN: k*(N/2)+n/2, nibble n%2. NK: n*(TotalK/2)+k/2, nibble k%2.
                    size_t offset = test.nk ? static_cast<size_t>(n) * (test.totalK / 2) + k / 2 :
                                              static_cast<size_t>(k) * (test.n / 2) + n / 2;
                    int shift = ((test.nk ? k : n) % 2) * 4;
                    uint8_t nibble = (packedB[offset] >> shift) & 15;
                    sumSixteenths += aEighthUnits(m, k, test.tileK) * decodeFp4HalfUnits(nibble);
                }
                // Each product/partial sum is a multiple of 1/16, |sum| <= 256*7*12/16.
                // Both Cube's FP32 accumulation and this independent integer sum are exact.
                expected[static_cast<size_t>(m) * test.n + n] = static_cast<float>(sumSixteenths) / 16.0f;
            }
        }

        const size_t cBytes = expected.size() * sizeof(float);
        std::vector<uint8_t> actual(cBytes + 2 * kGuardBytes, kCanary);
        DeviceBuffer aDevice(healthy), bDevice(healthy), cDevice(healthy);
        ASSERT_TRUE(aDevice.allocate(a.size()));
        ASSERT_TRUE(bDevice.allocate(packedB.size()));
        ASSERT_TRUE(cDevice.allocate(actual.size()));
        ASSERT_TRUE(checkAcl(
            aclrtMemcpy(aDevice.data, a.size(), a.data(), a.size(), ACL_MEMCPY_HOST_TO_DEVICE), "A H2D", healthy));
        ASSERT_TRUE(checkAcl(
            aclrtMemcpy(bDevice.data, packedB.size(), packedB.data(), packedB.size(), ACL_MEMCPY_HOST_TO_DEVICE),
            "packed B H2D", healthy));
        ASSERT_TRUE(checkAcl(
            aclrtMemcpy(cDevice.data, actual.size(), actual.data(), actual.size(), ACL_MEMCPY_HOST_TO_DEVICE),
            "C/guards H2D", healthy));
        launchFp4LutV2c(id, cDevice.data + kGuardBytes, aDevice.data, bDevice.data, stream);
        ASSERT_TRUE(checkAcl(aclrtGetLastError(ACL_RT_THREAD_LEVEL), "kernel launch", healthy));
        ASSERT_TRUE(checkAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream", healthy));
        ASSERT_TRUE(checkAcl(
            aclrtMemcpy(actual.data(), actual.size(), cDevice.data, actual.size(), ACL_MEMCPY_DEVICE_TO_HOST),
            "C/guards D2H", healthy));
        size_t mismatches = 0;
        for (size_t element = 0; element < expected.size(); ++element) {
            float value;
            std::memcpy(&value, actual.data() + kGuardBytes + element * sizeof(float), sizeof(float));
            if (value != expected[element]) {
                if (mismatches < 8) {
                    ADD_FAILURE() << "C[" << element / test.n << ',' << element % test.n << "]=" << value
                                  << ", expected " << expected[element];
                }
                ++mismatches;
            }
        }
        EXPECT_EQ(mismatches, 0u) << "All " << expected.size() << " C elements compared exactly";
        for (size_t guard = 0; guard < kGuardBytes; ++guard) {
            EXPECT_EQ(actual[guard], kCanary) << "Leading GM guard " << guard;
            EXPECT_EQ(actual[kGuardBytes + cBytes + guard], kCanary) << "Trailing GM guard " << guard;
        }
        std::vector<uint8_t> aAfter(a.size()), bAfter(packedB.size());
        ASSERT_TRUE(checkAcl(
            aclrtMemcpy(aAfter.data(), aAfter.size(), aDevice.data, a.size(), ACL_MEMCPY_DEVICE_TO_HOST), "A D2H",
            healthy));
        ASSERT_TRUE(checkAcl(
            aclrtMemcpy(bAfter.data(), bAfter.size(), bDevice.data, packedB.size(), ACL_MEMCPY_DEVICE_TO_HOST),
            "packed B D2H", healthy));
        EXPECT_EQ(aAfter, a);
        EXPECT_EQ(bAfter, packedB);
    }

    aclrtStream stream = nullptr;
    inline static aclrtContext context = nullptr;
    inline static bool initialized = false;
    inline static bool deviceOpened = false;
    inline static bool healthy = false;
};

#define FP4_LUT_V2C_TEST(Name, ...) \
    TEST_F(Fp4LutV2cTest, Name) { runCase(static_cast<int>(Fp4LutV2cCaseId::Name)); }
FP4_LUT_V2C_CASES(FP4_LUT_V2C_TEST)
#undef FP4_LUT_V2C_TEST
} // namespace
