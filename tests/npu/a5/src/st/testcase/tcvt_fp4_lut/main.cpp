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
constexpr std::array<Fp4LutCaseId, 6> kBulkRowCases = {Fp4LutCaseId::Bulk64ND,        Fp4LutCaseId::Bulk64NZ,
                                                       Fp4LutCaseId::Bulk64CompactNZ, Fp4LutCaseId::Bulk128ND,
                                                       Fp4LutCaseId::Bulk128NZ,       Fp4LutCaseId::Bulk128CompactNZ};

float decodeFp4(uint8_t code)
{
    int exponent = (code >> 1) & 3;
    int mantissa = code & 1;
    float magnitude = exponent == 0 ? 0.5f * mantissa : std::ldexp(2.0f + mantissa, exponent - 2);
    return std::copysign(magnitude, (code & 8) ? -1.0f : 1.0f);
}

// Independently decode FP4 numerically and encode normalized E4M3; no byte LUT.
uint8_t encodeE4M3(float value)
{
    uint8_t sign = std::signbit(value) ? 0x80 : 0;
    if (value == 0) {
        return sign;
    }
    int exponent;
    float fraction = std::frexp(std::fabs(value), &exponent);
    int mantissa = static_cast<int>(std::lround((fraction * 2.0f - 1.0f) * 8.0f));
    return sign | static_cast<uint8_t>(((exponent + 6) << 3) | mantissa);
}

class DeviceBuffer {
public:
    ~DeviceBuffer()
    {
        if (data != nullptr) {
            EXPECT_EQ(aclrtFree(data), ACL_SUCCESS) << "aclrtFree";
        }
    }
    aclError allocate(size_t bytes)
    {
        return aclrtMalloc(reinterpret_cast<void**>(&data), bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    uint8_t* data = nullptr;
};

class Fp4LutTest : public testing::Test {
protected:
    static void SetUpTestSuite()
    {
        aclError status = aclInit(nullptr);
        ASSERT_EQ(status, ACL_SUCCESS) << "aclInit";
        initialized = true;
        status = aclrtSetDevice(0);
        ASSERT_EQ(status, ACL_SUCCESS) << "aclrtSetDevice (infrastructure failure; no device tests started)";
        deviceOpened = true;
        status = aclrtCreateContext(&context, 0);
        ASSERT_EQ(status, ACL_SUCCESS) << "aclrtCreateContext";
        healthy = true;
    }
    static void TearDownTestSuite()
    {
        if (context != nullptr) {
            EXPECT_EQ(aclrtDestroyContext(context), ACL_SUCCESS) << "aclrtDestroyContext";
        }
        if (deviceOpened) {
            EXPECT_EQ(aclrtResetDevice(0), ACL_SUCCESS) << "aclrtResetDevice";
        }
        if (initialized) {
            EXPECT_EQ(aclFinalize(), ACL_SUCCESS) << "aclFinalize";
        }
    }
    void SetUp() override
    {
        ASSERT_TRUE(healthy) << "Device work stopped after an ACL setup/runtime failure";
        aclError status = aclrtSetCurrentContext(context);
        healthy = status == ACL_SUCCESS;
        ASSERT_EQ(status, ACL_SUCCESS) << "aclrtSetCurrentContext";
        status = aclrtCreateStream(&stream);
        healthy = status == ACL_SUCCESS;
        ASSERT_EQ(status, ACL_SUCCESS) << "aclrtCreateStream";
    }
    void TearDown() override
    {
        if (stream != nullptr) {
            aclError status = aclrtDestroyStream(stream);
            healthy = healthy && status == ACL_SUCCESS;
            EXPECT_EQ(status, ACL_SUCCESS) << "aclrtDestroyStream";
        }
    }
    void runBytes(
        int caseId, std::vector<uint8_t>& source, const std::vector<uint8_t>& expected, int rows, int cols, int round,
        int api, int regressionPair = -1, int primitiveSourceOffset = 0, bool primitiveRotation = false)
    {
        DeviceBuffer input, output;
        ASSERT_EQ(input.allocate(source.size()), ACL_SUCCESS) << "aclrtMalloc input";
        ASSERT_EQ(output.allocate(expected.size()), ACL_SUCCESS) << "aclrtMalloc output";
        std::vector<uint8_t> actual(expected.size(), kCanary);
        aclError status =
            aclrtMemcpy(input.data, source.size(), source.data(), source.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        healthy = status == ACL_SUCCESS;
        ASSERT_EQ(status, ACL_SUCCESS) << "aclrtMemcpy input H2D";
        status = aclrtMemcpy(output.data, actual.size(), actual.data(), actual.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        healthy = status == ACL_SUCCESS;
        ASSERT_EQ(status, ACL_SUCCESS) << "aclrtMemcpy output H2D";
        if (regressionPair < 0) {
            launchFp4Lut(caseId, output.data, input.data, rows, cols, round, api, stream);
        } else if (regressionPair == 2) {
            launchFp4LutBulkPrimitive(output.data, input.data, primitiveSourceOffset, primitiveRotation, stream);
        } else {
            launchFp4LutRegression(regressionPair, output.data, input.data, stream);
        }
        status = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
        healthy = status == ACL_SUCCESS;
        ASSERT_EQ(status, ACL_SUCCESS) << "kernel launch";
        status = aclrtSynchronizeStream(stream);
        healthy = status == ACL_SUCCESS;
        ASSERT_EQ(status, ACL_SUCCESS) << "aclrtSynchronizeStream";
        status = aclrtMemcpy(actual.data(), actual.size(), output.data, actual.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        healthy = status == ACL_SUCCESS;
        ASSERT_EQ(status, ACL_SUCCESS) << "aclrtMemcpy output D2H";
        for (size_t byte = 0; byte < expected.size(); ++byte) {
            ASSERT_EQ(actual[byte], expected[byte])
                << "physical byte " << byte << ", round " << round << ", API " << api;
        }
        std::vector<uint8_t> sourceAfter(source.size());
        status =
            aclrtMemcpy(sourceAfter.data(), sourceAfter.size(), input.data, source.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        healthy = status == ACL_SUCCESS;
        ASSERT_EQ(status, ACL_SUCCESS) << "aclrtMemcpy input D2H";
        ASSERT_EQ(sourceAfter, source) << "Source bytes changed";
    }
    void runCase(int id, int round = 1, int api = 0, int runtimeRows = -1, int runtimeCols = -1)
    {
        const auto& test = kFp4LutCases[id];
        SCOPED_TRACE(test.name);
        int rows = runtimeRows < 0 ? test.validRows : runtimeRows;
        int cols = runtimeCols < 0 ? test.validCols : runtimeCols;
        const size_t sourcePitch = test.sourceCols / 2;
        std::vector<uint8_t> source(test.sourceRows * sourcePitch);
        for (int row = 0; row < test.sourceRows; ++row) {
            for (size_t col = 0; col < sourcePitch; ++col) {
                source[row * sourcePitch + col] = static_cast<uint8_t>(row * 73 + col * 37 + (col / 256) * 11);
            }
        }
        if (id >= static_cast<int>(Fp4LutCaseId::Bulk64ND)) {
            for (size_t byte = 0; byte < source.size(); ++byte) {
                source[byte] = static_cast<uint8_t>(byte * 37 + (byte / 256) * 11);
            }
        }
        std::vector<uint8_t> expected(test.destinationRows * test.destinationCols + 2 * kGuardBytes, kCanary);
        size_t panelRows = test.compact ? ((rows + 15) / 16) * 16 : test.destinationRows;
        std::array<bool, 256> seen{};
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                uint8_t packed = source[row * sourcePitch + col / 2];
                seen[packed] = true;
                uint8_t nibble = (col & 1) ? (packed >> 4) : (packed & 15);
                size_t physical = test.nz ? (col / 32) * panelRows * 32 + row * 32 + col % 32 :
                                            row * static_cast<size_t>(test.destinationCols) + col;
                ASSERT_LT(physical, expected.size() - 2 * kGuardBytes);
                expected[kGuardBytes + physical] = encodeE4M3(decodeFp4(nibble));
            }
        }
        bool fullBulkFixture =
            id >= static_cast<int>(Fp4LutCaseId::Bulk64ND) && rows == test.validRows && cols == test.validCols;
        if (id == static_cast<int>(Fp4LutCaseId::MainND) || id == static_cast<int>(Fp4LutCaseId::MainNZ) ||
            fullBulkFixture) {
            for (int packed = 0; packed < 256; ++packed) {
                ASSERT_TRUE(seen[packed]) << "Missing packed FP4 byte " << packed;
            }
        }
        ASSERT_NO_FATAL_FAILURE(runBytes(id, source, expected, rows, cols, round, api));
    }
    aclrtStream stream = nullptr;
    inline static aclrtContext context = nullptr;
    inline static bool initialized = false;
    inline static bool deviceOpened = false;
    inline static bool healthy = false;
};

#define FP4_LUT_TEST(Name, ...) \
    TEST_F(Fp4LutTest, Name) { runCase(static_cast<int>(Fp4LutCaseId::Name)); }
FP4_LUT_CASES(FP4_LUT_TEST)
#undef FP4_LUT_TEST

TEST_F(Fp4LutTest, PublicOverloadsRoundSaturation)
{
    // RoundMode encodings 0..6 cover NONE/RINT/ROUND/FLOOR/CEIL/TRUNC/ODD.
    for (int layout = 0; layout < 2; ++layout) {
        for (int round = 0; round <= 6; ++round) {
            for (int api = 0; api < 6; ++api) {
                ASSERT_NO_FATAL_FAILURE(runCase(layout, round, api));
            }
        }
    }
}

TEST_F(Fp4LutTest, BulkRowsEveryValidRowCount)
{
    for (auto id : kBulkRowCases) {
        const auto& test = kFp4LutCases[static_cast<int>(id)];
        for (int rows = 0; rows <= test.sourceRows; ++rows) {
            SCOPED_TRACE(testing::Message() << "valid rows " << rows);
            ASSERT_NO_FATAL_FAILURE(runCase(static_cast<int>(id), 1, 0, rows, test.sourceCols));
        }
    }
}

TEST_F(Fp4LutTest, BulkRowsPartialColumns)
{
    for (auto id : kBulkRowCases) {
        const auto& test = kFp4LutCases[static_cast<int>(id)];
        for (int rows : {1, 3, 4, 17, 20}) {
            for (int cols : {0, 1, 63, test.sourceCols - 1}) {
                SCOPED_TRACE(testing::Message() << "valid rows " << rows << ", valid columns " << cols);
                ASSERT_NO_FATAL_FAILURE(runCase(static_cast<int>(id), 1, 0, rows, cols));
            }
        }
    }
}

TEST_F(Fp4LutTest, BulkRowsPublicOverloadsRoundSaturation)
{
    for (auto id : {Fp4LutCaseId::Bulk64CompactNZ, Fp4LutCaseId::Bulk128ND}) {
        for (int round = 0; round <= 6; ++round) {
            for (int api = 0; api < 6; ++api) {
                ASSERT_NO_FATAL_FAILURE(runCase(static_cast<int>(id), round, api));
            }
        }
    }
}

TEST_F(Fp4LutTest, EveryMinimumRowTail)
{
    for (int cols = 1; cols <= 64; ++cols) {
        ASSERT_NO_FATAL_FAILURE(runCase(static_cast<int>(Fp4LutCaseId::SingleNibbleND), 1, 0, 1, cols));
        ASSERT_NO_FATAL_FAILURE(runCase(static_cast<int>(Fp4LutCaseId::SingleNibbleNZ), 1, 0, 1, cols));
    }
}

TEST_F(Fp4LutTest, EmptyDynamicRegions)
{
    for (auto id : {Fp4LutCaseId::StridedDynamicND, Fp4LutCaseId::CompactDynamicNZ}) {
        ASSERT_NO_FATAL_FAILURE(runCase(static_cast<int>(id), 1, 0, 0, 579));
        ASSERT_NO_FATAL_FAILURE(runCase(static_cast<int>(id), 1, 0, 3, 0));
    }
}

TEST_F(Fp4LutTest, ExistingFp4ToBf16)
{
    constexpr int columns = 128;
    std::vector<uint8_t> source(columns / 2);
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<uint8_t>(i * 37);
    }
    std::vector<uint8_t> expected(columns * 2 + 2 * kGuardBytes, kCanary);
    for (int col = 0; col < columns; ++col) {
        uint8_t packed = source[col / 2];
        float value = decodeFp4((col & 1) ? packed >> 4 : packed & 15);
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        uint16_t bf16 = static_cast<uint16_t>(bits >> 16);
        std::memcpy(expected.data() + kGuardBytes + col * 2, &bf16, sizeof(bf16));
    }
    runBytes(-1, source, expected, 1, columns, 1, 0, 0);
}

TEST_F(Fp4LutTest, ExistingFp32ToFp8)
{
    constexpr int columns = 128;
    std::vector<uint8_t> source(columns * sizeof(float));
    std::vector<uint8_t> expected(columns + 2 * kGuardBytes, kCanary);
    for (int col = 0; col < columns; ++col) {
        float value = decodeFp4(static_cast<uint8_t>(col & 15));
        std::memcpy(source.data() + col * sizeof(float), &value, sizeof(value));
        expected[kGuardBytes + col] = encodeE4M3(value);
    }
    runBytes(-1, source, expected, 1, columns, 1, 0, 1);
}

TEST_F(Fp4LutTest, BulkPrimitiveQualification)
{
    constexpr size_t sourceBytes = 768;
    constexpr size_t registerBytes = 256;
    for (int sourceOffset : {0, 32, 64, 128}) {
        for (bool rotation : {false, true}) {
            SCOPED_TRACE(
                testing::Message() << "source offset " << sourceOffset << ", mode "
                                   << (rotation ? "rotate128" : "identity"));
            std::vector<uint8_t> source(sourceBytes);
            for (size_t byte = 0; byte < source.size(); ++byte) {
                source[byte] = static_cast<uint8_t>(byte);
            }
            std::vector<uint8_t> expected(registerBytes + 2 * kGuardBytes, kCanary);
            for (size_t byte = 0; byte < registerBytes; ++byte) {
                size_t index = (byte + (rotation ? 128 : 0)) % registerBytes;
                expected[kGuardBytes + byte] = source[sourceOffset + index];
            }
            ASSERT_NO_FATAL_FAILURE(runBytes(-1, source, expected, 1, registerBytes, 1, 0, 2, sourceOffset, rotation));
        }
    }
}
} // namespace
