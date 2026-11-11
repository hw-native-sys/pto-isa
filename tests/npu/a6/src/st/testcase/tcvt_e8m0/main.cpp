/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include "acl/acl.h"

#include "cases.h"

namespace {
constexpr size_t kGuardBytes = 32;
constexpr uint8_t kCanary = 0xA5;
// RoundMode encodings (pto/common/type.hpp): NONE=0 RINT=1 ROUND=2 FLOOR=3 CEIL=4 TRUNC=5 ODD=6.
constexpr int kRoundNone = 0;
constexpr int kRoundRint = 1;
constexpr int kRoundRound = 2;
constexpr int kRoundFloor = 3;
constexpr int kRoundCeil = 4;
constexpr int kRoundTrunc = 5;
constexpr int kMantissas[] = {0x00, 0x01, 0x11, 0x20, 0x3F, 0x40, 0x7E, 0x7F};

// E8M0 encodes a power of two as an unsigned 8-bit exponent (value = 2^(E-127),
// 0xFF = NaN). bf16 -> e8m0 operates on the value magnitude (the sign bit is
// dropped, matching the CANN reference emulation) and quantizes to the e8m0
// grid of powers of two:
//   - truncating rounds (NONE/RINT/FLOOR/TRUNC/ODD): E = exponent field
//   - ceil-style rounds (CEIL, ROUND): E = ceiling of the value, so a nonzero
//     mantissa bumps the exponent, subnormals stay at E = 0 while their value
//     is <= 2^-127 (mantissa <= 0x40), and +/-Inf and NaN map to 0xFF.
uint8_t encodeE8m0(uint16_t bits, int round)
{
    const uint32_t exponent = (bits >> 7) & 0xFF;
    const uint32_t mantissa = bits & 0x7F;
    const bool roundUp = round == kRoundCeil || round == kRoundRound;
    if (exponent == 0xFF) {
        return 0xFF;
    }
    if (roundUp) {
        if (exponent == 0) {
            return mantissa > 0x40 ? 1u : 0u;
        }
        return exponent + (mantissa != 0 ? 1u : 0u);
    }
    return static_cast<uint8_t>(exponent);
}

// e8m0 -> bf16 widens exactly: the exponent byte becomes the bf16 exponent field
// (bits = E << 7), E = 0 (2^-127, below the bf16 normal range) becomes the
// subnormal 0x0040, and 0xFF (NaN) becomes the all-ones-mantissa NaN 0x7FFF
// (verified on the dav_9201 simulator; not the quiet NaN 0x7FC0).
uint16_t decodeE8m0(uint8_t code)
{
    if (code == 0xFF) {
        return 0x7FFF;
    }
    if (code == 0x00) {
        return 0x0040;
    }
    return static_cast<uint16_t>(code) << 7;
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

class E8m0CvtTest : public testing::Test {
protected:
    void SetUp() override
    {
        ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS) << "aclInit";
        ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS) << "aclrtSetDevice";
        ASSERT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS) << "aclrtCreateStream";
    }
    void TearDown() override
    {
        if (stream != nullptr) {
            EXPECT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS) << "aclrtDestroyStream";
        }
        EXPECT_EQ(aclrtResetDevice(0), ACL_SUCCESS) << "aclrtResetDevice";
        EXPECT_EQ(aclFinalize(), ACL_SUCCESS) << "aclFinalize";
    }

    // Dispatches the buffered fixture to one of the three kernel families and
    // compares every physical byte (valid region, padding and guards) exactly.
    void compareBytes(const std::vector<uint8_t>& source, const std::vector<uint8_t>& expected)
    {
        DeviceBuffer input, output;
        ASSERT_EQ(input.allocate(source.size()), ACL_SUCCESS) << "aclrtMalloc input";
        ASSERT_EQ(output.allocate(expected.size()), ACL_SUCCESS) << "aclrtMalloc output";
        std::vector<uint8_t> actual(expected.size(), kCanary);
        aclError status =
            aclrtMemcpy(input.data, source.size(), source.data(), source.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        ASSERT_EQ(status, ACL_SUCCESS) << "aclrtMemcpy input H2D";
        status = aclrtMemcpy(output.data, actual.size(), actual.data(), actual.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        ASSERT_EQ(status, ACL_SUCCESS) << "aclrtMemcpy output H2D";
        if (mode == Mode::AllBytes) {
            launchE8m0ToBf16AllBytes(output.data, input.data, stream);
        } else if (mode == Mode::Specials) {
            launchBf16ToE8m0Specials(output.data, input.data, round, stream);
        } else {
            launchE8m0Cvt(caseId, output.data, input.data, rows, cols, round, api, stream);
        }
        status = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
        ASSERT_EQ(status, ACL_SUCCESS) << "kernel launch";
        status = aclrtSynchronizeStream(stream);
        ASSERT_EQ(status, ACL_SUCCESS) << "aclrtSynchronizeStream";
        status = aclrtMemcpy(actual.data(), actual.size(), output.data, actual.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        ASSERT_EQ(status, ACL_SUCCESS) << "aclrtMemcpy output D2H";
        for (size_t byte = 0; byte < expected.size(); ++byte) {
            EXPECT_EQ(actual[byte], expected[byte])
                << "physical byte " << byte << ", round " << round << ", API " << api;
        }
        std::vector<uint8_t> sourceAfter(source.size());
        status =
            aclrtMemcpy(sourceAfter.data(), sourceAfter.size(), input.data, source.size(), ACL_MEMCPY_DEVICE_TO_HOST);
        ASSERT_EQ(status, ACL_SUCCESS) << "aclrtMemcpy input D2H";
        ASSERT_EQ(sourceAfter, source) << "Source bytes changed";
    }

    void runCase(int id, int roundMode = kRoundRint, int apiMode = 0, int runtimeRows = -1, int runtimeCols = -1)
    {
        const auto& test = kE8m0CvtCases[id];
        SCOPED_TRACE(test.name);
        mode = Mode::Cases;
        caseId = id;
        round = roundMode;
        api = apiMode;
        const int rowCap = test.sourceRows < test.destinationRows ? test.sourceRows : test.destinationRows;
        const int colCap = test.sourceCols < test.destinationCols ? test.sourceCols : test.destinationCols;
        rows = runtimeRows < 0 ? test.validRows : runtimeRows;
        cols = runtimeCols < 0 ? test.validCols : runtimeCols;
        // A valid region larger than its tiles is an API misuse: clamp the
        // boundary sweep to the physical tile dimensions (the smaller side
        // binds when the source and destination tiles differ in size).
        rows = rows > rowCap ? rowCap : rows;
        cols = cols > colCap ? colCap : cols;
        const size_t elementSize = test.toE8m0 ? 2u : 1u;
        const size_t sourcePitch = static_cast<size_t>(test.sourceCols) * elementSize;
        std::vector<uint8_t> source(static_cast<size_t>(test.sourceRows) * sourcePitch);
        for (int row = 0; row < test.sourceRows; ++row) {
            for (int col = 0; col < test.sourceCols; ++col) {
                const int index = row * test.sourceCols + col;
                if (test.toE8m0) {
                    // Positive finite bf16 values with exponent <= 0xFD so the
                    // ceil golden never reaches the 0xFF saturation band.
                    const uint16_t exponent = static_cast<uint16_t>((index / 8) % 254);
                    const uint16_t mantissa = static_cast<uint16_t>(kMantissas[index % 8]);
                    uint16_t bits = static_cast<uint16_t>((exponent << 7) | mantissa);
                    std::memcpy(source.data() + row * sourcePitch + col * 2, &bits, sizeof(bits));
                } else {
                    source[row * sourcePitch + col] = static_cast<uint8_t>(index * 37);
                }
            }
        }
        const size_t destinationSize =
            static_cast<size_t>(test.destinationRows) * test.destinationCols * (test.toE8m0 ? 1u : 2u);
        std::vector<uint8_t> expected(destinationSize + 2 * kGuardBytes, kCanary);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                const size_t physical = static_cast<size_t>(row) * test.destinationCols + col;
                if (test.toE8m0) {
                    uint16_t bits;
                    std::memcpy(&bits, source.data() + row * sourcePitch + col * 2, sizeof(bits));
                    expected[kGuardBytes + physical] = encodeE8m0(bits, round);
                } else {
                    uint16_t bits = decodeE8m0(source[row * sourcePitch + col]);
                    std::memcpy(expected.data() + kGuardBytes + physical * 2, &bits, sizeof(bits));
                }
            }
        }
        ASSERT_NO_FATAL_FAILURE(compareBytes(source, expected));
    }

    void runE8m0ToBf16AllBytes()
    {
        constexpr int columns = 256;
        mode = Mode::AllBytes;
        std::vector<uint8_t> source(columns);
        for (int i = 0; i < columns; ++i) {
            source[i] = static_cast<uint8_t>(i);
        }
        std::vector<uint8_t> expected(columns * 2 + 2 * kGuardBytes, kCanary);
        for (int i = 0; i < columns; ++i) {
            uint16_t bits = decodeE8m0(source[i]);
            std::memcpy(expected.data() + kGuardBytes + i * 2, &bits, sizeof(bits));
        }
        ASSERT_NO_FATAL_FAILURE(compareBytes(source, expected));
    }

    void runBf16ToE8m0Specials(int roundMode)
    {
        constexpr int columns = 256;
        mode = Mode::Specials;
        round = roundMode;
        std::vector<uint8_t> source(columns * 2);
        for (int i = 0; i < columns; ++i) {
            uint16_t bits;
            if (i < 64) {
                // negative subnormals and small normals, full mantissa sweep
                bits = static_cast<uint16_t>((1u << 15) | (static_cast<uint32_t>(i >> 3) << 7) | kMantissas[i & 7]);
            } else if (i < 128) {
                // negative values with full mantissa
                bits = static_cast<uint16_t>((1u << 15) | (static_cast<uint32_t>(i - 64) << 7) | 0x7F);
            } else if (i < 192) {
                // 0xFE exponent band, both signs, mantissa sweep
                bits = static_cast<uint16_t>(((i & 1u) << 15) | (0xFEu << 7) | static_cast<uint32_t>(i - 128));
            } else {
                // +/-Inf (mantissa 0) and +/-NaN (mantissa != 0)
                bits = static_cast<uint16_t>(((i & 16u) << 11) | (0xFFu << 7) | static_cast<uint32_t>(i & 15));
            }
            std::memcpy(source.data() + i * 2, &bits, sizeof(bits));
        }
        std::vector<uint8_t> expected(columns + 2 * kGuardBytes, kCanary);
        for (int i = 0; i < columns; ++i) {
            uint16_t bits;
            std::memcpy(&bits, source.data() + i * 2, sizeof(bits));
            expected[kGuardBytes + i] = encodeE8m0(bits, round);
        }
        ASSERT_NO_FATAL_FAILURE(compareBytes(source, expected));
    }

    aclrtStream stream = nullptr;
    enum class Mode { Cases, AllBytes, Specials };
    Mode mode = Mode::Cases;
    int caseId = -1;
    int rows = 0;
    int cols = 0;
    int round = kRoundRint;
    int api = 0;
};

#define E8M0_CVT_TEST(Name, ...) \
    TEST_F(E8m0CvtTest, Name) { runCase(static_cast<int>(E8m0CvtCaseId::Name)); }
E8M0_CVT_CASES(E8M0_CVT_TEST)
#undef E8M0_CVT_TEST

TEST_F(E8m0CvtTest, PublicOverloadsRoundSaturation)
{
    // Every RoundMode encoding and API overload; the saturation operand is
    // ignored by the hardware for bf16 <-> e8m0, so all must match the golden.
    for (int id = 0; id < static_cast<int>(sizeof(kE8m0CvtCases) / sizeof(kE8m0CvtCases[0])); ++id) {
        for (int round = 0; round <= 6; ++round) {
            for (int api = 0; api < 6; ++api) {
                ASSERT_NO_FATAL_FAILURE(runCase(id, round, api));
            }
        }
    }
}

TEST_F(E8m0CvtTest, DynamicBoundaries)
{
    const int dynamicCases[] = {
        static_cast<int>(E8m0CvtCaseId::Bf16ToE8m0Tail), static_cast<int>(E8m0CvtCaseId::E8m0ToBf16Tail),
        static_cast<int>(E8m0CvtCaseId::Bf16ToE8m0Dynamic), static_cast<int>(E8m0CvtCaseId::E8m0ToBf16Dynamic)};
    for (int id : dynamicCases) {
        for (int rows : {0, 1, 2, 3, 5}) {
            for (int cols : {0, 1, 127, 128, 129, 257, 512, 579}) {
                SCOPED_TRACE(
                    testing::Message() << "case " << id << ", valid rows " << rows << ", valid columns " << cols);
                ASSERT_NO_FATAL_FAILURE(runCase(id, kRoundRint, 0, rows, cols));
            }
        }
    }
}

TEST_F(E8m0CvtTest, AllBytesE8m0ToBf16) { ASSERT_NO_FATAL_FAILURE(runE8m0ToBf16AllBytes()); }

TEST_F(E8m0CvtTest, Bf16ToE8m0RoundModes)
{
    // All RoundMode encodings: CEIL/ROUND take the ROUND_C path (exponent + 1
    // when the mantissa is nonzero), the remaining modes truncate. The fixture
    // keeps exponents <= 0xFD so the ceil golden never saturates to 0xFF.
    for (int round = 0; round <= 6; ++round) {
        SCOPED_TRACE(testing::Message() << "round " << round);
        ASSERT_NO_FATAL_FAILURE(runCase(static_cast<int>(E8m0CvtCaseId::Bf16ToE8m0Full), round, 0));
    }
}

TEST_F(E8m0CvtTest, ApiOverloads)
{
    // API variants (saturation on/off, tmp-buffer, TCVT<false> needSetCtrl):
    // saturation is ignored for bf16 <-> e8m0, so every overload must match.
    for (int api = 0; api < 6; ++api) {
        SCOPED_TRACE(testing::Message() << "api " << api);
        ASSERT_NO_FATAL_FAILURE(runCase(static_cast<int>(E8m0CvtCaseId::Bf16ToE8m0Full), kRoundCeil, api));
        ASSERT_NO_FATAL_FAILURE(runCase(static_cast<int>(E8m0CvtCaseId::E8m0ToBf16Full), kRoundRint, api));
    }
}

TEST_F(E8m0CvtTest, Bf16ToE8m0SpecialValues)
{
    // Truncating rounds only keep the exponent field; CEIL also exercises the
    // value ceiling on signs, subnormals, Inf/NaN and the 0xFE band, which the
    // generic (positive, exponent <= 0xFD) fixture cannot pin down.
    for (int round : {kRoundNone, kRoundRint, kRoundFloor, kRoundTrunc, kRoundCeil}) {
        SCOPED_TRACE(testing::Message() << "round " << round);
        ASSERT_NO_FATAL_FAILURE(runBf16ToE8m0Specials(round));
    }
}
} // namespace
