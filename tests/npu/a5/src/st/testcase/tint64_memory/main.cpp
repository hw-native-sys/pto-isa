/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#ifndef __CPU_SIM
#include <acl/acl.h>
#endif

class Int64MemoryTest : public testing::Test {
protected:
#ifndef __CPU_SIM
    static void SetUpTestSuite()
    {
        ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
        ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    }
    static void TearDownTestSuite()
    {
        EXPECT_EQ(aclrtResetDevice(0), ACL_SUCCESS);
        EXPECT_EQ(aclFinalize(), ACL_SUCCESS);
    }
#endif
};

template <typename T>
void Execute(
    void (*launch)(T*, T*, uint32_t*, void*), std::vector<T>& source, std::vector<uint32_t>& indices,
    std::vector<T>& output, const std::vector<T>& expected, int comparedBytes = -1)
{
#ifdef __CPU_SIM
    launch(output.data(), source.data(), indices.data(), nullptr);
#else
    void* stream = nullptr;
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS);
    T *srcDevice = nullptr, *dstDevice = nullptr;
    uint32_t* idxDevice = nullptr;
    const size_t srcBytes = source.size() * sizeof(T), dstBytes = output.size() * sizeof(T);
    const size_t idxBytes = indices.size() * sizeof(uint32_t);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&srcDevice), srcBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&dstDevice), dstBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMalloc(reinterpret_cast<void**>(&idxDevice), idxBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(srcDevice, srcBytes, source.data(), srcBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(dstDevice, dstBytes, output.data(), dstBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(idxDevice, idxBytes, indices.data(), idxBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    launch(dstDevice, srcDevice, idxDevice, stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    ASSERT_EQ(aclrtMemcpy(output.data(), dstBytes, dstDevice, dstBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    EXPECT_EQ(aclrtFree(idxDevice), ACL_SUCCESS);
    EXPECT_EQ(aclrtFree(dstDevice), ACL_SUCCESS);
    EXPECT_EQ(aclrtFree(srcDevice), ACL_SUCCESS);
    EXPECT_EQ(aclrtDestroyStream(stream), ACL_SUCCESS);
#endif
    if (comparedBytes < 0) {
        EXPECT_EQ(output, expected);
    } else {
        // Packed comparison padding is target-defined; check valid bits and the adjacent allocation.
        EXPECT_EQ(std::memcmp(output.data(), expected.data(), comparedBytes), 0);
        EXPECT_EQ(std::memcmp(output.data() + output.size() - 64, expected.data() + expected.size() - 64, 512), 0);
    }
}

template <typename T, int Op, int Cols, int ValidCols>
void TestBoundary(void (*launch)(T*, T*, uint32_t*, void*))
{
    constexpr bool rowReduce = Op >= 6 && Op <= 8;
    constexpr int outCols = rowReduce ? 4 : Cols;
    constexpr T sentinel = static_cast<T>(0x5A5A5A5A5A5A5A5AULL);
    std::vector<T> source(Cols), output(outCols + 64, sentinel), expected = output;
    std::vector<uint32_t> indices((Cols + 7) / 8 * 8);
    for (int i = 0; i < Cols; ++i) {
        source[i] = static_cast<T>((1ULL << 40) + i * 17 + 9);
        indices[i] = i;
    }
    if constexpr (Op == 19 || Op == 21)
        std::fill_n(expected.begin(), outCols, T(0));
    for (int i = 0; i < ValidCols; ++i) {
        if constexpr (Op == 0 || Op == 12 || Op == 14)
            expected[i] = source[i] * 2;
        else if constexpr (Op == 1)
            expected[i] = source[i] + 3;
        else if constexpr (Op == 2)
            expected[i] = ~source[i];
        else if constexpr (Op == 3)
            expected[i] = 1;
        else if constexpr (Op == 4)
            expected[i] = source[i] / 3;
        else if constexpr (Op == 5)
            expected[i] = 0;
        else if constexpr (Op == 6)
            expected[0] = source[ValidCols - 1];
        else if constexpr (Op == 7)
            expected[0] = source[0];
        else if constexpr (Op == 8) {
            if (i == 0)
                expected[0] = 0;
            expected[0] += source[i];
        } else if constexpr (Op == 13)
            expected[i] = source[i] + source[0];
        else if constexpr (Op == 16)
            expected[i] = (0x5a & (1 << (i % 8))) ? source[i] : T(3);
        else
            expected[i] = source[i];
    }
    if constexpr (Op == 17 || Op == 18) {
        auto* bytes = reinterpret_cast<uint8_t*>(expected.data());
        for (int i = 0; i < (ValidCols + 7) / 8; ++i)
            bytes[i] = 0xff;
        if constexpr (ValidCols % 8)
            bytes[ValidCols / 8] = (1 << (ValidCols % 8)) - 1;
        Execute(launch, source, indices, output, expected, (ValidCols + 7) / 8);
    } else
        Execute(launch, source, indices, output, expected);
}

template <typename T, int Op, int Rows, int Cols, int ValidRows, int ValidCols>
void TestLayout(void (*launch)(T*, T*, uint32_t*, void*))
{
    constexpr int count = Rows * Cols;
    constexpr T sentinel = static_cast<T>(0x5A5A5A5A5A5A5A5AULL);
    std::vector<T> source(count), output(count + 64, sentinel), expected = output;
    std::vector<uint32_t> indices(count, 0);
    for (int i = 0; i < count; ++i)
        source[i] = static_cast<T>(0x8000000000000000ULL ^ (uint64_t(i + 1) << 37) ^ (17ULL * i + 3));
    if constexpr (Op >= 2) {
        std::fill_n(expected.begin(), count, T(0));
#ifdef __CPU_SIM
        // CPU indexed scatter leaves unselected elements unchanged.
        std::fill_n(output.begin(), count, T(0));
#endif
    }
    for (int r = 0; r < ValidRows; ++r) {
        for (int c = 0; c < ValidCols; ++c) {
            const int selected = ValidRows * ValidCols - 1 - (r * ValidCols + c);
            const int physical = selected / ValidCols * Cols + selected % ValidCols;
            const int idxOffset = (Op == 0 || Op == 2) ? c * Rows + r : r * Cols + c;
            indices[idxOffset] = physical;
            if constexpr (Op < 2)
                expected[Op == 1 ? c * Rows + r : r * Cols + c] = source[physical];
            else
                expected[physical] = source[Op == 3 ? c * Rows + r : r * Cols + c];
        }
    }
    Execute(launch, source, indices, output, expected);
}

void TestEmpty(void (*launch)(uint64_t*, uint64_t*, uint32_t*, void*))
{
    std::vector<uint64_t> source(32, 0), output(80, 0x5A5A5A5A5A5A5A5AULL), expected = output;
    std::vector<uint32_t> indices(8, 0);
    Execute(launch, source, indices, output, expected);
}

#define PTO_BOUND_CASE(Name, T, Op, Cols, ValidCols) \
    void Launch_##Name(T*, T*, uint32_t*, void*);    \
    TEST_F(Int64MemoryTest, Name) { TestBoundary<T, Op, Cols, ValidCols>(Launch_##Name); }
#define PTO_LAYOUT_CASE(Name, T, Op, Rows, Cols, ValidRows, ValidCols) \
    void Launch_##Name(T*, T*, uint32_t*, void*);                      \
    TEST_F(Int64MemoryTest, Name) { TestLayout<T, Op, Rows, Cols, ValidRows, ValidCols>(Launch_##Name); }
#define PTO_EMPTY_CASE(Name, Offset, Rows, Cols)                \
    void Launch_##Name(uint64_t*, uint64_t*, uint32_t*, void*); \
    TEST_F(Int64MemoryTest, Name) { TestEmpty(Launch_##Name); }
#include "cases.inc"
