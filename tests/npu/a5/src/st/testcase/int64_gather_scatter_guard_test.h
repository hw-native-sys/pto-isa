/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INT64_GATHER_SCATTER_GUARD_TEST_H
#define INT64_GATHER_SCATTER_GUARD_TEST_H

#include <algorithm>
#include <cstdint>
#include <vector>

template <
    typename T, int Operation, int Rows, int SrcCols, int ValidRows, int ValidCols, int DstCols,
    pto::MaskPattern Pattern = pto::MaskPattern::P1010>
void LaunchInt64GatherScatterGuard(T* out, T* input, uint32_t* indices, void* stream);

template <
    typename T, int Operation, int Rows, int SrcCols, int ValidRows, int ValidCols, int DstCols,
    pto::MaskPattern Pattern = pto::MaskPattern::P1010>
void TestInt64GatherScatterGuard()
{
    constexpr int times = Pattern == pto::MaskPattern::P1111 ?
                              1 :
                              (Pattern == pto::MaskPattern::P0101 || Pattern == pto::MaskPattern::P1010 ? 2 : 4);
    constexpr int offset = Pattern == pto::MaskPattern::P1010 || Pattern == pto::MaskPattern::P0010 ?
                               1 :
                               (Pattern == pto::MaskPattern::P0100 ? 2 : (Pattern == pto::MaskPattern::P1000 ? 3 : 0));
    constexpr int dstRows = Operation == 4 ? Rows * times : Rows;
    constexpr int physicalElements = dstRows * DstCols;
    constexpr int outputElements = physicalElements + 64;
    constexpr int indexCols = (ValidCols + 7) / 8 * 8;
    constexpr T sentinel = static_cast<T>(0x5A5A5A5A5A5A5A5AULL);
    constexpr uint64_t edgeValues[] = {
        0, UINT64_MAX, 0x8000000000000000ULL, 0x7FFFFFFFFFFFFFFFULL, 0xFFFFFFFF00000000ULL, 0x00000000FFFFFFFFULL};
    std::vector<T> source(Rows * SrcCols);
    std::vector<uint32_t> indices(Rows * indexCols, 0);
    std::vector<T> output(outputElements, sentinel);
    std::vector<T> golden = output;
    if constexpr (Operation >= 2) {
        std::fill_n(golden.begin(), physicalElements, static_cast<T>(0));
    }
    for (int i = 0; i < Rows * SrcCols; ++i) {
        uint64_t bits =
            i < 6 ? edgeValues[i] : (0x8000000000000000ULL ^ (static_cast<uint64_t>(i + 1) << 37) ^ (17ULL * i + 3));
        source[i] = static_cast<T>(bits);
    }
    int packedIndex = 0;
    for (int row = 0; row < ValidRows; ++row) {
        for (int col = 0; col < ValidCols; ++col) {
            int linear = row * ValidCols + col;
            int selected = (17 * linear + 11) % (ValidRows * ValidCols);
            if constexpr (Operation == 0) {
                int sourceIndex = (selected / ValidCols) * SrcCols + selected % ValidCols;
                indices[row * indexCols + col] = sourceIndex;
                golden[row * DstCols + col] = source[sourceIndex];
            } else if constexpr (Operation == 1) {
                if (col % times == offset) {
                    golden[packedIndex++] = source[row * SrcCols + col];
                }
            } else if constexpr (Operation == 2) {
                int target = (selected / ValidCols) * DstCols + selected % ValidCols;
                indices[row * indexCols + col] = target;
                golden[target] = source[row * SrcCols + col];
            } else if constexpr (Operation == 3) {
                golden[row * DstCols + col * times + offset] = source[row * SrcCols + col];
            } else {
                golden[(row * times + offset) * DstCols + col] = source[row * SrcCols + col];
            }
        }
    }
    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
    ASSERT_EQ(aclrtSetDevice(0), ACL_SUCCESS);
    aclrtStream stream;
    ASSERT_EQ(aclrtCreateStream(&stream), ACL_SUCCESS);
    T* sourceDevice = nullptr;
    T* outputDevice = nullptr;
    uint32_t* indicesDevice = nullptr;
    size_t sourceBytes = source.size() * sizeof(T);
    size_t outputBytes = output.size() * sizeof(T);
    size_t indexBytes = indices.size() * sizeof(uint32_t);
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&sourceDevice), sourceBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&outputDevice), outputBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMalloc(reinterpret_cast<void**>(&indicesDevice), indexBytes, ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMemcpy(sourceDevice, sourceBytes, source.data(), sourceBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMemcpy(outputDevice, outputBytes, output.data(), outputBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMemcpy(indicesDevice, indexBytes, indices.data(), indexBytes, ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    LaunchInt64GatherScatterGuard<T, Operation, Rows, SrcCols, ValidRows, ValidCols, DstCols, Pattern>(
        outputDevice, sourceDevice, indicesDevice, stream);
    ASSERT_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
    ASSERT_EQ(
        aclrtMemcpy(output.data(), outputBytes, outputDevice, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    EXPECT_TRUE(PtoTestCommon::ResultCmpExact(golden, output.data()));
    aclrtFree(indicesDevice);
    aclrtFree(outputDevice);
    aclrtFree(sourceDevice);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

#endif
