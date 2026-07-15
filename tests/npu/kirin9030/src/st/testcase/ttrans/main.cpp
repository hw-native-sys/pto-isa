/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <type_traits>
#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace PtoTestCommon;

// #define DEBUG_PRINT

#ifdef DEBUG_PRINT
template <typename T>
void PrintFirst64(const char* name, const T* data, size_t totalSize)
{
    size_t count = std::min(totalSize / sizeof(T), static_cast<size_t>(64));
    std::cout << "\n=== " << name << " (first " << count << " values) ===" << std::endl;
    for (size_t i = 0; i < count; ++i) {
        if constexpr (std::is_same_v<T, aclFloat16>) {
            std::cout << std::setw(10) << aclFloat16ToFloat(data[i]);
        } else {
            std::cout << std::setw(10) << static_cast<float>(data[i]);
        }
        if ((i + 1) % 8 == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl;
}
#endif

template <typename T, int dstTRows, int dstTCols, int srcTRows, int srcTCols, int vRows, int vCols>
void LaunchTTRANS(T* out, T* src, void* stream);

template <int dstTRows, int dstTCols, int srcTRows, int srcTCols, int vRows, int vCols>
void LaunchTTRANSHalf(aclFloat16* out, aclFloat16* src, void* stream);

class TTRANSTest : public testing::Test {
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

template <typename T, int dstTRows, int dstTCols, int srcTRows, int srcTCols, int vRows, int vCols, bool isHalf = false>
void test_ttrans()
{
    size_t srcFileSize = srcTRows * srcTCols * sizeof(T);
    size_t dstFileSize = dstTRows * dstTCols * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&srcHost), srcFileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);
    aclrtMemset(dstHost, dstFileSize, 0, dstFileSize);

    aclrtMemcpy(dstDevice, dstFileSize, dstHost, dstFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if constexpr (isHalf) {
        LaunchTTRANSHalf<dstTRows, dstTCols, srcTRows, srcTCols, vRows, vCols>(dstDevice, srcDevice, stream);
    } else {
        LaunchTTRANS<T, dstTRows, dstTCols, srcTRows, srcTCols, vRows, vCols>(dstDevice, srcDevice, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize / sizeof(T));
    std::vector<T> result(dstFileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

#ifdef DEBUG_PRINT
    std::vector<T> inputData(srcFileSize / sizeof(T));
    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, inputData.data(), srcFileSize);
    PrintFirst64("INPUT", inputData.data(), srcFileSize);
    PrintFirst64("OUTPUT", result.data(), dstFileSize);
    PrintFirst64("GOLDEN", golden.data(), dstFileSize);
#endif

    bool ret = ResultCmp(golden, result, 0.000f);

    EXPECT_TRUE(ret);
}

TEST_F(TTRANSTest, case_float_8x8_2x8_2x8) { test_ttrans<float, 8, 8, 2, 8, 2, 8>(); }
TEST_F(TTRANSTest, case_half_16x16_16x16_16x16) { test_ttrans<aclFloat16, 16, 16, 16, 16, 16, 16, true>(); }
TEST_F(TTRANSTest, case_float_16x32_32x16_31x15) { test_ttrans<float, 16, 32, 32, 16, 31, 15>(); }
TEST_F(TTRANSTest, case_half_32x32_32x32_31x31) { test_ttrans<aclFloat16, 32, 32, 32, 32, 31, 31, true>(); }
TEST_F(TTRANSTest, case_float_8x8_4x8_4x8) { test_ttrans<float, 8, 8, 4, 8, 4, 8>(); }
TEST_F(TTRANSTest, case_float_512x16_9x512_9x512) { test_ttrans<float, 512, 16, 9, 512, 9, 512>(); }
TEST_F(TTRANSTest, case_float_66x88_9x16_7x15) { test_ttrans<float, 66, 88, 9, 16, 7, 15>(); }
TEST_F(TTRANSTest, case_float_16x32_32x16_23x15) { test_ttrans<float, 16, 32, 32, 16, 23, 15>(); }
TEST_F(TTRANSTest, case_float_128x64_64x128_27x77) { test_ttrans<float, 128, 64, 64, 128, 27, 77>(); }
TEST_F(TTRANSTest, case_half_64x112_100x64_64x64) { test_ttrans<aclFloat16, 64, 112, 100, 64, 64, 64, true>(); }
TEST_F(TTRANSTest, case_half_64x128_128x64_64x64) { test_ttrans<aclFloat16, 64, 128, 128, 64, 64, 64, true>(); }
TEST_F(TTRANSTest, case_half_64x128_128x64_100x64) { test_ttrans<aclFloat16, 64, 128, 128, 64, 100, 64, true>(); }
TEST_F(TTRANSTest, case_float_32x512_512x32_512x2) { test_ttrans<float, 32, 512, 512, 32, 512, 2>(); }
TEST_F(TTRANSTest, case_float_16x8_1x16_1x16) { test_ttrans<float, 16, 8, 1, 16, 1, 16>(); }
TEST_F(TTRANSTest, case_float_64x64_64x64_36x64) { test_ttrans<float, 64, 64, 64, 64, 36, 64>(); }
TEST_F(TTRANSTest, case_float_8x8_8x8_8x8) { test_ttrans<float, 8, 8, 8, 8, 8, 8>(); }
TEST_F(TTRANSTest, case_uint8_32x32_32x32_32x32) { test_ttrans<uint8_t, 32, 32, 32, 32, 32, 32>(); }
TEST_F(TTRANSTest, case_uint8_64x64_64x64_22x63) { test_ttrans<uint8_t, 64, 64, 64, 64, 22, 63>(); }
TEST_F(TTRANSTest, case_float_8x8_1x8_1x8) { test_ttrans<float, 8, 8, 1, 8, 1, 8>(); }
