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
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <typename T, int tRows, int tCols, int vRows, int vCols>
void LaunchTTRANS(T *out, T *src, void *stream);

template <typename T, int tRows, int tCols, int vRows, int vCols>
void LaunchTTRANSInplace(T *out, T *src, void *stream);

class TTRANSTest : public testing::Test {
protected:
    void SetUp() override
    {}
    void TearDown() override
    {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo *testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    std::string fullPath = "../" + suiteName + "." + caseName;
    return fullPath;
}

template <typename T, int tRows, int tCols, int vRows, int vCols, bool inplace = false>
void test_ttrans()
{
    uint32_t M = tRows;
    uint32_t N = tCols;
    size_t srcFileSize = M * N * sizeof(T);
    size_t dstFileSize = M * N * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if constexpr (inplace) {
        LaunchTTRANSInplace<T, tRows, tCols, vRows, vCols>(dstDevice, srcDevice, stream);
    } else {
        LaunchTTRANS<T, tRows, tCols, vRows, vCols>(dstDevice, srcDevice, stream);
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

    bool ret = ResultCmp(golden, result, 0.001f);

    EXPECT_TRUE(ret);
}

// tile shape of dst and src can be different
// here we just test the same cases
TEST_F(TTRANSTest, case1_float_16_8_16_8)
{
    test_ttrans<float, 16, 8, 16, 8>();
}
TEST_F(TTRANSTest, case2_half_16_16_16_16)
{
    test_ttrans<aclFloat16, 16, 16, 16, 16>();
}
TEST_F(TTRANSTest, case3_float_32_16_31_15)
{
    test_ttrans<float, 32, 16, 31, 15>();
}
TEST_F(TTRANSTest, case4_half_32_32_31_31)
{
    test_ttrans<aclFloat16, 32, 32, 31, 31>();
}
TEST_F(TTRANSTest, case5_float_2_512_2_512)
{
    test_ttrans<float, 2, 512, 2, 512>();
}
TEST_F(TTRANSTest, case6_float_9_512_9_512)
{
    test_ttrans<float, 9, 512, 9, 512>();
}
TEST_F(TTRANSTest, case7_float_32_16_23_15)
{
    test_ttrans<float, 32, 16, 23, 15>();
}
TEST_F(TTRANSTest, case8_float_64_128_27_77)
{
    test_ttrans<float, 64, 128, 27, 77>();
}
TEST_F(TTRANSTest, case9_half_100_64_64_64)
{
    test_ttrans<aclFloat16, 100, 64, 64, 64>();
}
TEST_F(TTRANSTest, case10_half_128_64_64_64)
{
    test_ttrans<aclFloat16, 128, 64, 64, 64>();
}
TEST_F(TTRANSTest, case11_half_128_64_100_64)
{
    test_ttrans<aclFloat16, 128, 64, 100, 64>();
}
TEST_F(TTRANSTest, case12_float_512_32_512_2)
{
    test_ttrans<float, 512, 32, 512, 2>();
}
TEST_F(TTRANSTest, case13_float_64_64_64_64)
{
    test_ttrans<float, 64, 64, 64, 64>();
}
TEST_F(TTRANSTest, case14_float_64_32_64_32)
{
    test_ttrans<float, 64, 32, 64, 32>();
}
TEST_F(TTRANSTest, case15_float_64_64_36_64)
{
    test_ttrans<float, 64, 64, 36, 64>();
}
TEST_F(TTRANSTest, case16_float_2_16_2_16)
{
    test_ttrans<float, 2, 16, 2, 16>();
}
TEST_F(TTRANSTest, case17_int8_32_32_32_32)
{
    test_ttrans<uint8_t, 32, 32, 32, 32>();
}
TEST_F(TTRANSTest, case18_int8_64_64_22_63)
{
    test_ttrans<uint8_t, 64, 64, 22, 63>();
}
TEST_F(TTRANSTest, case19_float_8_8_8_8)
{
    test_ttrans<float, 8, 8, 8, 8>();
}
// In-place regression: dst tile aliases the src tile's UB buffer. The [8,8] FP32
// transpose takes the scalar ttrans path (dst cols 8, not a multiple of 16); the
// direct-write fallback corrupted dst == src, the tmp-staging fix makes it correct.
TEST_F(TTRANSTest, case_inplace_float_8_8_8_8)
{
    test_ttrans<float, 8, 8, 8, 8, true>();
}
