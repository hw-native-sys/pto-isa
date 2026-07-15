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
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <
    typename T, int format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gShape5,
    int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void LaunchTTRANSConv(T* out, T* src, void* stream);

template <
    typename T, int format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gShape5, int gShape6,
    int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, int gWholeShape5>
void LaunchTTRANSGroupConv(T* out, T* src, void* stream);

class TTRANSConvTest : public testing::Test {
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

template <
    typename T, int format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gShape5,
    int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void test_ttrans()
{
    size_t srcFileSize = gShape0 * gShape1 * gShape2 * gShape3 * gShape4 * gShape5 * sizeof(T);
    size_t dstFileSize = srcFileSize;

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

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTTRANSConv<
        T, format, gShape0, gShape1, gShape2, gShape3, gShape4, gShape5, gWholeShape0, gWholeShape1, gWholeShape2,
        gWholeShape3, gWholeShape4>(dstDevice, srcDevice, stream);

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

    std::vector<T> golden(dstFileSize);
    std::vector<T> result(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

    bool ret = ResultCmp(golden, result, 0.001f);

    EXPECT_TRUE(ret);
}

template <
    typename T, int format, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gShape5, int gShape6,
    int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, int gWholeShape5>
void test_ttrans_group()
{
    size_t srcFileSize = gShape0 * gShape1 * gShape2 * gShape3 * gShape4 * gShape5 * gShape6 * sizeof(T);
    size_t dstFileSize = srcFileSize;

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

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTTRANSGroupConv<
        T, format, gShape0, gShape1, gShape2, gShape3, gShape4, gShape5, gShape6, gWholeShape0, gWholeShape1,
        gWholeShape2, gWholeShape3, gWholeShape4, gWholeShape5>(dstDevice, srcDevice, stream);

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

    std::vector<T> golden(dstFileSize);
    std::vector<T> result(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

    bool ret = ResultCmp(golden, result, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TTRANSConvTest, int32_1_8_1_8) { test_ttrans<int32_t, 0, 1, 1, 1, 8, 8, 1, 1, 1, 8, 1, 8>(); }

TEST_F(TTRANSConvTest, float32_5_57_4_16) { test_ttrans<float, 0, 5, 4, 4, 16, 16, 1, 1, 5, 57, 4, 16>(); }

TEST_F(TTRANSConvTest, half_1_30_2_16) { test_ttrans<aclFloat16, 0, 1, 2, 2, 16, 16, 1, 1, 1, 30, 2, 16>(); }

TEST_F(TTRANSConvTest, int16_7_53_6_16) { test_ttrans<int16_t, 0, 7, 4, 6, 16, 16, 1, 1, 7, 53, 6, 16>(); }

TEST_F(TTRANSConvTest, int8_3_64_2_64) { test_ttrans<int8_t, 0, 3, 2, 2, 64, 32, 1, 1, 3, 64, 2, 64>(); }

TEST_F(TTRANSConvTest, int8_1_63_2_128) { test_ttrans<int8_t, 0, 1, 2, 2, 128, 32, 1, 1, 1, 63, 2, 128>(); }

TEST_F(TTRANSConvTest, int8_5_58_2_16) { test_ttrans<int8_t, 0, 5, 2, 2, 16, 32, 1, 1, 5, 58, 2, 16>(); }

TEST_F(TTRANSConvTest, uint8_9_87_6_16) { test_ttrans<uint8_t, 0, 9, 3, 6, 16, 32, 1, 1, 9, 87, 6, 16>(); }

TEST_F(TTRANSConvTest, float32_1_32_6_48) { test_ttrans<float, 0, 1, 8, 6, 48, 4, 1, 1, 1, 32, 6, 48>(); }

TEST_F(TTRANSConvTest, uint16_1_26_2_16) { test_ttrans<uint16_t, 0, 1, 7, 2, 16, 4, 1, 1, 1, 26, 2, 16>(); }

TEST_F(TTRANSConvTest, int8_5_18_2_16) { test_ttrans<int8_t, 0, 5, 5, 2, 16, 4, 1, 1, 5, 18, 2, 16>(); }

// NC1HWC0 -> C1HWN0N1C0
TEST_F(TTRANSConvTest, float32_3_2_2_16_4) { test_ttrans<float, 1, 2, 2, 16, 2, 2, 4, 3, 2, 2, 16, 4>(); }

TEST_F(TTRANSConvTest, int32_37_2_3_10_8) { test_ttrans<int32_t, 1, 2, 3, 10, 3, 16, 8, 37, 2, 3, 10, 8>(); }

TEST_F(TTRANSConvTest, float16_7_2_1_8_16) { test_ttrans<aclFloat16, 1, 2, 1, 8, 1, 16, 16, 7, 2, 1, 8, 16>(); }

TEST_F(TTRANSConvTest, float16_7_2_1_8_4) { test_ttrans<aclFloat16, 1, 2, 1, 8, 1, 16, 4, 7, 2, 1, 8, 4>(); }

TEST_F(TTRANSConvTest, uint16_45_3_2_7_16) { test_ttrans<uint16_t, 1, 3, 2, 7, 3, 16, 16, 45, 3, 2, 7, 16>(); }

TEST_F(TTRANSConvTest, int8_25_5_1_6_32) { test_ttrans<int8_t, 1, 5, 1, 6, 2, 16, 32, 25, 5, 1, 6, 32>(); }

TEST_F(TTRANSConvTest, uint8_11_2_7_7_32) { test_ttrans<uint8_t, 1, 2, 7, 7, 1, 16, 32, 11, 2, 7, 7, 32>(); }

TEST_F(TTRANSConvTest, int32_4_1_8_1_8) { test_ttrans_group<int32_t, 0, 4, 1, 1, 1, 8, 8, 1, 1, 4, 1, 8, 1, 8>(); }

TEST_F(TTRANSConvTest, float32_2_5_30_4_16)
{
    test_ttrans_group<float, 0, 2, 5, 2, 4, 16, 16, 1, 1, 2, 5, 30, 4, 16>();
}

TEST_F(TTRANSConvTest, half_1_1_30_2_16)
{
    test_ttrans_group<aclFloat16, 0, 1, 1, 2, 2, 16, 16, 1, 1, 1, 1, 30, 2, 16>();
}

TEST_F(TTRANSConvTest, float32_2_1_32_6_12) { test_ttrans_group<float, 0, 2, 1, 8, 6, 12, 4, 1, 1, 2, 1, 32, 6, 12>(); }

TEST_F(TTRANSConvTest, float32_1_3_2_2_16_4) { test_ttrans_group<float, 1, 1, 2, 2, 16, 2, 2, 4, 1, 3, 2, 2, 16, 4>(); }

TEST_F(TTRANSConvTest, float32_2_3_2_2_16_4) { test_ttrans_group<float, 1, 2, 2, 2, 16, 2, 2, 4, 2, 3, 2, 2, 16, 4>(); }

TEST_F(TTRANSConvTest, float32_2_4_2_2_16_4) { test_ttrans_group<float, 1, 2, 2, 2, 16, 2, 2, 4, 2, 4, 2, 2, 16, 4>(); }

TEST_F(TTRANSConvTest, float16_1_7_2_1_8_16)
{
    test_ttrans_group<aclFloat16, 1, 1, 2, 1, 8, 1, 16, 16, 1, 7, 2, 1, 8, 16>();
}

TEST_F(TTRANSConvTest, float16_4_7_2_1_8_4)
{
    test_ttrans_group<aclFloat16, 1, 4, 2, 1, 8, 1, 16, 4, 4, 7, 2, 1, 8, 4>();
}