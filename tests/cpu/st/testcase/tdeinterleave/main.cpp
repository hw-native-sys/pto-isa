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
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

class TDEINTERLEAVETest : public testing::Test {
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

template <typename T, int row, int col>
void LaunchTDEINTERLEAVE(T *dst1, T *dst0, T *src1, T *src0, void *stream);

template <typename T, int row, int col>
void LaunchTDEINTERLEAVE(T *dst1, T *dst0, T *src, void *stream);

template <typename T, int row, int col>
void test_TDEINTERLEAVE()
{
    size_t dstFileSize = row * col * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dst1Host, *dst0Host, *src1Host, *src0Host;
    T *dst1Device, *dst0Device, *src1Device, *src0Device;

    aclrtMallocHost((void **)(&dst1Host), dstFileSize);
    aclrtMallocHost((void **)(&dst0Host), dstFileSize);
    aclrtMallocHost((void **)(&src1Host), dstFileSize);
    aclrtMallocHost((void **)(&src0Host), dstFileSize);

    aclrtMalloc((void **)&dst1Device, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dst0Device, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input0.bin", dstFileSize, src0Host, dstFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", dstFileSize, src1Host, dstFileSize));

    aclrtMemcpy(src0Device, dstFileSize, src0Host, dstFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, dstFileSize, src1Host, dstFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTDEINTERLEAVE<T, row, col>(dst1Device, dst0Device, src1Device, src0Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dst1Host, dstFileSize, dst1Device, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(dst0Host, dstFileSize, dst0Device, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    CHECK_RESULT_GTEST(WriteFile(GetGoldenDir() + "/output1.bin", dst1Host, dstFileSize));
    CHECK_RESULT_GTEST(WriteFile(GetGoldenDir() + "/output0.bin", dst0Host, dstFileSize));

    aclrtFree(dst1Device);
    aclrtFree(dst0Device);
    aclrtFree(src1Device);
    aclrtFree(src0Device);

    aclrtFreeHost(dst1Host);
    aclrtFreeHost(dst0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src0Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden0(row * col);
    std::vector<T> output0(row * col);

    std::vector<T> golden1(row * col);
    std::vector<T> output1(row * col);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/dst0.bin", dstFileSize, golden0.data(), dstFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output0.bin", dstFileSize, output0.data(), dstFileSize));
    bool ret0 = ResultCmp<T>(golden0, output0, 0.001f);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/dst1.bin", dstFileSize, golden1.data(), dstFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output1.bin", dstFileSize, output1.data(), dstFileSize));
    bool ret1 = ResultCmp<T>(golden1, output1, 0.001f);

    EXPECT_TRUE(ret0 && ret1);
}

template <typename T, int row, int col>
void test_TDEINTERLEAVE_MODE_1()
{
    size_t dstFileSize = row * col * sizeof(T);
    size_t srcFileSize = 2 * dstFileSize;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dst1Host, *dst0Host, *srcHost;
    T *dst1Device, *dst0Device, *srcDevice;

    aclrtMallocHost((void **)(&dst1Host), dstFileSize);
    aclrtMallocHost((void **)(&dst0Host), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);

    aclrtMalloc((void **)&dst1Device, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dst0Device, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/src.bin", srcFileSize, srcHost, srcFileSize));

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTDEINTERLEAVE<T, row, col>(dst1Device, dst0Device, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dst1Host, dstFileSize, dst1Device, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(dst0Host, dstFileSize, dst0Device, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    CHECK_RESULT_GTEST(WriteFile(GetGoldenDir() + "/output1.bin", dst1Host, dstFileSize));
    CHECK_RESULT_GTEST(WriteFile(GetGoldenDir() + "/output0.bin", dst0Host, dstFileSize));

    aclrtFree(dst1Device);
    aclrtFree(dst0Device);
    aclrtFree(srcDevice);

    aclrtFreeHost(dst1Host);
    aclrtFreeHost(dst0Host);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden0(row * col);
    std::vector<T> output0(row * col);

    std::vector<T> golden1(row * col);
    std::vector<T> output1(row * col);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/dst0.bin", dstFileSize, golden0.data(), dstFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output0.bin", dstFileSize, output0.data(), dstFileSize));
    bool ret0 = ResultCmp<T>(golden0, output0, 0.001f);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/dst1.bin", dstFileSize, golden1.data(), dstFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output1.bin", dstFileSize, output1.data(), dstFileSize));
    bool ret1 = ResultCmp<T>(golden1, output1, 0.001f);

    EXPECT_TRUE(ret0 && ret1);
}

TEST_F(TDEINTERLEAVETest, case_1)
{
    test_TDEINTERLEAVE<float, 64, 64>();
}
TEST_F(TDEINTERLEAVETest, case_2)
{
    test_TDEINTERLEAVE<int32_t, 16, 64>();
}
TEST_F(TDEINTERLEAVETest, case_3)
{
    test_TDEINTERLEAVE<int16_t, 32, 128>();
}
TEST_F(TDEINTERLEAVETest, case_4)
{
    test_TDEINTERLEAVE<aclFloat16, 20, 16>();
}

TEST_F(TDEINTERLEAVETest, case_5)
{
    test_TDEINTERLEAVE_MODE_1<int32_t, 16, 64>();
}
