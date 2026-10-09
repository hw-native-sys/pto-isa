/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include "test_common.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

enum class TStoreConvCase {
    NDC1HWC0Static,
    NC1HWC0Static,
    NDC1HWC0DynN,
    NDC1HWC0DynND,
    NDC1HWC0DynNDGlobal,
};

template <typename T, TStoreConvCase caseKey, int s0, int s1, int s2, int s3, int s4>
void LaunchTStoreConv(T* out, T* src, int64_t n, int64_t d, void* stream);

class TStoreConvTest : public testing::Test {
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

template <typename T, typename LaunchFn>
void RunTStoreConvCase(size_t srcFileSize, size_t dstFileSize, LaunchFn launch)
{
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
    aclrtMemset(dstDevice, dstFileSize, 0, dstFileSize);

    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launch(dstDevice, srcDevice, stream);

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

    std::vector<T> golden(dstFileSize / sizeof(T), 0);
    std::vector<T> result(dstFileSize / sizeof(T), 0);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, result.data(), dstFileSize);

    bool ret = ResultCmp<T>(golden, result, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename T, TStoreConvCase caseKey, int s0, int s1, int s2, int s3, int s4>
void test_tstore_conv(int64_t n = 0, int64_t d = 0)
{
    constexpr size_t kC0 = 32 / sizeof(T);
    size_t fileSize = s0 * s1 * s2 * s3 * s4 * kC0 * sizeof(T);

    RunTStoreConvCase<T>(fileSize, fileSize, [&](T* dst, T* src, void* stream) {
        LaunchTStoreConv<T, caseKey, s0, s1, s2, s3, s4>(dst, src, n, d, stream);
    });
}

/*-----------------Static shapes------------------*/
TEST_F(TStoreConvTest, NDC1HWC0_1) { test_tstore_conv<float, TStoreConvCase::NDC1HWC0Static, 1, 1, 1, 2, 8>(); }

TEST_F(TStoreConvTest, NDC1HWC0_2) { test_tstore_conv<float, TStoreConvCase::NDC1HWC0Static, 2, 3, 4, 1, 7>(); }

TEST_F(TStoreConvTest, NDC1HWC0_3) { test_tstore_conv<int32_t, TStoreConvCase::NDC1HWC0Static, 1, 2, 4, 16, 8>(); }

/*---------Static shapes, unaligned sizes---------*/
TEST_F(TStoreConvTest, NDC1HWC0_4) { test_tstore_conv<float, TStoreConvCase::NDC1HWC0Static, 3, 2, 3, 3, 5>(); }

TEST_F(TStoreConvTest, NDC1HWC0_5) { test_tstore_conv<int32_t, TStoreConvCase::NDC1HWC0Static, 3, 3, 5, 3, 3>(); }

TEST_F(TStoreConvTest, NC1HWC0_1) { test_tstore_conv<float, TStoreConvCase::NC1HWC0Static, 3, 3, 5, 3, 1>(); }

/*------------------Dynamic tile dims------------------*/
TEST_F(TStoreConvTest, NDC1HWC0_Dyn1) { test_tstore_conv<float, TStoreConvCase::NDC1HWC0DynN, 4, 1, 1, 2, 8>(2); }

TEST_F(TStoreConvTest, NDC1HWC0_Dyn2) { test_tstore_conv<float, TStoreConvCase::NDC1HWC0DynND, 4, 3, 1, 7, 8>(2, 1); }

TEST_F(TStoreConvTest, NDC1HWC0_Dyn3) { test_tstore_conv<int32_t, TStoreConvCase::NDC1HWC0DynN, 3, 2, 4, 8, 8>(2); }

TEST_F(TStoreConvTest, NDC1HWC0_Dyn4)
{
    test_tstore_conv<float, TStoreConvCase::NDC1HWC0DynNDGlobal, 4, 3, 2, 4, 8>(3, 2);
}

/*-------Dynamic tile dims, unaligned sizes-------*/
TEST_F(TStoreConvTest, NDC1HWC0_Dyn5) { test_tstore_conv<float, TStoreConvCase::NDC1HWC0DynN, 4, 3, 3, 3, 5>(3); }
