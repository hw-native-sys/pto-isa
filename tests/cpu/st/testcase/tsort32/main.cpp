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
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

template <typename T0, typename T1, int kGRows, int kGCols, int kTRows, int kTCols, int validRow, int validCol>
void launchTSort32(T0 *out, T0 *src, T1 *idx, aclrtStream stream);

class TSORT32Test : public testing::Test {
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

template <typename T0, typename T1, int kGRows, int kGCols, int kTRows, int kTCols, int validRow, int validCol>
bool TSort32Test()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t dstByteSize = kGRows * kGCols * 8;
    size_t srcByteSize = kGRows * kGCols * sizeof(T0);
    size_t idxByteSize = kGRows * kGCols * sizeof(T1);

    T0 *dstHost = nullptr;
    T0 *srcHost = nullptr;
    T1 *idxHost = nullptr;
    T0 *dstDevice = nullptr;
    T0 *srcDevice = nullptr;
    T1 *idxDevice = nullptr;

    aclrtMallocHost((void **)(&dstHost), dstByteSize);
    aclrtMallocHost((void **)(&srcHost), srcByteSize);
    aclrtMallocHost((void **)(&idxHost), idxByteSize);

    aclrtMalloc((void **)(&dstDevice), dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)(&srcDevice), srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)(&idxDevice), idxByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", srcByteSize, srcHost, srcByteSize);
    ReadFile(GetGoldenDir() + "/input1.bin", idxByteSize, idxHost, idxByteSize);

    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(idxDevice, idxByteSize, idxHost, idxByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    PrintData(srcHost, kGRows * kGCols, INT16_T, kGCols);
    PrintData(idxHost, kGRows * kGCols, UINT32_T, kGCols);
    launchTSort32<T0, T1, kGRows, kGCols, kTRows, kTCols, validRow, validCol>(dstDevice, srcDevice, idxDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    PrintData(dstHost, kGRows * kGCols * 8 / sizeof(T0), INT16_T, kGCols * 8 / sizeof(T0));

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(idxDevice);
    aclrtFree(dstHost);
    aclrtFree(srcHost);
    aclrtFree(idxHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T0> golden(dstByteSize);
    std::vector<T0> devFinal(dstByteSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);
    return ResultCmp<T0>(golden, devFinal, 0.001f);
}

TEST_F(TSORT32Test, test0)
{
    bool res = TSort32Test<aclFloat16, uint32_t, 16, 16, 16, 16, 16, 16>();
    EXPECT_TRUE(res);
}

TEST_F(TSORT32Test, test1)
{
    bool res = TSort32Test<float, uint32_t, 8, 32, 8, 32, 8, 32>();
    EXPECT_TRUE(res);
}

TEST_F(TSORT32Test, test2)
{
    bool res = TSort32Test<float, uint32_t, 7, 32, 7, 32, 7, 32>();
    EXPECT_TRUE(res);
}

TEST_F(TSORT32Test, test3)
{
    bool res = TSort32Test<aclFloat16, uint32_t, 32, 16, 32, 16, 32, 16>();
    EXPECT_TRUE(res);
}