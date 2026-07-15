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

template <
    typename T, int format, int srcShape0, int srcShape1, int srcShape2, int srcShape3, int srcShape4, int dstShape0,
    int dstShape1, int dstShape2, int dstShape3, int dstShape4, int groupN>
void LaunchTStoreConv(T* out, T* src, void* stream);

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

template <
    typename T, int format, int srcShape0, int srcShape1, int srcShape2, int srcShape3, int srcShape4, int dstShape0,
    int dstShape1, int dstShape2, int dstShape3, int dstShape4, int groupN = 1>
void test_tstore()
{
    size_t srcFileSize = srcShape0 * srcShape1 * srcShape2 * srcShape3 * srcShape4 * groupN * sizeof(T);
    size_t dstFileSize = dstShape0 * dstShape1 * dstShape2 * dstShape3 * dstShape4 * groupN * sizeof(T);

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

    LaunchTStoreConv<
        T, format, srcShape0, srcShape1, srcShape2, srcShape3, srcShape4, dstShape0, dstShape1, dstShape2, dstShape3,
        dstShape4, groupN>(dstDevice, srcDevice, stream);

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

TEST_F(TStoreConvTest, NDC1HWC0_1) { test_tstore<float, 2, 1, 1, 1, 2, 8, 1, 1, 1, 2, 8, 1>(); }

TEST_F(TStoreConvTest, NDC1HWC0_2) { test_tstore<float, 2, 3, 4, 1, 7, 8, 3, 4, 1, 7, 8, 2>(); }
