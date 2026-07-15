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

class TAXPYTest : public testing::Test {
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

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTAxpy(T* out, T* src0, float scalar, void* stream);

template <typename T, typename U, int kTRows_, int kTCols_, int vRows, int vCols>
void LaunchTAxpy(T* out, U* src0, float scalar, void* stream);

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols, typename U = T>
void test_taxpy()
{
    size_t dstFileSize = kTRows_ * kTCols_ * sizeof(T);
    size_t srcFileSize = kTRows_ * kTCols_ * sizeof(U);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *dstDevice;
    U *src0Host, *src0Device;
    float scalar;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&src0Host), srcFileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", dstFileSize, dstHost, dstFileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", srcFileSize, src0Host, srcFileSize);
    std::string scalar_file = GetGoldenDir() + "/scalar.bin";
    std::ifstream file(scalar_file, std::ios::binary);

    file.read(reinterpret_cast<char*>(&scalar), 4);
    file.close();

    aclrtMemcpy(src0Device, srcFileSize, src0Host, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dstDevice, dstFileSize, dstHost, dstFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if constexpr (std::is_same_v<T, U>) {
        LaunchTAxpy<T, kTRows_, kTCols_, vRows, vCols>(dstDevice, src0Device, scalar, stream);
    } else {
        LaunchTAxpy<T, U, kTRows_, kTCols_, vRows, vCols>(dstDevice, src0Device, scalar, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstFileSize);
    std::vector<T> devFinal(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<T>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TAXPYTest, case1) { test_taxpy<aclFloat16, 64, 64, 64, 64>(); }

TEST_F(TAXPYTest, case2) { test_taxpy<aclFloat16, 64, 64, 63, 63>(); }

TEST_F(TAXPYTest, case3) { test_taxpy<aclFloat16, 1, 16384, 1, 16384>(); }

TEST_F(TAXPYTest, case4) { test_taxpy<aclFloat16, 2048, 16, 2048, 16>(); }

TEST_F(TAXPYTest, case5) { test_taxpy<float, 64, 64, 64, 64>(); }

TEST_F(TAXPYTest, case6) { test_taxpy<float, 64, 64, 63, 63>(); }

TEST_F(TAXPYTest, case7) { test_taxpy<float, 64, 64, 63, 63, aclFloat16>(); }

TEST_F(TAXPYTest, case8) { test_taxpy<float, 4, 1024, 4, 1023, aclFloat16>(); }

TEST_F(TAXPYTest, case9) { test_taxpy<float, 256, 16, 256, 15, aclFloat16>(); }