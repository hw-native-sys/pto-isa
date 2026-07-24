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
#include <gtest/gtest.h>
#include <acl/acl.h>

using namespace std;
using namespace PtoTestCommon;

template <
    typename T, int kGRowsD_, int kGColsD_, int kGRowsS0_, int kGColsS0_, int kGRowsS1_, int kGColsS1_, int kTRowsD_,
    int kTColsD_, int kTRowsS0_, int kTColsS0_, int kTRowsS1_, int kTColsS1_>
void LaunchTPartMax(T* out, T* src0, T* src1, aclrtStream stream);

class TPARTMAXTest : public testing::Test {
public:
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
    typename T, int kGRowsD_, int kGColsD_, int kGRowsS0_, int kGColsS0_, int kGRowsS1_, int kGColsS1_, int kTRowsD_,
    int kTColsD_, int kTRowsS0_, int kTColsS0_, int kTRowsS1_, int kTColsS1_>
bool TPartMaxTest()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t dstByteSize = kGRowsD_ * kGColsD_ * sizeof(T);
    size_t src0ByteSize = kGRowsS0_ * kGColsS0_ * sizeof(T) + 1;
    size_t src1ByteSize = kGRowsS1_ * kGColsS1_ * sizeof(T) + 1;

    T* dstHost = nullptr;
    T* src0Host = nullptr;
    T* src1Host = nullptr;
    T* dstDevice = nullptr;
    T* src0Device = nullptr;
    T* src1Device = nullptr;

    aclrtMallocHost((void**)(&dstHost), dstByteSize);
    aclrtMallocHost((void**)(&src0Host), src0ByteSize);
    aclrtMallocHost((void**)(&src1Host), src1ByteSize);

    aclrtMalloc((void**)(&dstDevice), dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)(&src0Device), src0ByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)(&src1Device), src1ByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input0.bin", src0ByteSize, src0Host, src0ByteSize);
    ReadFile(GetGoldenDir() + "/input1.bin", src1ByteSize, src1Host, src1ByteSize);

    aclrtMemcpy(src0Device, src0ByteSize, src0Host, src0ByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1ByteSize, src1Host, src1ByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTPartMax<
        T, kGRowsD_, kGColsD_, kGRowsS0_, kGColsS0_, kGRowsS1_, kGColsS1_, kTRowsD_, kTColsD_, kTRowsS0_, kTColsS0_,
        kTRowsS1_, kTColsS1_>(dstDevice, src0Device, src1Device, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(dstHost);
    aclrtFree(src0Host);
    aclrtFree(src1Host);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(dstByteSize);
    std::vector<float> devFinal(dstByteSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);
    return ResultCmp(golden, devFinal, 0.001f);
}

TEST_F(TPARTMAXTest, test0)
{
    bool res = TPartMaxTest<float, 16, 32, 16, 16, 16, 32, 16, 32, 16, 16, 16, 32>();
    EXPECT_TRUE(res);
}
TEST_F(TPARTMAXTest, test1)
{
    bool res = TPartMaxTest<float, 22, 32, 22, 32, 16, 32, 22, 32, 22, 32, 16, 32>();
    EXPECT_TRUE(res);
}
TEST_F(TPARTMAXTest, test2)
{
    bool res = TPartMaxTest<float, 22, 40, 22, 40, 22, 32, 22, 40, 22, 40, 22, 32>();
    EXPECT_TRUE(res);
}
TEST_F(TPARTMAXTest, test3)
{
    bool res = TPartMaxTest<float, 22, 40, 22, 40, 8, 40, 22, 40, 22, 40, 8, 40>();
    EXPECT_TRUE(res);
}
TEST_F(TPARTMAXTest, test4)
{
    bool res = TPartMaxTest<float, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128, 64, 128>();
    EXPECT_TRUE(res);
}
TEST_F(TPARTMAXTest, testEmpty0)
{
    bool res = TPartMaxTest<float, 16, 32, 16, 0, 16, 32, 16, 32, 16, 8, 16, 32>();
    EXPECT_TRUE(res);
}
TEST_F(TPARTMAXTest, testEmpty1)
{
    bool res = TPartMaxTest<float, 16, 32, 0, 32, 16, 32, 16, 32, 8, 32, 16, 32>();
    EXPECT_TRUE(res);
}
TEST_F(TPARTMAXTest, testEmpty2)
{
    bool res = TPartMaxTest<float, 16, 32, 16, 32, 16, 0, 16, 32, 16, 32, 16, 8>();
    EXPECT_TRUE(res);
}
TEST_F(TPARTMAXTest, testEmpty3)
{
    bool res = TPartMaxTest<float, 16, 32, 16, 32, 0, 32, 16, 32, 16, 32, 8, 32>();
    EXPECT_TRUE(res);
}
