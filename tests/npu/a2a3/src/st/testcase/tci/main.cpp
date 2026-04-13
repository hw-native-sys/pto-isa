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
#include "tci_common.h"

using namespace std;
using namespace PtoTestCommon;

template <uint32_t GROW, uint32_t GCOL, uint32_t TROW, uint32_t TCOL, uint32_t descending, uint32_t mode>
void launchTCI_demo_b32(int32_t *out, void *stream);
template <uint32_t GROW, uint32_t GCOL, uint32_t TROW, uint32_t TCOL, uint32_t descending, uint32_t mode>
void launchTCI_demo_b16(int16_t *out, void *stream);

class TCITest : public testing::Test {
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
    std::cout << fullPath << std::endl;
    return fullPath;
}

template <typename T, uint32_t ROW, uint32_t COL, uint32_t descending, uint32_t start, uint32_t mode>
void test_vci_b32()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t FileSize = ROW * COL * sizeof(T);

    int32_t *dstHost;
    int32_t *dstDevice;

    aclrtMallocHost((void **)(&dstHost), FileSize);
    aclrtMalloc((void **)&dstDevice, FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    launchTCI_demo_b32<ROW, COL, ROW, COL, descending, mode>(dstDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, FileSize, dstDevice, FileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, FileSize);

    aclrtFree(dstDevice);
    aclrtFreeHost(dstHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<int32_t> golden(FileSize);
    std::vector<int32_t> devFinal(FileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", FileSize, golden.data(), FileSize);
    ReadFile(GetGoldenDir() + "/output.bin", FileSize, devFinal.data(), FileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

template <typename T, uint32_t ROW, uint32_t COL, uint32_t descending, uint32_t start, uint32_t mode>
void test_vci_b16()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    size_t FileSize = ROW * COL * sizeof(T);

    int16_t *dstHost;
    int16_t *dstDevice;

    aclrtMallocHost((void **)(&dstHost), FileSize);
    aclrtMalloc((void **)&dstDevice, FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    launchTCI_demo_b16<ROW, COL, ROW, COL, descending, mode>(dstDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, FileSize, dstDevice, FileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, FileSize);

    aclrtFree(dstDevice);
    aclrtFreeHost(dstHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<int16_t> golden(FileSize);
    std::vector<int16_t> devFinal(FileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", FileSize, golden.data(), FileSize);
    ReadFile(GetGoldenDir() + "/output.bin", FileSize, devFinal.data(), FileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TCITest, case1_int32)
{
    test_vci_b32<int32_t, 1, 128, 0, 0, 0>();
}

TEST_F(TCITest, case2_int32)
{
    test_vci_b32<int32_t, 1, 600, 0, 0, 0>();
}

TEST_F(TCITest, case3_int32)
{
    test_vci_b32<int32_t, 1, 32, 1, 0, 0>();
}

TEST_F(TCITest, case4_int32)
{
    test_vci_b32<int32_t, 1, 2000, 1, 0, 0>();
}

TEST_F(TCITest, case5_int16)
{
    test_vci_b16<int16_t, 1, 256, 0, 0, 0>();
}

TEST_F(TCITest, case6_int16)
{
    test_vci_b16<int16_t, 1, 800, 1, 0, 0>();
}

TEST_F(TCITest, case7_int16)
{
    test_vci_b16<int16_t, 1, 64, 0, 0, 0>();
}

TEST_F(TCITest, case8_int16)
{
    test_vci_b16<int16_t, 1, 5120, 1, 0, 0>();
}

TEST_F(TCITest, case9_int32)
{
    test_vci_b32<int32_t, 1, 128, 0, 0, 1>();
}

TEST_F(TCITest, case10_int32)
{
    test_vci_b32<int32_t, 1, 32, 1, 0, 1>();
}

TEST_F(TCITest, case11_int16)
{
    test_vci_b16<int16_t, 1, 256, 0, 0, 1>();
}

TEST_F(TCITest, case12_int16)
{
    test_vci_b16<int16_t, 1, 800, 1, 0, 1>();
}

TEST_F(TCITest, case13_int16)
{
    test_vci_b16<int16_t, 1, 3328, 1, 0, 1>();
}

TEST_F(TCITest, case14_int16)
{
    test_vci_b16<int16_t, 1, 64, 0, 0, 1>();
}

TEST_F(TCITest, case15_int16)
{
    test_vci_b16<int16_t, 1, 32, 1, 0, 1>();
}