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
    typename T, int src_row, int src_col, int src_validCol, int dst_row, int dst_col, int dst_validRow,
    int dst_validCol>
void launchTCOLEXPAND(T* out, T* src, void* stream);

class TCOLEXPANDTest : public testing::Test {
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
    typename T, int src_row, int src_col, int src_validCol, int dst_row, int dst_col, int dst_validRow,
    int dst_validCol>
bool TCOLEXPANDTestFramework()
{
    size_t dstByteSize = dst_row * dst_col * sizeof(T);
    size_t srcByteSize = src_row * src_col * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *srcHost;
    T *dstDevice, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), dstByteSize);
    aclrtMallocHost((void**)(&srcHost), srcByteSize);

    aclrtMalloc((void**)&dstDevice, dstByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcByteSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcByteSize, srcHost, srcByteSize);

    aclrtMemcpy(srcDevice, srcByteSize, srcHost, srcByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTCOLEXPAND<T, src_row, src_col, src_validCol, dst_row, dst_col, dst_validRow, dst_validCol>(
        dstDevice, srcDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstByteSize, dstDevice, dstByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstByteSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(dstByteSize);
    std::vector<T> devFinal(dstByteSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstByteSize, golden.data(), dstByteSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstByteSize, devFinal.data(), dstByteSize);

    return ResultCmp(golden, devFinal, 0.001f);
}

TEST_F(TCOLEXPANDTest, case1)
{
    bool ret = TCOLEXPANDTestFramework<int16_t, 32, 32, 8, 32, 32, 16, 8>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLEXPANDTest, case2)
{
    bool ret = TCOLEXPANDTestFramework<int32_t, 16, 16, 8, 24, 16, 16, 8>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLEXPANDTest, case3)
{
    bool ret = TCOLEXPANDTestFramework<float, 16, 16, 8, 24, 16, 16, 8>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLEXPANDTest, case4)
{
    bool ret = TCOLEXPANDTestFramework<int16_t, 8, 128, 127, 16, 128, 8, 127>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLEXPANDTest, case5)
{
    bool ret = TCOLEXPANDTestFramework<int32_t, 3, 64, 63, 16, 64, 15, 63>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLEXPANDTest, case6)
{
    bool ret = TCOLEXPANDTestFramework<float, 3, 64, 63, 16, 64, 15, 63>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLEXPANDTest, case7)
{
    bool ret = TCOLEXPANDTestFramework<int16_t, 16, 256, 256, 12, 256, 6, 256>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLEXPANDTest, case8)
{
    bool ret = TCOLEXPANDTestFramework<int32_t, 4, 256, 256, 16, 256, 15, 256>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLEXPANDTest, case9)
{
    bool ret = TCOLEXPANDTestFramework<float, 6, 64, 64, 16, 64, 15, 64>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLEXPANDTest, case10)
{
    bool ret = TCOLEXPANDTestFramework<int16_t, 16, 256, 255, 16, 256, 7, 255>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLEXPANDTest, case11)
{
    bool ret = TCOLEXPANDTestFramework<int32_t, 8, 256, 255, 32, 256, 31, 255>();
    EXPECT_TRUE(ret);
}

TEST_F(TCOLEXPANDTest, case12)
{
    bool ret = TCOLEXPANDTestFramework<float, 1, 64, 63, 1, 64, 1, 63>();
    EXPECT_TRUE(ret);
}
