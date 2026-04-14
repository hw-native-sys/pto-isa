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

template <typename T, int validRow, int validCol, int Row, int Col, bool src0eqdst>
void launchTRowExpandMin(T *out, T *src0, T *src1, void *stream);

template <typename T, int validRow, int validCol, int Row, int Col, bool src0eqdst>
void launchTRowExpandMin2(T *out, T *src0, T *src1, void *stream);

template <typename T, int validRow, int validCol, int Row, int Col, bool src0eqdst>
void launchTRowExpandMin3(T *out, T *src0, T *src1, void *stream);

class TROWEXPANDMINTest : public testing::Test {
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

template <typename T, int validRow, int validCol, int Row, int Col, bool src0eqdst, bool isRowMajor,
          bool declTmp = false>
void test_trowexpandmin()
{
    size_t dstFileSize = Row * Col * sizeof(T);
    size_t src1FileSize = ((validRow * sizeof(T) + 31) / 32) * 32;
    if (isRowMajor) {
        src1FileSize = Row * 32;
    }

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *dstHost, *src0Host, *src1Host;
    T *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&src0Host), dstFileSize);
    aclrtMallocHost((void **)(&src1Host), src1FileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", dstFileSize, src0Host, dstFileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", src1FileSize, src1Host, src1FileSize);

    aclrtMemcpy(src0Device, dstFileSize, src0Host, dstFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (declTmp) {
        launchTRowExpandMin3<T, validRow, validCol, Row, Col, src0eqdst>(dstDevice, src0Device, src1Device, stream);
    } else if (isRowMajor) {
        launchTRowExpandMin2<T, validRow, validCol, Row, Col, src0eqdst>(dstDevice, src0Device, src1Device, stream);
    } else {
        launchTRowExpandMin<T, validRow, validCol, Row, Col, src0eqdst>(dstDevice, src0Device, src1Device, stream);
    }

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
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

TEST_F(TROWEXPANDMINTest, case1)
{
    test_trowexpandmin<float, 16, 16, 16, 16, true, false>();
}

TEST_F(TROWEXPANDMINTest, case2)
{
    test_trowexpandmin<float, 16, 16, 32, 32, true, false>();
}
TEST_F(TROWEXPANDMINTest, case3)
{
    test_trowexpandmin<aclFloat16, 16, 16, 16, 16, true, false>();
}

TEST_F(TROWEXPANDMINTest, case4)
{
    test_trowexpandmin<aclFloat16, 16, 16, 32, 32, true, false>();
}
TEST_F(TROWEXPANDMINTest, case5)
{
    test_trowexpandmin<float, 1, 16384, 1, 16384, true, false>();
}

TEST_F(TROWEXPANDMINTest, case6)
{
    test_trowexpandmin<float, 2048, 1, 2048, 8, true, false>();
}

TEST_F(TROWEXPANDMINTest, case7)
{
    test_trowexpandmin<float, 16, 16, 16, 16, true, true>();
}

TEST_F(TROWEXPANDMINTest, case8)
{
    test_trowexpandmin<float, 16, 16, 32, 32, true, true>();
}
TEST_F(TROWEXPANDMINTest, case9)
{
    test_trowexpandmin<aclFloat16, 16, 16, 16, 16, true, true>();
}

TEST_F(TROWEXPANDMINTest, case10)
{
    test_trowexpandmin<aclFloat16, 16, 16, 32, 32, true, true>();
}
TEST_F(TROWEXPANDMINTest, case11)
{
    test_trowexpandmin<float, 1, 16384, 1, 16384, true, true>();
}

TEST_F(TROWEXPANDMINTest, case12)
{
    test_trowexpandmin<float, 2048, 1, 2048, 8, true, true>();
}

TEST_F(TROWEXPANDMINTest, case13)
{
    test_trowexpandmin<float, 16, 16, 16, 16, false, false>();
}

TEST_F(TROWEXPANDMINTest, case14)
{
    test_trowexpandmin<float, 16, 16, 16, 16, false, true>();
}

TEST_F(TROWEXPANDMINTest, case15)
{
    test_trowexpandmin<float, 16, 16, 32, 32, true, false, true>();
}

TEST_F(TROWEXPANDMINTest, case16)
{
    test_trowexpandmin<aclFloat16, 16, 16, 16, 16, true, false, true>();
}

TEST_F(TROWEXPANDMINTest, case17)
{
    test_trowexpandmin<float, 1, 16384, 1, 16384, true, false, true>();
}

TEST_F(TROWEXPANDMINTest, case18)
{
    test_trowexpandmin<float, 2048, 1, 2048, 8, true, false, true>();
}

TEST_F(TROWEXPANDMINTest, case19)
{
    test_trowexpandmin<int32_t, 16, 16, 16, 16, true, false>();
}

TEST_F(TROWEXPANDMINTest, case20)
{
    test_trowexpandmin<int32_t, 16, 16, 16, 16, true, true>();
}

TEST_F(TROWEXPANDMINTest, case21)
{
    test_trowexpandmin<int16_t, 16, 16, 16, 16, true, false>();
}

TEST_F(TROWEXPANDMINTest, case22)
{
    test_trowexpandmin<int16_t, 16, 16, 16, 16, true, true>();
}
