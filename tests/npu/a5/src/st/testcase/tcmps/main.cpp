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
#include <pto/common/type.hpp>

#include "acl/acl.h"

using namespace std;
using namespace PtoTestCommon;
using namespace pto;

class TCMPSTest : public testing::Test {
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

template <typename T, int Row, int Col, int ValidRow, int ValidCol, CmpMode cmpMode, bool isSrc1Tile>
void LaunchTCmps(uint8_t *out, T *src0, T *src1, void *stream);

template <typename T, int Row, int Col, int ValidRow, int ValidCol, CmpMode cmpMode, bool isSrc1Tile = false>
void test_tcmps()
{
    size_t src0FileSize = Row * Col * sizeof(T);
    size_t src1FileSize = sizeof(T);
    size_t dstFileSize = Row * ((Col + 7) / 8);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    T *src0Host, *src0Device;
    T *src1Host, *src1Device;

    uint8_t *dstHost, *dstDevice;
    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&src0Host), src0FileSize);
    aclrtMallocHost((void **)(&src1Host), src1FileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, src0FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, src1FileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input1.bin", src0FileSize, src0Host, src0FileSize);
    ReadFile(GetGoldenDir() + "/input2.bin", src1FileSize, src1Host, src1FileSize);

    aclrtMemcpy(src0Device, src0FileSize, src0Host, src0FileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, src1FileSize, src1Host, src1FileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTCmps<T, Row, Col, ValidRow, ValidCol, cmpMode, isSrc1Tile>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, src0FileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

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

    std::vector<uint8_t> golden(Row * ((Col + 7) / 8));
    std::vector<uint8_t> devFinal(Row * ((Col + 7) / 8));
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<uint8_t>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TCMPSTest, case_half_32x32_32x32)
{
    test_tcmps<uint16_t, 32, 32, 32, 32, CmpMode::EQ>();
}
TEST_F(TCMPSTest, case_float_8x64_8x64)
{
    test_tcmps<float, 8, 64, 8, 64, CmpMode::GT, true>();
}
TEST_F(TCMPSTest, case_int32_4x64_4x64)
{
    test_tcmps<int32_t, 4, 64, 4, 64, CmpMode::NE>();
}
TEST_F(TCMPSTest, case_int32_128x128_64x64)
{
    test_tcmps<int32_t, 128, 128, 64, 64, CmpMode::LT, true>();
}
TEST_F(TCMPSTest, case_int32_64x64_32x32)
{
    test_tcmps<int32_t, 64, 64, 32, 32, CmpMode::EQ>();
}
TEST_F(TCMPSTest, case_int32_16x32_16x32)
{
    test_tcmps<int32_t, 16, 32, 16, 32, CmpMode::EQ, true>();
}
TEST_F(TCMPSTest, case_float_128x128_64x64)
{
    test_tcmps<float, 128, 128, 64, 64, CmpMode::LE>();
}
TEST_F(TCMPSTest, case_int32_77x80_32x32)
{
    test_tcmps<int32_t, 77, 80, 32, 32, CmpMode::EQ, true>();
}
TEST_F(TCMPSTest, case_int32_32x32_32x32)
{
    test_tcmps<int32_t, 32, 32, 32, 32, CmpMode::EQ>();
}
TEST_F(TCMPSTest, case_int16_32x32_16x322)
{
    test_tcmps<int16_t, 32, 32, 16, 32, CmpMode::EQ, true>();
}
TEST_F(TCMPSTest, case_int16_77x80_32x32)
{
    test_tcmps<int16_t, 77, 80, 32, 32, CmpMode::LE>();
}
