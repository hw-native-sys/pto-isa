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
#include <cstdint>

using namespace std;
using namespace PtoTestCommon;

template <
    typename T, int format, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4, int WN5,
    int BASEM, int BASEK>
void launchTLOADMIX(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

class TLOADSHAPE2DTest : public testing::Test {
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
    typename T, int format, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4, int WN5,
    int BASEM, int BASEK>
void TLOADMIXFUNC()
{
    size_t aFileSize = WN1 * WN2 * WN3 * WN4 * WN5 * sizeof(T);
    size_t bFileSize = N4 * N5 * sizeof(T);
    size_t cFileSize = BASEM * BASEK * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTLOADMIX<T, format, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
        dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<T> golden(cFileSize);
    std::vector<T> devFinal(cFileSize);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TLOADSHAPE2DTest, 1_1_1_128_128_half_ND2NZ)
{
    TLOADMIXFUNC<uint16_t, 0, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>();
}
TEST_F(TLOADSHAPE2DTest, 1_1_1_128_128_int8_t_ND2NZ)
{
    TLOADMIXFUNC<int8_t, 0, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>();
}

TEST_F(TLOADSHAPE2DTest, 1_1_1_128_128_float_ND2NZ)
{
    TLOADMIXFUNC<float, 0, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>();
}

TEST_F(TLOADSHAPE2DTest, 1_1_1_64_128_half_DN2NZ)
{
    TLOADMIXFUNC<uint16_t, 1, 1, 1, 1, 64, 128, 1, 1, 1, 64, 128, 64, 128>();
}

TEST_F(TLOADSHAPE2DTest, 1_1_1_63_127_half_ND2NZ)
{
    TLOADMIXFUNC<uint16_t, 0, 1, 1, 1, 63, 127, 1, 1, 1, 63, 127, 64, 128>();
}

TEST_F(TLOADSHAPE2DTest, 1_1_1_128_128_float_ND2ND)
{
    TLOADMIXFUNC<float, 2, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>();
}

TEST_F(TLOADSHAPE2DTest, 1_1_1_37_126_int8_t_ND2ND)
{
    TLOADMIXFUNC<int8_t, 2, 1, 1, 1, 37, 126, 1, 1, 1, 37, 126, 37, 128>();
}

TEST_F(TLOADSHAPE2DTest, 1_1_1_33_99_1_1_1_64_128_48_112_half_ND2NZ)
{
    TLOADMIXFUNC<uint16_t, 0, 1, 1, 1, 33, 99, 1, 1, 1, 64, 128, 48, 112>();
}
TEST_F(TLOADSHAPE2DTest, 1_1_1_59_119_1_1_1_64_128_64_128_int8_t_ND2NZ)
{
    TLOADMIXFUNC<int8_t, 0, 1, 1, 1, 59, 119, 1, 1, 1, 64, 128, 64, 128>();
}

TEST_F(TLOADSHAPE2DTest, 1_1_1_51_123_1_1_1_64_128_64_128_float_DN2NZ)
{
    TLOADMIXFUNC<float, 1, 1, 1, 1, 51, 123, 1, 1, 1, 64, 128, 64, 128>();
}

TEST_F(TLOADSHAPE2DTest, 1_1_1_63_127_1_1_1_63_127_64_128_half_DN2NZ)
{
    TLOADMIXFUNC<uint16_t, 1, 1, 1, 1, 63, 127, 1, 1, 1, 63, 127, 64, 128>();
}

// 3:DN2DN
TEST_F(TLOADSHAPE2DTest, 1_1_1_128_128_1_1_1_128_128_128_128_float_DN2DN)
{
    TLOADMIXFUNC<float, 3, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>();
}
TEST_F(TLOADSHAPE2DTest, 1_1_1_37_126_1_1_1_37_126_64_126_int8_t_DN2DN)
{
    TLOADMIXFUNC<int8_t, 3, 1, 1, 1, 37, 126, 1, 1, 1, 37, 126, 64, 126>();
}

// NZ2NZ
TEST_F(TLOADSHAPE2DTest, 1_10_8_16_16_1_11_9_16_16_128_160_half_NZ2NZ)
{
    TLOADMIXFUNC<uint16_t, 4, 1, 10, 8, 16, 16, 1, 11, 9, 16, 16, 128, 160>();
}
TEST_F(TLOADSHAPE2DTest, 1_8_4_16_32_1_9_4_16_32_80_256_int8_t_NZ2NZ)
{
    TLOADMIXFUNC<int8_t, 4, 1, 8, 4, 16, 32, 1, 9, 4, 16, 32, 80, 256>();
}

TEST_F(TLOADSHAPE2DTest, 1_1_1_59_119_1_1_1_59_124_59_120_int64_t_ND2ND)
{
    TLOADMIXFUNC<int64_t, 2, 1, 1, 1, 59, 119, 1, 1, 1, 59, 124, 59, 120>();
}