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

namespace TloadMixTestFormat {
constexpr int ND2NZ = 0;
constexpr int ND2ND = 2;
constexpr int NZ2NZ = 4;
constexpr int NC1HWC02NC1HWC0 = 6;
constexpr int FZ2FZ = 7;
constexpr int FZ4D2FZ4D = 8;
} // namespace TloadMixTestFormat

template <typename T, int format, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4, int WN5,
          int BASEM, int BASEK>
void launchTLOADMIX(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

class TLOADMIXTest : public testing::Test {
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

template <typename T, int format, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4, int WN5,
          int BASEM, int BASEK>
void TLOADMIXFUNC()
{
    constexpr uint32_t c0SizeByte = 32;
    constexpr uint32_t n0 = 16;
    size_t aFileSize = WN1 * WN2 * WN3 * WN4 * WN5 * sizeof(T);
    size_t bFileSize = N4 * N5 * sizeof(T);
    size_t cFileSize = BASEM * BASEK * sizeof(T);
    if constexpr (format == TloadMixTestFormat::NC1HWC02NC1HWC0 || format == TloadMixTestFormat::FZ4D2FZ4D ||
                  format == TloadMixTestFormat::FZ2FZ) {
        cFileSize = N1 * N2 * N3 * N4 * N5 * sizeof(T);
    }

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, src1Host, bFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTLOADMIX<T, format, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(dstDevice, src0Device,
                                                                                         src1Device, stream);

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

// format 0:ND2NZ 2:ND2ND 4 NZ2NZ
TEST_F(TLOADMIXTest, 1_1_1_128_128_half_ND2NZ)
{
    TLOADMIXFUNC<uint16_t, 0, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>();
}
TEST_F(TLOADMIXTest, 1_1_1_128_128_int8_t_ND2NZ)
{
    TLOADMIXFUNC<int8_t, 0, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>();
}

TEST_F(TLOADMIXTest, 1_1_1_128_128_float_ND2NZ)
{
    TLOADMIXFUNC<float, 0, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>();
}

TEST_F(TLOADMIXTest, 1_1_1_63_127_half_ND2NZ)
{
    TLOADMIXFUNC<uint16_t, 0, 1, 1, 1, 63, 127, 1, 1, 1, 63, 127, 64, 128>();
}

TEST_F(TLOADMIXTest, 1_1_1_128_128_float_ND2ND)
{
    TLOADMIXFUNC<float, 2, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>();
}

TEST_F(TLOADMIXTest, 1_1_1_37_126_int8_t_ND2ND)
{
    TLOADMIXFUNC<int8_t, 2, 1, 1, 1, 37, 126, 1, 1, 1, 37, 126, 37, 128>();
}

TEST_F(TLOADMIXTest, 1_2_3_64_128_1_3_4_128_128_384_128_half_ND2ND)
{
    TLOADMIXFUNC<uint16_t, 2, 1, 2, 3, 64, 128, 1, 3, 4, 128, 128, 384, 128>();
}

TEST_F(TLOADMIXTest, 1_2_3_33_99_1_2_3_33_99_int8_t_ND2ND)
{
    TLOADMIXFUNC<int8_t, 2, 1, 2, 3, 33, 99, 1, 2, 3, 33, 99, 198, 128>();
}

TEST_F(TLOADMIXTest, 1_1_1_33_99_1_1_1_64_128_48_112_half_ND2NZ)
{
    TLOADMIXFUNC<uint16_t, 0, 1, 1, 1, 33, 99, 1, 1, 1, 64, 128, 48, 112>();
}
TEST_F(TLOADMIXTest, 1_1_1_59_119_1_1_1_64_128_64_128_int8_t_ND2NZ)
{
    TLOADMIXFUNC<int8_t, 0, 1, 1, 1, 59, 119, 1, 1, 1, 64, 128, 64, 128>();
}

// 4 NZ2NZ
TEST_F(TLOADMIXTest, 2_2_4_16_8_2_2_4_16_8_80_48_float_NZ2NZ)
{
    TLOADMIXFUNC<float, 4, 2, 2, 4, 16, 8, 2, 2, 4, 16, 8, 80, 48>();
}
TEST_F(TLOADMIXTest, 1_10_8_16_16_1_11_9_16_16_128_160_half_NZ2NZ)
{
    TLOADMIXFUNC<uint16_t, 4, 1, 10, 8, 16, 16, 1, 11, 9, 16, 16, 128, 160>();
}
TEST_F(TLOADMIXTest, 1_8_4_16_32_1_9_4_16_32_80_256_int8_t_NZ2NZ)
{
    TLOADMIXFUNC<int8_t, 4, 1, 8, 4, 16, 32, 1, 9, 4, 16, 32, 80, 256>();
}

TEST_F(TLOADMIXTest, 1_1_1_59_119_1_1_1_59_124_59_120_int64_t_ND2ND)
{
    TLOADMIXFUNC<int64_t, 2, 1, 1, 1, 59, 119, 1, 1, 1, 59, 124, 59, 120>();
}
TEST_F(TLOADMIXTest, 1_2_1_64_128_1_3_4_128_128_128_128_uint64_t_ND2ND)
{
    TLOADMIXFUNC<uint64_t, 2, 1, 2, 1, 64, 128, 1, 3, 4, 128, 128, 128, 128>();
}

// 6: NC1HWC02NC1HWC0
TEST_F(TLOADMIXTest, NC1HWC02NC1HWC0_int8_t_1_3_16_128_32_3_4_1024_1024_32)
{
    TLOADMIXFUNC<int8_t, 6, 1, 3, 16, 128, 32, 3, 4, 1024, 1024, 32, 1, 1>();
}
TEST_F(TLOADMIXTest, NC1HWC02NC1HWC0_int8_t_3_2_128_8_32_3_2_128_128_32)
{
    TLOADMIXFUNC<int8_t, 6, 3, 2, 128, 8, 32, 3, 2, 128, 128, 32, 1, 1>();
}
TEST_F(TLOADMIXTest, NC1HWC02NC1HWC0_int8_t_3_2_8_128_32_3_8_8_128_32)
{
    TLOADMIXFUNC<int8_t, 6, 3, 2, 8, 128, 32, 3, 8, 8, 128, 32, 1, 1>();
}
TEST_F(TLOADMIXTest, NC1HWC02NC1HWC0_float16_1_6_10_100_16_1_6_100_100_16)
{
    TLOADMIXFUNC<uint16_t, 6, 1, 6, 10, 100, 16, 1, 6, 100, 100, 16, 1, 1>();
}
TEST_F(TLOADMIXTest, NC1HWC02NC1HWC0_float16_10_16_16_2_16_256_16_100_16_16)
{
    TLOADMIXFUNC<uint16_t, 6, 10, 16, 16, 2, 16, 256, 16, 100, 16, 16, 1, 1>();
}
TEST_F(TLOADMIXTest, NC1HWC02NC1HWC0_float16_1_1_1_8192_16_8_16_16_8192_16)
{
    TLOADMIXFUNC<uint16_t, 6, 1, 1, 1, 8192, 16, 8, 16, 16, 8192, 16, 1, 1>();
}
TEST_F(TLOADMIXTest, NC1HWC02NC1HWC0_float_1_1_56_112_8_2_3_224_224_8)
{
    TLOADMIXFUNC<float, 6, 1, 1, 56, 112, 8, 2, 3, 224, 224, 8, 1, 1>();
}

TEST_F(TLOADMIXTest, FZ2FZ_float16_1_7_7_20_16_3_7_7_100_16)
{
    TLOADMIXFUNC<uint16_t, 7, 1, 7, 7, 20, 16, 3, 7, 7, 100, 16, 1, 1>();
}
TEST_F(TLOADMIXTest, FZ2FZ_float16_64_7_7_2_16_256_7_7_16_16)
{
    TLOADMIXFUNC<uint16_t, 7, 64, 7, 7, 2, 16, 256, 7, 7, 16, 16, 1, 1>();
}
TEST_F(TLOADMIXTest, FZ2FZ_float16_96_3_3_8_16_256_3_3_8_16)
{
    TLOADMIXFUNC<uint16_t, 7, 96, 3, 3, 8, 16, 256, 3, 3, 8, 16, 1, 1>();
}
TEST_F(TLOADMIXTest, FZ2FZ_int8_t_1_3_3_64_32_3_3_3_128_32)
{
    TLOADMIXFUNC<int8_t, 7, 2, 3, 3, 64, 32, 3, 3, 3, 128, 32, 1, 1>();
}
TEST_F(TLOADMIXTest, FZ2FZ_int8_t_8_5_5_32_32_8_5_5_128_32)
{
    TLOADMIXFUNC<int8_t, 7, 8, 5, 5, 32, 32, 8, 5, 5, 128, 32, 1, 1>();
}
TEST_F(TLOADMIXTest, FZ2FZ_float_70_7_7_2_8_256_7_7_256_8)
{
    TLOADMIXFUNC<float, 7, 70, 7, 7, 2, 8, 256, 7, 7, 256, 8, 1, 1>();
}

// 8: FZ4D2FZ4D
TEST_F(TLOADMIXTest, FZ4D2FZ4D_float16_1_49_7_16_16_1_980_32_16_16)
{
    TLOADMIXFUNC<uint16_t, 8, 1, 49, 7, 16, 16, 1, 980, 32, 16, 16, 1, 1>();
}
TEST_F(TLOADMIXTest, FZ4D2FZ4D_float16_1_81_3_16_16_1_90_3_16_16)
{
    TLOADMIXFUNC<uint16_t, 8, 1, 81, 3, 16, 16, 1, 90, 3, 16, 16, 1, 1>();
}
TEST_F(TLOADMIXTest, FZ4D2FZ4D_int8_t_1_63_3_16_32_1_63_9_16_32)
{
    TLOADMIXFUNC<int8_t, 8, 1, 63, 3, 16, 32, 1, 63, 9, 16, 32, 1, 1>();
}
TEST_F(TLOADMIXTest, FZ4D2FZ4D_int8_t_1_125_3_16_32_1_250_5_16_32)
{
    TLOADMIXFUNC<int8_t, 8, 1, 125, 3, 16, 32, 1, 250, 5, 16, 32, 1, 1>();
}
TEST_F(TLOADMIXTest, FZ4D2FZ4D_float_1_126_3_16_8_1_4704_7_16_8)
{
    TLOADMIXFUNC<float, 8, 1, 126, 3, 16, 8, 1, 4704, 7, 16, 8, 1, 1>();
}
