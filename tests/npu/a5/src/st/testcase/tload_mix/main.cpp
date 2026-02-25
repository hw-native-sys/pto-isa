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
constexpr int ND2NZ = 0; // format 0:ND2NZ 1:DN2NZ 2:ND2ND 3:DN2DN 4 NZ2NZ 5 DN2ZN
constexpr int DN2NZ = 1;
constexpr int ND2ND = 2;
constexpr int DN2DN = 3;
constexpr int NZ2NZ = 4;
constexpr int DN2ZN = 5;
constexpr int NC1HWC02NC1HWC0 = 6;
constexpr int FZ2FZ = 7;
constexpr int FZ4D2FZ4D = 8;
constexpr int NHWC2NC1HWC0 = 9;
constexpr int NCHW2NC1HWC0 = 10;
constexpr int NCHW2FZ4D = 11;
constexpr int NCDHW2NDC1HWC0 = 12;
constexpr int NCDHW2FZ3D = 13;
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
                  format == TloadMixTestFormat::FZ2FZ || format == TloadMixTestFormat::NHWC2NC1HWC0 ||
                  format == TloadMixTestFormat::NCHW2NC1HWC0) {
        cFileSize = N1 * N2 * N3 * N4 * N5 * sizeof(T);
    } else if constexpr (format == TloadMixTestFormat::NCHW2FZ4D) {
        cFileSize = N1 * N2 * N3 * N4 * sizeof(T);
        aFileSize = WN4 * WN5 * BASEM * BASEK * sizeof(T);
    } else if constexpr (format == TloadMixTestFormat::NCDHW2NDC1HWC0) {
        cFileSize = N1 * N2 * N3 * N4 * N5 * c0SizeByte;
        aFileSize = WN1 * WN2 * WN3 * WN4 * WN5 * sizeof(T);
    } else if constexpr (format == TloadMixTestFormat::NCDHW2FZ3D) {
        cFileSize = BASEM * BASEK * n0 * c0SizeByte;
        aFileSize = WN1 * WN2 * WN3 * WN4 * WN5 * sizeof(T);
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

// format 0:ND2NZ 1:DN2NZ 2:ND2ND 3:DN2DN 4 NZ2NZ 5 DN2ZN
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

TEST_F(TLOADMIXTest, 1_1_1_64_128_half_DN2NZ)
{
    TLOADMIXFUNC<uint16_t, 1, 1, 1, 1, 64, 128, 1, 1, 1, 64, 128, 64, 128>();
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

TEST_F(TLOADMIXTest, 1_1_1_51_123_1_1_1_64_128_64_128_float_DN2NZ)
{
    TLOADMIXFUNC<float, 1, 1, 1, 1, 51, 123, 1, 1, 1, 64, 128, 64, 128>();
}

TEST_F(TLOADMIXTest, 1_1_1_63_127_1_1_1_63_127_64_128_half_DN2NZ)
{
    TLOADMIXFUNC<uint16_t, 1, 1, 1, 1, 63, 127, 1, 1, 1, 63, 127, 64, 128>();
}

// 3:DN2DN
TEST_F(TLOADMIXTest, 1_1_1_128_128_1_1_1_128_128_128_128_float_DN2DN)
{
    TLOADMIXFUNC<float, 3, 1, 1, 1, 128, 128, 1, 1, 1, 128, 128, 128, 128>();
}
TEST_F(TLOADMIXTest, 1_1_1_37_126_1_1_1_37_126_64_126_int8_t_DN2DN)
{
    TLOADMIXFUNC<int8_t, 3, 1, 1, 1, 37, 126, 1, 1, 1, 37, 126, 64, 126>();
}
TEST_F(TLOADMIXTest, 1_2_3_64_128_1_3_4_96_128_64_768_half_DN2DN)
{
    TLOADMIXFUNC<uint16_t, 3, 1, 2, 3, 64, 128, 1, 3, 4, 96, 128, 64, 768>();
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

// 5 DN2ZN
TEST_F(TLOADMIXTest, 1_1_1_33_99_1_1_1_64_128_48_112_half_DN2ZN)
{
    TLOADMIXFUNC<uint16_t, 5, 1, 1, 1, 33, 99, 1, 1, 1, 64, 128, 48, 112>();
}
TEST_F(TLOADMIXTest, 1_1_1_59_119_1_1_1_64_128_64_128_int8_t_DN2ZN)
{
    TLOADMIXFUNC<int8_t, 5, 1, 1, 1, 59, 119, 1, 1, 1, 64, 128, 64, 128>();
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
TEST_F(TLOADMIXTest, NC1HWC02NC1HWC0_bfloat16_1_6_10_100_16_1_6_100_100_16)
{
    TLOADMIXFUNC<uint16_t, 6, 1, 6, 10, 100, 16, 1, 6, 100, 100, 16, 1, 1>();
}
TEST_F(TLOADMIXTest, NC1HWC02NC1HWC0_bfloat16_10_16_16_2_16_256_16_100_16_16)
{
    TLOADMIXFUNC<uint16_t, 6, 10, 16, 16, 2, 16, 256, 16, 100, 16, 16, 1, 1>();
}
TEST_F(TLOADMIXTest, NC1HWC02NC1HWC0_bfloat16_1_1_1_8192_16_8_16_16_8192_16)
{
    TLOADMIXFUNC<uint16_t, 6, 1, 1, 1, 8192, 16, 8, 16, 16, 8192, 16, 1, 1>();
}
TEST_F(TLOADMIXTest, NC1HWC02NC1HWC0_float_1_1_56_112_8_2_3_224_224_8)
{
    TLOADMIXFUNC<float, 6, 1, 1, 56, 112, 8, 2, 3, 224, 224, 8, 1, 1>();
}

TEST_F(TLOADMIXTest, FZ2FZ_bfloat16_1_7_7_20_16_3_7_7_100_16)
{
    TLOADMIXFUNC<uint16_t, 7, 1, 7, 7, 20, 16, 3, 7, 7, 100, 16, 1, 1>();
}
TEST_F(TLOADMIXTest, FZ2FZ_bfloat16_64_7_7_2_16_256_7_7_16_16)
{
    TLOADMIXFUNC<uint16_t, 7, 64, 7, 7, 2, 16, 256, 7, 7, 16, 16, 1, 1>();
}
TEST_F(TLOADMIXTest, FZ2FZ_bfloat16_96_3_3_8_16_256_3_3_8_16)
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
TEST_F(TLOADMIXTest, FZ4D2FZ4D_bfloat16_1_49_7_16_16_1_980_32_16_16)
{
    TLOADMIXFUNC<uint16_t, 8, 1, 49, 7, 16, 16, 1, 980, 32, 16, 16, 1, 1>();
}
TEST_F(TLOADMIXTest, FZ4D2FZ4D_bfloat16_1_81_3_16_16_1_90_3_16_16)
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

// 9 : NHWC2NC1HWC0
TEST_F(TLOADMIXTest, NHWC2NC1HWC0_int8_t_1_3_11_109_32_1_3_1023_1000_111)
{
    TLOADMIXFUNC<int8_t, 9, 1, 3, 11, 109, 32, 1, 3, 1023, 1000, 111, 1, 1>();
}
TEST_F(TLOADMIXTest, NHWC2NC1HWC0_int8_t_3_2_121_9_32_1_3_128_127_65)
{
    TLOADMIXFUNC<int8_t, 9, 3, 2, 121, 9, 32, 1, 3, 128, 127, 65, 1, 1>();
}
TEST_F(TLOADMIXTest, NHWC2NC1HWC0_bfloat16_1_6_10_100_16_1_1_100_100_96)
{
    TLOADMIXFUNC<uint16_t, 9, 1, 6, 10, 100, 16, 1, 1, 100, 100, 96, 1, 1>();
}
TEST_F(TLOADMIXTest, NHWC2NC1HWC0_bfloat16_10_16_16_2_16_1_256_100_16_255)
{
    TLOADMIXFUNC<uint16_t, 9, 10, 16, 16, 2, 16, 1, 256, 100, 16, 255, 1, 1>();
}
TEST_F(TLOADMIXTest, NHWC2NC1HWC0_float_1_1_56_112_8_1_2_224_224_25)
{
    TLOADMIXFUNC<float, 9, 1, 1, 56, 112, 8, 1, 2, 224, 224, 25, 1, 1>();
}
TEST_F(TLOADMIXTest, NHWC2NC1HWC0_float_2_1_56_43_8_1_3_333_188_19)
{
    TLOADMIXFUNC<float, 9, 2, 1, 56, 43, 8, 1, 3, 333, 188, 19, 1, 1>();
}

// 10 : NCHW2NC1HWC0
TEST_F(TLOADMIXTest, NCHW2NC1HWC0_int8_t_1_3_11_109_32_1_3_111_1023_109)
{
    TLOADMIXFUNC<int8_t, 10, 1, 3, 11, 109, 32, 1, 3, 111, 1023, 109, 1, 1>();
}
TEST_F(TLOADMIXTest, NCHW2NC1HWC0_int8_t_3_2_121_9_32_1_3_65_128_127)
{
    TLOADMIXFUNC<int8_t, 10, 3, 2, 121, 9, 32, 1, 3, 65, 128, 127, 1, 1>();
}
TEST_F(TLOADMIXTest, NCHW2NC1HWC0_bfloat16_1_6_10_100_16_1_1_96_100_100)
{
    TLOADMIXFUNC<uint16_t, 10, 1, 6, 10, 100, 16, 1, 1, 96, 100, 100, 1, 1>();
}
TEST_F(TLOADMIXTest, NCHW2NC1HWC0_bfloat16_10_16_16_2_16_1_256_255_100_16)
{
    TLOADMIXFUNC<uint16_t, 10, 10, 16, 16, 2, 16, 1, 256, 255, 100, 16, 1, 1>();
}
TEST_F(TLOADMIXTest, NCHW2NC1HWC0_float_1_1_56_112_8_1_2_25_224_112)
{
    TLOADMIXFUNC<float, 10, 1, 1, 56, 112, 8, 1, 2, 25, 224, 112, 1, 1>();
}
TEST_F(TLOADMIXTest, NCHW2NC1HWC0_float_2_1_56_43_8_1_3_19_333_188)
{
    TLOADMIXFUNC<float, 10, 2, 1, 56, 43, 8, 1, 3, 19, 333, 188, 1, 1>();
}

// 11. NCHW -> Fractal_Z4D [C1HW,N/16,16,C0,srcN,srcC,srcH,srcW,N,C,H,W]
TEST_F(TLOADMIXTest, NCHW2FZ4D_int8_t_75_3_16_32_48_95_5_5_50_111_5_5)
{
    TLOADMIXFUNC<int8_t, 11, 75, 3, 16, 32, 48, 95, 5, 5, 50, 111, 5, 5>();
}
TEST_F(TLOADMIXTest, NCHW2FZ4D_int8_t_98_4_16_32_64_58_7_7_121_127_7_7)
{
    TLOADMIXFUNC<int8_t, 11, 98, 4, 16, 32, 64, 58, 7, 7, 121, 127, 7, 7>();
}
TEST_F(TLOADMIXTest, NCHW2FZ4D_bfloat16_63_6_16_16_96_111_3_3_220_96_3_3)
{
    TLOADMIXFUNC<uint16_t, 11, 63, 6, 16, 16, 96, 111, 3, 3, 220, 112, 3, 3>();
}
TEST_F(TLOADMIXTest, NCHW2FZ4D_bfloat16_75_4_16_16_64_48_5_5_100_50_5_5)
{
    TLOADMIXFUNC<uint16_t, 11, 75, 4, 16, 16, 64, 48, 5, 5, 100, 50, 5, 5>();
}
TEST_F(TLOADMIXTest, NCHW2FZ4D_float_50_3_16_8_48_14_5_5_224_224_5_5)
{
    TLOADMIXFUNC<float, 11, 50, 3, 16, 8, 48, 14, 5, 5, 224, 224, 5, 5>();
}
TEST_F(TLOADMIXTest, NCHW2FZ4D_float_27_2_16_8_32_24_3_3_333_188_3_3)
{
    TLOADMIXFUNC<float, 11, 27, 2, 16, 8, 32, 24, 3, 3, 333, 188, 3, 3>();
}

// 12 : NCDHW2NDC1HWC0   N D C1 H W  N C D H W
TEST_F(TLOADMIXTest, NCDHW2NDC1HWC0_int8_t_1_2_3_11_109_3_111_2_1023_109)
{
    TLOADMIXFUNC<int8_t, 12, 1, 2, 3, 11, 109, 3, 111, 2, 1023, 109, 1, 1>();
}
TEST_F(TLOADMIXTest, NCDHW2NDC1HWC0_int8_t_3_3_2_15_9_3_65_4_30_50)
{
    TLOADMIXFUNC<int8_t, 12, 3, 3, 2, 15, 9, 3, 65, 4, 30, 50, 1, 1>();
}
TEST_F(TLOADMIXTest, NCDHW2NDC1HWC0_bfloat16_1_4_6_10_10_1_96_6_100_10)
{
    TLOADMIXFUNC<uint16_t, 12, 1, 4, 6, 10, 10, 1, 96, 6, 100, 10, 1, 1>();
}
TEST_F(TLOADMIXTest, NCDHW2NDC1HWC0_bfloat16_10_2_8_16_2_256_128_2_100_7)
{
    TLOADMIXFUNC<uint16_t, 12, 10, 2, 8, 16, 2, 256, 128, 2, 100, 7, 1, 1>();
}
TEST_F(TLOADMIXTest, NCDHW2NDC1HWC0_float_1_5_1_25_31_2_25_7_112_31)
{
    TLOADMIXFUNC<float, 12, 1, 5, 1, 25, 31, 2, 25, 7, 112, 31, 1, 1>();
}
TEST_F(TLOADMIXTest, NCDHW2NDC1HWC0_float_2_2_1_43_43_3_19_2_155_180)
{
    TLOADMIXFUNC<float, 12, 2, 2, 1, 43, 43, 3, 19, 2, 155, 180, 1, 1>();
}

// 13. NCDHW -> Fractal_Z3D  srcN srcC srcD, srcH srcW N C D H W C1DHW N/16
TEST_F(TLOADMIXTest, NCDHW2FZ3D_int8_t_48_95_2_5_5_50_111_4_5_5_150_3)
{
    TLOADMIXFUNC<int8_t, 13, 48, 95, 2, 5, 5, 50, 111, 4, 5, 5, 150, 3>();
}
TEST_F(TLOADMIXTest, NCDHW2FZ3D_int8_t_32_58_2_7_7_63_127_2_7_7_196_2)
{
    TLOADMIXFUNC<int8_t, 13, 32, 58, 2, 7, 7, 63, 127, 2, 7, 7, 196, 2>();
}
TEST_F(TLOADMIXTest, NCDHW2FZ3D_bfloat16_48_111_2_3_3_110_112_2_3_3_126_3)
{
    TLOADMIXFUNC<uint16_t, 13, 48, 111, 2, 3, 3, 110, 112, 2, 3, 3, 126, 3>();
}
TEST_F(TLOADMIXTest, NCDHW2FZ3D_bfloat16_32_48_3_3_3_70_50_4_3_3_81_2)
{
    TLOADMIXFUNC<uint16_t, 13, 32, 48, 3, 3, 3, 70, 50, 4, 3, 3, 81, 2>();
}
TEST_F(TLOADMIXTest, NCDHW2FZ3D_float_48_14_5_2_2_224_224_7_2_2_40_3)
{
    TLOADMIXFUNC<float, 13, 48, 14, 5, 2, 2, 224, 224, 7, 2, 2, 40, 3>();
}
TEST_F(TLOADMIXTest, NCDHW2FZ3D_float_32_24_2_3_3_333_188_2_3_3_54_2)
{
    TLOADMIXFUNC<float, 13, 32, 24, 2, 3, 3, 333, 188, 2, 3, 3, 54, 2>();
}

template <typename T, int format, int dtype, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4,
          int WN5, int BASEM, int BASEK>
void launchTLOADMIX_B4(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <typename T, int format, int dtype, int N1, int N2, int N3, int N4, int N5, int WN1, int WN2, int WN3, int WN4,
          int WN5, int BASEM, int BASEK>
void TLOADMIXFUNCB4()
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
    launchTLOADMIX_B4<T, format, dtype, N1, N2, N3, N4, N5, WN1, WN2, WN3, WN4, WN5, BASEM, BASEK>(
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

// format 0:ND2NZ 1:DN2NZ 2:ND2ND 3:DN2DN 4 NZ2NZ
TEST_F(TLOADMIXTest, 1_2_1_64_128_1_3_4_128_128_128_128_fp4x2_e1m2_t_ND2ND)
{
    // T固定uint8，dtype=0 表示e1m2 dtype=1 表示e2m1 内部处理的时候最内轴按cols * 2处理
    TLOADMIXFUNCB4<uint8_t, 2, 0, 1, 2, 1, 64, 128, 1, 3, 4, 128, 128, 128, 128>();
}
TEST_F(TLOADMIXTest, 1_1_1_59_119_1_1_1_64_128_64_128_fp4x2_e2m1_t_ND2NZ)
{
    // T固定uint8，dtype=0 表示e1m2 dtype=1 表示e2m1 内部处理的时候最内轴按cols * 2处理
    TLOADMIXFUNCB4<uint8_t, 0, 1, 1, 1, 1, 59, 119, 1, 1, 1, 64, 128, 64, 128>();
}

TEST_F(TLOADMIXTest, 1_8_4_16_32_1_9_4_16_32_80_256_fp4x2_e1m2_t_NZ2NZ)
{
    // T固定uint8，dtype=0 表示e1m2 dtype=1 表示e2m1 内部处理的时候最内轴按cols * 2处理
    TLOADMIXFUNCB4<uint8_t, 4, 0, 1, 8, 4, 16, 32, 1, 9, 4, 16, 32, 80, 256>();
}

TEST_F(TLOADMIXTest, 1_1_1_37_126_1_1_1_37_126_64_126_fp4x2_e1m2_t_DN2DN)
{
    // T固定uint8，dtype=0 表示e1m2 dtype=1 表示e2m1 内部处理的时候最内轴按cols * 2处理
    TLOADMIXFUNCB4<uint8_t, 3, 0, 1, 1, 1, 37, 126, 1, 1, 1, 37, 126, 64, 126>();
}

TEST_F(TLOADMIXTest, 1_1_1_59_119_1_1_1_64_128_64_128_fp4x2_e1m2_t_DN2ZN)
{
    // T固定uint8，dtype=0 表示e1m2 dtype=1 表示e2m1 内部处理的时候最内轴按cols * 2处理
    TLOADMIXFUNCB4<uint8_t, 5, 0, 1, 1, 1, 59, 119, 1, 1, 1, 64, 128, 64, 128>();
}
