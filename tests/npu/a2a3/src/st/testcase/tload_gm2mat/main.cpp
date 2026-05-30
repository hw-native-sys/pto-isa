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

template <int format, typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
          int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void LaunchTLoad(T *out, T *src, void *stream);

class TLoadGM2L1Test : public testing::Test {
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

template <int format, typename DataType, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4,
          int gWholeShape0, int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4>
void TestTload()
{
    // format = 0: ND2ND
    // foramt = 1: DN2DN
    // format = 2: NZ2NZ
    // format = 3: ND2NZ
    // format = 4: DN2ZN
    // format = 5: NC1HWC02NC1HWC0
    // format = 6: FRACTALZ2FRACTALZ
    // format = 7: FRACTALZ4D2FRACTALZ4D
    // format = 8: NDC1HWC02NDC1HWC0
    size_t srcDataSize = static_cast<size_t>(gWholeShape0) * static_cast<size_t>(gWholeShape1) *
                         static_cast<size_t>(gWholeShape2) * static_cast<size_t>(gWholeShape3) *
                         static_cast<size_t>(gWholeShape4) * sizeof(DataType);

    size_t dstDataSize = static_cast<size_t>(gShape0) * static_cast<size_t>(gShape1) * static_cast<size_t>(gShape2) *
                         static_cast<size_t>(gShape3) * static_cast<size_t>(gShape4) * sizeof(DataType);
    constexpr int c0Size = 32 / sizeof(DataType);
    if (format == 3) {
        int gShape4Align = (gShape4 + c0Size - 1) / c0Size * c0Size;
        dstDataSize = static_cast<size_t>(gShape0) * static_cast<size_t>(gShape1) * static_cast<size_t>(gShape2) *
                      static_cast<size_t>(gShape3) * static_cast<size_t>(gShape4Align) * sizeof(DataType);
    } else if (format == 4) {
        int gShape3Align = (gShape3 + c0Size - 1) / c0Size * c0Size;
        dstDataSize = static_cast<size_t>(gShape0) * static_cast<size_t>(gShape1) * static_cast<size_t>(gShape2) *
                      static_cast<size_t>(gShape3Align) * static_cast<size_t>(gShape4) * sizeof(DataType);
    } else if (format == 8) {
        srcDataSize = static_cast<size_t>(gWholeShape0) * static_cast<size_t>(gWholeShape1) *
                      static_cast<size_t>(gWholeShape2) * static_cast<size_t>(gWholeShape3) *
                      static_cast<size_t>(gWholeShape4) * static_cast<size_t>(c0Size) * sizeof(DataType);
        dstDataSize = static_cast<size_t>(gShape0) * static_cast<size_t>(gShape1) * static_cast<size_t>(gShape2) *
                      static_cast<size_t>(gShape3) * static_cast<size_t>(gShape4) * static_cast<size_t>(c0Size) *
                      sizeof(DataType);
    }

    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    DataType *dstHost, *srcHost;
    DataType *dstDevice, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstDataSize);
    aclrtMallocHost((void **)(&srcHost), srcDataSize);

    aclrtMalloc((void **)&dstDevice, dstDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ReadFile(GetGoldenDir() + "/input.bin", srcDataSize, srcHost, srcDataSize);
    aclrtMemcpy(srcDevice, srcDataSize, srcHost, srcDataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTLoad<format, DataType, gShape0, gShape1, gShape2, gShape3, gShape4, gWholeShape0, gWholeShape1, gWholeShape2,
                gWholeShape3, gWholeShape4>(dstDevice, srcDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstDataSize, dstDevice, dstDataSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstDataSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<DataType> golden(dstDataSize);
    std::vector<DataType> devFinal(dstDataSize);
    ReadFile(GetGoldenDir() + "/golden.bin", dstDataSize, golden.data(), dstDataSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstDataSize, devFinal.data(), dstDataSize);

    bool ret = ResultCmp<DataType>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TLoadGM2L1Test, ND_float_1_1_1_3_128_3_3_3_32_128)
{
    TestTload<0, float, 1, 1, 1, 3, 128, 3, 3, 3, 32, 128>();
}

TEST_F(TLoadGM2L1Test, ND_int16_t_2_2_1_2_32_3_3_3_111_64)
{
    TestTload<0, int16_t, 2, 2, 1, 2, 32, 3, 3, 3, 111, 64>();
}

TEST_F(TLoadGM2L1Test, ND_int8_t_1_2_1_11_32_1_3_2_93_32)
{
    TestTload<0, int8_t, 1, 2, 1, 11, 32, 1, 3, 2, 93, 32>();
}

TEST_F(TLoadGM2L1Test, ND_int8_t_1_1_1_1_201_1_1_1_1_201)
{
    TestTload<0, int8_t, 1, 1, 1, 1, 201, 1, 1, 1, 1, 201>();
}

TEST_F(TLoadGM2L1Test, DN_float_1_1_1_128_3_3_3_3_128_32)
{
    TestTload<1, float, 1, 1, 1, 128, 3, 3, 3, 3, 128, 32>();
}

TEST_F(TLoadGM2L1Test, DN_int16_t_2_2_1_32_2_3_3_3_64_111)
{
    TestTload<1, int16_t, 2, 2, 1, 32, 2, 3, 3, 3, 64, 111>();
}

TEST_F(TLoadGM2L1Test, DN_int8_t_1_2_1_32_11_1_3_2_32_93)
{
    TestTload<1, int8_t, 1, 2, 1, 32, 11, 1, 3, 2, 32, 93>();
}

TEST_F(TLoadGM2L1Test, DN_float_1_1_1_156_1_1_1_1_156_1)
{
    TestTload<1, float, 1, 1, 1, 156, 1, 1, 1, 1, 156, 1>();
}

TEST_F(TLoadGM2L1Test, NZ_float_1_5_21_16_8_1_5_21_16_8)
{
    TestTload<2, float, 1, 5, 21, 16, 8, 1, 5, 21, 16, 8>();
}

TEST_F(TLoadGM2L1Test, NZ_int16_t_2_16_11_16_16_3_23_13_16_16)
{
    TestTload<2, int16_t, 2, 15, 11, 16, 16, 3, 23, 13, 16, 16>();
}

TEST_F(TLoadGM2L1Test, NZ_int8_t_1_16_32_16_32_1_32_32_16_32)
{
    TestTload<2, int8_t, 1, 16, 32, 16, 32, 1, 32, 32, 16, 32>();
}

TEST_F(TLoadGM2L1Test, ND2NZ_float_t_1_1_1_49_35_1_1_1_49_35)
{
    TestTload<3, float, 1, 1, 1, 49, 35, 1, 1, 1, 49, 35>();
}

TEST_F(TLoadGM2L1Test, ND2NZ_int16_t_1_1_1_155_250_1_1_1_752_1000)
{
    TestTload<3, int16_t, 1, 1, 1, 155, 250, 1, 1, 1, 752, 1000>();
}

TEST_F(TLoadGM2L1Test, ND2NZ_int8_t_1_1_1_1023_511_1_1_1_1024_1024)
{
    TestTload<3, int8_t, 1, 1, 1, 1023, 511, 1, 1, 1, 1024, 1024>();
}

TEST_F(TLoadGM2L1Test, ND2NZ_bfloat16_t_1_1_1_1023_51_1_1_1_1024_1024)
{
    TestTload<3, uint16_t, 1, 1, 1, 1023, 51, 1, 1, 1, 1024, 1024>();
}

TEST_F(TLoadGM2L1Test, ND_bfloat16_t_1_1_1_128_128_1_1_1_256_256)
{
    TestTload<0, uint16_t, 1, 1, 1, 128, 128, 1, 1, 1, 256, 256>();
}

TEST_F(TLoadGM2L1Test, DN_bfloat16_t_1_2_2_128_311_4_3_3_256_400)
{
    TestTload<1, uint16_t, 1, 2, 2, 128, 311, 4, 3, 3, 256, 400>();
}

TEST_F(TLoadGM2L1Test, NZ_bfloat16_t_2_4_5_16_16_7_7_7_16_16)
{
    TestTload<2, uint16_t, 2, 4, 5, 16, 16, 7, 7, 7, 16, 16>();
}

TEST_F(TLoadGM2L1Test, ND2NZ_bfloat16_t_1_1_1_1_1_1_1_1_1_1)
{
    TestTload<3, uint16_t, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>();
}

TEST_F(TLoadGM2L1Test, ND2NZ_bfloat16_t_1_1_1_1_1_1_1_1_16_16)
{
    TestTload<3, uint16_t, 1, 1, 1, 1, 1, 1, 1, 1, 16, 16>();
}

TEST_F(TLoadGM2L1Test, ND2NZ_bfloat16_t_1_1_1_256_1024_1_1_1_256_1024)
{
    TestTload<3, uint16_t, 1, 1, 1, 256, 1024, 1, 1, 1, 256, 1024>();
}

TEST_F(TLoadGM2L1Test, ND_int64_1_1_1_3_128_3_3_3_32_128)
{
    TestTload<0, int64_t, 1, 1, 1, 3, 128, 3, 3, 3, 32, 128>();
}

TEST_F(TLoadGM2L1Test, ND_uint64_2_2_1_2_32_3_3_3_111_64)
{
    TestTload<0, uint64_t, 2, 2, 1, 2, 32, 3, 3, 3, 111, 64>();
}

TEST_F(TLoadGM2L1Test, ND_int64_1_2_1_11_32_1_3_2_93_32)
{
    TestTload<0, int64_t, 1, 2, 1, 11, 32, 1, 3, 2, 93, 32>();
}

TEST_F(TLoadGM2L1Test, DN_uint64_1_1_1_128_3_3_3_3_128_32)
{
    TestTload<1, uint64_t, 1, 1, 1, 128, 3, 3, 3, 3, 128, 32>();
}

TEST_F(TLoadGM2L1Test, DN_int64_2_2_1_32_2_3_3_3_64_111)
{
    TestTload<1, int64_t, 2, 2, 1, 32, 2, 3, 3, 3, 64, 111>();
}

TEST_F(TLoadGM2L1Test, DN_uint64_1_2_1_32_11_1_3_2_32_93)
{
    TestTload<1, uint64_t, 1, 2, 1, 32, 11, 1, 3, 2, 32, 93>();
}

TEST_F(TLoadGM2L1Test, DN2ZN_bfloat16_t_1_1_1_256_1024_1_1_1_256_1024)
{
    TestTload<4, uint16_t, 1, 1, 1, 256, 1024, 1, 1, 1, 256, 1024>();
}

TEST_F(TLoadGM2L1Test, DN2ZN_float_t_1_1_1_49_35_1_1_1_49_35)
{
    TestTload<4, float, 1, 1, 1, 49, 35, 1, 1, 1, 49, 35>();
}

TEST_F(TLoadGM2L1Test, DN2ZN_int16_t_1_1_1_155_250_1_1_1_752_1000)
{
    TestTload<4, int16_t, 1, 1, 1, 155, 250, 1, 1, 1, 752, 1000>();
}

TEST_F(TLoadGM2L1Test, DN2ZN_int8_t_1_1_1_1023_511_1_1_1_1024_1024)
{
    TestTload<4, int8_t, 1, 1, 1, 1023, 511, 1, 1, 1, 1024, 1024>();
}

TEST_F(TLoadGM2L1Test, NC1HWC02NC1HWC0_int8_t_2_3_16_128_32_3_4_1024_1024_32)
{
    TestTload<5, int8_t, 2, 3, 16, 128, 32, 3, 4, 1024, 1024, 32>();
}
TEST_F(TLoadGM2L1Test, NC1HWC02NC1HWC0_int8_t_3_4_128_8_32_3_4_128_128_32)
{
    TestTload<5, int8_t, 3, 4, 128, 8, 32, 3, 4, 128, 128, 32>();
}
TEST_F(TLoadGM2L1Test, NC1HWC02NC1HWC0_int8_t_3_4_8_128_32_3_8_8_128_32)
{
    TestTload<5, int8_t, 3, 4, 8, 128, 32, 3, 8, 8, 128, 32>();
}

TEST_F(TLoadGM2L1Test, NC1HWC02NC1HWC0_bfloat16_1_16_10_100_16_1_16_100_100_16)
{
    TestTload<5, uint16_t, 1, 16, 10, 100, 16, 1, 16, 100, 100, 16>();
}
TEST_F(TLoadGM2L1Test, NC1HWC02NC1HWC0_bfloat16_10_16_16_2_16_256_16_100_16_16)
{
    TestTload<5, uint16_t, 10, 16, 16, 2, 16, 256, 16, 100, 16, 16>();
}
TEST_F(TLoadGM2L1Test, NC1HWC02NC1HWC0_bfloat16_1_1_1_8192_16_8_16_16_8192_16)
{
    TestTload<5, uint16_t, 1, 1, 1, 8192, 16, 8, 16, 16, 8192, 16>();
}
TEST_F(TLoadGM2L1Test, NC1HWC02NC1HWC0_float_1_1_112_112_8_2_3_224_224_8)
{
    TestTload<5, float, 1, 1, 112, 112, 8, 2, 3, 224, 224, 8>();
}

TEST_F(TLoadGM2L1Test, FZ2FZ_bfloat16_1_7_7_20_16_3_7_7_100_16)
{
    TestTload<6, uint16_t, 1, 7, 7, 20, 16, 3, 7, 7, 100, 16>();
}
TEST_F(TLoadGM2L1Test, FZ2FZ_bfloat16_128_7_7_2_16_256_7_7_16_16)
{
    TestTload<6, uint16_t, 128, 7, 7, 2, 16, 256, 7, 7, 16, 16>();
}
TEST_F(TLoadGM2L1Test, FZ2FZ_bfloat16_192_3_3_8_16_256_3_3_8_16)
{
    TestTload<6, uint16_t, 192, 3, 3, 8, 16, 256, 3, 3, 8, 16>();
}
TEST_F(TLoadGM2L1Test, FZ2FZ_int8_t_2_3_3_64_32_3_3_3_128_32)
{
    TestTload<6, int8_t, 2, 3, 3, 64, 32, 3, 3, 3, 128, 32>();
}
TEST_F(TLoadGM2L1Test, FZ2FZ_int8_t_3_5_5_128_32_8_5_5_128_32)
{
    TestTload<6, int8_t, 3, 5, 5, 128, 32, 8, 5, 5, 128, 32>();
}
TEST_F(TLoadGM2L1Test, FZ2FZ_float_96_7_7_2_8_256_7_7_256_8)
{
    TestTload<6, float, 96, 7, 7, 2, 8, 256, 7, 7, 256, 8>();
}

TEST_F(TLoadGM2L1Test, FZ4D2FZ4D_bfloat16_1_49_7_16_16_1_980_32_16_16)
{
    TestTload<7, uint16_t, 1, 49, 7, 16, 16, 1, 980, 32, 16, 16>();
}
TEST_F(TLoadGM2L1Test, FZ4D2FZ4D_bfloat16_1_81_3_16_16_1_90_3_16_16)
{
    TestTload<7, uint16_t, 1, 81, 3, 16, 16, 1, 90, 3, 16, 16>();
}
TEST_F(TLoadGM2L1Test, FZ4D2FZ4D_int8_t_1_63_3_16_32_1_63_9_16_32)
{
    TestTload<7, int8_t, 1, 63, 3, 16, 32, 1, 63, 9, 16, 32>();
}
TEST_F(TLoadGM2L1Test, FZ4D2FZ4D_int8_t_1_125_3_16_32_1_250_5_16_32)
{
    TestTload<7, int8_t, 1, 125, 3, 16, 32, 1, 250, 5, 16, 32>();
}
TEST_F(TLoadGM2L1Test, FZ4D2FZ4D_float_1_256_3_16_8_1_4704_7_16_8)
{
    TestTload<7, float, 1, 256, 3, 16, 8, 1, 4704, 7, 16, 8>();
}

TEST_F(TLoadGM2L1Test, NDC1HWC02NDC1HWC0_int8_t_1_2_3_16_128_2_3_4_1024_1024)
{
    TestTload<8, int8_t, 1, 2, 3, 16, 128, 2, 3, 4, 1024, 1024>();
}
TEST_F(TLoadGM2L1Test, NDC1HWC02NDC1HWC0_int8_t_1_3_4_128_8_2_3_4_128_128)
{
    TestTload<8, int8_t, 1, 3, 4, 128, 8, 2, 3, 4, 128, 128>();
}
TEST_F(TLoadGM2L1Test, NDC1HWC02NDC1HWC0_int8_t_1_3_4_8_128_2_3_8_8_128)
{
    TestTload<8, int8_t, 1, 3, 4, 8, 128, 2, 3, 8, 8, 128>();
}

TEST_F(TLoadGM2L1Test, NDC1HWC02NDC1HWC0_bfloat16_1_1_16_10_100_2_1_16_100_100)
{
    TestTload<8, uint16_t, 1, 1, 16, 10, 100, 2, 1, 16, 100, 100>();
}
TEST_F(TLoadGM2L1Test, NDC1HWC02NDC1HWC0_bfloat16_1_1_10_16_2_2_2_256_16_100)
{
    TestTload<8, uint16_t, 1, 1, 10, 16, 2, 2, 2, 256, 16, 100>();
}
TEST_F(TLoadGM2L1Test, NDC1HWC02NDC1HWC0_bfloat16_1_1_1_1_8192_2_8_16_16_8192)
{
    TestTload<8, uint16_t, 1, 1, 1, 1, 8192, 2, 8, 16, 16, 8192>();
}
TEST_F(TLoadGM2L1Test, NDC1HWC02NDC1HWC0_float_1_1_1_112_112_2_2_3_224_224)
{
    TestTload<8, float, 1, 1, 1, 112, 112, 2, 2, 3, 224, 224>();
}
