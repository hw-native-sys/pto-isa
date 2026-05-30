/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <gtest/gtest.h>
#include "acl/acl.h"
#include "test_common.h"
#include <pto/common/type.hpp>

using namespace std;
using namespace PtoTestCommon;
using pto::HistByte;

// ---------------------------------------------------------------------------
// uint16 launch declaration
// ---------------------------------------------------------------------------
template <int validRows, int validCols, HistByte byte>
void LaunchTHistogramU16(uint16_t *src, uint32_t *dst, void *stream, uint8_t *idx);

// ---------------------------------------------------------------------------
// uint32 launch declaration
// ---------------------------------------------------------------------------
template <int validRows, int validCols, HistByte byte>
void LaunchTHistogramU32(uint32_t *src, uint32_t *dst, void *stream, uint8_t *idx);

class THISTOGRAMTest : public testing::Test {
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

// ---------------------------------------------------------------------------
// uint16 test helper
// ---------------------------------------------------------------------------
template <int validRows, int validCols, HistByte byte>
void test_thistogram()
{
    constexpr bool isMSB = (byte == HistByte::BYTE_1);
    size_t srcFileSize = validRows * validCols * sizeof(uint16_t);
    size_t dstFileSize = validRows * 256 * sizeof(uint32_t);
    const size_t idxFileSize = validRows * sizeof(uint8_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint16_t *srcHost, *srcDevice;
    uint8_t *idxHost, *idxDevice;
    uint32_t *dstHost, *dstDevice;

    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMallocHost((void **)(&idxHost), idxFileSize);
    aclrtMalloc((void **)&idxDevice, idxFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // Read input data and copy to device
    size_t readSize = srcFileSize;
    ReadFile(GetGoldenDir() + "/input.bin", readSize, srcHost, srcFileSize);
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if constexpr (isMSB) {
        idxHost[0] = 0;
    } else {
        size_t idxReadSize = idxFileSize;
        ReadFile(GetGoldenDir() + "/idx.bin", idxReadSize, idxHost, idxFileSize);
    }
    aclrtMemcpy(idxDevice, idxFileSize, idxHost, idxFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTHistogramU16<validRows, validCols, byte>(srcDevice, dstDevice, stream, idxDevice);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(srcDevice);
    aclrtFree(idxDevice);
    aclrtFree(dstDevice);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(idxHost);
    aclrtFreeHost(dstHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    constexpr size_t numBinsPerRow = 256;
    std::vector<uint32_t> golden(validRows * numBinsPerRow);
    std::vector<uint32_t> devFinal(validRows * numBinsPerRow);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<uint32_t>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

// ---------------------------------------------------------------------------
// uint32 test helper
// ---------------------------------------------------------------------------
template <int validRows, int validCols, HistByte byte>
void test_thistogram_u32()
{
    constexpr int byteVal = static_cast<int>(byte);
    constexpr int numIdxRows = 3 - byteVal;
    size_t srcFileSize = validRows * validCols * sizeof(uint32_t);
    size_t dstFileSize = validRows * 256 * sizeof(uint32_t);
    // idx shape: (numIdxRows, validCols) for byte < 3, else unused
    const size_t idxFileSize = numIdxRows > 0 ? numIdxRows * validCols * sizeof(uint8_t) : sizeof(uint8_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint32_t *srcHost, *srcDevice;
    uint8_t *idxHost, *idxDevice;
    uint32_t *dstHost, *dstDevice;

    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMallocHost((void **)(&idxHost), idxFileSize);
    aclrtMalloc((void **)&idxDevice, idxFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t readSize = srcFileSize;
    ReadFile(GetGoldenDir() + "/input.bin", readSize, srcHost, srcFileSize);
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if constexpr (byte != HistByte::BYTE_3) {
        size_t idxReadSize = idxFileSize;
        ReadFile(GetGoldenDir() + "/idx.bin", idxReadSize, idxHost, idxFileSize);
    } else {
        idxHost[0] = 0;
    }
    aclrtMemcpy(idxDevice, idxFileSize, idxHost, idxFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTHistogramU32<validRows, validCols, byte>(srcDevice, dstDevice, stream, idxDevice);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output.bin", dstHost, dstFileSize);

    aclrtFree(srcDevice);
    aclrtFree(idxDevice);
    aclrtFree(dstDevice);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(idxHost);
    aclrtFreeHost(dstHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    constexpr size_t numBinsPerRow = 256;
    std::vector<uint32_t> golden(validRows * numBinsPerRow);
    std::vector<uint32_t> devFinal(validRows * numBinsPerRow);
    ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize);

    bool ret = ResultCmp<uint32_t>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

// ===========================================================================
// uint16 test cases — BYTE_1 = MSB (bits 15-8), BYTE_0 = LSB (bits 7-0)
// ===========================================================================
TEST_F(THISTOGRAMTest, case_2x128_b1)
{
    test_thistogram<2, 128, HistByte::BYTE_1>();
}
TEST_F(THISTOGRAMTest, case_4x64_b1)
{
    test_thistogram<4, 64, HistByte::BYTE_1>();
}
TEST_F(THISTOGRAMTest, case_8x128_b1)
{
    test_thistogram<8, 128, HistByte::BYTE_1>();
}
TEST_F(THISTOGRAMTest, case_1x256_b1)
{
    test_thistogram<1, 256, HistByte::BYTE_1>();
}
TEST_F(THISTOGRAMTest, case_4x256_b1)
{
    test_thistogram<4, 256, HistByte::BYTE_1>();
}
TEST_F(THISTOGRAMTest, case_2x100_b1)
{
    test_thistogram<2, 100, HistByte::BYTE_1>();
}

TEST_F(THISTOGRAMTest, case_2x128_b0_k108)
{
    test_thistogram<2, 128, HistByte::BYTE_0>();
}

TEST_F(THISTOGRAMTest, case_4x64_b0_k52)
{
    test_thistogram<4, 64, HistByte::BYTE_0>();
}

TEST_F(THISTOGRAMTest, case_8x128_b0_k104)
{
    test_thistogram<8, 128, HistByte::BYTE_0>();
}

TEST_F(THISTOGRAMTest, case_1x256_b0_k210)
{
    test_thistogram<1, 256, HistByte::BYTE_0>();
}

TEST_F(THISTOGRAMTest, case_4x256_b0_k220)
{
    test_thistogram<4, 256, HistByte::BYTE_0>();
}

TEST_F(THISTOGRAMTest, case_2x100_b0_k82)
{
    test_thistogram<2, 100, HistByte::BYTE_0>();
}

// ===========================================================================
// uint32 test cases
// ===========================================================================

// BYTE_3: histogram of byte3 (MSB), no filtering
TEST_F(THISTOGRAMTest, case_u32_1x128_b3_k64)
{
    test_thistogram_u32<1, 128, HistByte::BYTE_3>();
}
TEST_F(THISTOGRAMTest, case_u32_1x256_b3_k128)
{
    test_thistogram_u32<1, 256, HistByte::BYTE_3>();
}
TEST_F(THISTOGRAMTest, case_u32_2x128_b3_k100)
{
    test_thistogram_u32<2, 128, HistByte::BYTE_3>();
}
TEST_F(THISTOGRAMTest, case_u32_2x4096_b3_k96)
{
    test_thistogram_u32<2, 4096, HistByte::BYTE_3>();
}
TEST_F(THISTOGRAMTest, case_u32_4x4096_b3_k128)
{
    test_thistogram_u32<4, 4096, HistByte::BYTE_3>();
}
TEST_F(THISTOGRAMTest, case_u32_2x192_b3_k64)
{
    test_thistogram_u32<2, 192, HistByte::BYTE_3>();
}
TEST_F(THISTOGRAMTest, case_u32_6x912_b3_k64)
{
    test_thistogram_u32<6, 912, HistByte::BYTE_3>();
}

// BYTE_2: histogram of byte2, filtered by byte3
TEST_F(THISTOGRAMTest, case_u32_1x128_b2_k64)
{
    test_thistogram_u32<1, 128, HistByte::BYTE_2>();
}
TEST_F(THISTOGRAMTest, case_u32_1x256_b2_k128)
{
    test_thistogram_u32<1, 256, HistByte::BYTE_2>();
}
TEST_F(THISTOGRAMTest, case_u32_2x128_b2_k100)
{
    test_thistogram_u32<2, 128, HistByte::BYTE_2>();
}
TEST_F(THISTOGRAMTest, case_u32_2x4096_b2_k96)
{
    test_thistogram_u32<2, 4096, HistByte::BYTE_2>();
}
TEST_F(THISTOGRAMTest, case_u32_4x4096_b2_k128)
{
    test_thistogram_u32<4, 4096, HistByte::BYTE_2>();
}
TEST_F(THISTOGRAMTest, case_u32_2x192_b2_k64)
{
    test_thistogram_u32<2, 192, HistByte::BYTE_2>();
}
TEST_F(THISTOGRAMTest, case_u32_6x912_b2_k64)
{
    test_thistogram_u32<6, 912, HistByte::BYTE_2>();
}

// BYTE_1: histogram of byte1, filtered by byte3 & byte2
TEST_F(THISTOGRAMTest, case_u32_1x128_b1_k64)
{
    test_thistogram_u32<1, 128, HistByte::BYTE_1>();
}
TEST_F(THISTOGRAMTest, case_u32_1x256_b1_k128)
{
    test_thistogram_u32<1, 256, HistByte::BYTE_1>();
}
TEST_F(THISTOGRAMTest, case_u32_2x4096_b1_k96)
{
    test_thistogram_u32<2, 4096, HistByte::BYTE_1>();
}
TEST_F(THISTOGRAMTest, case_u32_2x192_b1_k64)
{
    test_thistogram_u32<2, 192, HistByte::BYTE_1>();
}
TEST_F(THISTOGRAMTest, case_u32_6x912_b1_k64)
{
    test_thistogram_u32<6, 912, HistByte::BYTE_1>();
}

// BYTE_0: histogram of byte0 (LSB), filtered by all upper bytes
TEST_F(THISTOGRAMTest, case_u32_1x128_b0_k64)
{
    test_thistogram_u32<1, 128, HistByte::BYTE_0>();
}
TEST_F(THISTOGRAMTest, case_u32_1x256_b0_k128)
{
    test_thistogram_u32<1, 256, HistByte::BYTE_0>();
}
TEST_F(THISTOGRAMTest, case_u32_2x4096_b0_k96)
{
    test_thistogram_u32<2, 4096, HistByte::BYTE_0>();
}
TEST_F(THISTOGRAMTest, case_u32_2x192_b0_k64)
{
    test_thistogram_u32<2, 192, HistByte::BYTE_0>();
}
TEST_F(THISTOGRAMTest, case_u32_6x912_b0_k64)
{
    test_thistogram_u32<6, 912, HistByte::BYTE_0>();
}
