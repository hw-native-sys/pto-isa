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
#include <type_traits>
#include "acl/acl.h"
#include "test_common.h"

#define DIV_ROUNDUP(a, b) (((a) + (b)-1) / (b))

using namespace std;
using namespace PtoTestCommon;

namespace pto {
enum class QuantType
{
    MXFP8,
    INT8_SYM,
    INT8_ASYM
};
}

namespace TQuantTest {

template <int validRows, int validCols, int mode>
void LaunchTQuantMXFP8(uint8_t *dst, float *src, uint8_t *dst_exp, void *stream);

template <int validRows, int validCols, int mode, pto::QuantType quantType>
void LaunchTQuantInt8(std::conditional_t<quantType == pto::QuantType::INT8_SYM, int8_t, uint8_t> *dst, float *src,
                      float *scale, void *stream, float *offset = nullptr);

template <int validRows, int validCols, int mode>
void LaunchTQuantMXFP8_BF16(uint8_t *dst, uint16_t *src, uint8_t *dst_exp, void *stream);

template <int validRows, int validCols, int mode>
void LaunchTQuantMXFP8_FP16(uint8_t *dst, uint16_t *src, uint8_t *dst_exp, void *stream);

class TQUANTTEST : public testing::Test {
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

template <int validRows, int validCols, int mode>
void test_tquant_mxfp8()
{
    constexpr int paddedCols = ((validCols + 31) / 32) * 32;
    size_t srcFileSize = validRows * validCols * sizeof(float);
    size_t dstExpFileSize = DIV_ROUNDUP(validRows * paddedCols, 32) * sizeof(uint8_t);
    size_t dstFileSize = validRows * validCols * sizeof(uint8_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *dstDevice, *dstExpHost, *dstExpDevice;
    float *srcHost, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&dstExpHost), dstExpFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstExpDevice, dstExpFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTQuantMXFP8<validRows, validCols, mode>(dstDevice, srcDevice, dstExpDevice, stream);

    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed (ret=" << syncRet
                                    << "): " << aclGetRecentErrMsg();
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(dstExpHost, dstExpFileSize, dstExpDevice, dstExpFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_e4m3.bin", dstHost, dstFileSize);
    WriteFile(GetGoldenDir() + "/output_e8m0.bin", dstExpHost, dstExpFileSize);

    aclrtFree((void *)dstDevice);
    aclrtFree((void *)dstExpDevice);
    aclrtFree((void *)srcDevice);

    aclrtFreeHost((void *)dstHost);
    aclrtFreeHost((void *)dstExpHost);
    aclrtFreeHost((void *)srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<uint8_t> golden_fp8(dstFileSize);
    std::vector<uint8_t> dev_fp8(dstFileSize);
    std::vector<uint8_t> golden_e8m0(dstExpFileSize);
    std::vector<uint8_t> dev_e8m0(dstExpFileSize);

    ReadFile(GetGoldenDir() + "/golden_fp8.bin", dstFileSize, golden_fp8.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/golden_e8m0.bin", dstExpFileSize, golden_e8m0.data(), dstExpFileSize);
    ReadFile(GetGoldenDir() + "/output_e4m3.bin", dstFileSize, dev_fp8.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output_e8m0.bin", dstExpFileSize, dev_e8m0.data(), dstExpFileSize);

    bool ret_fp8 = ResultCmp<uint8_t>(golden_fp8, dev_fp8, 0.0f);
    bool ret_e8m0 = ResultCmp<uint8_t>(golden_e8m0, dev_e8m0, 0.0f);

    EXPECT_TRUE(ret_e8m0);
    EXPECT_TRUE(ret_fp8);
}

template <int validRows, int validCols, int mode>
void test_tquant_mxfp8_bf16()
{
    constexpr int paddedCols = ((validCols + 31) / 32) * 32;
    size_t srcFileSize = validRows * validCols * sizeof(uint16_t); // bf16 = 2 bytes
    size_t dstExpFileSize = DIV_ROUNDUP(validRows * paddedCols, 32) * sizeof(uint8_t);
    size_t dstFileSize = validRows * validCols * sizeof(uint8_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *dstDevice, *dstExpHost, *dstExpDevice;
    uint16_t *srcHost, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&dstExpHost), dstExpFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstExpDevice, dstExpFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTQuantMXFP8_BF16<validRows, validCols, mode>(dstDevice, srcDevice, dstExpDevice, stream);

    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed (ret=" << syncRet
                                    << "): " << aclGetRecentErrMsg();
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(dstExpHost, dstExpFileSize, dstExpDevice, dstExpFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_e4m3.bin", dstHost, dstFileSize);
    WriteFile(GetGoldenDir() + "/output_e8m0.bin", dstExpHost, dstExpFileSize);

    aclrtFree((void *)dstDevice);
    aclrtFree((void *)dstExpDevice);
    aclrtFree((void *)srcDevice);

    aclrtFreeHost((void *)dstHost);
    aclrtFreeHost((void *)dstExpHost);
    aclrtFreeHost((void *)srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<uint8_t> golden_fp8(dstFileSize);
    std::vector<uint8_t> dev_fp8(dstFileSize);
    std::vector<uint8_t> golden_e8m0(dstExpFileSize);
    std::vector<uint8_t> dev_e8m0(dstExpFileSize);

    ReadFile(GetGoldenDir() + "/golden_fp8.bin", dstFileSize, golden_fp8.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/golden_e8m0.bin", dstExpFileSize, golden_e8m0.data(), dstExpFileSize);
    ReadFile(GetGoldenDir() + "/output_e4m3.bin", dstFileSize, dev_fp8.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output_e8m0.bin", dstExpFileSize, dev_e8m0.data(), dstExpFileSize);

    bool ret_fp8 = ResultCmp<uint8_t>(golden_fp8, dev_fp8, 0.0f);
    bool ret_e8m0 = ResultCmp<uint8_t>(golden_e8m0, dev_e8m0, 0.0f);

    EXPECT_TRUE(ret_e8m0);
    EXPECT_TRUE(ret_fp8);
}

template <int validRows, int validCols, int mode>
void test_tquant_mxfp8_fp16()
{
    constexpr int paddedCols = ((validCols + 31) / 32) * 32;
    size_t srcFileSize = validRows * validCols * sizeof(uint16_t); // fp16 = 2 bytes
    size_t dstExpFileSize = DIV_ROUNDUP(validRows * paddedCols, 32) * sizeof(uint8_t);
    size_t dstFileSize = validRows * validCols * sizeof(uint8_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *dstDevice, *dstExpHost, *dstExpDevice;
    uint16_t *srcHost, *srcDevice;

    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&dstExpHost), dstExpFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);

    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&dstExpDevice, dstExpFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTQuantMXFP8_FP16<validRows, validCols, mode>(dstDevice, srcDevice, dstExpDevice, stream);

    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed (ret=" << syncRet
                                    << "): " << aclGetRecentErrMsg();
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(dstExpHost, dstExpFileSize, dstExpDevice, dstExpFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_e4m3.bin", dstHost, dstFileSize);
    WriteFile(GetGoldenDir() + "/output_e8m0.bin", dstExpHost, dstExpFileSize);

    aclrtFree((void *)dstDevice);
    aclrtFree((void *)dstExpDevice);
    aclrtFree((void *)srcDevice);

    aclrtFreeHost((void *)dstHost);
    aclrtFreeHost((void *)dstExpHost);
    aclrtFreeHost((void *)srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<uint8_t> golden_fp8(dstFileSize);
    std::vector<uint8_t> dev_fp8(dstFileSize);
    std::vector<uint8_t> golden_e8m0(dstExpFileSize);
    std::vector<uint8_t> dev_e8m0(dstExpFileSize);

    ReadFile(GetGoldenDir() + "/golden_fp8.bin", dstFileSize, golden_fp8.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/golden_e8m0.bin", dstExpFileSize, golden_e8m0.data(), dstExpFileSize);
    ReadFile(GetGoldenDir() + "/output_e4m3.bin", dstFileSize, dev_fp8.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output_e8m0.bin", dstExpFileSize, dev_e8m0.data(), dstExpFileSize);

    bool ret_fp8 = ResultCmp<uint8_t>(golden_fp8, dev_fp8, 0.0f);
    bool ret_e8m0 = ResultCmp<uint8_t>(golden_e8m0, dev_e8m0, 0.0f);

    EXPECT_TRUE(ret_e8m0);
    EXPECT_TRUE(ret_fp8);
}

template <int validRows, int validCols, int mode>
void test_tquant_int8_sym()
{
    size_t srcFileSize = validRows * validCols * sizeof(float);
    size_t dstFileSize = validRows * validCols * sizeof(int8_t);
    size_t scaleFileSize = validRows * sizeof(float);
    int8_t *dstHost, *dstDevice;
    float *srcHost, *srcDevice, *scaleHost, *scaleDevice;

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    aclrtMallocHost((void **)(&dstHost), dstFileSize);
    aclrtMallocHost((void **)(&srcHost), srcFileSize);
    aclrtMallocHost((void **)(&scaleHost), scaleFileSize);
    aclrtMalloc((void **)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&scaleDevice, scaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    ReadFile(GetGoldenDir() + "/inv_scale_fp32.bin", scaleFileSize, scaleHost, scaleFileSize);
    aclrtMemcpy(scaleDevice, scaleFileSize, scaleHost, scaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTQuantInt8<validRows, validCols, mode, pto::QuantType::INT8_SYM>(dstDevice, srcDevice, scaleDevice, stream);

    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed (ret=" << syncRet
                                    << "): " << aclGetRecentErrMsg();
    aclrtMemcpy(dstHost, dstFileSize, dstDevice, dstFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output_s8.bin", dstHost, dstFileSize);
    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(scaleDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(scaleHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<int8_t> golden_s8(dstFileSize);
    std::vector<int8_t> dev_s8(dstFileSize);
    ReadFile(GetGoldenDir() + "/golden_s8.bin", dstFileSize, golden_s8.data(), dstFileSize);
    ReadFile(GetGoldenDir() + "/output_s8.bin", dstFileSize, dev_s8.data(), dstFileSize);

    EXPECT_TRUE(ResultCmp<int8_t>(golden_s8, dev_s8, 0.0f));
}

template <int validRows, int validCols, int mode>
void test_tquant_int8_asym()
{
    size_t srcSize = validRows * validCols * sizeof(float);
    size_t dstSize = validRows * validCols * sizeof(uint8_t);
    size_t scaleSize = validRows * sizeof(float);
    size_t offSize = validRows * sizeof(float);
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    uint8_t *dstHost, *dstDev;
    float *srcHost, *srcDev, *scaleHost, *scaleDev, *offHost, *offDev;
    aclrtMallocHost((void **)&dstHost, dstSize);
    aclrtMallocHost((void **)&srcHost, srcSize);
    aclrtMallocHost((void **)&scaleHost, scaleSize);
    aclrtMallocHost((void **)&offHost, offSize);
    aclrtMalloc((void **)&dstDev, dstSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDev, srcSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&scaleDev, scaleSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&offDev, offSize, ACL_MEM_MALLOC_HUGE_FIRST);
    ReadFile(GetGoldenDir() + "/input.bin", srcSize, srcHost, srcSize);
    ReadFile(GetGoldenDir() + "/inv_scale_fp32.bin", scaleSize, scaleHost, scaleSize);
    ReadFile(GetGoldenDir() + "/offset_fp32.bin", offSize, offHost, offSize);
    aclrtMemcpy(srcDev, srcSize, srcHost, srcSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(scaleDev, scaleSize, scaleHost, scaleSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(offDev, offSize, offHost, offSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTQuantInt8<validRows, validCols, mode, pto::QuantType::INT8_ASYM>(dstDev, srcDev, scaleDev, stream, offDev);
    aclError syncRet = aclrtSynchronizeStream(stream);
    ASSERT_EQ(syncRet, ACL_SUCCESS) << "aclrtSynchronizeStream failed (ret=" << syncRet
                                    << "): " << aclGetRecentErrMsg();
    aclrtMemcpy(dstHost, dstSize, dstDev, dstSize, ACL_MEMCPY_DEVICE_TO_HOST);
    WriteFile(GetGoldenDir() + "/output_u8.bin", dstHost, dstSize);
    aclrtFree(dstDev);
    aclrtFree(srcDev);
    aclrtFree(scaleDev);
    aclrtFree(offDev);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(scaleHost);
    aclrtFreeHost(offHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    std::vector<uint8_t> golden_u8(dstSize), dev_u8(dstSize);
    ReadFile(GetGoldenDir() + "/golden_u8.bin", dstSize, golden_u8.data(), dstSize);
    ReadFile(GetGoldenDir() + "/output_u8.bin", dstSize, dev_u8.data(), dstSize);
    EXPECT_TRUE(ResultCmp<uint8_t>(golden_u8, dev_u8, 0.0f));
}

// MXFP8
TEST_F(TQUANTTEST, case_mxfp8_fp32_32x32_nd)
{
    test_tquant_mxfp8<32, 32, 0>();
}
TEST_F(TQUANTTEST, case_mxfp8_fp32_32x64_nd)
{
    test_tquant_mxfp8<32, 64, 0>();
}
TEST_F(TQUANTTEST, case_mxfp8_fp32_64x128_nd)
{
    test_tquant_mxfp8<64, 128, 0>();
}
TEST_F(TQUANTTEST, case_mxfp8_fp32_128x128_nd)
{
    test_tquant_mxfp8<128, 128, 0>();
}

TEST_F(TQUANTTEST, case_mxfp8_fp32_32x64_nz)
{
    test_tquant_mxfp8<32, 64, 1>();
}
TEST_F(TQUANTTEST, case_mxfp8_fp32_64x128_nz)
{
    test_tquant_mxfp8<64, 128, 1>();
}
TEST_F(TQUANTTEST, case_mxfp8_fp32_128x128_nz)
{
    test_tquant_mxfp8<128, 128, 1>();
}
TEST_F(TQUANTTEST, case_mxfp8_fp32_64x256_nz)
{
    test_tquant_mxfp8<64, 256, 1>();
}
TEST_F(TQUANTTEST, case_mxfp8_fp32_64x512_nz)
{
    test_tquant_mxfp8<64, 512, 1>();
}

TEST_F(TQUANTTEST, case_mxfp8_fp32_15x32_nd)
{
    test_tquant_mxfp8<15, 32, 0>();
}
TEST_F(TQUANTTEST, case_mxfp8_fp32_7x64_nd)
{
    test_tquant_mxfp8<7, 64, 0>();
}
TEST_F(TQUANTTEST, case_mxfp8_fp32_33x64_nd)
{
    test_tquant_mxfp8<33, 64, 0>();
}

// MXFP8 BF16
TEST_F(TQUANTTEST, case_mxfp8_bf16_32x128_nd)
{
    test_tquant_mxfp8_bf16<32, 128, 0>();
}
TEST_F(TQUANTTEST, case_mxfp8_bf16_64x128_nd)
{
    test_tquant_mxfp8_bf16<64, 128, 0>();
}
TEST_F(TQUANTTEST, case_mxfp8_bf16_128x128_nd)
{
    test_tquant_mxfp8_bf16<128, 128, 0>();
}
TEST_F(TQUANTTEST, case_mxfp8_bf16_14x16_nd)
{
    test_tquant_mxfp8_bf16<14, 16, 0>();
}
TEST_F(TQUANTTEST, case_mxfp8_bf16_7x48_nd)
{
    test_tquant_mxfp8_bf16<7, 48, 0>();
}

// Diagnostic cases for board failure root-cause analysis.
TEST_F(TQUANTTEST, case_mxfp8_bf16_1x32_nd)
{
    test_tquant_mxfp8_bf16<1, 32, 0>();
}
TEST_F(TQUANTTEST, case_mxfp8_bf16_2x16_nd)
{
    test_tquant_mxfp8_bf16<2, 16, 0>();
}
TEST_F(TQUANTTEST, case_mxfp8_bf16_3x32_nd)
{
    test_tquant_mxfp8_bf16<3, 32, 0>();
}
TEST_F(TQUANTTEST, case_mxfp8_bf16_5x96_nd)
{
    test_tquant_mxfp8_bf16<5, 96, 0>();
}
TEST_F(TQUANTTEST, case_mxfp8_bf16_1x16_nd)
{
    test_tquant_mxfp8_bf16<1, 16, 0>();
}
// Multi-flush vstas coverage: loop_num odd >= 3 => 16B pending in st_align at final vstas.
// 3x256 => padded 768 elements => loop_num = ceil(768/256) = 3 (odd).
TEST_F(TQUANTTEST, case_mxfp8_bf16_3x256_nd)
{
    test_tquant_mxfp8_bf16<3, 256, 0>();
}
// 5x256 => padded 1280 elements => loop_num = 5 (odd), exercises more vstus iterations.
TEST_F(TQUANTTEST, case_mxfp8_bf16_5x256_nd)
{
    test_tquant_mxfp8_bf16<5, 256, 0>();
}
// Additional padding coverage (non-multiple-of-32 cols / large padding).
TEST_F(TQUANTTEST, case_mxfp8_bf16_18x138_nd)
{
    test_tquant_mxfp8_bf16<18, 138, 0>();
}
TEST_F(TQUANTTEST, case_mxfp8_bf16_1x192_nd)
{
    test_tquant_mxfp8_bf16<1, 192, 0>();
}
TEST_F(TQUANTTEST, case_mxfp8_bf16_1x198_nd)
{
    test_tquant_mxfp8_bf16<1, 198, 0>();
}

TEST_F(TQUANTTEST, case_mxfp8_bf16_32x128_nz)
{
    test_tquant_mxfp8_bf16<32, 128, 1>();
}
TEST_F(TQUANTTEST, case_mxfp8_bf16_64x128_nz)
{
    test_tquant_mxfp8_bf16<64, 128, 1>();
}
TEST_F(TQUANTTEST, case_mxfp8_bf16_128x128_nz)
{
    test_tquant_mxfp8_bf16<128, 128, 1>();
}

// MXFP8 FP16
TEST_F(TQUANTTEST, case_mxfp8_fp16_32x128_nd)
{
    test_tquant_mxfp8_fp16<32, 128, 0>();
}
TEST_F(TQUANTTEST, case_mxfp8_fp16_64x128_nd)
{
    test_tquant_mxfp8_fp16<64, 128, 0>();
}
TEST_F(TQUANTTEST, case_mxfp8_fp16_128x128_nd)
{
    test_tquant_mxfp8_fp16<128, 128, 0>();
}

TEST_F(TQUANTTEST, case_mxfp8_fp16_32x128_nz)
{
    test_tquant_mxfp8_fp16<32, 128, 1>();
}
TEST_F(TQUANTTEST, case_mxfp8_fp16_64x128_nz)
{
    test_tquant_mxfp8_fp16<64, 128, 1>();
}
TEST_F(TQUANTTEST, case_mxfp8_fp16_128x128_nz)
{
    test_tquant_mxfp8_fp16<128, 128, 1>();
}

// // INT8 - Sym cases
TEST_F(TQUANTTEST, case_int8_sym_fp32_64x128_nd)
{
    test_tquant_int8_sym<64, 128, 0>();
}
TEST_F(TQUANTTEST, case_int8_sym_fp32_128x128_nd)
{
    test_tquant_int8_sym<128, 128, 0>();
}
TEST_F(TQUANTTEST, case_int8_sym_fp32_256x128_nd)
{
    test_tquant_int8_sym<256, 128, 0>();
}

// //INT8 - Asym cases
TEST_F(TQUANTTEST, case_int8_asym_fp32_64x128_nd)
{
    test_tquant_int8_asym<64, 128, 0>();
}
TEST_F(TQUANTTEST, case_int8_asym_fp32_128x128_nd)
{
    test_tquant_int8_asym<128, 128, 0>();
}
TEST_F(TQUANTTEST, case_int8_asym_fp32_256x128_nd)
{
    test_tquant_int8_asym<256, 128, 0>();
}

} // namespace TQuantTest