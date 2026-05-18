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

#define DIV_ROUNDUP(a, b) (((a) + (b) - 1) / (b))

using namespace std;
using namespace PtoTestCommon;

namespace TQuantTest {

template <int validRows, int validCols, int mode, bool isInt8Sym>
void LaunchTQuantInt8(std::conditional_t<isInt8Sym, int8_t, uint8_t> *dst, float *src, float *scale, void *stream,
                      float *offset = nullptr);

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

    LaunchTQuantInt8<validRows, validCols, mode, true>(dstDevice, srcDevice, scaleDevice, stream);

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
    LaunchTQuantInt8<validRows, validCols, mode, false>(dstDev, srcDev, scaleDev, stream, offDev);
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

// // INT8 - Sym cases
TEST_F(TQUANTTEST, case_int8_sym_fp32_64x128_nd)
{
    test_tquant_int8_sym<64, 128, 0>();
}
TEST_F(TQUANTTEST, case_int8_sym_fp32_128x128_nd)
{
    test_tquant_int8_sym<128, 128, 0>();
}
TEST_F(TQUANTTEST, case_int8_sym_fp32_192x128_nd)
{
    test_tquant_int8_sym<192, 128, 0>();
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
TEST_F(TQUANTTEST, case_int8_asym_fp32_192x128_nd)
{
    test_tquant_int8_asym<192, 128, 0>();
}
} // namespace TQuantTest
