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
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <gtest/gtest.h>

using namespace std;
using namespace pto;
using namespace PtoTestCommon;

template <int32_t tilingKey>
void launchTCMP_demo(uint8_t* out, uint8_t* src, void* stream);

class TCMPTest : public testing::Test {
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

template <typename T, typename TDst, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void LaunchTCmp(TDst* out, T* src0, T* src1, pto::CmpMode mode, void* stream);

template <typename T, typename TDst, int kGRows_, int kGCols_, int kTRows_, int kTCols_>
void test_tcmp(pto::CmpMode mode)
{
    size_t srcFileSize = kGRows_ * kGCols_ * sizeof(T);
    size_t dstFileSize = kGRows_ * kGCols_ * sizeof(TDst);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    TDst *dstHost, *dstDevice;
    T *src0Host, *src1Host;
    T *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), dstFileSize);
    aclrtMallocHost((void**)(&src0Host), srcFileSize);
    aclrtMallocHost((void**)(&src1Host), srcFileSize);

    aclrtMalloc((void**)&dstDevice, dstFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input1.bin", srcFileSize, src0Host, srcFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/input2.bin", srcFileSize, src1Host, srcFileSize));

    aclrtMemcpy(src0Device, srcFileSize, src0Host, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, srcFileSize, src1Host, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchTCmp<T, TDst, kGRows_, kGCols_, kTRows_, kTCols_>(dstDevice, src0Device, src1Device, mode, stream);

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

    std::vector<TDst> golden(dstFileSize / sizeof(TDst));
    std::vector<TDst> devFinal(dstFileSize / sizeof(TDst));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/golden.bin", dstFileSize, golden.data(), dstFileSize));
    CHECK_RESULT_GTEST(ReadFile(GetGoldenDir() + "/output.bin", dstFileSize, devFinal.data(), dstFileSize));

    bool ret = ResultCmp<TDst>(golden, devFinal, 0.001f);

    EXPECT_TRUE(ret);
}

TEST_F(TCMPTest, case_float_uint8_64x64_64x64_64x64_EQ)
{
    test_tcmp<float, uint8_t, NUM_64, NUM_64, NUM_64, NUM_64>(pto::CmpMode::EQ);
}
TEST_F(TCMPTest, case_int32_uint8_64x64_64x64_64x64_NE)
{
    test_tcmp<int32_t, uint8_t, NUM_64, NUM_64, NUM_64, NUM_64>(pto::CmpMode::NE);
}
TEST_F(TCMPTest, case_half_uint8_16x256_16x256_16x256_GT)
{
    test_tcmp<aclFloat16, uint8_t, NUM_16, NUM_256, NUM_16, NUM_256>(pto::CmpMode::GT);
}
TEST_F(TCMPTest, case_uint32_uint32_64x64_64x64_64x64_GE)
{
    test_tcmp<uint32_t, uint32_t, NUM_64, NUM_64, NUM_64, NUM_64>(pto::CmpMode::GE);
}
TEST_F(TCMPTest, case_int32_uint32_64x64_64x64_64x64_LT)
{
    test_tcmp<int32_t, uint32_t, NUM_64, NUM_64, NUM_64, NUM_64>(pto::CmpMode::LT);
}
TEST_F(TCMPTest, case_uint16_uint32_64x64_64x64_64x64_LE)
{
    test_tcmp<uint16_t, uint32_t, NUM_64, NUM_64, NUM_64, NUM_64>(pto::CmpMode::LE);
}
TEST_F(TCMPTest, case_int16_uint32_64x64_64x64_64x64_EQ)
{
    test_tcmp<int16_t, uint32_t, NUM_64, NUM_64, NUM_64, NUM_64>(pto::CmpMode::EQ);
}
TEST_F(TCMPTest, case_uint8_uint32_64x64_64x64_64x64_LT)
{
    test_tcmp<uint8_t, uint32_t, NUM_64, NUM_64, NUM_64, NUM_64>(pto::CmpMode::LT);
}
TEST_F(TCMPTest, case_int8_uint32_64x64_64x64_64x64_GT)
{
    test_tcmp<int8_t, uint32_t, NUM_64, NUM_64, NUM_64, NUM_64>(pto::CmpMode::GT);
}
TEST_F(TCMPTest, case_float_uint32_64x64_64x64_64x64_NE)
{
    test_tcmp<float, uint32_t, NUM_64, NUM_64, NUM_64, NUM_64>(pto::CmpMode::NE);
}
TEST_F(TCMPTest, case_half_uint32_16x256_16x256_16x256_LE)
{
    test_tcmp<aclFloat16, uint32_t, NUM_16, NUM_256, NUM_16, NUM_256>(pto::CmpMode::LE);
}
#ifdef CPU_SIM_BFLOAT_ENABLED
TEST_F(TCMPTest, case_bf16_uint32_16x256_16x256_16x256_GE)
{
    test_tcmp<bfloat16_t, uint32_t, NUM_16, NUM_256, NUM_16, NUM_256>(pto::CmpMode::GE);
}
#endif
