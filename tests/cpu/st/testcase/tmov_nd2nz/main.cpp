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
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

// fp4 kind selected by enum tag (same enum defined in the kernel TU):
// E1M2 = e1m2, E2M1 = e2m1. The host driver never names a fp4 C++ type
// (it lives in a __CPU_SIM-only header).
enum class Fp4Kind {
    E1M2,
    E2M1,
};
template <Fp4Kind kFp4Kind, int kRows, int kCols>
void launchTMOV_nd2nz_f4(uint8_t* out, uint8_t* src, void* stream);

class TMovNd2NzTest : public testing::Test {
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

// CPU-SIM analog of the A5 tmov_nd2nz fp4 ST cases: dense ND fp4 GM input, NZ
// fractal output over the panel-aligned width, byte-exact comparison against
// the numpy golden from gen_data.py.
template <Fp4Kind kFp4Kind, int kRows, int kCols>
void test_tmov_nd2nz_f4()
{
    constexpr int c0 = 64;                                  // f4 nibbles per NZ panel row (32 bytes)
    constexpr int alignedCols = (kCols + c0 - 1) / c0 * c0; // padded to full NZ panels
    size_t inputSize = kRows * kCols / 2;                   // dense GM feed, 2 nibbles/byte
    size_t outputSize = kRows * alignedCols / 2;            // NZ over the padded width

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *dstDevice;
    uint8_t *srcHost, *srcDevice;

    aclrtMallocHost((void**)(&dstHost), outputSize);
    aclrtMallocHost((void**)(&srcHost), inputSize);
    aclrtMalloc((void**)(&dstDevice), outputSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)(&srcDevice), inputSize, ACL_MEM_MALLOC_HUGE_FIRST);

    size_t inputFileSize = 0;
    size_t goldenFileSize = 0;
    size_t outputFileSize = 0;
    ASSERT_TRUE(ReadFile(GetGoldenDir() + "/input_arr.bin", inputFileSize, srcHost, inputSize));
    aclrtMemset(dstDevice, outputSize, 0, outputSize);

    aclrtMemcpy(srcDevice, inputSize, srcHost, inputSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTMOV_nd2nz_f4<kFp4Kind, kRows, kCols>(dstDevice, srcDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outputSize, dstDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, outputSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<uint8_t> golden(outputSize);
    std::vector<uint8_t> devFinal(outputSize);
    ASSERT_TRUE(ReadFile(GetGoldenDir() + "/golden.bin", goldenFileSize, golden.data(), outputSize));
    ASSERT_TRUE(ReadFile(GetGoldenDir() + "/output_z.bin", outputFileSize, devFinal.data(), outputSize));

    bool ret = ResultCmp<uint8_t>(golden, devFinal, 0.0f);
    EXPECT_TRUE(ret);
}

TEST_F(TMovNd2NzTest, case_f4e1m2_16x32) { test_tmov_nd2nz_f4<Fp4Kind::E1M2, 16, 32>(); }

TEST_F(TMovNd2NzTest, case_f4e2m1_16x32) { test_tmov_nd2nz_f4<Fp4Kind::E2M1, 16, 32>(); }
