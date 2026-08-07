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

using namespace std;
using namespace PtoTestCommon;

namespace TMovZZTest {

template <int validRows, int validCols>
void LaunchTMovZZ(uint8_t* dstFp8Nz, uint8_t* src, void* stream);

class TMOVZZTest : public testing::Test {
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

template <int validRows, int validCols>
void test_tmov_zz()
{
    constexpr int paddedCols = ((validCols + 31) / 32) * 32;
    constexpr int paddedRows = ((validRows + 15) / 16) * 16;
    size_t srcFileSize = validRows * validCols * sizeof(uint8_t);
    size_t dstFp8FileSize = paddedRows * paddedCols * sizeof(uint8_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstFp8Host, *dstFp8Device;
    uint8_t *srcHost, *srcDevice;

    aclrtMallocHost((void**)(&dstFp8Host), dstFp8FileSize);
    aclrtMallocHost((void**)(&srcHost), srcFileSize);

    aclrtMalloc((void**)&dstFp8Device, dstFp8FileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcDevice, srcFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input.bin", srcFileSize, srcHost, srcFileSize);
    aclrtMemcpy(srcDevice, srcFileSize, srcHost, srcFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    aclrtMemset(dstFp8Host, dstFp8FileSize, 0, dstFp8FileSize);
    aclrtMemcpy(dstFp8Device, dstFp8FileSize, dstFp8Host, dstFp8FileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTMovZZ<validRows, validCols>(dstFp8Device, srcDevice, stream);

    aclError syncRet = aclrtSynchronizeStream(stream);

    aclrtMemcpy(dstFp8Host, dstFp8FileSize, dstFp8Device, dstFp8FileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_fp8_nz.bin", dstFp8Host, dstFp8FileSize);

    aclrtFree(dstFp8Device);
    aclrtFree(srcDevice);
    aclrtFreeHost(dstFp8Host);
    aclrtFreeHost(srcHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<uint8_t> goldenFp8(dstFp8FileSize);
    std::vector<uint8_t> outFp8(dstFp8FileSize);

    ReadFile(GetGoldenDir() + "/golden_fp8_nz.bin", dstFp8FileSize, goldenFp8.data(), dstFp8FileSize);
    ReadFile(GetGoldenDir() + "/output_fp8_nz.bin", dstFp8FileSize, outFp8.data(), dstFp8FileSize);

    EXPECT_TRUE(ResultCmp<uint8_t>(goldenFp8, outFp8, 0.0f));
}

TEST_F(TMOVZZTest, case_fp32_32x64) { test_tmov_zz<32, 64>(); }

TEST_F(TMOVZZTest, case_fp32_64x64) { test_tmov_zz<64, 64>(); }

TEST_F(TMOVZZTest, case_fp32_64x128) { test_tmov_zz<64, 128>(); }

TEST_F(TMOVZZTest, case_fp32_64x192) { test_tmov_zz<64, 192>(); }

TEST_F(TMOVZZTest, case_fp32_64x256) { test_tmov_zz<64, 256>(); }

TEST_F(TMOVZZTest, case_fp32_64x320) { test_tmov_zz<64, 320>(); }

TEST_F(TMOVZZTest, case_fp32_64x384) { test_tmov_zz<64, 384>(); }

TEST_F(TMOVZZTest, case_fp32_64x448) { test_tmov_zz<64, 448>(); }

TEST_F(TMOVZZTest, case_fp32_64x512) { test_tmov_zz<64, 512>(); }

TEST_F(TMOVZZTest, case_fp32_64x576) { test_tmov_zz<64, 576>(); }

TEST_F(TMOVZZTest, case_fp32_64x640) { test_tmov_zz<64, 640>(); }

TEST_F(TMOVZZTest, case_fp32_64x704) { test_tmov_zz<64, 704>(); }

TEST_F(TMOVZZTest, case_fp32_64x768) { test_tmov_zz<64, 768>(); }

TEST_F(TMOVZZTest, case_fp32_64x832) { test_tmov_zz<64, 832>(); }

TEST_F(TMOVZZTest, case_fp32_64x896) { test_tmov_zz<64, 896>(); }

TEST_F(TMOVZZTest, case_fp32_128x128) { test_tmov_zz<128, 128>(); }

TEST_F(TMOVZZTest, case_fp32_128x256) { test_tmov_zz<128, 256>(); }

TEST_F(TMOVZZTest, case_fp32_128x384) { test_tmov_zz<128, 384>(); }

TEST_F(TMOVZZTest, case_fp32_256x192) { test_tmov_zz<256, 192>(); }

TEST_F(TMOVZZTest, case_fp32_8x64) { test_tmov_zz<8, 64>(); }

TEST_F(TMOVZZTest, case_fp32_6x64) { test_tmov_zz<6, 64>(); }

TEST_F(TMOVZZTest, case_fp32_13x64) { test_tmov_zz<13, 64>(); }

TEST_F(TMOVZZTest, case_fp32_3x64) { test_tmov_zz<3, 64>(); }

TEST_F(TMOVZZTest, case_fp32_29x64) { test_tmov_zz<29, 64>(); }

TEST_F(TMOVZZTest, case_fp32_31x64) { test_tmov_zz<31, 64>(); }

TEST_F(TMOVZZTest, case_fp32_47x64) { test_tmov_zz<47, 64>(); }

TEST_F(TMOVZZTest, case_fp32_31x128) { test_tmov_zz<31, 128>(); }

TEST_F(TMOVZZTest, case_fp32_47x128) { test_tmov_zz<47, 128>(); }

TEST_F(TMOVZZTest, case_fp32_31x256) { test_tmov_zz<31, 256>(); }

TEST_F(TMOVZZTest, case_fp32_47x256) { test_tmov_zz<47, 256>(); }

} // namespace TMovZZTest
