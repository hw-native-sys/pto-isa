/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include "test_common.h"
#include "acl_env.h"
#include "acl/acl.h"
#include <gtest/gtest.h>

using namespace std;
using namespace PtoTestCommon;

void launchTExtractNd2xNz(
    int key, uint8_t* out0, uint8_t* out1, uint8_t* src, uint16_t ir0, uint16_t ic0, uint16_t ir1, uint16_t ic1,
    void* stream);

void launchTExtractNd2xNz1x1(
    int key, uint8_t* out0, uint8_t* out1, uint8_t* src, uint16_t ir0, uint16_t ic0, uint16_t ir1, uint16_t ic1,
    void* stream);

class TExtractNd2xNzTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

std::string GetGoldenDir()
{
    const testing::TestInfo* testInfo = testing::UnitTest::GetInstance()->current_test_info();
    const std::string caseName = testInfo->name();
    std::string suiteName = testInfo->test_suite_name();
    return "../" + suiteName + "." + caseName;
}

static void test_ndto2xnz(int key, int esize, uint16_t ir0, uint16_t ic0, uint16_t ir1, uint16_t ic1)
{
    constexpr int SR = 64, SC = 128, R0 = 32, C0 = 64, R1 = 32, C1 = 64;
    size_t srcSize = static_cast<size_t>(SR) * SC * esize;
    size_t out0Size = static_cast<size_t>(R0) * C0 * esize;
    size_t out1Size = static_cast<size_t>(R1) * C1 * esize;

    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *src0Host, *srcDevice;
    uint8_t *out0Host, *out0Device;
    uint8_t *out1Host, *out1Device;

    aclrtMallocHost((void**)(&src0Host), srcSize);
    aclrtMallocHost((void**)(&out0Host), out0Size);
    aclrtMallocHost((void**)(&out1Host), out1Size);
    aclrtMalloc((void**)(&srcDevice), srcSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)(&out0Device), out0Size, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)(&out1Device), out1Size, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input_arr.bin", srcSize, src0Host, srcSize);
    aclrtMemcpy(srcDevice, srcSize, src0Host, srcSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTExtractNd2xNz(key, out0Device, out1Device, srcDevice, ir0, ic0, ir1, ic1, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(out0Host, out0Size, out0Device, out0Size, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(out1Host, out1Size, out1Device, out1Size, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(srcDevice);
    aclrtFree(out0Device);
    aclrtFree(out1Device);

    std::vector<uint8_t> golden0(out0Size);
    std::vector<uint8_t> golden1(out1Size);
    std::vector<uint8_t> dev0(out0Host, out0Host + out0Size);
    std::vector<uint8_t> dev1(out1Host, out1Host + out1Size);
    ReadFile(GetGoldenDir() + "/golden0.bin", out0Size, golden0.data(), out0Size);
    ReadFile(GetGoldenDir() + "/golden1.bin", out1Size, golden1.data(), out1Size);

    aclrtFreeHost(src0Host);
    aclrtFreeHost(out0Host);
    aclrtFreeHost(out1Host);
    aclrtDestroyStream(stream);

    bool ret0 = ResultCmp<uint8_t>(golden0, dev0, 0.0f);
    bool ret1 = ResultCmp<uint8_t>(golden1, dev1, 0.0f);
    EXPECT_TRUE(ret0);
    EXPECT_TRUE(ret1);
}

static void test_ndto2xnz_1x1(int key, int esize, uint16_t ir0, uint16_t ic0, uint16_t ir1, uint16_t ic1)
{
    constexpr int SR = 64, SC = 128, N0 = 16;
    if (esize <= 0) {
        FAIL() << "esize must be greater than 0";
        return;
    }
    int c0 = 32 / esize;
    size_t srcSize = static_cast<size_t>(SR) * SC * esize;
    size_t outSize = static_cast<size_t>(N0) * c0 * esize;
    size_t goldenSize = static_cast<size_t>(esize);
    size_t devCmpSize = static_cast<size_t>(esize);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *src0Host, *srcDevice;
    uint8_t *out0Host, *out0Device;
    uint8_t *out1Host, *out1Device;

    aclrtMallocHost((void**)(&src0Host), srcSize);
    aclrtMallocHost((void**)(&out0Host), outSize);
    aclrtMallocHost((void**)(&out1Host), outSize);
    aclrtMalloc((void**)(&srcDevice), srcSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)(&out0Device), outSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)(&out1Device), outSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/input_arr.bin", srcSize, src0Host, srcSize);
    aclrtMemcpy(srcDevice, srcSize, src0Host, srcSize, ACL_MEMCPY_HOST_TO_DEVICE);

    launchTExtractNd2xNz1x1(key, out0Device, out1Device, srcDevice, ir0, ic0, ir1, ic1, stream);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(out0Host, outSize, out0Device, outSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(out1Host, outSize, out1Device, outSize, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(srcDevice);
    aclrtFree(out0Device);
    aclrtFree(out1Device);

    std::vector<uint8_t> golden0(esize);
    std::vector<uint8_t> golden1(esize);
    std::vector<uint8_t> dev0(out0Host, out0Host + devCmpSize);
    std::vector<uint8_t> dev1(out1Host, out1Host + devCmpSize);
    ReadFile(GetGoldenDir() + "/golden0.bin", goldenSize, golden0.data(), goldenSize);
    ReadFile(GetGoldenDir() + "/golden1.bin", goldenSize, golden1.data(), goldenSize);

    aclrtFreeHost(src0Host);
    aclrtFreeHost(out0Host);
    aclrtFreeHost(out1Host);
    aclrtDestroyStream(stream);

    bool ret0 = ResultCmp<uint8_t>(golden0, dev0, 0.0f);
    bool ret1 = ResultCmp<uint8_t>(golden1, dev1, 0.0f);
    EXPECT_TRUE(ret0);
    EXPECT_TRUE(ret1);
}

TEST_F(TExtractNd2xNzTest, case_half_aligned) { test_ndto2xnz(0, 2, 0, 0, 32, 64); }

TEST_F(TExtractNd2xNzTest, case_half_unaligned) { test_ndto2xnz(0, 2, 0, 0, 16, 8); }

TEST_F(TExtractNd2xNzTest, case_float_aligned) { test_ndto2xnz(1, 4, 0, 0, 32, 64); }

TEST_F(TExtractNd2xNzTest, case_float_unaligned) { test_ndto2xnz(1, 4, 0, 0, 16, 4); }

TEST_F(TExtractNd2xNzTest, case_bf16_aligned) { test_ndto2xnz(2, 2, 0, 0, 32, 64); }

TEST_F(TExtractNd2xNzTest, case_int8_aligned) { test_ndto2xnz(3, 1, 0, 0, 32, 64); }

TEST_F(TExtractNd2xNzTest, case_int8_unaligned) { test_ndto2xnz(3, 1, 0, 0, 16, 16); }

TEST_F(TExtractNd2xNzTest, case_int32_aligned) { test_ndto2xnz(4, 4, 0, 0, 32, 64); }

TEST_F(TExtractNd2xNzTest, case_hif8_aligned) { test_ndto2xnz(5, 1, 0, 0, 32, 64); }

TEST_F(TExtractNd2xNzTest, case_fp8e4m3_aligned) { test_ndto2xnz(6, 1, 0, 0, 32, 64); }

TEST_F(TExtractNd2xNzTest, case_fp8e5m2_aligned) { test_ndto2xnz(7, 1, 0, 0, 32, 64); }

TEST_F(TExtractNd2xNzTest, case_fp8e8m0_aligned) { test_ndto2xnz(8, 1, 0, 0, 32, 64); }

TEST_F(TExtractNd2xNzTest, case_half_1x1) { test_ndto2xnz_1x1(0, 2, 0, 0, 5, 7); }

TEST_F(TExtractNd2xNzTest, case_float_1x1) { test_ndto2xnz_1x1(1, 4, 0, 0, 10, 3); }

TEST_F(TExtractNd2xNzTest, case_int8_1x1) { test_ndto2xnz_1x1(3, 1, 0, 0, 20, 17); }

TEST_F(TExtractNd2xNzTest, case_int8_tail_unaligned) { test_ndto2xnz(9, 1, 0, 0, 0, 2); }

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    testing::AddGlobalTestEnvironment(new AclEnv());
    return RUN_ALL_TESTS();
}
