/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "acl/acl.h"
#include "test_common.h"
#include <gtest/gtest.h>

using namespace PtoTestCommon;

template <int32_t tilingKey>
void LaunchTPushPopFixpipe(uint8_t* out, uint8_t* srcA, uint8_t* srcB, uint8_t* srcQuant, void* stream);

class TPushPopFixpipeTest : public testing::Test {
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

template <int32_t key>
void TPushPopFixpipeTestFunc(uint32_t m, uint32_t k, uint32_t n)
{
    const size_t aElementCount = static_cast<size_t>(m) * k;
    const size_t bElementCount = static_cast<size_t>(k) * n;
    const size_t cElementCount = static_cast<size_t>(m) * n;
    size_t aFileSize = aElementCount * sizeof(aclFloat16);
    size_t bFileSize = bElementCount * sizeof(aclFloat16);
    size_t cFileSize = cElementCount * sizeof(aclFloat16);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t* dstHost = nullptr;
    uint8_t* srcAHost = nullptr;
    uint8_t* srcBHost = nullptr;
    uint8_t* dstDevice = nullptr;
    uint8_t* srcADevice = nullptr;
    uint8_t* srcBDevice = nullptr;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&srcAHost), aFileSize);
    aclrtMallocHost((void**)(&srcBHost), bFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, srcAHost, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, srcBHost, bFileSize);

    aclrtMemcpy(srcADevice, aFileSize, srcAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcBDevice, bFileSize, srcBHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTPushPopFixpipe<key>(dstDevice, srcADevice, srcBDevice, nullptr, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcADevice);
    aclrtFree(srcBDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcAHost);
    aclrtFreeHost(srcBHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<aclFloat16> golden(cElementCount);
    std::vector<aclFloat16> devFinal(cElementCount);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

template <int32_t key>
void TPushPopFixpipeDEQF16TestFunc(uint32_t m, uint32_t k, uint32_t n)
{
    const size_t aElementCount = static_cast<size_t>(m) * k;
    const size_t bElementCount = static_cast<size_t>(k) * n;
    const size_t cElementCount = static_cast<size_t>(m) * n;
    size_t aFileSize = aElementCount * sizeof(int8_t);
    size_t bFileSize = bElementCount * sizeof(int8_t);
    size_t cFileSize = cElementCount * sizeof(aclFloat16);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t* dstHost = nullptr;
    uint8_t* srcAHost = nullptr;
    uint8_t* srcBHost = nullptr;
    uint8_t* dstDevice = nullptr;
    uint8_t* srcADevice = nullptr;
    uint8_t* srcBDevice = nullptr;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&srcAHost), aFileSize);
    aclrtMallocHost((void**)(&srcBHost), bFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, srcAHost, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, srcBHost, bFileSize);

    aclrtMemcpy(srcADevice, aFileSize, srcAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcBDevice, bFileSize, srcBHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTPushPopFixpipe<key>(dstDevice, srcADevice, srcBDevice, nullptr, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcADevice);
    aclrtFree(srcBDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcAHost);
    aclrtFreeHost(srcBHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<aclFloat16> golden(cElementCount);
    std::vector<aclFloat16> devFinal(cElementCount);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

template <int32_t key>
void TPushPopFixpipeVDEQF16TestFunc(uint32_t m, uint32_t k, uint32_t n)
{
    const size_t aElementCount = static_cast<size_t>(m) * k;
    const size_t bElementCount = static_cast<size_t>(k) * n;
    const size_t cElementCount = static_cast<size_t>(m) * n;
    const size_t quantElementCount = n;
    size_t aFileSize = aElementCount * sizeof(int8_t);
    size_t bFileSize = bElementCount * sizeof(int8_t);
    size_t cFileSize = cElementCount * sizeof(aclFloat16);
    size_t quantFileSize = quantElementCount * sizeof(uint64_t);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t* dstHost = nullptr;
    uint8_t* srcAHost = nullptr;
    uint8_t* srcBHost = nullptr;
    uint8_t* srcQuantHost = nullptr;
    uint8_t* dstDevice = nullptr;
    uint8_t* srcADevice = nullptr;
    uint8_t* srcBDevice = nullptr;
    uint8_t* srcQuantDevice = nullptr;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&srcAHost), aFileSize);
    aclrtMallocHost((void**)(&srcBHost), bFileSize);
    aclrtMallocHost((void**)(&srcQuantHost), quantFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcQuantDevice, quantFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, srcAHost, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, srcBHost, bFileSize);
    ReadFile(GetGoldenDir() + "/quant_gm.bin", quantFileSize, srcQuantHost, quantFileSize);

    aclrtMemcpy(srcADevice, aFileSize, srcAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcBDevice, bFileSize, srcBHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcQuantDevice, quantFileSize, srcQuantHost, quantFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTPushPopFixpipe<key>(dstDevice, srcADevice, srcBDevice, srcQuantDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcADevice);
    aclrtFree(srcBDevice);
    aclrtFree(srcQuantDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcAHost);
    aclrtFreeHost(srcBHost);
    aclrtFreeHost(srcQuantHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<aclFloat16> golden(cElementCount);
    std::vector<aclFloat16> devFinal(cElementCount);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

template <int32_t key>
void TPushPopFixpipeF322BF16TestFunc(uint32_t m, uint32_t k, uint32_t n)
{
    const size_t aElementCount = static_cast<size_t>(m) * k;
    const size_t bElementCount = static_cast<size_t>(k) * n;
    const size_t cElementCount = static_cast<size_t>(m) * n;
    size_t aFileSize = aElementCount * sizeof(aclFloat16);
    size_t bFileSize = bElementCount * sizeof(aclFloat16);
    size_t cFileSize = cElementCount * sizeof(aclFloat16);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t* dstHost = nullptr;
    uint8_t* srcAHost = nullptr;
    uint8_t* srcBHost = nullptr;
    uint8_t* dstDevice = nullptr;
    uint8_t* srcADevice = nullptr;
    uint8_t* srcBDevice = nullptr;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&srcAHost), aFileSize);
    aclrtMallocHost((void**)(&srcBHost), bFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&srcBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, srcAHost, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, srcBHost, bFileSize);

    aclrtMemcpy(srcADevice, aFileSize, srcAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcBDevice, bFileSize, srcBHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    LaunchTPushPopFixpipe<key>(dstDevice, srcADevice, srcBDevice, nullptr, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcADevice);
    aclrtFree(srcBDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcAHost);
    aclrtFreeHost(srcBHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<aclFloat16> golden(cElementCount);
    std::vector<aclFloat16> devFinal(cElementCount);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TPushPopFixpipeTest, case1_matmul_64x64_f322f16_tadds) { TPushPopFixpipeTestFunc<1>(64, 64, 64); }

TEST_F(TPushPopFixpipeTest, case2_matmul_64x64_f322bf16_tadds) { TPushPopFixpipeF322BF16TestFunc<2>(64, 64, 64); }

TEST_F(TPushPopFixpipeTest, case3_matmul_128x128_deqf16_tadds) { TPushPopFixpipeDEQF16TestFunc<3>(128, 128, 128); }

TEST_F(TPushPopFixpipeTest, case4_matmul_128x128_vdeqf16_tadds) { TPushPopFixpipeVDEQF16TestFunc<4>(128, 128, 128); }
