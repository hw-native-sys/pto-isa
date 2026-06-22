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
#include "runtime/rt.h"
#include "test_common.h"
#include <gtest/gtest.h>

using namespace PtoTestCommon;

template <int32_t tilingKey>
void LaunchTPushPopFixpipeA2A3(uint8_t *ffts, uint8_t *out, uint8_t *srcA, uint8_t *srcB, uint8_t *srcQuant,
                               uint8_t *fifoMem, void *stream);

class TPushPopFixpipeA2A3Test : public testing::Test {
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
    return "../" + suiteName + "." + caseName;
}

template <int32_t key>
void TPushPopFixpipeA2A3TestFunc(uint32_t m, uint32_t k, uint32_t n)
{
    size_t aFileSize = static_cast<size_t>(m) * k * sizeof(aclFloat16);
    size_t bFileSize = static_cast<size_t>(k) * n * sizeof(aclFloat16);
    size_t cFileSize = static_cast<size_t>(m) * n * sizeof(aclFloat16);
    size_t fifoFileSize = 2 * static_cast<size_t>(m) * n * sizeof(aclFloat16);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost = nullptr;
    uint8_t *srcAHost = nullptr;
    uint8_t *srcBHost = nullptr;
    uint8_t *dstDevice = nullptr;
    uint8_t *srcADevice = nullptr;
    uint8_t *srcBDevice = nullptr;
    uint8_t *fifoMemDevice = nullptr;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&srcAHost), aFileSize);
    aclrtMallocHost((void **)(&srcBHost), bFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&fifoMemDevice, fifoFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, srcAHost, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, srcBHost, bFileSize);

    aclrtMemcpy(srcADevice, aFileSize, srcAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcBDevice, bFileSize, srcBHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&ffts, &fftsLen);

    LaunchTPushPopFixpipeA2A3<key>((uint8_t *)ffts, dstDevice, srcADevice, srcBDevice, nullptr, fifoMemDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcADevice);
    aclrtFree(srcBDevice);
    aclrtFree(fifoMemDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcAHost);
    aclrtFreeHost(srcBHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    const size_t elementCount = static_cast<size_t>(m) * n;
    std::vector<aclFloat16> golden(elementCount);
    std::vector<aclFloat16> devFinal(elementCount);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

template <int32_t key>
void TPushPopFixpipeDEQF16A2A3TestFunc(uint32_t m, uint32_t k, uint32_t n)
{
    size_t aFileSize = static_cast<size_t>(m) * k * sizeof(int8_t);
    size_t bFileSize = static_cast<size_t>(k) * n * sizeof(int8_t);
    size_t cFileSize = static_cast<size_t>(m) * n * sizeof(aclFloat16);
    size_t fifoFileSize = 2 * static_cast<size_t>(m) * n * sizeof(aclFloat16);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost = nullptr;
    uint8_t *srcAHost = nullptr;
    uint8_t *srcBHost = nullptr;
    uint8_t *dstDevice = nullptr;
    uint8_t *srcADevice = nullptr;
    uint8_t *srcBDevice = nullptr;
    uint8_t *fifoMemDevice = nullptr;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&srcAHost), aFileSize);
    aclrtMallocHost((void **)(&srcBHost), bFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&fifoMemDevice, fifoFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, srcAHost, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, srcBHost, bFileSize);

    aclrtMemcpy(srcADevice, aFileSize, srcAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcBDevice, bFileSize, srcBHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&ffts, &fftsLen);

    LaunchTPushPopFixpipeA2A3<key>((uint8_t *)ffts, dstDevice, srcADevice, srcBDevice, nullptr, fifoMemDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcADevice);
    aclrtFree(srcBDevice);
    aclrtFree(fifoMemDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcAHost);
    aclrtFreeHost(srcBHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    const size_t elementCount = static_cast<size_t>(m) * n;
    std::vector<aclFloat16> golden(elementCount);
    std::vector<aclFloat16> devFinal(elementCount);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

template <int32_t key>
void TPushPopFixpipeVDEQF16A2A3TestFunc(uint32_t m, uint32_t k, uint32_t n)
{
    size_t aFileSize = static_cast<size_t>(m) * k * sizeof(int8_t);
    size_t bFileSize = static_cast<size_t>(k) * n * sizeof(int8_t);
    size_t cFileSize = static_cast<size_t>(m) * n * sizeof(aclFloat16);
    size_t quantFileSize = static_cast<size_t>(n) * sizeof(uint64_t);
    size_t fifoFileSize = 2 * static_cast<size_t>(m) * n * sizeof(aclFloat16);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost = nullptr;
    uint8_t *srcAHost = nullptr;
    uint8_t *srcBHost = nullptr;
    uint8_t *srcQuantHost = nullptr;
    uint8_t *dstDevice = nullptr;
    uint8_t *srcADevice = nullptr;
    uint8_t *srcBDevice = nullptr;
    uint8_t *srcQuantDevice = nullptr;
    uint8_t *fifoMemDevice = nullptr;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&srcAHost), aFileSize);
    aclrtMallocHost((void **)(&srcBHost), bFileSize);
    aclrtMallocHost((void **)(&srcQuantHost), quantFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcQuantDevice, quantFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&fifoMemDevice, fifoFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, srcAHost, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, srcBHost, bFileSize);
    ReadFile(GetGoldenDir() + "/quant_gm.bin", quantFileSize, srcQuantHost, quantFileSize);

    aclrtMemcpy(srcADevice, aFileSize, srcAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcBDevice, bFileSize, srcBHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcQuantDevice, quantFileSize, srcQuantHost, quantFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&ffts, &fftsLen);

    LaunchTPushPopFixpipeA2A3<key>((uint8_t *)ffts, dstDevice, srcADevice, srcBDevice, srcQuantDevice, fifoMemDevice,
                                   stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcADevice);
    aclrtFree(srcBDevice);
    aclrtFree(srcQuantDevice);
    aclrtFree(fifoMemDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcAHost);
    aclrtFreeHost(srcBHost);
    aclrtFreeHost(srcQuantHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    const size_t elementCount = static_cast<size_t>(m) * n;
    std::vector<aclFloat16> golden(elementCount);
    std::vector<aclFloat16> devFinal(elementCount);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

template <int32_t key>
void TPushPopFixpipeF322BF16A2A3TestFunc(uint32_t m, uint32_t k, uint32_t n)
{
    size_t aFileSize = static_cast<size_t>(m) * k * sizeof(aclFloat16);
    size_t bFileSize = static_cast<size_t>(k) * n * sizeof(aclFloat16);
    size_t cFileSize = static_cast<size_t>(m) * n * sizeof(aclFloat16);
    size_t fifoFileSize = 2 * static_cast<size_t>(m) * n * sizeof(aclFloat16);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost = nullptr;
    uint8_t *srcAHost = nullptr;
    uint8_t *srcBHost = nullptr;
    uint8_t *dstDevice = nullptr;
    uint8_t *srcADevice = nullptr;
    uint8_t *srcBDevice = nullptr;
    uint8_t *fifoMemDevice = nullptr;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&srcAHost), aFileSize);
    aclrtMallocHost((void **)(&srcBHost), bFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&fifoMemDevice, fifoFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile(GetGoldenDir() + "/x1_gm.bin", aFileSize, srcAHost, aFileSize);
    ReadFile(GetGoldenDir() + "/x2_gm.bin", bFileSize, srcBHost, bFileSize);

    aclrtMemcpy(srcADevice, aFileSize, srcAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(srcBDevice, bFileSize, srcBHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);

    uint64_t ffts{0};
    uint32_t fftsLen{0};
    rtGetC2cCtrlAddr(&ffts, &fftsLen);

    LaunchTPushPopFixpipeA2A3<key>((uint8_t *)ffts, dstDevice, srcADevice, srcBDevice, nullptr, fifoMemDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile(GetGoldenDir() + "/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcADevice);
    aclrtFree(srcBDevice);
    aclrtFree(fifoMemDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcAHost);
    aclrtFreeHost(srcBHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    const size_t elementCount = static_cast<size_t>(m) * n;
    std::vector<aclFloat16> golden(elementCount);
    std::vector<aclFloat16> devFinal(elementCount);
    ReadFile(GetGoldenDir() + "/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile(GetGoldenDir() + "/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    EXPECT_TRUE(ret);
}

TEST_F(TPushPopFixpipeA2A3Test, case1_matmul_64x64_f322f16_tadds)
{
    TPushPopFixpipeA2A3TestFunc<1>(64, 64, 64);
}

TEST_F(TPushPopFixpipeA2A3Test, case2_matmul_64x64_f322bf16)
{
    TPushPopFixpipeF322BF16A2A3TestFunc<2>(64, 64, 64);
}

TEST_F(TPushPopFixpipeA2A3Test, case3_matmul_128x128_deqf16_tadds)
{
    TPushPopFixpipeDEQF16A2A3TestFunc<3>(128, 128, 128);
}

TEST_F(TPushPopFixpipeA2A3Test, case4_matmul_128x128_vdeqf16_tadds)
{
    TPushPopFixpipeVDEQF16A2A3TestFunc<4>(128, 128, 128);
}
