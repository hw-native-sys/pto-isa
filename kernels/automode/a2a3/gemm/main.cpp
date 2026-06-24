/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "acl/acl.h"
#include "test_common.h"
using namespace std;
using namespace PtoTestCommon;

template <typename T>
void LaunchGEMME2E(uint8_t *out, uint8_t *src0, uint8_t *src1, void *stream);

template <typename T, typename U, typename S, uint32_t blockDim, uint32_t m, uint32_t k, uint32_t n,
          uint32_t singleCoreM, uint32_t singleCoreK, uint32_t singleCoreN, uint32_t baseM, uint32_t baseK,
          uint32_t baseN, uint32_t stepM, uint32_t stepKa, uint32_t stepKb, uint32_t stepN>
void GemmE2E()
{
    size_t aFileSize = m * k * sizeof(U); // uint16_t represent half
    size_t bFileSize = k * n * sizeof(S); // uint16_t represent half
    size_t cFileSize = m * n * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void **)(&dstHost), cFileSize);
    aclrtMallocHost((void **)(&src0Host), aFileSize);
    aclrtMallocHost((void **)(&src1Host), bFileSize);

    aclrtMalloc((void **)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile("../input/x2_gm.bin", bFileSize, src1Host, bFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchGEMME2E<U>(dstDevice, src0Device, src1Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    std::vector<float> golden(cFileSize);
    std::vector<float> devFinal(cFileSize);
    ReadFile("../output/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile("../output/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    if (ret) {
        printf("test success\n");
    } else {
        printf("test failed\n");
    }
}

int main()
{
    constexpr uint32_t m = 6144;
    constexpr uint32_t k = 6144;
    constexpr uint32_t n = 6144;
    constexpr uint32_t singleCoreM = 1536;
    constexpr uint32_t singleCoreK = 6144;
    constexpr uint32_t singleCoreN = 1024;
    constexpr uint32_t blockDim = 24;
    constexpr uint32_t baseM = 128;
    constexpr uint32_t baseK = 64;
    constexpr uint32_t baseN = 256;
    constexpr uint32_t stepM = 1;
    constexpr uint32_t stepKa = 4;
    constexpr uint32_t stepKb = 4;
    constexpr uint32_t stepN = 1;

    GemmE2E<float, uint16_t, uint16_t, blockDim, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM, baseK, baseN,
            stepM, stepKa, stepKb, stepN>();
}