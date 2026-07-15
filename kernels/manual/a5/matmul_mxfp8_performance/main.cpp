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
using namespace std;
using namespace PtoTestCommon;

void LaunchMxMatmul(uint8_t* out, uint8_t* src0, uint8_t* src1, uint8_t* src2, uint8_t* src3, void* stream);

template <typename T>
void VerifyResult(size_t cFileSize)
{
    std::vector<T> golden(cFileSize);
    std::vector<T> devFinal(cFileSize);
    ReadFile("../output/golden.bin", cFileSize, golden.data(), cFileSize);
    ReadFile("../output/output_z.bin", cFileSize, devFinal.data(), cFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    if (ret) {
        printf("test success\n");
    } else {
        printf("test failed\n");
    }
}

template <
    typename T, typename U, typename X, uint32_t blockDim, uint32_t m, uint32_t k, uint32_t n, uint32_t singleCoreM,
    uint32_t singleCoreK, uint32_t singleCoreN, uint32_t baseM, uint32_t baseK, uint32_t baseN, uint32_t stepM,
    uint32_t stepKa, uint32_t stepKb, uint32_t stepN>
void MxMatmul()
{
    size_t aFileSize = m * k * sizeof(U); // uint8_t represent fp8
    size_t bFileSize = k * n * sizeof(U);
    int sacleFactor = 32;
    size_t aScaleFileSize = m * k / sacleFactor * sizeof(X);
    size_t bScaleFileSize = k / sacleFactor * n * sizeof(X);
    size_t cFileSize = m * n * sizeof(T); // uint16_t represent bf16

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host, *src2Host, *src3Host;
    uint8_t *dstDevice, *src0Device, *src1Device, *src2Device, *src3Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);
    aclrtMallocHost((void**)(&src2Host), aScaleFileSize);
    aclrtMallocHost((void**)(&src3Host), bScaleFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src2Device, aScaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src3Device, bScaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile("../input/x2_gm.bin", bFileSize, src1Host, bFileSize);
    ReadFile("../input/x1_scale_gm.bin", aScaleFileSize, src2Host, aScaleFileSize);
    ReadFile("../input/x2_scale_gm.bin", bScaleFileSize, src3Host, bScaleFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src2Device, aScaleFileSize, src2Host, aScaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src3Device, bScaleFileSize, src3Host, bScaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    LaunchMxMatmul(dstDevice, src0Device, src1Device, src2Device, src3Device, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, cFileSize, dstDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_z.bin", dstHost, cFileSize);

    aclrtFree(dstDevice);
    aclrtFree(src0Device);
    aclrtFree(src1Device);
    aclrtFree(src2Device);
    aclrtFree(src3Device);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(src0Host);
    aclrtFreeHost(src1Host);
    aclrtFreeHost(src2Host);
    aclrtFreeHost(src3Host);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    VerifyResult<T>(cFileSize);
}

int main()
{
    constexpr uint32_t m = 6144;
    constexpr uint32_t k = 6144;
    constexpr uint32_t n = 6144;
    constexpr uint32_t singleCoreM = 1536;
    constexpr uint32_t singleCoreK = 6144;
    constexpr uint32_t singleCoreN = 768;
    constexpr uint32_t blockDim = 32;
    constexpr uint32_t baseM = 128;
    constexpr uint32_t baseK = 128;
    constexpr uint32_t baseN = 256;
    constexpr uint32_t stepM = 1;
    constexpr uint32_t stepKa = 4;
    constexpr uint32_t stepKb = 4;
    constexpr uint32_t stepN = 1;

    MxMatmul<
        uint16_t, uint8_t, uint8_t, blockDim, m, k, n, singleCoreM, singleCoreK, singleCoreN, baseM, baseK, baseN,
        stepM, stepKa, stepKb, stepN>();
}
