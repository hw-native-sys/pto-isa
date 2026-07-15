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
#include "acl/acl.h"
using namespace std;
using namespace PtoTestCommon;

template <typename T>
void launchConv2dForward(uint8_t* out, uint8_t* src0, uint8_t* src1, void* stream);

template <typename T, typename U>
void conv2d_forward(
    uint32_t fmapN, uint32_t fmapC1, uint32_t fmapH, uint32_t fmapW, uint32_t fmapC0, uint32_t filterDim3,
    uint32_t filterDim2, uint32_t filterDim1, uint32_t filterDim0, uint32_t hk, uint32_t wk, uint8_t dilationH = 1,
    uint8_t dilationW = 1, uint8_t strideH = 1, uint8_t strideW = 1, uint8_t padTop = 1, uint8_t padBottom = 1,
    uint8_t padLeft = 1, uint8_t padRight = 1)
{
    uint32_t widthOut = 0;
    uint32_t heightOut = 0;
    if (strideH != 0 && strideW != 0) {
        heightOut = (fmapH + padTop + padBottom - dilationH * (hk - 1) - 1) / strideH + 1;
        widthOut = (fmapW + padLeft + padRight - dilationW * (wk - 1) - 1) / strideW + 1;
    }
    size_t aFileSize = fmapN * fmapC1 * fmapH * fmapW * fmapC0 * sizeof(U);
    size_t bFileSize = filterDim3 * filterDim2 * filterDim1 * filterDim0 * sizeof(U);
    size_t cFileSize = fmapN * heightOut * widthOut * filterDim2 * filterDim1 * sizeof(T);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *src0Host, *src1Host;
    uint8_t *dstDevice, *src0Device, *src1Device;

    aclrtMallocHost((void**)(&dstHost), cFileSize);
    aclrtMallocHost((void**)(&src0Host), aFileSize);
    aclrtMallocHost((void**)(&src1Host), bFileSize);

    aclrtMalloc((void**)&dstDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src0Device, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&src1Device, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/x1_gm.bin", aFileSize, src0Host, aFileSize);
    ReadFile("../input/x2_gm.bin", bFileSize, src1Host, bFileSize);

    aclrtMemcpy(src0Device, aFileSize, src0Host, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(src1Device, bFileSize, src1Host, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchConv2dForward<U>(dstDevice, src0Device, src1Device, stream);

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

int main()
{
    constexpr uint32_t fmapN = 4;
    constexpr uint32_t fmapC1 = 32;
    constexpr uint32_t fmapH = 16;
    constexpr uint32_t fmapW = 96;
    constexpr uint32_t fmapC0 = 16;
    constexpr uint32_t filterDim3 = 288;
    constexpr uint32_t filterDim2 = 384;
    constexpr uint32_t filterDim1 = 16;
    constexpr uint32_t filterDim0 = 16;
    constexpr uint32_t hk = 3;
    constexpr uint32_t wk = 3;

    conv2d_forward<uint16_t, uint16_t>(
        fmapN, fmapC1, fmapH, fmapW, fmapC0, filterDim3, filterDim2, filterDim1, filterDim0, hk, wk);
}
