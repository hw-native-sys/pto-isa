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
void launchTopk(uint8_t *out, uint8_t *index, uint8_t *src, uint8_t *inIdx, void *stream);

template <typename T>
inline bool ValidateDataResults(size_t outFileSize)
{
    std::vector<T> golden(outFileSize);
    std::vector<T> devFinal(outFileSize);

    ReadFile("../output/golden_d.bin", outFileSize, golden.data(), outFileSize);
    ReadFile("../output/output_z.bin", outFileSize, devFinal.data(), outFileSize);

    bool ret = ResultCmp(golden, devFinal, 0.001f);
    if (ret) {
        printf("test data success\n");
    } else {
        printf("test data failed\n");
    }
    return ret;
}

inline bool ValidateIndexResults(size_t indexFileSize)
{
    std::vector<uint32_t> golden_i(indexFileSize);
    std::vector<uint32_t> devFinal_i(indexFileSize);

    ReadFile("../output/golden_i.bin", indexFileSize, golden_i.data(), indexFileSize);
    ReadFile("../output/index_z.bin", indexFileSize, devFinal_i.data(), indexFileSize);

    bool ret = ResultCmp(golden_i, devFinal_i, 0.001f);
    if (ret) {
        printf("test index success\n");
    } else {
        printf("test index failed\n");
    }
    return ret;
}

template <typename T, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
          int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, int topk>
void Topk()
{
    constexpr int rows = gWholeShape0 * gWholeShape1 * gWholeShape2 * gWholeShape3;
    constexpr int cols = gWholeShape4;
    constexpr int valid_row = gShape0 * gShape1 * gShape2 * gShape3;
    using indexT = uint32_t;
    size_t inFileSize = rows * cols * sizeof(T);
    size_t inIdxSize = cols * sizeof(indexT);
    constexpr int TYPE_COEF = sizeof(float) / sizeof(T);
    size_t outFileSize = valid_row * topk * sizeof(T);
    size_t indexFileSize = valid_row * topk * sizeof(indexT);

    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    uint8_t *dstHost, *srcHost, *indexHost, *inIdxHost;
    uint8_t *dstDevice, *srcDevice, *indexDevice, *inIdxDevice;

    aclrtMallocHost((void **)(&dstHost), outFileSize);
    aclrtMallocHost((void **)(&indexHost), indexFileSize);
    aclrtMallocHost((void **)(&srcHost), inFileSize);
    aclrtMallocHost((void **)(&inIdxHost), inIdxSize);

    aclrtMalloc((void **)&dstDevice, outFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&srcDevice, inFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&indexDevice, indexFileSize, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void **)&inIdxDevice, inIdxSize, ACL_MEM_MALLOC_HUGE_FIRST);

    ReadFile("../input/x1_gm.bin", inFileSize, srcHost, inFileSize);
    ReadFile("../input/x1_idx.bin", inIdxSize, inIdxHost, inIdxSize);

    aclrtMemcpy(srcDevice, inFileSize, srcHost, inFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(inIdxDevice, inIdxSize, inIdxHost, inIdxSize, ACL_MEMCPY_HOST_TO_DEVICE);
    launchTopk<T>(dstDevice, indexDevice, srcDevice, inIdxDevice, stream);

    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstHost, outFileSize, dstDevice, outFileSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(indexHost, indexFileSize, indexDevice, indexFileSize, ACL_MEMCPY_DEVICE_TO_HOST);

    WriteFile("../output/output_z.bin", dstHost, outFileSize);
    WriteFile("../output/index_z.bin", indexHost, indexFileSize);

    aclrtFree(dstDevice);
    aclrtFree(srcDevice);
    aclrtFree(indexDevice);
    aclrtFree(inIdxDevice);

    aclrtFreeHost(dstHost);
    aclrtFreeHost(srcHost);
    aclrtFreeHost(indexHost);
    aclrtFreeHost(inIdxHost);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    bool dataSuccess = ValidateDataResults<T>(outFileSize);
    bool indexSuccess = ValidateIndexResults(indexFileSize);
    if (dataSuccess && indexSuccess) {
        printf("All tests passed!\n");
    } else {
        printf("Some tests failed!\n");
    }
}

int main()
{
    constexpr int gShape3 = 4800;
    constexpr int gShape4 = 1024;
    constexpr int gWholeShape3 = 4800;
    constexpr int gWholeShape4 = 1280;
    constexpr int topk = 1000;
    Topk<float, 1, 1, 1, gShape3, gShape4, 1, 1, 1, gWholeShape3, gWholeShape4, topk>();
}
