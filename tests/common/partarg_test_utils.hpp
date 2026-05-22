/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PARTARG_TEST_UTILS_HPP
#define PARTARG_TEST_UTILS_HPP

#include <cstdint>
#include "test_common.h"

namespace pto {

template <typename T>
void AllocateAndLoadData(T *&host, T *&device, size_t size, const std::string &filename)
{
    aclrtMallocHost((void **)(&host), size);
    aclrtMalloc((void **)&device, size, ACL_MEM_MALLOC_HUGE_FIRST);
    size_t readSize = 0;
    CHECK_RESULT_GTEST(PtoTestCommon::ReadFile(filename, readSize, host, size));
    aclrtMemcpy(device, size, host, size, ACL_MEMCPY_HOST_TO_DEVICE);
}

template <typename LaunchFunc>
void RunPartArgKernelAndGetResults(LaunchFunc launchFunc, aclrtStream stream, float *dstValDevice, float *src0ValDevice,
                                   float *src1ValDevice, uint32_t *dstIdxDevice, uint32_t *src0IdxDevice,
                                   uint32_t *src1IdxDevice, float *dstValHost, uint32_t *dstIdxHost, size_t valSize,
                                   size_t idxSize)
{
    launchFunc(dstValDevice, src0ValDevice, src1ValDevice, dstIdxDevice, src0IdxDevice, src1IdxDevice, stream);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(dstValHost, valSize, dstValDevice, valSize, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(dstIdxHost, idxSize, dstIdxDevice, idxSize, ACL_MEMCPY_DEVICE_TO_HOST);
}

static inline bool VerifyPartArgResults(const std::string &goldenDir, size_t valSize, size_t idxSize, int rows,
                                        int cols, float epsilon)
{
    std::vector<float> golden_val(rows * cols);
    std::vector<float> out_val(rows * cols);
    std::vector<uint32_t> golden_idx(rows * cols);
    std::vector<uint32_t> out_idx(rows * cols);
    size_t readSize = 0;
    PtoTestCommon::ReadFile(goldenDir + "/golden_val.bin", readSize, golden_val.data(), valSize);
    PtoTestCommon::ReadFile(goldenDir + "/output_val.bin", readSize, out_val.data(), valSize);
    PtoTestCommon::ReadFile(goldenDir + "/golden_idx.bin", readSize, golden_idx.data(), idxSize);
    PtoTestCommon::ReadFile(goldenDir + "/output_idx.bin", readSize, out_idx.data(), idxSize);
    return PtoTestCommon::ResultCmp<float>(golden_val, out_val, epsilon) &&
           PtoTestCommon::ResultCmp<uint32_t>(golden_idx, out_idx, 0.0f);
}
} // namespace pto
#endif