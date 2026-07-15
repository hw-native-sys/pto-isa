/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#include <acl/acl.h>
#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace std;
using namespace pto;

template <typename T>
PTO_INTERNAL void runTASSIGN(__gm__ T** out, __gm__ T* src, int offset)
{
    using StaticDim2Shape = Shape<1, 1, 1, 8, 16>;
    using StaticDim2Stride = pto::Stride<1, 1, 512, 32, 1>;
    using GlobalData = GlobalTensor<T, StaticDim2Shape, StaticDim2Stride>;
    GlobalData srcGlobal(src, StaticDim2Shape(), StaticDim2Stride());

    TASSIGN(srcGlobal, src + offset);
    *out = srcGlobal.data();
}

extern "C" __global__ AICORE void launchTASSIGNCase1(__gm__ float** out, __gm__ float* src, int offset)
{
    runTASSIGN<float>(out, src, offset);
}

template <uint32_t caseId>
void launchTASSIGNTestCase(void* out, void* src, int offset, aclrtStream stream)
{
    switch (caseId) {
        case 1: {
            launchTASSIGNCase1<<<1, nullptr, stream>>>((float**)out, (float*)src, offset);
            break;
        }
        default: {
        }
    }
}

template void launchTASSIGNTestCase<1>(void* out, void* src, int offset, aclrtStream stream);