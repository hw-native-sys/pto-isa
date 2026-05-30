/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

template <typename T, int kGRows_, int kGCols_>
AICORE void runTPutAsync(__gm__ T __out__ *out, __gm__ T __in__ *input)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;

    pto::comm::AsyncSession session;

    GlobalData inputGlobal(input);
    GlobalData outputGlobal(out);

    comm::TPUT_ASYNC(outputGlobal, inputGlobal, session);
    out = outputGlobal.data();
}

template <typename T, int kGRows_, int kGCols_>
void LaunchTPutAsync(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTPutAsync<half, kGRows_, kGCols_>((half *)(out), (half *)(src));
    else
        runTPutAsync<T, kGRows_, kGCols_>(out, src);
}

template void LaunchTPutAsync<float, 64, 64>(float *out, float *src, void *stream);
template void LaunchTPutAsync<int32_t, 64, 64>(int32_t *out, int32_t *src, void *stream);
template void LaunchTPutAsync<aclFloat16, 16, 256>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTPutAsync<int16_t, 64, 64>(int16_t *out, int16_t *src, void *stream);
