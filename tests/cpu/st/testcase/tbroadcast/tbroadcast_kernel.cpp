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

/**
 * @tparam T Data type
 * @tparam D0, D1, D2, D3, D4 5D Shape
 * @tparam KN Number of copies (Group size)
 */
template <typename T, int D0, int D1, int D2, int D3, int D4, int KN>
AICORE void runTBroadcast(__gm__ T __out__ *out, __gm__ T __in__ *src)
{
    using FullShape = Shape<D0, D1, D2, D3, D4>;
    using FullStride = Stride<D1 * D2 * D3 * D4, D2 * D3 * D4, D3 * D4, D4, 1>;
    using GTf = GlobalTensor<T, FullShape, FullStride>;

    GTf tensors[KN];
    size_t sizePerCopy = D0 * D1 * D2 * D3 * D4;
    for (int i = 0; i < KN; ++i) {
        tensors[i] = GTf(out + (i * sizePerCopy));
    }
    comm::ParallelGroup<GTf> group(tensors, KN, 0);

    GTf srcGlobal(src);

    using TileT = Tile<TileType::Vec, T, D3, D4, BLayout::RowMajor, -1, -1>;
    TileT tempTile(D3, D4);

    TBROADCAST(group, srcGlobal, tempTile);
}

template <typename T, int D0, int D1, int D2, int D3, int D4, int KN>
void LaunchTBroadcast(T *out, T *src, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTBroadcast<T, D0, D1, D2, D3, D4, KN>((half *)(out), (half *)(src));
    else
        runTBroadcast<T, D0, D1, D2, D3, D4, KN>(out, src);
}

template void LaunchTBroadcast<float, 1, 2, 4, 64, 64, 5>(float *, float *, void *);
template void LaunchTBroadcast<int32_t, 1, 2, 4, 64, 64, 3>(int32_t *, int32_t *, void *);
template void LaunchTBroadcast<int16_t, 2, 2, 3, 64, 64, 2>(int16_t *, int16_t *, void *);
template void LaunchTBroadcast<aclFloat16, 1, 2, 1, 16, 256, 1>(aclFloat16 *, aclFloat16 *, void *);
#ifdef CPU_SIM_BFLOAT_ENABLED
template void LaunchTBroadcast<bfloat16_t, 1, 2, 1, 16, 256, 1>(bfloat16_t *, bfloat16_t *, void *);
#endif
