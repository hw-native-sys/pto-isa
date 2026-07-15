/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, bool isInPlace = false>
__global__ AICORE void runTRsqrt(__gm__ T __out__* out, __gm__ T __in__* src)
{
    using DynShapeDim5 = Shape<1, 1, 1, kGRows_, kGCols_>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kGCols_, 1>;
    using GlobalData = GlobalTensor<T, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, T, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;
    TileData srcTile(kTRows_, kTCols_);
    TileData dstTile(kTRows_, kTCols_);
    Tile<TileType::Vec, T, 1, BLOCK_BYTE_SIZE / sizeof(T)> tmpTile;
    TASSIGN(srcTile, 0x0);
    if constexpr (isInPlace) {
        TASSIGN(dstTile, 0x0);
        TASSIGN(tmpTile, kTRows_ * kTCols_ * sizeof(T));
    } else {
        TASSIGN(dstTile, kTRows_ * kTCols_ * sizeof(T));
    }

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    Event<Op::TLOAD, Op::TRSQRT> e0;
    Event<Op::TRSQRT, Op::TSTORE_VEC> e1;

    e0 = TLOAD(srcTile, srcGlobal);
    if constexpr (isInPlace) {
        e1 = TRSQRT(dstTile, srcTile, tmpTile, e0);
    } else {
        e1 = TRSQRT(dstTile, srcTile, e0);
    }
    TSTORE(dstGlobal, dstTile, e1);
    out = dstGlobal.data();
}

template <typename T, int kGRows_, int kGCols_, int kTRows_, int kTCols_, bool isInPlace = false>
void LaunchTRsqrt(T* out, T* src, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTRsqrt<half, kGRows_, kGCols_, kTRows_, kTCols_, isInPlace>
            <<<1, nullptr, stream>>>((half*)(out), (half*)(src));
    else
        runTRsqrt<T, kGRows_, kGCols_, kTRows_, kTCols_, isInPlace><<<1, nullptr, stream>>>(out, src);
}

template void LaunchTRsqrt<float, 64, 64, 64, 64, true>(float* out, float* src, void* stream);
template void LaunchTRsqrt<float, 64, 64, 64, 64, false>(float* out, float* src, void* stream);
template void LaunchTRsqrt<aclFloat16, 64, 64, 64, 64, true>(aclFloat16* out, aclFloat16* src, void* stream);
template void LaunchTRsqrt<aclFloat16, 64, 64, 64, 64, false>(aclFloat16* out, aclFloat16* src, void* stream);
