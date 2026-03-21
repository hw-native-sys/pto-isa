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
#include "acl/acl.h"

using namespace pto;

template <typename T, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC>
__global__ AICORE void runTPartMul(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1)
{
    constexpr uint16_t alignedSrc0VC = ((src0VC * sizeof(T) + 31) / 32) * (32 / sizeof(T));
    constexpr uint16_t alignedSrc1VC = ((src1VC * sizeof(T) + 31) / 32) * (32 / sizeof(T));
    using GlobalDataDst = GlobalTensor<T, Shape<1, 1, 1, dstVR, dstVC>, pto::Stride<1, 1, 1, dstVC, 1>>;
    using GlobalDataSrc0 = GlobalTensor<T, Shape<1, 1, 1, src0VR, src0VC>, pto::Stride<1, 1, 1, src0VC, 1>>;
    using GlobalDataSrc1 = GlobalTensor<T, Shape<1, 1, 1, src1VR, src1VC>, pto::Stride<1, 1, 1, src1VC, 1>>;
    using TileDataDst = Tile<TileType::Vec, T, dstVR, dstVC, BLayout::RowMajor, -1, -1>;
    using TileDataSrc0 = Tile<TileType::Vec, T, src0VR, alignedSrc0VC, BLayout::RowMajor, -1, -1>;
    using TileDataSrc1 = Tile<TileType::Vec, T, src1VR, alignedSrc1VC, BLayout::RowMajor, -1, -1>;
    TileDataSrc0 src0Tile(src0VR, src0VC);
    TileDataSrc1 src1Tile(src1VR, src1VC);
    TileDataDst dstTile(dstVR, dstVC);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataDst dstGlobal(out);

    Event<Op::TLOAD, Op::TPARTMUL> event0;
    Event<Op::TPARTMUL, Op::TSTORE_VEC> event1;

    TLOAD(src0Tile, src0Global);
    event0 = TLOAD(src1Tile, src1Global);
    event1 = TPARTMUL<TileDataDst, TileDataSrc0, TileDataSrc1>(dstTile, src0Tile, src1Tile, event0);
    TSTORE(dstGlobal, dstTile, event1);
    out = dstGlobal.data();
}

template <typename T, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC>
void LaunchTPartMul(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>)
        runTPartMul<half, dstVR, dstVC, src0VR, src0VC, src1VR, src1VC>
            <<<1, nullptr, stream>>>((half *)(out), (half *)(src0), (half *)(src1));
    else
        runTPartMul<T, dstVR, dstVC, src0VR, src0VC, src1VR, src1VC><<<1, nullptr, stream>>>(out, src0, src1);
}

template void LaunchTPartMul<float, 64, 64, 64, 64, 64, 64>(float *out, float *src0, float *src1, void *stream);
template void LaunchTPartMul<float, 64, 64, 8, 64, 64, 64>(float *out, float *src0, float *src1, void *stream);
template void LaunchTPartMul<float, 64, 64, 64, 8, 64, 64>(float *out, float *src0, float *src1, void *stream);
template void LaunchTPartMul<float, 64, 64, 64, 64, 8, 64>(float *out, float *src0, float *src1, void *stream);
template void LaunchTPartMul<float, 64, 64, 64, 64, 64, 8>(float *out, float *src0, float *src1, void *stream);
template void LaunchTPartMul<aclFloat16, 8, 48, 8, 16, 8, 48>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1,
                                                              void *stream);
template void LaunchTPartMul<aclFloat16, 8, 768, 8, 512, 8, 768>(aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1,
                                                                 void *stream);
template void LaunchTPartMul<int16_t, 8, 48, 8, 48, 8, 16>(int16_t *out, int16_t *src0, int16_t *src1, void *stream);
template void LaunchTPartMul<int32_t, 64, 64, 8, 64, 64, 64>(int32_t *out, int32_t *src0, int32_t *src1, void *stream);
