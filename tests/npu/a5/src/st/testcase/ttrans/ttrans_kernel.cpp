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
#include <pto/common/debug.h>
#include "acl/acl.h"

using namespace pto;

template <typename T, int dstTRows, int dstTCols, int srcTRows, int srcTCols>
__global__ AICORE void runTTRANS(__gm__ T __out__ *out, __gm__ T __in__ *src, int vRows, int vCols)
{
    using DynShapeSrc = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStrideSrc = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalDataSrc = GlobalTensor<T, DynShapeSrc, DynStrideSrc>;

    using DynShapeDst = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStrideDst = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalDataDst = GlobalTensor<T, DynShapeDst, DynStrideDst>;

    using TileDataSrc = Tile<TileType::Vec, T, srcTRows, srcTCols, BLayout::RowMajor, -1, -1>;
    using TileDataDst = Tile<TileType::Vec, T, dstTRows, dstTCols, BLayout::RowMajor, -1, -1>;
    using TileDataTmp = Tile<TileType::Vec, T, dstTRows, dstTCols, BLayout::RowMajor, dstTRows, dstTCols>;

    TileDataSrc srcTile(vRows, vCols);
    TileDataDst dstTile(vCols, vRows);
    TileDataTmp tmpTile;

    constexpr size_t srcUBAddr = 0;
    constexpr size_t srcUBSize = srcTRows * srcTCols * sizeof(T);
    constexpr size_t dstUBAddr = srcUBSize;
    constexpr size_t dstUBSize = dstTRows * dstTCols * sizeof(T);
    static_assert(dstUBAddr >= srcUBAddr + srcUBSize, "src and dst UB address ranges overlap");
    static_assert(srcUBAddr != dstUBAddr || srcUBSize == 0, "src and dst share same UB address");
    static_assert(srcUBSize + dstUBSize <= 256u * 1024u, "total UB usage exceeds A5 256KB UB limit");

    TASSIGN(srcTile, srcUBAddr);
    TASSIGN(dstTile, dstUBAddr);
    TASSIGN(tmpTile, 0);

    GlobalDataSrc srcGlobal(src, pto::Shape(1, 1, 1, vRows, vCols), pto::Stride(1, 1, 1, srcTCols, 1));
    GlobalDataDst dstGlobal(out, pto::Shape(1, 1, 1, vCols, vRows), pto::Stride(1, 1, 1, dstTCols, 1));

    Event<Op::TLOAD, Op::TTRANS> event0;
    Event<Op::TTRANS, Op::TSTORE_VEC> event1;

    event0 = TLOAD(srcTile, srcGlobal);
    event1 = TTRANS(dstTile, srcTile, tmpTile, event0);
    TSTORE(dstGlobal, dstTile, event1);
}

template <typename T, int dstTRows, int dstTCols, int srcTRows, int srcTCols, int vRows, int vCols>
void LaunchTTRANS(T *out, T *src, void *stream)
{
    runTTRANS<T, dstTRows, dstTCols, srcTRows, srcTCols><<<1, nullptr, stream>>>(out, src, vRows, vCols);
}

template <int dstTRows, int dstTCols, int srcTRows, int srcTCols, int vRows, int vCols>
void LaunchTTRANSHalf(aclFloat16 *out, aclFloat16 *src, void *stream)
{
    runTTRANS<half, dstTRows, dstTCols, srcTRows, srcTCols>
        <<<1, nullptr, stream>>>((half *)(out), (half *)(src), vRows, vCols);
}

template void LaunchTTRANS<float, 8, 8, 2, 8, 2, 8>(float *out, float *src, void *stream);
template void LaunchTTRANSHalf<16, 16, 16, 16, 16, 16>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTTRANS<float, 16, 32, 32, 16, 31, 15>(float *out, float *src, void *stream);
template void LaunchTTRANSHalf<32, 32, 32, 32, 31, 31>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTTRANS<float, 8, 8, 4, 8, 4, 8>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 512, 16, 9, 512, 9, 512>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 66, 88, 9, 16, 7, 15>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 16, 32, 32, 16, 23, 15>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 128, 64, 64, 128, 27, 77>(float *out, float *src, void *stream);
template void LaunchTTRANSHalf<64, 112, 100, 64, 64, 64>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTTRANSHalf<64, 128, 128, 64, 64, 64>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTTRANSHalf<64, 128, 128, 64, 100, 64>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTTRANS<float, 32, 512, 512, 32, 512, 2>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 16, 8, 1, 16, 1, 16>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 64, 64, 64, 64, 36, 64>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 8, 8, 8, 8, 8, 8>(float *out, float *src, void *stream);
template void LaunchTTRANS<uint8_t, 32, 32, 32, 32, 32, 32>(uint8_t *out, uint8_t *src, void *stream);
template void LaunchTTRANS<uint8_t, 64, 64, 64, 64, 22, 63>(uint8_t *out, uint8_t *src, void *stream);
template void LaunchTTRANS<float, 8, 8, 1, 8, 1, 8>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 8, 536, 532, 8, 532, 8>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 8, 624, 618, 8, 618, 8>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 8, 536, 532, 8, 400, 4>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 128, 128, 128, 128, 100, 100>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 8, 256, 256, 8, 256, 8>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 8, 512, 512, 8, 512, 8>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 256, 8, 8, 256, 8, 256>(float *out, float *src, void *stream);
template void LaunchTTRANSHalf<16, 256, 256, 16, 256, 16>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTTRANS<uint8_t, 64, 256, 256, 64, 200, 60>(uint8_t *out, uint8_t *src, void *stream);
template void LaunchTTRANS<float, 8, 640, 640, 8, 640, 8>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 8, 1024, 1024, 8, 1024, 8>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 8, 4096, 4096, 8, 4096, 8>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 8, 2048, 2048, 8, 1500, 4>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 8, 520, 513, 8, 513, 8>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 8, 448, 448, 8, 300, 6>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 1024, 8, 1, 1024, 1, 1024>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 512, 8, 2, 512, 2, 512>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 2048, 8, 8, 2048, 8, 2048>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 176, 176, 176, 176, 150, 150>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 256, 64, 64, 256, 50, 200>(float *out, float *src, void *stream);
template void LaunchTTRANSHalf<16, 512, 512, 16, 512, 16>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTTRANSHalf<16, 2064, 2064, 16, 2064, 16>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTTRANSHalf<512, 16, 16, 512, 10, 400>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTTRANSHalf<128, 128, 128, 128, 128, 128>(aclFloat16 *out, aclFloat16 *src, void *stream);
template void LaunchTTRANS<uint8_t, 32, 1024, 1024, 32, 1024, 32>(uint8_t *out, uint8_t *src, void *stream);
template void LaunchTTRANS<uint8_t, 1024, 32, 32, 1024, 32, 1024>(uint8_t *out, uint8_t *src, void *stream);
template void LaunchTTRANS<uint8_t, 64, 128, 128, 64, 100, 50>(uint8_t *out, uint8_t *src, void *stream);
template void LaunchTTRANS<float, 8, 64, 64, 8, 1, 1>(float *out, float *src, void *stream);
template void LaunchTTRANS<float, 8, 16, 16, 8, 16, 1>(float *out, float *src, void *stream);