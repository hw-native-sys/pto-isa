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

template <typename T, typename U, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC, int dstTR,
          int dstTC, int src0TR, int src0TC, int src1TR, int src1TC>
__global__ AICORE void runTPartArgMax(__gm__ T __out__ *out, __gm__ T __in__ *src0, __gm__ T __in__ *src1,
                                      __gm__ U __out__ *outIdx, __gm__ U __in__ *src0Idx, __gm__ U __in__ *src1Idx)
{
    using GlobalDataDst = GlobalTensor<T, Shape<1, 1, 1, dstVR, dstVC>, pto::Stride<1, 1, dstTR, dstTC, 1>>;
    using GlobalDataSrc0 = GlobalTensor<T, Shape<1, 1, 1, src0VR, src0VC>, pto::Stride<1, 1, src0TR, src0TC, 1>>;
    using GlobalDataSrc1 = GlobalTensor<T, Shape<1, 1, 1, src1VR, src1VC>, pto::Stride<1, 1, src1TR, src1TC, 1>>;
    using GlobalDataDstIdx = GlobalTensor<U, Shape<1, 1, 1, dstVR, dstVC>, pto::Stride<1, 1, dstTR, dstTC, 1>>;
    using GlobalDataSrc0Idx = GlobalTensor<U, Shape<1, 1, 1, src0VR, src0VC>, pto::Stride<1, 1, src0TR, src0TC, 1>>;
    using GlobalDataSrc1Idx = GlobalTensor<U, Shape<1, 1, 1, src1VR, src1VC>, pto::Stride<1, 1, src1TR, src1TC, 1>>;

    using TileDataDst = Tile<TileType::Vec, T, dstTR, dstTC, BLayout::RowMajor, -1, -1>;
    using TileDataSrc0 = Tile<TileType::Vec, T, src0TR, src0TC, BLayout::RowMajor, -1, -1>;
    using TileDataSrc1 = Tile<TileType::Vec, T, src1TR, src1TC, BLayout::RowMajor, -1, -1>;
    using TileDataDstIdx = Tile<TileType::Vec, U, dstTR, dstTC, BLayout::RowMajor, -1, -1>;
    using TileDataSrc0Idx = Tile<TileType::Vec, U, src0TR, src0TC, BLayout::RowMajor, -1, -1>;
    using TileDataSrc1Idx = Tile<TileType::Vec, U, src1TR, src1TC, BLayout::RowMajor, -1, -1>;

    TileDataSrc0 src0Tile(src0VR, src0VC);
    TileDataSrc1 src1Tile(src1VR, src1VC);
    TileDataDst dstTile(dstVR, dstVC);
    TileDataSrc0Idx src0TileIdx(src0VR, src0VC);
    TileDataSrc1Idx src1TileIdx(src1VR, src1VC);
    TileDataDstIdx dstTileIdx(dstVR, dstVC);

    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, src0TR * src0TC * sizeof(T));
    TASSIGN(dstTile, (src0TR * src0TC + src1TR * src1TC) * sizeof(T));
    TASSIGN(src0TileIdx, (src0TR * src0TC + src1TR * src1TC + dstTR * dstTC) * sizeof(T));
    TASSIGN(src1TileIdx, (src0TR * src0TC + src1TR * src1TC + dstTR * dstTC) * sizeof(T) + src0TR * src0TC * sizeof(U));
    TASSIGN(dstTileIdx, (src0TR * src0TC + src1TR * src1TC + dstTR * dstTC) * sizeof(T) +
                            (src0TR * src0TC + src1TR * src1TC) * sizeof(U));

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataDst dstGlobal(out);
    GlobalDataSrc0Idx src0GlobalIdx(src0Idx);
    GlobalDataSrc1Idx src1GlobalIdx(src1Idx);
    GlobalDataDstIdx dstGlobalIdx(outIdx);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TLOAD(src0TileIdx, src0GlobalIdx);
    TLOAD(src1TileIdx, src1GlobalIdx);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TPARTARGMAX<TileDataDst, TileDataSrc0, TileDataSrc1, TileDataDstIdx, TileDataSrc0Idx, TileDataSrc1Idx>(
        dstTile, src0Tile, src1Tile, dstTileIdx, src0TileIdx, src1TileIdx);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);
    TSTORE(dstGlobalIdx, dstTileIdx);
    out = dstGlobal.data();
    outIdx = dstGlobalIdx.data();
}

template <typename T, typename U, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC, int dstTR,
          int dstTC, int src0TR, int src0TC, int src1TR, int src1TC, bool isHalf = false>
void LaunchTPartArgMax(T *out, T *src0, T *src1, U *outIdx, U *src0Idx, U *src1Idx, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16> && isHalf == true) {
        runTPartArgMax<half, U, dstVR, dstVC, src0VR, src0VC, src1VR, src1VC, dstTR, dstTC, src0TR, src0TC, src1TR,
                       src1TC>
            <<<1, nullptr, stream>>>((half *)out, (half *)src0, (half *)src1, outIdx, src0Idx, src1Idx);
    } else {
        runTPartArgMax<T, U, dstVR, dstVC, src0VR, src0VC, src1VR, src1VC, dstTR, dstTC, src0TR, src0TC, src1TR, src1TC>
            <<<1, nullptr, stream>>>(out, src0, src1, outIdx, src0Idx, src1Idx);
    }
}

template void LaunchTPartArgMax<float, uint32_t, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64>(
    float *out, float *src0, float *src1, uint32_t *outIdx, uint32_t *src0Idx, uint32_t *src1Idx, void *stream);
template void LaunchTPartArgMax<float, int32_t, 2, 24, 2, 24, 2, 8, 4, 32, 3, 24, 2, 16>(
    float *out, float *src0, float *src1, int32_t *outIdx, int32_t *src0Idx, int32_t *src1Idx, void *stream);
template void LaunchTPartArgMax<float, int32_t, 12, 63, 12, 63, 6, 60, 12, 64, 12, 64, 6, 64>(
    float *out, float *src0, float *src1, int32_t *outIdx, int32_t *src0Idx, int32_t *src1Idx, void *stream);
template void LaunchTPartArgMax<aclFloat16, int16_t, 10, 31, 8, 16, 10, 31, 10, 32, 8, 32, 12, 32, true>(
    aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, int16_t *outIdx, int16_t *src0Idx, int16_t *src1Idx,
    void *stream);
template void LaunchTPartArgMax<aclFloat16, uint16_t, 5, 33, 5, 33, 5, 30, 8, 48, 5, 48, 6, 48, true>(
    aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, uint16_t *outIdx, uint16_t *src0Idx, uint16_t *src1Idx,
    void *stream);
