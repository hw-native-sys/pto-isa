/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <type_traits>
#include <pto/pto-inst.hpp>
#include "acl/acl.h"

using namespace pto;
#define PTO_CEIL(x, y) ((((x) + (y)-1) / (y)) * (y))

namespace TPartMaxTest {

template <typename T, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC, int dstTR, int dstTC,
          int src0TR, int src0TC, int src1TR, int src1TC>
__global__ AICORE void runTPartMax(__gm__ T *out, __gm__ T *src0, __gm__ T *src1)
{
    using GlobalDataDst = GlobalTensor<T, Shape<1, 1, 1, dstVR, dstVC>, pto::Stride<1, 1, dstVR, dstVC, 1>>;
    using GlobalDataSrc0 = GlobalTensor<T, Shape<1, 1, 1, src0VR, src0VC>, pto::Stride<1, 1, src0VR, src0VC, 1>>;
    using GlobalDataSrc1 = GlobalTensor<T, Shape<1, 1, 1, src1VR, src1VC>, pto::Stride<1, 1, src1VR, src1VC, 1>>;

    using TileDataDst = Tile<TileType::Vec, T, dstTR, dstTC, BLayout::RowMajor, -1, -1>;
    using TileDataSrc0 = Tile<TileType::Vec, T, src0TR, src0TC, BLayout::RowMajor, -1, -1>;
    using TileDataSrc1 = Tile<TileType::Vec, T, src1TR, src1TC, BLayout::RowMajor, -1, -1>;

    TileDataSrc0 src0Tile(src0VR, src0VC);
    TileDataSrc1 src1Tile(src1VR, src1VC);
    TileDataDst dstTile(dstVR, dstVC);

    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, src0TR * src0TC * sizeof(T));
    TASSIGN(dstTile, src0TR * src0TC * sizeof(T) + src1TR * src1TC * sizeof(T));

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataDst dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TPARTMAX<TileDataDst, TileDataSrc0, TileDataSrc1>(dstTile, src0Tile, src1Tile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC, bool isHalf = false>
void LaunchTPartMax(T *out, T *src0, T *src1, void *stream)
{
    constexpr int blockSize = 32;
    constexpr int alignedSrc0VC = PTO_CEIL(src0VC, blockSize / sizeof(T));
    constexpr int alignedSrc1VC = PTO_CEIL(src1VC, blockSize / sizeof(T));
    constexpr int alignedDstVC = PTO_CEIL(dstVC, blockSize / sizeof(T));
    if constexpr (std::is_same_v<T, aclFloat16> && isHalf == true) {
        runTPartMax<half, dstVR, dstVC, src0VR, src0VC, src1VR, src1VC, dstVR, alignedDstVC, src0VR, alignedSrc0VC,
                    src1VR, alignedSrc1VC><<<1, nullptr, stream>>>((half *)out, (half *)src0, (half *)src1);
    } else {
        runTPartMax<T, dstVR, dstVC, src0VR, src0VC, src1VR, src1VC, dstVR, alignedDstVC, src0VR, alignedSrc0VC, src1VR,
                    alignedSrc1VC><<<1, nullptr, stream>>>(out, src0, src1);
    }
}

template <typename T, int dstVR, int dstVC, int src0VR, int src0VC, int src1VR, int src1VC, int dstTR, int dstTC,
          int src0TR, int src0TC, int src1TR, int src1TC, bool isHalf = false>
void LaunchTPartMax(T *out, T *src0, T *src1, void *stream)
{
    if constexpr (std::is_same_v<T, aclFloat16> && isHalf == true) {
        runTPartMax<half, dstVR, dstVC, src0VR, src0VC, src1VR, src1VC, dstTR, dstTC, src0TR, src0TC, src1TR, src1TC>
            <<<1, nullptr, stream>>>((half *)out, (half *)src0, (half *)src1);
    } else {
        runTPartMax<T, dstVR, dstVC, src0VR, src0VC, src1VR, src1VC, dstTR, dstTC, src0TR, src0TC, src1TR, src1TC>
            <<<1, nullptr, stream>>>(out, src0, src1);
    }
}
} // namespace TPartMaxTest

template void TPartMaxTest::LaunchTPartMax<float, 64, 64, 64, 64, 64, 64>(float *out, float *src0, float *src1,
                                                                          void *stream);
template void TPartMaxTest::LaunchTPartMax<float, 2, 24, 2, 24, 2, 8>(float *out, float *src0, float *src1,
                                                                      void *stream);
template void TPartMaxTest::LaunchTPartMax<float, 128, 64, 128, 64, 96, 64>(float *out, float *src0, float *src1,
                                                                            void *stream);
template void TPartMaxTest::LaunchTPartMax<float, 95, 95, 95, 95, 95, 95>(float *out, float *src0, float *src1,
                                                                          void *stream);
template void TPartMaxTest::LaunchTPartMax<float, 122, 123, 104, 123, 122, 110>(float *out, float *src0, float *src1,
                                                                                void *stream);
template void TPartMaxTest::LaunchTPartMax<aclFloat16, 122, 123, 104, 123, 122, 110, true>(aclFloat16 *out,
                                                                                           aclFloat16 *src0,
                                                                                           aclFloat16 *src1,
                                                                                           void *stream);
template void TPartMaxTest::LaunchTPartMax<int16_t, 122, 123, 104, 123, 122, 110>(int16_t *out, int16_t *src0,
                                                                                  int16_t *src1, void *stream);
template void TPartMaxTest::LaunchTPartMax<int32_t, 122, 123, 104, 123, 122, 110>(int32_t *out, int32_t *src0,
                                                                                  int32_t *src1, void *stream);
template void TPartMaxTest::LaunchTPartMax<uint16_t, 122, 123, 104, 123, 122, 110>(uint16_t *out, uint16_t *src0,
                                                                                   uint16_t *src1, void *stream);
template void TPartMaxTest::LaunchTPartMax<uint32_t, 122, 123, 104, 123, 122, 110>(uint32_t *out, uint32_t *src0,
                                                                                   uint32_t *src1, void *stream);
template void TPartMaxTest::LaunchTPartMax<int8_t, 122, 123, 104, 123, 122, 110>(int8_t *out, int8_t *src0,
                                                                                 int8_t *src1, void *stream);
template void TPartMaxTest::LaunchTPartMax<uint8_t, 122, 123, 104, 123, 122, 110>(uint8_t *out, uint8_t *src0,
                                                                                  uint8_t *src1, void *stream);
template void TPartMaxTest::LaunchTPartMax<aclFloat16, 5, 33, 5, 33, 5, 33, 6, 1520, 6, 1520, 6, 464, true>(
    aclFloat16 *out, aclFloat16 *src0, aclFloat16 *src1, void *stream);
