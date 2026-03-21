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

template <typename T, uint64_t dstS1, uint64_t dstS0, uint64_t offsetS1, uint64_t offsetS0, uint64_t srcS1,
          uint64_t srcS0>
__global__ AICORE void runTGATHERB(__gm__ T __out__ *out, __gm__ T __in__ *src, __gm__ uint32_t __in__ *offset)
{
    using GlobalDataDst = GlobalTensor<T, pto::Shape<1, 1, 1, dstS1, dstS0>, pto::Stride<1, 1, 1, dstS0, 1>>;

    constexpr unsigned blockSizeElem = pto::BLOCK_BYTE_SIZE / sizeof(T);
    using GlobalDataOffset =
        GlobalTensor<uint32_t, pto::Shape<1, 1, 1, offsetS1, offsetS0>, pto::Stride<1, 1, 1, offsetS0, 1>>;
    using GlobalDataSrc = GlobalTensor<T, pto::Shape<1, 1, 1, srcS1, srcS0>, pto::Stride<1, 1, 1, srcS0, 1>>;

    using TileDataDst = Tile<TileType::Vec, T, dstS1, dstS0, BLayout::RowMajor, dstS1, dstS0>;
    using TileDataOffset = Tile<TileType::Vec, uint32_t, offsetS1, offsetS0, BLayout::RowMajor, offsetS1, offsetS0>;
    using TileDataSrc = Tile<TileType::Vec, T, srcS1, srcS0, BLayout::RowMajor, srcS1, srcS0>;

    TileDataSrc srcTile;
    TileDataOffset offsetTile;
    TileDataDst dstTile;

    constexpr uint64_t srcSize = srcS1 * srcS0 * sizeof(T);
    constexpr uint64_t offsetSize = dstS1 * offsetS0 * 4;
    constexpr uint64_t dstSize = dstS1 * dstS0 * sizeof(T);
    constexpr uint64_t totalSize = srcSize + offsetSize + dstSize;
    static_assert(totalSize < 192 * 1024, "UB size overflow, should be less than 192KB.");
    TASSIGN(srcTile, 0x0);
    TASSIGN(offsetTile, srcSize);
    TASSIGN(dstTile, srcSize + offsetSize);

    GlobalDataSrc srcGlobal(src);
    GlobalDataOffset offsetGlobal(offset);
    GlobalDataDst dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    TLOAD(offsetTile, offsetGlobal);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TGATHERB<TileDataDst, TileDataSrc, TileDataOffset>(dstTile, srcTile, offsetTile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

template <typename T, uint64_t dstS1, uint64_t dstS0, uint64_t offsetS1, uint64_t offsetS0, uint64_t srcS1,
          uint64_t srcS0>
void LaunchTGatherB(T *out, T *src, uint32_t *offset, void *stream)
{
    runTGATHERB<T, dstS1, dstS0, offsetS1, offsetS0, srcS1, srcS0><<<1, nullptr, stream>>>(out, src, offset);
}

template void LaunchTGatherB<float, 2, 128, 2, 16, 2, 128>(float *out, float *src, uint32_t *offset, void *stream);
template void LaunchTGatherB<int32_t, 2, 128, 2, 16, 2, 128>(int32_t *out, int32_t *src, uint32_t *offset,
                                                             void *stream);
template void LaunchTGatherB<uint32_t, 2, 128, 2, 16, 2, 128>(uint32_t *out, uint32_t *src, uint32_t *offset,
                                                              void *stream);
template void LaunchTGatherB<int16_t, 1, 32768, 1, 2048, 1, 32768>(int16_t *out, int16_t *src, uint32_t *offset,
                                                                   void *stream);
template void LaunchTGatherB<uint16_t, 257, 128, 257, 8, 257, 128>(uint16_t *out, uint16_t *src, uint32_t *offset,
                                                                   void *stream);
template void LaunchTGatherB<half, 1, 32768, 1, 2048, 1, 32768>(half *out, half *src, uint32_t *offset, void *stream);
template void LaunchTGatherB<int8_t, 2, 256, 2, 8, 2, 256>(int8_t *out, int8_t *src, uint32_t *offset, void *stream);
template void LaunchTGatherB<int8_t, 2, 32768, 2, 1024, 2, 32768>(int8_t *out, int8_t *src, uint32_t *offset,
                                                                  void *stream);
template void LaunchTGatherB<uint8_t, 2, 32768, 2, 1024, 2, 32768>(uint8_t *out, uint8_t *src, uint32_t *offset,
                                                                   void *stream);