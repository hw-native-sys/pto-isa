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
#include <pto/npu/a2a3/TQuant.hpp>
#include "acl/acl.h"
#include <type_traits>

using namespace pto;

#define PTO_CEIL(x, y) ((((x) + (y)-1) / (y)) * (y))

namespace TQuantTest {

// FP32 --> INT8 SYM
template <int validRows, int validCols, int mode>
__global__ AICORE void runTQuantInt8Sym(__gm__ int8_t __out__ *out_s8, __gm__ float __in__ *src,
                                        __gm__ float __in__ *scale)
{
    constexpr int paddedCols_b32 = PTO_CEIL(validCols, BLOCK_BYTE_SIZE / sizeof(float));
    constexpr int paddedCols_b8 = PTO_CEIL(validCols, BLOCK_BYTE_SIZE / sizeof(int8_t));
    using SrcGlobal = GlobalTensor<float, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using DstGlobal = GlobalTensor<int8_t, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using ParaGlobal = GlobalTensor<float, Shape<1, 1, 1, validRows, 1>, pto::Stride<1, 1, 1, 1, 1>, pto::Layout::DN>;

    using SrcTile = Tile<TileType::Vec, float, validRows, paddedCols_b32, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, int8_t, validRows, paddedCols_b8, BLayout::RowMajor, -1, -1>;
    using ParaTile = Tile<TileType::Vec, float, validRows, 1, BLayout::ColMajor, -1, -1>;

    SrcTile srcTile(validRows, validCols);
    DstTile dstS8Tile(validRows, validCols);
    ParaTile scaleTile(validRows, 1);

    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(out_s8);
    ParaGlobal scaleGlobal(scale);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstS8Tile, 0x0);
    TASSIGN(scaleTile, 0x20100);

    TLOAD(srcTile, srcGlobal);
    TLOAD(scaleTile, scaleGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    TQUANT<pto::QuantType::INT8_SYM, DstTile, SrcTile, ParaTile>(dstS8Tile, srcTile, scaleTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, dstS8Tile);
}

// FP32 --> INT8 ASYM
template <int validRows, int validCols, int mode>
__global__ AICORE void runTQuantInt8Asym(__gm__ uint8_t __out__ *out_u8, __gm__ float __in__ *src,
                                         __gm__ float __in__ *scale, __gm__ float __in__ *offset)
{
    // pad each row to multiple of 32 elements
    constexpr int paddedCols_b32 = PTO_CEIL(validCols, BLOCK_BYTE_SIZE / sizeof(float));
    constexpr int paddedCols_b8 = PTO_CEIL(validCols, BLOCK_BYTE_SIZE / sizeof(uint8_t));
    using SrcGlobal = GlobalTensor<float, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using DstGlobal = GlobalTensor<uint8_t, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    using ParaGlobal = GlobalTensor<float, Shape<1, 1, 1, validRows, 1>, pto::Stride<1, 1, 1, 1, 1>, pto::Layout::DN>;

    using SrcTile = Tile<TileType::Vec, float, validRows, paddedCols_b32, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, uint8_t, validRows, paddedCols_b8, BLayout::RowMajor, -1, -1>;
    using ParaTile = Tile<TileType::Vec, float, validRows, 1, BLayout::ColMajor, -1, -1>;

    SrcTile srcTile(validRows, validCols);
    DstTile dstU8Tile(validRows, validCols);
    ParaTile scaleTile(validRows, 1);
    ParaTile offsetTile(validRows, 1);

    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(out_u8);
    ParaGlobal scaleGlobal(scale);
    ParaGlobal offsetGlobal(offset);

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstU8Tile, 0x0);
    TASSIGN(scaleTile, 0x20100);
    TASSIGN(offsetTile, 0x26000);

    TLOAD(srcTile, srcGlobal);
    TLOAD(scaleTile, scaleGlobal);
    TLOAD(offsetTile, offsetGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    TQUANT<pto::QuantType::INT8_ASYM, DstTile, SrcTile, ParaTile>(dstU8Tile, srcTile, scaleTile, &offsetTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, dstU8Tile);
}

template <int validRows, int validCols, int mode, pto::QuantType quantType>
void LaunchTQuantInt8(std::conditional_t<quantType == pto::QuantType::INT8_SYM, int8_t, uint8_t> *dst, float *src,
                      float *scale, void *stream, float *offset = nullptr)
{
    if constexpr (quantType == pto::QuantType::INT8_SYM) {
        runTQuantInt8Sym<validRows, validCols, mode><<<1, nullptr, stream>>>(dst, src, scale);
    } else {
        runTQuantInt8Asym<validRows, validCols, mode><<<1, nullptr, stream>>>(dst, src, scale, offset);
    }
}

} // namespace TQuantTest

// INT8 SYM cases
template void TQuantTest::LaunchTQuantInt8<64, 128, 0, pto::QuantType::INT8_SYM>(int8_t *dst, float *src, float *scale,
                                                                                 void *stream, float *offset);
template void TQuantTest::LaunchTQuantInt8<128, 128, 0, pto::QuantType::INT8_SYM>(int8_t *dst, float *src, float *scale,
                                                                                  void *stream, float *offset);
template void TQuantTest::LaunchTQuantInt8<256, 128, 0, pto::QuantType::INT8_SYM>(int8_t *dst, float *src, float *scale,
                                                                                  void *stream, float *offset);
// INT8 ASYM cases
template void TQuantTest::LaunchTQuantInt8<64, 128, 0, pto::QuantType::INT8_ASYM>(uint8_t *dst, float *src,
                                                                                  float *scale, void *stream,
                                                                                  float *offset);
template void TQuantTest::LaunchTQuantInt8<128, 128, 0, pto::QuantType::INT8_ASYM>(uint8_t *dst, float *src,
                                                                                   float *scale, void *stream,
                                                                                   float *offset);
template void TQuantTest::LaunchTQuantInt8<256, 128, 0, pto::QuantType::INT8_ASYM>(uint8_t *dst, float *src,
                                                                                   float *scale, void *stream,
                                                                                   float *offset);