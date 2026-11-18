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

using namespace std;
using namespace pto;

// Sign combinations for TMADD_SIGN (A6 fused multiply-accumulate, swap form):
//   <NP, NA>  formula                        instruction
//   <f, f>    dst =  s0*dst + s1             vmadd
//   <t, f>    dst = -(s0*dst) + s1           vmsub
//   <t, t>    dst = -(s0*dst) - s1           vnmadd
//   <f, t>    dst =  s0*dst - s1             vnmsub

template <typename T, int tileRows, int tileCols, bool NP, bool NA>
__global__ AICORE void runTMAddSign(__gm__ T* out, __gm__ T* src0, __gm__ T* src1, __gm__ T* acc)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    constexpr int total = tileRows * tileCols;
    GlobalData src0G(src0, DynShape(1, 1, 1, tileRows, tileCols), DynStride(total, total, total, tileCols, 1));
    GlobalData src1G(src1, DynShape(1, 1, 1, tileRows, tileCols), DynStride(total, total, total, tileCols, 1));
    GlobalData dstG(acc, DynShape(1, 1, 1, tileRows, tileCols), DynStride(total, total, total, tileCols, 1));
    GlobalData outG(out, DynShape(1, 1, 1, tileRows, tileCols), DynStride(total, total, total, tileCols, 1));
    using TileData = Tile<TileType::Vec, T, tileRows, tileCols, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(tileRows, tileCols);
    TileData src1Tile(tileRows, tileCols);
    TileData dstTile(tileRows, tileCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, total * sizeof(T));
    TASSIGN(dstTile, 2 * total * sizeof(T));
    TLOAD(src0Tile, src0G);
    TLOAD(src1Tile, src1G);
    TLOAD(dstTile, dstG);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TMADD<NP, NA>(dstTile, src0Tile, src1Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(outG, dstTile);
}

template <typename T, int tileRows, int tileCols, bool NP, bool NA>
void LaunchTMAddSign(T* out, T* src0, T* src1, T* acc, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTMAddSign<half, tileRows, tileCols, NP, NA>
            <<<1, nullptr, stream>>>((half*)out, (half*)src0, (half*)src1, (half*)acc);
    } else {
        runTMAddSign<T, tileRows, tileCols, NP, NA><<<1, nullptr, stream>>>(out, src0, src1, acc);
    }
}

// float, 4x16: all 4 sign combinations
extern "C" void LaunchTMAddSign_f32_ff(float* out, float* s0, float* s1, float* acc, void* st)
{
    LaunchTMAddSign<float, 4, 16, false, false>(out, s0, s1, acc, st);
}
extern "C" void LaunchTMAddSign_f32_tf(float* out, float* s0, float* s1, float* acc, void* st)
{
    LaunchTMAddSign<float, 4, 16, true, false>(out, s0, s1, acc, st);
}
extern "C" void LaunchTMAddSign_f32_tt(float* out, float* s0, float* s1, float* acc, void* st)
{
    LaunchTMAddSign<float, 4, 16, true, true>(out, s0, s1, acc, st);
}
extern "C" void LaunchTMAddSign_f32_ft(float* out, float* s0, float* s1, float* acc, void* st)
{
    LaunchTMAddSign<float, 4, 16, false, true>(out, s0, s1, acc, st);
}

// half, 4x16
extern "C" void LaunchTMAddSign_f16_ff(aclFloat16* out, aclFloat16* s0, aclFloat16* s1, aclFloat16* acc, void* st)
{
    LaunchTMAddSign<aclFloat16, 4, 16, false, false>(out, s0, s1, acc, st);
}
extern "C" void LaunchTMAddSign_f16_tf(aclFloat16* out, aclFloat16* s0, aclFloat16* s1, aclFloat16* acc, void* st)
{
    LaunchTMAddSign<aclFloat16, 4, 16, true, false>(out, s0, s1, acc, st);
}
extern "C" void LaunchTMAddSign_f16_tt(aclFloat16* out, aclFloat16* s0, aclFloat16* s1, aclFloat16* acc, void* st)
{
    LaunchTMAddSign<aclFloat16, 4, 16, true, true>(out, s0, s1, acc, st);
}
extern "C" void LaunchTMAddSign_f16_ft(aclFloat16* out, aclFloat16* s0, aclFloat16* s1, aclFloat16* acc, void* st)
{
    LaunchTMAddSign<aclFloat16, 4, 16, false, true>(out, s0, s1, acc, st);
}

// Boundary variant: tile is tileRows×tileCols static, but valid is vRows×vCols
// (vRows ≤ tileRows, vCols ≤ tileCols). Tests predicate masking and tail handling.
template <typename T, int tileRows, int tileCols, int vRows, int vCols, bool NP, bool NA>
__global__ AICORE void runTMAddSignBounded(__gm__ T* out, __gm__ T* src0, __gm__ T* src1, __gm__ T* acc)
{
    using DynShape = pto::Shape<-1, -1, -1, -1, -1>;
    using DynStride = pto::Stride<-1, -1, -1, -1, -1>;
    using GlobalData = GlobalTensor<T, DynShape, DynStride>;
    constexpr int total = tileRows * tileCols;
    GlobalData src0G(src0, DynShape(1, 1, 1, vRows, vCols), DynStride(total, total, total, tileCols, 1));
    GlobalData src1G(src1, DynShape(1, 1, 1, vRows, vCols), DynStride(total, total, total, tileCols, 1));
    GlobalData dstG(acc, DynShape(1, 1, 1, vRows, vCols), DynStride(total, total, total, tileCols, 1));
    GlobalData outG(out, DynShape(1, 1, 1, vRows, vCols), DynStride(total, total, total, tileCols, 1));
    using TileData = Tile<TileType::Vec, T, tileRows, tileCols, BLayout::RowMajor, -1, -1>;
    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, total * sizeof(T));
    TASSIGN(dstTile, 2 * total * sizeof(T));
    TLOAD(src0Tile, src0G);
    TLOAD(src1Tile, src1G);
    TLOAD(dstTile, dstG);
#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif
    TMADD<NP, NA>(dstTile, src0Tile, src1Tile);
#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif
    TSTORE(outG, dstTile);
}

template <typename T, int tileRows, int tileCols, int vRows, int vCols, bool NP, bool NA>
void LaunchTMAddSignB(T* out, T* src0, T* src1, T* acc, void* stream)
{
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTMAddSignBounded<half, tileRows, tileCols, vRows, vCols, NP, NA>
            <<<1, nullptr, stream>>>((half*)out, (half*)src0, (half*)src1, (half*)acc);
    } else {
        runTMAddSignBounded<T, tileRows, tileCols, vRows, vCols, NP, NA><<<1, nullptr, stream>>>(out, src0, src1, acc);
    }
}

// boundary shapes
extern "C" void LaunchTMAddSign_f32_tf_1x16(float* out, float* s0, float* s1, float* acc, void* st)
{
    LaunchTMAddSign<float, 1, 16, true, false>(out, s0, s1, acc, st);
}
extern "C" void LaunchTMAddSign_f32_tf_8x32(float* out, float* s0, float* s1, float* acc, void* st)
{
    LaunchTMAddSign<float, 8, 32, true, false>(out, s0, s1, acc, st);
}
extern "C" void LaunchTMAddSign_f32_tf_1x4096(float* out, float* s0, float* s1, float* acc, void* st)
{
    LaunchTMAddSign<float, 1, 4096, true, false>(out, s0, s1, acc, st);
}

// valid < static: tile 4x32, valid 4x20
extern "C" void LaunchTMAddSign_f32_tf_v4x20(float* out, float* s0, float* s1, float* acc, void* st)
{
    LaunchTMAddSignB<float, 4, 32, 4, 20, true, false>(out, s0, s1, acc, st);
}
// valid < static: tile 8x16, valid 5x12
extern "C" void LaunchTMAddSign_f32_tt_v5x12(float* out, float* s0, float* s1, float* acc, void* st)
{
    LaunchTMAddSignB<float, 8, 16, 5, 12, true, true>(out, s0, s1, acc, st);
}
// f16 valid < static: tile 4x64, valid 4x37
extern "C" void LaunchTMAddSign_f16_tf_v4x37(aclFloat16* out, aclFloat16* s0, aclFloat16* s1, aclFloat16* acc, void* st)
{
    LaunchTMAddSignB<aclFloat16, 4, 64, 4, 37, true, false>(out, s0, s1, acc, st);
}

// extreme values: same shape, different data (gen_data controls the values)
extern "C" void LaunchTMAddSign_f32_tf_extreme(float* out, float* s0, float* s1, float* acc, void* st)
{
    LaunchTMAddSign<float, 4, 16, true, false>(out, s0, s1, acc, st);
}

// Multi-VL and stress sizes
extern "C" void LaunchTMAddSign_f32_tf_64x64(float* out, float* s0, float* s1, float* acc, void* st)
{
    LaunchTMAddSign<float, 64, 64, true, false>(out, s0, s1, acc, st);
}
extern "C" void LaunchTMAddSign_f32_tt_16x512(float* out, float* s0, float* s1, float* acc, void* st)
{
    LaunchTMAddSign<float, 16, 512, true, true>(out, s0, s1, acc, st);
}
extern "C" void LaunchTMAddSign_f32_ff_2728x8(float* out, float* s0, float* s1, float* acc, void* st)
{
    LaunchTMAddSign<float, 2728, 8, false, false>(out, s0, s1, acc, st);
}
extern "C" void LaunchTMAddSign_f16_tf_64x128(
    aclFloat16* out, aclFloat16* s0, aclFloat16* s1, aclFloat16* acc, void* st)
{
    LaunchTMAddSign<aclFloat16, 64, 128, true, false>(out, s0, s1, acc, st);
}
// f32 valid < static on a multi-VL tile: tile 4x256, valid 3x200
extern "C" void LaunchTMAddSign_f32_tf_v3x200(float* out, float* s0, float* s1, float* acc, void* st)
{
    LaunchTMAddSignB<float, 4, 256, 3, 200, true, false>(out, s0, s1, acc, st);
}

// More extreme values: dedicated per-category
extern "C" void LaunchTMAddSign_f32_ff_zeros(float* out, float* s0, float* s1, float* acc, void* st)
{
    LaunchTMAddSign<float, 4, 16, false, false>(out, s0, s1, acc, st);
}
extern "C" void LaunchTMAddSign_f32_tf_inf(float* out, float* s0, float* s1, float* acc, void* st)
{
    LaunchTMAddSign<float, 4, 16, true, false>(out, s0, s1, acc, st);
}
extern "C" void LaunchTMAddSign_f32_tt_denorm(float* out, float* s0, float* s1, float* acc, void* st)
{
    LaunchTMAddSign<float, 4, 16, true, true>(out, s0, s1, acc, st);
}
extern "C" void LaunchTMAddSign_f32_ft_overflow(float* out, float* s0, float* s1, float* acc, void* st)
{
    LaunchTMAddSign<float, 4, 16, false, true>(out, s0, s1, acc, st);
}
