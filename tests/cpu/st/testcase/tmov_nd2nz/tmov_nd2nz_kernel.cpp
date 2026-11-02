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
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

// TMOV ND->NZ kernel for fp4 types (float4_e1m2x2_t / float4_e2m1x2_t), ported from
// the A5 tmov_nd2nz ST test.
template <typename T, int kRows, int kCols>
__global__ AICORE void runTMOV_nd2nz_f4(__gm__ T __out__* out, __gm__ T __in__* src)
{
    constexpr int c0 = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T)) * 2; // 64 nibbles = 32 bytes
    constexpr int alignedCols = (kCols + c0 - 1) / c0 * c0;
    constexpr int C1 = alignedCols / c0;
    constexpr int N1 = kRows / FRACTAL_NZ_ROW;
    constexpr int srcBytes = kRows * alignedCols / 2; // fp4: 2 nibbles/byte

    // Dense GM input [kRows, kCols] nibbles — NOT pre-padded.
    using SrcShape = Shape<1, 1, 1, kRows, kCols>;
    using SrcStride = pto::Stride<1, 1, 1, kCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;

    // Output GM: NZ fractal [C1, N1, 16, c0] over the padded width.
    using DstShape = Shape<1, C1, N1, FRACTAL_NZ_ROW, c0>;
    using DstStride = pto::Stride<C1 * kRows * c0, kRows * c0, FRACTAL_NZ_ROW * c0, c0, 1>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride, Layout::NZ>;

    // Src UB: alignedCols wide (passes 32B-align assert), ColValid=kCols (dense).
    // PadVal::Null -> TLOAD leaves the per-row gap [kCols, alignedCols) untouched
    // (the DMA only pads the sub-block tail), so the pre-filled 0x11 gap below
    // survives like loop-carried UB leftovers would on hardware.
    using SrcTile = Tile<
        TileType::Vec, T, kRows, alignedCols, BLayout::RowMajor, kRows, kCols, SLayout::NoneBox,
        TileConfig::fractalABSize, PadValue::Null>;
    // Dst UB: alignedCols wide, ColValid=alignedCols.
    using DstTile =
        Tile<TileType::Vec, T, kRows, alignedCols, BLayout::ColMajor, kRows, alignedCols, SLayout::RowMajor>;

    SrcTile srcTile;
    DstTile dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, srcBytes);

    // Dirty-UB regression: pre-fill both tiles with non-zero garbage. The 0x11
    // src pads (one nibble pair per byte) must be copied verbatim into the dst
    // pads, and the 0x55 dst fill must be fully overwritten by TMOV over the
    // whole destination valid width -- matching A5, which derives the copy
    // extent (vsstb predicate) from dst.GetValidCol().
    constexpr int dstBytes = kRows * alignedCols / 2; // fp4: 2 nibbles/byte
    std::fill(reinterpret_cast<uint8_t*>(srcTile.data()), reinterpret_cast<uint8_t*>(srcTile.data()) + dstBytes, 0x11);
    std::fill(reinterpret_cast<uint8_t*>(dstTile.data()), reinterpret_cast<uint8_t*>(dstTile.data()) + dstBytes, 0x55);

    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(out);

    TLOAD(srcTile, srcGlobal);

#ifndef __PTO_AUTO__
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
#endif

    TMOV<DstTile, SrcTile>(dstTile, srcTile);

#ifndef __PTO_AUTO__
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#endif

    TSTORE(dstGlobal, dstTile);
    out = dstGlobal.data();
}

// Host-visible launch wrappers (uint8_t* since fp4 is opaque on host).
// fp4 kind is selected by an enum tag (not the C++ type) so the host test driver
// never has to name a fp4 type, which lives in MXTypes.hpp (only included under
// __CPU_SIM): Fp4Kind::E1M2 = float4_e1m2x2_t, Fp4Kind::E2M1 = float4_e2m1x2_t.
enum class Fp4Kind {
    E1M2,
    E2M1,
};

template <Fp4Kind kFp4Kind, int kRows, int kCols>
void launchTMOV_nd2nz_f4(uint8_t* out, uint8_t* src, void* stream)
{
    (void)stream;
    if constexpr (kFp4Kind == Fp4Kind::E1M2) {
        runTMOV_nd2nz_f4<float4_e1m2x2_t, kRows, kCols>((float4_e1m2x2_t*)out, (float4_e1m2x2_t*)src);
    } else {
        runTMOV_nd2nz_f4<float4_e2m1x2_t, kRows, kCols>((float4_e2m1x2_t*)out, (float4_e2m1x2_t*)src);
    }
}

// fp4 regression cases: 16x32 x {e1m2, e2m1}
template void launchTMOV_nd2nz_f4<Fp4Kind::E1M2, 16, 32>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2nz_f4<Fp4Kind::E2M1, 16, 32>(uint8_t*, uint8_t*, void*);
