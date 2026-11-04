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
#include "acl/acl.h"

using namespace pto;

// TMOV ND→NZ kernel for hifloat8_t.
// Sizes must be pre-aligned: kRows % 16 == 0 and kCols % 32 == 0.
template <int kRows, int kCols>
__global__ AICORE void runTMOV_nd2nz(__gm__ hifloat8_t __out__* out, __gm__ hifloat8_t __in__* src)
{
    using T = hifloat8_t;
    constexpr int c0 = CUBE_BLOCK_SIZE / (FRACTAL_NZ_ROW * sizeof(T)); // 32

    // Input GM: ND row-major
    using SrcShape = Shape<1, 1, 1, kRows, kCols>;
    using SrcStride = pto::Stride<1, 1, 1, kCols, 1>;
    using SrcGlobal = GlobalTensor<T, SrcShape, SrcStride>;

    // Output GM: NZ fractal [C1, N1, 16, c0]
    constexpr int C1 = kCols / c0;
    constexpr int N1 = kRows / FRACTAL_NZ_ROW;
    using DstShape = Shape<1, C1, N1, FRACTAL_NZ_ROW, c0>;
    using DstStride = pto::Stride<C1 * kRows * c0, kRows * c0, FRACTAL_NZ_ROW * c0, c0, 1>;
    using DstGlobal = GlobalTensor<T, DstShape, DstStride, Layout::NZ>;

    // UB tiles: src ND, dst NZ
    using SrcTile = Tile<TileType::Vec, T, kRows, kCols, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, T, kRows, kCols, BLayout::ColMajor, -1, -1, SLayout::RowMajor>;

    SrcTile srcTile(kRows, kCols);
    DstTile dstTile(kRows, kCols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

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

// TMOV ND→NZ kernel for fp4 types (float4_e1m2x2_t / float4_e2m1x2_t).
// kRows/kCols are in f4 element (nibble) units. Uses TLOAD MTE zero-pad:
// GM stays dense [kRows, kCols]; the src UB tile is alignedCols wide (32-byte-
// aligned rows) with PadVal::Zero so TLOAD copies only kCols nibbles per row
// and zero-fills the per-row gap. The dst NZ tile covers the full alignedCols
// so the zero pad flows into the last NZ panel deterministically.
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

    // Src UB: alignedCols wide (passes 32B-align assert), ColValid=kCols (dense),
    // PadVal::Zero → TLOAD zero-fills the per-row gap in MTE (cheap, no extra UB).
    using SrcTile = Tile<
        TileType::Vec, T, kRows, alignedCols, BLayout::RowMajor, kRows, kCols, SLayout::NoneBox,
        TileConfig::fractalABSize, PadValue::Zero>;
    // Dst UB: alignedCols wide, ColValid=alignedCols (full panel incl. zero pad).
    using DstTile =
        Tile<TileType::Vec, T, kRows, alignedCols, BLayout::ColMajor, kRows, alignedCols, SLayout::RowMajor>;

    SrcTile srcTile;
    DstTile dstTile;
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, srcBytes);

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

// Host-visible launch wrappers (uint8_t* since fp4 / hifloat8 are opaque on host).
// fp4 kind is selected by an int tag (not the C++ type) so the host test driver
// never has to name a fp4 type, which lives in MXTypes.hpp (only included under
// __CPU_SIM): kFp4E1M2 = float4_e1m2x2_t, kFp4E2M1 = float4_e2m1x2_t.
constexpr int kFp4E1M2 = 0;
constexpr int kFp4E2M1 = 1;

template <int kRows, int kCols>
void launchTMOV_nd2nz(uint8_t* out, uint8_t* src, void* stream)
{
    runTMOV_nd2nz<kRows, kCols><<<1, nullptr, stream>>>((hifloat8_t*)out, (hifloat8_t*)src);
}

template <int kFp4Kind, int kRows, int kCols>
void launchTMOV_nd2nz_f4(uint8_t* out, uint8_t* src, void* stream)
{
    if constexpr (kFp4Kind == kFp4E1M2) {
        runTMOV_nd2nz_f4<float4_e1m2x2_t, kRows, kCols>
            <<<1, nullptr, stream>>>((float4_e1m2x2_t*)out, (float4_e1m2x2_t*)src);
    } else {
        runTMOV_nd2nz_f4<float4_e2m1x2_t, kRows, kCols>
            <<<1, nullptr, stream>>>((float4_e2m1x2_t*)out, (float4_e2m1x2_t*)src);
    }
}

template void launchTMOV_nd2nz<32, 32>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2nz<32, 64>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2nz<64, 64>(uint8_t*, uint8_t*, void*);

// 12 fp4 regression cases: 6 sizes × {e1m2, e2m1}
template void launchTMOV_nd2nz_f4<kFp4E1M2, 16, 32>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2nz_f4<kFp4E1M2, 16, 8160>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2nz_f4<kFp4E1M2, 4080, 32>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2nz_f4<kFp4E1M2, 32, 32>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2nz_f4<kFp4E1M2, 32, 64>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2nz_f4<kFp4E1M2, 64, 64>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2nz_f4<kFp4E2M1, 16, 32>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2nz_f4<kFp4E2M1, 16, 8160>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2nz_f4<kFp4E2M1, 4080, 32>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2nz_f4<kFp4E2M1, 32, 32>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2nz_f4<kFp4E2M1, 32, 64>(uint8_t*, uint8_t*, void*);
template void launchTMOV_nd2nz_f4<kFp4E2M1, 64, 64>(uint8_t*, uint8_t*, void*);
