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

using namespace pto;

#ifndef PTO_CEIL
#define PTO_CEIL(x, y) ((((x) + (y) - 1) / (y)) * (y))
#endif

namespace TQuantDNTest {

// Stage 1: after TQUANT (FP8 ND + E8M0 DN)
// Stage 2: after TQUANT + FP8 ND->NZ
// Stage 3: full pipeline including E8 DN->ZZ
template <int Stage, typename T, int M, int N, int N_pad>
__global__ AICORE void runTQuantDN(__gm__ T __in__ *src_gm, __gm__ int8_t __out__ *fp8_nd_gm,
                                   __gm__ uint8_t __out__ *e8_dn_gm, __gm__ int8_t __out__ *fp8_nz_gm,
                                   __gm__ uint8_t __out__ *e8_zz_gm, __gm__ T __out__ *max_dn_gm)
{
    static_assert(Stage >= 1 && Stage <= 3, "Stage must be 1 (quant), 2 (nz), or 3 (zz).");

    constexpr uint32_t grpSize = 32;
    constexpr uint32_t hatM = M / grpSize;
    constexpr uint32_t paddedCols = N_pad;
    constexpr uint32_t groupedColsValid = paddedCols / 32;
    constexpr uint32_t numGroupsFlat = M * groupedColsValid;
    constexpr uint32_t numGroupsFlatAligned = PTO_CEIL(numGroupsFlat, 32);
    constexpr uint32_t paddedRows16 = PTO_CEIL(M, FRACTAL_NZ_ROW);
    constexpr uint32_t virtualRow = paddedRows16 + 1;

    using SrcTile =
        Tile<TileType::Vec, T, M, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    using DstFP8Tile = Tile<TileType::Vec, int8_t, M, paddedCols, BLayout::RowMajor, M, paddedCols, SLayout::NoneBox,
                            512, PadValue::Zero>;
    using MaxTile =
        Tile<TileType::Vec, T, hatM, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    using ScalingTile =
        Tile<TileType::Vec, T, hatM, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    using E8NdTile = Tile<TileType::Vec, uint8_t, hatM, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                          PadValue::Zero>;

    using E8DnTile = Tile<TileType::Vec, uint8_t, hatM, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                          PadValue::Zero>;

    using E8ZzTile = Tile<TileType::Vec, uint8_t, paddedRows16, groupedColsValid, BLayout::RowMajor, -1, -1,
                          SLayout::RowMajor, 32, PadValue::Zero>;
    using E8StoreTile = Tile<TileType::Vec, uint8_t, 1, numGroupsFlatAligned, BLayout::RowMajor, -1, -1,
                             SLayout::NoneBox, 512, PadValue::Zero>;

    using Fp8NZTile = Tile<TileType::Vec, int8_t, virtualRow, paddedCols, BLayout::ColMajor, M, paddedCols,
                           SLayout::RowMajor, 512, PadValue::Null, CompactMode::RowPlusOne>;

    constexpr uint32_t colBlkCount = paddedCols / 16;
    constexpr uint32_t hatP = hatM / 2;
    constexpr uint32_t tmpBufSize =
        (BLOCK_SIZE / sizeof(uint16_t) +
         (colBlkCount > hatP ? colBlkCount : hatP) * (hatP > colBlkCount ? hatP : colBlkCount) +
         BLOCK_SIZE / sizeof(uint16_t)) *
        sizeof(uint16_t);
    constexpr uint32_t tmpBufSizeAligned = PTO_CEIL(tmpBufSize, 32);

    using TmpTile = Tile<TileType::Vec, uint8_t, 1, tmpBufSizeAligned, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
                         PadValue::Zero>;

    SrcTile srcTile(M, paddedCols);
    DstFP8Tile fp8Tile;
    MaxTile maxPerGpTile(hatM, paddedCols);
    ScalingTile scalingTile(hatM, paddedCols);
    E8NdTile e8Tile(hatM, paddedCols);
    E8DnTile e8DnTile(hatM, paddedCols);
    E8ZzTile e8ZzTile(paddedRows16, groupedColsValid);
    E8StoreTile e8StoreTile(1, numGroupsFlatAligned);
    Fp8NZTile fp8TileNZ;
    TmpTile tmpTile(1, tmpBufSizeAligned);

    using SrcGlobal = GlobalTensor<T, Shape<1, 1, 1, M, N_pad>, pto::Stride<1, 1, 1, N_pad, 1>>;
    SrcGlobal srcGlobal(src_gm);

    using DstFp8NdGlobal = GlobalTensor<int8_t, Shape<1, 1, 1, M, paddedCols>, pto::Stride<1, 1, 1, paddedCols, 1>>;
    DstFp8NdGlobal fp8NdGlobal(fp8_nd_gm);

    using DstMaxGlobal = GlobalTensor<T, Shape<1, 1, 1, hatM, paddedCols>, pto::Stride<1, 1, 1, paddedCols, 1>>;
    DstMaxGlobal maxGlobal(max_dn_gm);

    using DstE8DnGlobal = GlobalTensor<uint8_t, Shape<1, 1, 1, hatM, paddedCols>, pto::Stride<1, 1, 1, paddedCols, 1>>;
    DstE8DnGlobal e8DnGlobal(e8_dn_gm);

    using DstE8Global =
        GlobalTensor<uint8_t, Shape<1, 1, 1, 1, numGroupsFlatAligned>, pto::Stride<1, 1, 1, numGroupsFlatAligned, 1>>;
    DstE8Global e8Global(e8_zz_gm);

    using DstFp8GlobalNZ = GlobalTensor<int8_t, TileShape2D<int8_t, M, paddedCols, Layout::NZ>,
                                        BaseShape2D<int8_t, M, paddedCols, Layout::NZ>, Layout::NZ>;
    DstFp8GlobalNZ fp8GlobalNZ((__gm__ int8_t *)fp8_nz_gm);

    constexpr uint32_t srcTileBytes = M * paddedCols * sizeof(T);
    constexpr uint32_t maxTileBytes = hatM * paddedCols * sizeof(T);
    constexpr uint32_t scalingTileBytes = hatM * paddedCols * sizeof(T);
    constexpr uint32_t e8TileBytes = hatM * paddedCols;
    constexpr uint32_t e8DnTileBytes = hatM * paddedCols;
    constexpr uint32_t fp8TileBytes = M * paddedCols;

    // Keep source and destination UB tiles orthogonal: place fp8Tile after all
    // input/work tiles so TQuant reads src and writes dst to non-overlapping
    // regions (on-board store ordering is not guaranteed).
    constexpr uint32_t srcTileAddr = 0x0;
    constexpr uint32_t maxTileAddr = PTO_CEIL(srcTileAddr + srcTileBytes, 0x20);
    constexpr uint32_t scalingTileAddr = PTO_CEIL(maxTileAddr + maxTileBytes, 0x20);
    constexpr uint32_t e8TileAddr = PTO_CEIL(scalingTileAddr + scalingTileBytes, 0x20);
    constexpr uint32_t e8DnTileAddr = PTO_CEIL(e8TileAddr + e8TileBytes, 0x20);
    constexpr uint32_t fp8TileAddr = PTO_CEIL(e8DnTileAddr + e8DnTileBytes, 0x20);
    constexpr uint32_t C0_SIZE_B = 32;
    constexpr uint32_t nColGroupsNZ = paddedCols / C0_SIZE_B;
    constexpr uint32_t fp8NZTileBytes =
        (nColGroupsNZ > 1) ? (nColGroupsNZ - 1) * (paddedRows16 + 1) * C0_SIZE_B + paddedRows16 * C0_SIZE_B :
                             paddedRows16 * C0_SIZE_B;
    constexpr uint32_t fp8NZTileAddr = PTO_CEIL(fp8TileAddr + fp8TileBytes, 0x20);
    // workTileEnd marks the end of the TQuant input-side tiles; fp8Tile now lives
    // after it, so use fp8NZEnd to find where the ZZ/tmp scratch area can start.
    constexpr uint32_t workTileEnd = e8DnTileAddr + e8DnTileBytes;
    constexpr uint32_t fp8NZEnd = fp8NZTileAddr + fp8NZTileBytes;
    constexpr uint32_t zzTmpStart = PTO_CEIL(workTileEnd > fp8NZEnd ? workTileEnd : fp8NZEnd, 0x20);
    constexpr uint32_t e8ZzTileAddr = zzTmpStart;
    constexpr uint32_t e8StoreTileAddr = zzTmpStart;
    constexpr uint32_t tmpTileAddr = PTO_CEIL(e8ZzTileAddr + numGroupsFlatAligned, 0x20);
    constexpr uint32_t layoutEnd = PTO_CEIL(tmpTileAddr + tmpBufSizeAligned, 0x100);
    static_assert(layoutEnd <= 0x40000, "UB layout exceeds 256 KB.");

    TASSIGN(srcTile, srcTileAddr);
    TASSIGN(maxPerGpTile, maxTileAddr);
    TASSIGN(scalingTile, scalingTileAddr);
    TASSIGN(e8Tile, e8TileAddr);
    TASSIGN(e8DnTile, e8DnTileAddr);
    TASSIGN(e8ZzTile, e8ZzTileAddr);
    TASSIGN(e8StoreTile, e8StoreTileAddr);
    TASSIGN(fp8Tile, fp8TileAddr);
    TASSIGN(fp8TileNZ, fp8NZTileAddr);
    TASSIGN(tmpTile, tmpTileAddr);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TQUANT<QuantType::MXFP8, QuantScaleAlg::OCP>(fp8Tile, srcTile, &e8Tile, &maxPerGpTile, &scalingTile, &e8DnTile);

    // TQuant writes the exponent tile (e8Tile) in row-major [hatM, paddedCols].
    // Copy it to e8DnTile so the UB tile shape matches the GM shape exactly.
    TMOV(e8DnTile, e8Tile);

    if constexpr (Stage == 1) {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(fp8NdGlobal, fp8Tile);
        TSTORE(e8DnGlobal, e8DnTile);
        TSTORE(maxGlobal, maxPerGpTile);
        return;
    }

    TMOV(fp8TileNZ, fp8Tile);

    if constexpr (Stage == 2) {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(fp8GlobalNZ, fp8TileNZ);
        return;
    }

    TMOV(e8ZzTile, e8DnTile, tmpTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(e8Global, e8StoreTile);
    TSTORE(fp8GlobalNZ, fp8TileNZ);
}

template <int Stage, int M, int N, int N_pad>
void LaunchTQuantDN(uint16_t *src, int8_t *fp8_nd, uint8_t *e8_dn, int8_t *fp8_nz, uint8_t *e8_zz, uint16_t *max_dn,
                    void *stream)
{
    runTQuantDN<Stage, bfloat16_t, M, N, N_pad>
        <<<1, nullptr, stream>>>((bfloat16_t *)src, fp8_nd, e8_dn, fp8_nz, e8_zz, (bfloat16_t *)max_dn);
}

template <int Stage, int M, int N, int N_pad>
void LaunchTQuantDN_fp32(uint32_t *src, int8_t *fp8_nd, uint8_t *e8_dn, int8_t *fp8_nz, uint8_t *e8_zz,
                         uint32_t *max_dn, void *stream)
{
    runTQuantDN<Stage, float, M, N, N_pad>
        <<<1, nullptr, stream>>>((float *)src, fp8_nd, e8_dn, fp8_nz, e8_zz, (float *)max_dn);
}

#define INSTANTIATE_TQUANT_DN_STAGE(S, M, N, NP) \
    template void LaunchTQuantDN<S, M, N, NP>(uint16_t *, int8_t *, uint8_t *, int8_t *, uint8_t *, uint16_t *, void *)

#define INSTANTIATE_TQUANT_DN_STAGE_FP32(S, M, N, NP)                                                                \
    template void LaunchTQuantDN_fp32<S, M, N, NP>(uint32_t *, int8_t *, uint8_t *, int8_t *, uint8_t *, uint32_t *, \
                                                   void *)

INSTANTIATE_TQUANT_DN_STAGE(1, 128, 128, 128);
INSTANTIATE_TQUANT_DN_STAGE(1, 64, 128, 128);
INSTANTIATE_TQUANT_DN_STAGE(1, 64, 256, 256);
INSTANTIATE_TQUANT_DN_STAGE(1, 128, 256, 256);
INSTANTIATE_TQUANT_DN_STAGE(1, 64, 64, 64);
INSTANTIATE_TQUANT_DN_STAGE(1, 128, 64, 64);
INSTANTIATE_TQUANT_DN_STAGE(1, 256, 64, 64);
INSTANTIATE_TQUANT_DN_STAGE(1, 256, 128, 128);
INSTANTIATE_TQUANT_DN_STAGE_FP32(1, 64, 128, 128);
INSTANTIATE_TQUANT_DN_STAGE_FP32(1, 128, 128, 128);
INSTANTIATE_TQUANT_DN_STAGE_FP32(1, 64, 256, 256);

#undef INSTANTIATE_TQUANT_DN_STAGE

} // namespace TQuantDNTest
