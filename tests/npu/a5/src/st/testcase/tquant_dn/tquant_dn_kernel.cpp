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

template <
    MxQuantAlg Alg, int StaticRows, int StaticCols, int ValidRows, int ValidCols, typename T, typename SrcTile,
    typename DstTile, typename ExpTile, typename MaxTile, typename SrcGlobal, typename DstGlobal, typename ExpGlobal>
PTO_INTERNAL void RunTQuantDNValidShapeBody(
    __gm__ T __in__* src, __gm__ int8_t __out__* dst, __gm__ uint8_t __out__* exp)
{
    constexpr int fp8StaticCols = PTO_CEIL(StaticCols, 32);
    constexpr int expStaticRows = StaticRows / 64;
    constexpr int expStaticCols = PTO_CEIL(StaticCols * 2, 32);
    constexpr int maxStaticRows = StaticRows / 32;
    constexpr int expValidRows = ValidRows / 64;
    constexpr int expValidCols = ValidCols * 2;
    constexpr int maxValidRows = ValidRows / 32;
    constexpr uint32_t srcAddr = 0;
    constexpr uint32_t srcBytes = StaticRows * StaticCols * sizeof(T);
    constexpr uint32_t dstAddr = PTO_CEIL(srcAddr + srcBytes, 32);
    constexpr uint32_t dstBytes = StaticRows * fp8StaticCols;
    constexpr uint32_t expAddr = PTO_CEIL(dstAddr + dstBytes, 32);
    constexpr uint32_t expBytes = expStaticRows * expStaticCols;
    constexpr uint32_t maxAddr = PTO_CEIL(expAddr + expBytes, 32);
    constexpr uint32_t maxBytes = maxStaticRows * StaticCols * sizeof(T);
    constexpr uint32_t scalingAddr = PTO_CEIL(maxAddr + maxBytes, 32);
    static_assert(scalingAddr + maxBytes <= 0x40000, "validShape test UB layout exceeds 256 KB.");

    SrcTile srcTile(ValidRows, ValidCols);
    DstTile dstTile(ValidRows, ValidCols);
    ExpTile expTile(expValidRows, expValidCols);
    MaxTile maxTile(maxValidRows, ValidCols);
    MaxTile scalingTile(maxValidRows, ValidCols);
    SrcGlobal srcGlobal(src);
    DstGlobal dstGlobal(dst);
    ExpGlobal expGlobal(exp);
    TASSIGN(srcTile, srcAddr);
    TASSIGN(dstTile, dstAddr);
    TASSIGN(expTile, expAddr);
    TASSIGN(maxTile, maxAddr);
    TASSIGN(scalingTile, scalingAddr);
    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TQUANT<0, Alg, true>(dstTile, srcTile, &expTile, &maxTile, &scalingTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    TSTORE(expGlobal, expTile);
}

// Reproduce the static tile and runtime validShape combinations emitted by
// PyPTO for actual=[1472,1010], view=[1408,34], tile=[896,1212].
template <typename T, MxQuantAlg Alg, int StaticRows, int StaticCols, int ValidRows, int ValidCols>
__global__ AICORE void runTQuantDNValidShape(
    __gm__ T __in__* src, __gm__ int8_t __out__* dst, __gm__ uint8_t __out__* exp)
{
    static_assert(StaticRows % 64 == 0, "DN static rows must be 64-aligned.");
    static_assert(ValidRows % 64 == 0, "DN valid rows must be 64-aligned.");
    static_assert(ValidRows <= StaticRows && ValidCols <= StaticCols, "validShape must fit in the static tile.");

    constexpr int fp8StaticCols = PTO_CEIL(StaticCols, 32);
    constexpr int expStaticRows = StaticRows / 64;
    constexpr int expStaticCols = PTO_CEIL(StaticCols * 2, 32);
    constexpr int maxStaticRows = StaticRows / 32;
    constexpr int expValidRows = ValidRows / 64;
    constexpr int expValidCols = ValidCols * 2;
    constexpr int maxValidRows = ValidRows / 32;

    using SrcTile = Tile<
        TileType::Vec, T, StaticRows, StaticCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    using DstTile = Tile<
        TileType::Vec, int8_t, StaticRows, fp8StaticCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
        PadValue::Zero>;
    using ExpTile = Tile<
        TileType::Vec, uint8_t, expStaticRows, expStaticCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
        PadValue::Zero>;
    using MaxTile = Tile<
        TileType::Vec, T, maxStaticRows, StaticCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    using SrcGlobal = GlobalTensor<T, Shape<1, 1, 1, ValidRows, ValidCols>, pto::Stride<1, 1, 1, ValidCols, 1>>;
    using DstGlobal = GlobalTensor<int8_t, Shape<1, 1, 1, ValidRows, ValidCols>, pto::Stride<1, 1, 1, ValidCols, 1>>;
    using ExpGlobal =
        GlobalTensor<uint8_t, Shape<1, 1, 1, expValidRows, expValidCols>, pto::Stride<1, 1, 1, expValidCols, 1>>;

    RunTQuantDNValidShapeBody<
        Alg, StaticRows, StaticCols, ValidRows, ValidCols, T, SrcTile, DstTile, ExpTile, MaxTile, SrcGlobal, DstGlobal,
        ExpGlobal>(src, dst, exp);
}

template <typename T, MxQuantAlg Alg, int StaticRows, int StaticCols, int ValidRows, int ValidCols, typename SrcT>
void LaunchTQuantDNValidShape(SrcT* src, int8_t* dst, uint8_t* exp, void* stream)
{
    runTQuantDNValidShape<T, Alg, StaticRows, StaticCols, ValidRows, ValidCols>
        <<<1, nullptr, stream>>>((T*)src, dst, exp);
}

template <int StaticRows, int StaticCols, int ValidRows, int ValidCols>
void LaunchTQuantDNValidShapeFP16(uint16_t* src, int8_t* dst, uint8_t* exp, void* stream)
{
    LaunchTQuantDNValidShape<half, MxQuantAlg::OcpMxFp8E4M3, StaticRows, StaticCols, ValidRows, ValidCols>(
        src, dst, exp, stream);
}

template <int StaticRows, int StaticCols, int ValidRows, int ValidCols>
void LaunchTQuantDNValidShapeBF16(uint16_t* src, int8_t* dst, uint8_t* exp, void* stream)
{
    LaunchTQuantDNValidShape<bfloat16_t, MxQuantAlg::OcpMxFp8E4M3, StaticRows, StaticCols, ValidRows, ValidCols>(
        src, dst, exp, stream);
}

template <int StaticRows, int StaticCols, int ValidRows, int ValidCols>
void LaunchTQuantDNValidShapeNVFP16(uint16_t* src, int8_t* dst, uint8_t* exp, void* stream)
{
    LaunchTQuantDNValidShape<half, MxQuantAlg::NvMxFp8E4M3, StaticRows, StaticCols, ValidRows, ValidCols>(
        src, dst, exp, stream);
}

template <int StaticRows, int StaticCols, int ValidRows, int ValidCols>
void LaunchTQuantDNValidShapeNVBF16(uint16_t* src, int8_t* dst, uint8_t* exp, void* stream)
{
    LaunchTQuantDNValidShape<bfloat16_t, MxQuantAlg::NvMxFp8E4M3, StaticRows, StaticCols, ValidRows, ValidCols>(
        src, dst, exp, stream);
}

template <int StaticRows, int StaticCols, int ValidRows, int ValidCols>
void LaunchTQuantDNValidShapeFP32(uint32_t* src, int8_t* dst, uint8_t* exp, void* stream)
{
    LaunchTQuantDNValidShape<float, MxQuantAlg::OcpMxFp8E4M3, StaticRows, StaticCols, ValidRows, ValidCols>(
        src, dst, exp, stream);
}

template <int StaticRows, int StaticCols, int ValidRows, int ValidCols>
void LaunchTQuantDNValidShapeNVFP32(uint32_t* src, int8_t* dst, uint8_t* exp, void* stream)
{
    LaunchTQuantDNValidShape<float, MxQuantAlg::NvMxFp8E4M3, StaticRows, StaticCols, ValidRows, ValidCols>(
        src, dst, exp, stream);
}

// Full DN vector pipeline: TQUANT(DN) + TMOV(ND->NZ) + TMOV<0>(DN->ZZ).
// Stores FP8 ND, E8M0 DN, per-group max, FP8 NZ, and E8M0 ZZ to GM for comparison.
template <typename T, int M, int N, int N_pad, bool InterleaveExp = false, MxQuantAlg Alg = MxQuantAlg::OcpMxFp8E4M3>
__global__ AICORE void runTQuantDN(
    __gm__ T __in__* src_gm, __gm__ int8_t __out__* fp8_nd_gm, __gm__ uint8_t __out__* e8_dn_gm,
    __gm__ int8_t __out__* fp8_nz_gm, __gm__ uint8_t __out__* e8_zz_gm, __gm__ T __out__* max_dn_gm)
{
    constexpr uint32_t grpSize = 32;
    constexpr uint32_t hatM = M / grpSize;
    constexpr uint32_t pairRows = (M + 63) / 64;
    constexpr uint32_t paddedCols = N_pad;
    constexpr uint32_t groupedColsValid = paddedCols / 32;
    constexpr uint32_t zzGroupedCols = InterleaveExp ? 2 : groupedColsValid;
    constexpr uint32_t numGroupsFlat = M * groupedColsValid;
    constexpr uint32_t numGroupsFlatAligned = PTO_CEIL(numGroupsFlat, 32);
    constexpr uint32_t paddedRows16 = PTO_CEIL(M, FRACTAL_NZ_ROW);
    constexpr uint32_t virtualRow = paddedRows16 + 1;

    using SrcTile =
        Tile<TileType::Vec, T, M, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    using DstFP8Tile = Tile<
        TileType::Vec, int8_t, M, paddedCols, BLayout::RowMajor, M, paddedCols, SLayout::NoneBox, 512, PadValue::Zero>;
    using MaxTile =
        Tile<TileType::Vec, T, hatM, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    using ScalingTile =
        Tile<TileType::Vec, T, hatM, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    using E8NdTile = Tile<
        TileType::Vec, uint8_t, hatM, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;

    constexpr uint32_t e8DnRows = InterleaveExp ? pairRows : hatM;
    constexpr uint32_t e8DnCols = InterleaveExp ? 2 * paddedCols : paddedCols;
    using E8DnTile = Tile<
        TileType::Vec, uint8_t, e8DnRows, e8DnCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;

    using E8ZzTile = Tile<
        TileType::Vec, uint8_t, paddedRows16, zzGroupedCols, BLayout::RowMajor, -1, -1, SLayout::RowMajor, 32,
        PadValue::Zero>;
    using E8StoreTile = Tile<
        TileType::Vec, uint8_t, 1, numGroupsFlatAligned, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
        PadValue::Zero>;

    using Fp8NZTile = Tile<
        TileType::Vec, int8_t, virtualRow, paddedCols, BLayout::ColMajor, M, paddedCols, SLayout::RowMajor, 512,
        PadValue::Null, CompactMode::RowPlusOne>;

    constexpr uint32_t colBlkCount = paddedCols / 16;
    constexpr uint32_t hatP = hatM / 2;
    constexpr uint32_t tmpBufSize =
        (BLOCK_SIZE / sizeof(uint16_t) +
         (colBlkCount > hatP ? colBlkCount : hatP) * (hatP > colBlkCount ? hatP : colBlkCount) +
         BLOCK_SIZE / sizeof(uint16_t)) *
        sizeof(uint16_t);
    constexpr uint32_t tmpBufSizeAligned = PTO_CEIL(tmpBufSize, 32);

    using TmpTile = Tile<
        TileType::Vec, uint8_t, 1, tmpBufSizeAligned, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;

    SrcTile srcTile(M, paddedCols);
    DstFP8Tile fp8Tile;
    MaxTile maxPerGpTile(hatM, paddedCols);
    ScalingTile scalingTile(hatM, paddedCols);
    E8NdTile e8Tile(hatM, paddedCols);
    E8DnTile e8DnTile(e8DnRows, e8DnCols);
    E8ZzTile e8ZzTile(paddedRows16, zzGroupedCols);
    E8StoreTile e8StoreTile(1, numGroupsFlatAligned);
    Fp8NZTile fp8TileNZ;
    TmpTile tmpTile(1, tmpBufSizeAligned);

    using SrcGlobal = GlobalTensor<T, Shape<1, 1, 1, M, N_pad>, pto::Stride<1, 1, 1, N_pad, 1>>;
    SrcGlobal srcGlobal(src_gm);

    using DstFp8NdGlobal = GlobalTensor<int8_t, Shape<1, 1, 1, M, paddedCols>, pto::Stride<1, 1, 1, paddedCols, 1>>;
    DstFp8NdGlobal fp8NdGlobal(fp8_nd_gm);

    using DstMaxGlobal = GlobalTensor<T, Shape<1, 1, 1, hatM, paddedCols>, pto::Stride<1, 1, 1, paddedCols, 1>>;
    DstMaxGlobal maxGlobal(max_dn_gm);

    using DstE8DnGlobal = GlobalTensor<uint8_t, Shape<1, 1, 1, e8DnRows, e8DnCols>, pto::Stride<1, 1, 1, e8DnCols, 1>>;
    DstE8DnGlobal e8DnGlobal(e8_dn_gm);

    using DstE8Global =
        GlobalTensor<uint8_t, Shape<1, 1, 1, 1, numGroupsFlatAligned>, pto::Stride<1, 1, 1, numGroupsFlatAligned, 1>>;
    DstE8Global e8Global(e8_zz_gm);

    using DstFp8GlobalNZ = GlobalTensor<
        int8_t, TileShape2D<int8_t, M, paddedCols, Layout::NZ>, BaseShape2D<int8_t, M, paddedCols, Layout::NZ>,
        Layout::NZ>;
    DstFp8GlobalNZ fp8GlobalNZ((__gm__ int8_t*)fp8_nz_gm);

    constexpr uint32_t srcTileBytes = M * paddedCols * sizeof(T);
    constexpr uint32_t maxTileBytes = hatM * paddedCols * sizeof(T);
    constexpr uint32_t scalingTileBytes = hatM * paddedCols * sizeof(T);
    constexpr uint32_t e8TileBytes = hatM * paddedCols;
    constexpr uint32_t e8DnTileBytes = hatM * paddedCols;
    constexpr uint32_t fp8TileBytes = M * paddedCols;

    // Keep source and destination UB tiles orthogonal: place fp8Tile after all
    // input/work tiles so TQUANT reads src and writes dst to non-overlapping
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
    // workTileEnd marks the end of the TQUANT input-side tiles; fp8Tile now lives
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

    // Generic DN API: grp_axis=0 (groups on axis 0), single MxQuantAlg tag. The
    // exponent is written into e8Tile; copy it into e8DnTile so the UB tile shape
    // matches the GM shape exactly.
    if constexpr (InterleaveExp) {
        TQUANT<0, Alg, true>(fp8Tile, srcTile, &e8DnTile, &maxPerGpTile, &scalingTile);
    } else {
        TQUANT<0, Alg>(fp8Tile, srcTile, &e8Tile, &maxPerGpTile, &scalingTile);
        TMOV(e8DnTile, e8Tile);
    }

    // Data ND->NZ (stock 2-arg TMOV) and exponent DN->ZZ (grp_axis=0 TMOV).
    TMOV(fp8TileNZ, fp8Tile);
    if constexpr (!InterleaveExp) {
        TMOV<0>(e8ZzTile, e8DnTile, tmpTile);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(fp8NdGlobal, fp8Tile);
    TSTORE(e8DnGlobal, e8DnTile);
    TSTORE(maxGlobal, maxPerGpTile);
    TSTORE(fp8GlobalNZ, fp8TileNZ);
    TSTORE(e8Global, e8StoreTile);
}

template <int M, int N, int N_pad>
void LaunchTQuantDNInterleaved(
    uint16_t* src, int8_t* fp8_nd, uint8_t* e8_dn, int8_t* fp8_nz, uint8_t* e8_zz, uint16_t* max_dn, void* stream)
{
    runTQuantDN<bfloat16_t, M, N, N_pad, true>
        <<<1, nullptr, stream>>>((bfloat16_t*)src, fp8_nd, e8_dn, fp8_nz, e8_zz, (bfloat16_t*)max_dn);
}

template <int M, int N, int N_pad>
void LaunchTQuantDN(
    uint16_t* src, int8_t* fp8_nd, uint8_t* e8_dn, int8_t* fp8_nz, uint8_t* e8_zz, uint16_t* max_dn, void* stream)
{
    runTQuantDN<bfloat16_t, M, N, N_pad>
        <<<1, nullptr, stream>>>((bfloat16_t*)src, fp8_nd, e8_dn, fp8_nz, e8_zz, (bfloat16_t*)max_dn);
}

template <int M, int N, int N_pad>
void LaunchTQuantDN_fp32(
    uint32_t* src, int8_t* fp8_nd, uint8_t* e8_dn, int8_t* fp8_nz, uint8_t* e8_zz, uint32_t* max_dn, void* stream)
{
    runTQuantDN<float, M, N, N_pad><<<1, nullptr, stream>>>((float*)src, fp8_nd, e8_dn, fp8_nz, e8_zz, (float*)max_dn);
}

template <int M, int N, int N_pad>
void LaunchTQuantDN_fp32_interleaved(
    uint32_t* src, int8_t* fp8_nd, uint8_t* e8_dn, int8_t* fp8_nz, uint8_t* e8_zz, uint32_t* max_dn, void* stream)
{
    runTQuantDN<float, M, N, N_pad, true>
        <<<1, nullptr, stream>>>((float*)src, fp8_nd, e8_dn, fp8_nz, e8_zz, (float*)max_dn);
}

template <int M, int N, int N_pad>
void LaunchTQuantDN_nv_interleaved(
    uint16_t* src, int8_t* fp8_nd, uint8_t* e8_dn, int8_t* fp8_nz, uint8_t* e8_zz, uint16_t* max_dn, void* stream)
{
    runTQuantDN<bfloat16_t, M, N, N_pad, true, MxQuantAlg::NvMxFp8E4M3>
        <<<1, nullptr, stream>>>((bfloat16_t*)src, fp8_nd, e8_dn, fp8_nz, e8_zz, (bfloat16_t*)max_dn);
}

template <int M, int N, int N_pad>
void LaunchTQuantDN_nv(
    uint16_t* src, int8_t* fp8_nd, uint8_t* e8_dn, int8_t* fp8_nz, uint8_t* e8_zz, uint16_t* max_dn, void* stream)
{
    runTQuantDN<bfloat16_t, M, N, N_pad, false, MxQuantAlg::NvMxFp8E4M3>
        <<<1, nullptr, stream>>>((bfloat16_t*)src, fp8_nd, e8_dn, fp8_nz, e8_zz, (bfloat16_t*)max_dn);
}

template <int M, int N, int N_pad, bool InterleaveExp>
void LaunchTQuantDN_fp32_nv(
    uint32_t* src, int8_t* fp8_nd, uint8_t* e8_dn, int8_t* fp8_nz, uint8_t* e8_zz, uint32_t* max_dn, void* stream)
{
    runTQuantDN<float, M, N, N_pad, InterleaveExp, MxQuantAlg::NvMxFp8E4M3>
        <<<1, nullptr, stream>>>((float*)src, fp8_nd, e8_dn, fp8_nz, e8_zz, (float*)max_dn);
}

// MXFP4 (E2M1) DN kernel: quantizes src[M,N_pad] to packed FP4 plus per-group
// e8m0/max tiles. TQUANT writes FP4 as a flat float4_e2m1x2_t tile; a uint8_t
// TSTORE tile is assigned via TASSIGN to the same UB region so TSTORE reads it in-place
// (no copy/intrinsics needed).
template <typename T, int M, int N, int N_pad, bool InterleaveExp = false, MxQuantAlg Alg = MxQuantAlg::OcpMxFp4E2M1>
__global__ AICORE void runTQuantDN_MXFP4(
    __gm__ T __in__* src_gm, __gm__ uint8_t __out__* fp4_nd_gm, __gm__ uint8_t __out__* e8_dn_gm,
    __gm__ uint8_t __out__* fp4_nz_gm, __gm__ T __out__* max_dn_gm)
{
    constexpr uint32_t grpSize = 32;
    constexpr uint32_t hatM = M / grpSize;
    constexpr uint32_t pairRows = (M + 63) / 64;
    constexpr uint32_t paddedCols = N_pad;
    constexpr uint32_t packedCols = paddedCols / 2;
    constexpr uint32_t fp4FlatAligned = PTO_CEIL(M * packedCols, 32);

    using SrcTile =
        Tile<TileType::Vec, T, M, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    using MaxTile =
        Tile<TileType::Vec, T, hatM, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    using ScalingTile =
        Tile<TileType::Vec, T, hatM, paddedCols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    constexpr uint32_t e8Rows = InterleaveExp ? pairRows : hatM;
    constexpr uint32_t e8Cols = InterleaveExp ? 2 * paddedCols : paddedCols;
    using E8Tile =
        Tile<TileType::Vec, uint8_t, e8Rows, e8Cols, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    // TQUANT output tile: flat float4_e2m1x2_t (element = 0.5 byte -> bytes = M*packedCols).
    using DstFP4Tile = Tile<
        TileType::Vec, float4_e2m1x2_t, 1, fp4FlatAligned, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512,
        PadValue::Zero>;
    // 2D RowMajor view of the same packed FP4 UB data for ND->NZ.
    // Use uint8_t for the view so the standard byte ND->NZ lowering is used.
    using Fp4Tile2D = Tile<
        TileType::Vec, uint8_t, M, packedCols, BLayout::RowMajor, M, packedCols, SLayout::NoneBox, 512, PadValue::Zero>;
    // NZ tile for packed FP4: use uint8_t as the element type so the standard
    // byte ND->NZ lowering (32B blocks) is used. Each byte holds 2 FP4 values.
    constexpr uint32_t paddedRows16 = PTO_CEIL(M, FRACTAL_NZ_ROW);
    constexpr uint32_t virtualRow = paddedRows16 + 1;
    using Fp4NZTile = Tile<
        TileType::Vec, uint8_t, virtualRow, packedCols, BLayout::ColMajor, M, packedCols, SLayout::RowMajor, 512,
        PadValue::Null, CompactMode::RowPlusOne>;
    // TSTORE tile: uint8_t [M, packedCols] view over the same UB region that TQUANT
    // wrote the packed FP4 data into (TASSIGN to fp4Addr). No copy needed — TSTORE
    // reads the bytes in-place. Valid extents are DYNAMIC for runtime sizing.
    using DstBytesTile = Tile<
        TileType::Vec, uint8_t, M, packedCols, BLayout::RowMajor, M, packedCols, SLayout::NoneBox, 512, PadValue::Zero>;

    constexpr uint32_t srcBytes = M * paddedCols * sizeof(T);
    constexpr uint32_t maxBytes = hatM * paddedCols * sizeof(T);
    constexpr uint32_t scalingBytes = hatM * paddedCols * sizeof(T);
    constexpr uint32_t e8Bytes = hatM * paddedCols;
    constexpr uint32_t fp4Bytes = M * packedCols;
    constexpr uint32_t fp4NZBytes = virtualRow * packedCols;

    constexpr uint32_t srcAddr = 0x0;
    constexpr uint32_t maxAddr = PTO_CEIL(srcAddr + srcBytes, 0x20);
    constexpr uint32_t scalingAddr = PTO_CEIL(maxAddr + maxBytes, 0x20);
    constexpr uint32_t e8Addr = PTO_CEIL(scalingAddr + scalingBytes, 0x20);
    constexpr uint32_t fp4Addr = PTO_CEIL(e8Addr + e8Bytes, 0x20);
    constexpr uint32_t fp4NZAddr = PTO_CEIL(fp4Addr + fp4Bytes, 0x20);
    constexpr uint32_t layoutEnd = PTO_CEIL(fp4NZAddr + fp4NZBytes, 0x100);
    static_assert(layoutEnd <= 0x40000, "MXFP4 DN UB layout exceeds 256 KB.");

    SrcTile srcTile(M, paddedCols);
    MaxTile maxTile(hatM, paddedCols);
    ScalingTile scalingTile(hatM, paddedCols);
    E8Tile e8Tile(e8Rows, e8Cols);
    DstFP4Tile fp4Tile;
    Fp4Tile2D fp4Tile2D;
    Fp4NZTile fp4NZTile;
    DstBytesTile fp4BytesTile;

    TASSIGN(srcTile, srcAddr);
    TASSIGN(maxTile, maxAddr);
    TASSIGN(scalingTile, scalingAddr);
    TASSIGN(e8Tile, e8Addr);
    TASSIGN(fp4Tile, fp4Addr);
    TASSIGN(fp4Tile2D, fp4Addr);
    TASSIGN(fp4NZTile, fp4NZAddr);
    TASSIGN(fp4BytesTile, fp4Addr);

    using SrcGlobal = GlobalTensor<T, Shape<1, 1, 1, M, N_pad>, pto::Stride<1, 1, 1, N_pad, 1>>;
    using Fp4Global = GlobalTensor<uint8_t, Shape<1, 1, 1, M, packedCols>, pto::Stride<1, 1, 1, packedCols, 1>>;
    using Fp4GlobalNZ = GlobalTensor<
        uint8_t, TileShape2D<uint8_t, M, packedCols, Layout::NZ>, BaseShape2D<uint8_t, M, packedCols, Layout::NZ>,
        Layout::NZ>;
    using MaxGlobal = GlobalTensor<T, Shape<1, 1, 1, hatM, paddedCols>, pto::Stride<1, 1, 1, paddedCols, 1>>;
    using E8Global = GlobalTensor<uint8_t, Shape<1, 1, 1, e8Rows, e8Cols>, pto::Stride<1, 1, 1, e8Cols, 1>>;

    SrcGlobal srcGlobal(src_gm);
    Fp4Global fp4Global(fp4_nd_gm);
    Fp4GlobalNZ fp4GlobalNZ(fp4_nz_gm);
    MaxGlobal maxGlobal(max_dn_gm);
    E8Global e8Global(e8_dn_gm);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Generic DN API: grp_axis=0, single MxQuantAlg tag. TQUANT writes the exponent
    // into e8Tile; the kernel TSTOREs e8Tile directly (shape already matches GM).
    if constexpr (InterleaveExp)
        TQUANT<0, Alg, true>(fp4Tile, srcTile, &e8Tile, &maxTile, &scalingTile);
    else
        TQUANT<0, Alg>(fp4Tile, srcTile, &e8Tile, &maxTile, &scalingTile);

    // Packed FP4 ND->NZ: source is RowMajor [M, packedCols] of float4_e2m1x2_t.
    TMOV(fp4NZTile, fp4Tile2D);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(fp4Global, fp4BytesTile);
    TSTORE(fp4GlobalNZ, fp4NZTile);
    TSTORE(maxGlobal, maxTile);
    TSTORE(e8Global, e8Tile);
}

template <int M, int N, int N_pad>
void LaunchTQuantDN_MXFP4_bf16(
    uint16_t* src, uint8_t* fp4_nd, uint8_t* e8_dn, uint8_t* fp4_nz, uint16_t* max_dn, void* stream)
{
    runTQuantDN_MXFP4<bfloat16_t, M, N, N_pad>
        <<<1, nullptr, stream>>>((bfloat16_t*)src, fp4_nd, e8_dn, fp4_nz, (bfloat16_t*)max_dn);
}

template <int M, int N, int N_pad>
void LaunchTQuantDN_MXFP4_fp16(
    uint16_t* src, uint8_t* fp4_nd, uint8_t* e8_dn, uint8_t* fp4_nz, uint16_t* max_dn, void* stream)
{
    runTQuantDN_MXFP4<half, M, N, N_pad><<<1, nullptr, stream>>>((half*)src, fp4_nd, e8_dn, fp4_nz, (half*)max_dn);
}

template <int M, int N, int N_pad>
void LaunchTQuantDN_MXFP4_bf16_interleaved(
    uint16_t* src, uint8_t* fp4_nd, uint8_t* e8_dn, uint8_t* fp4_nz, uint16_t* max_dn, void* stream)
{
    runTQuantDN_MXFP4<bfloat16_t, M, N, N_pad, true>
        <<<1, nullptr, stream>>>((bfloat16_t*)src, fp4_nd, e8_dn, fp4_nz, (bfloat16_t*)max_dn);
}

template <int M, int N, int N_pad>
void LaunchTQuantDN_MXFP4_fp16_interleaved(
    uint16_t* src, uint8_t* fp4_nd, uint8_t* e8_dn, uint8_t* fp4_nz, uint16_t* max_dn, void* stream)
{
    runTQuantDN_MXFP4<half, M, N, N_pad, true>
        <<<1, nullptr, stream>>>((half*)src, fp4_nd, e8_dn, fp4_nz, (half*)max_dn);
}

template <typename T, int M, int N, int N_pad, bool InterleaveExp>
void LaunchTQuantDN_MXFP4_nv(
    uint16_t* src, uint8_t* fp4_nd, uint8_t* e8_dn, uint8_t* fp4_nz, uint16_t* max_dn, void* stream)
{
    runTQuantDN_MXFP4<T, M, N, N_pad, InterleaveExp, MxQuantAlg::NvMxFp4E2M1>
        <<<1, nullptr, stream>>>((T*)src, fp4_nd, e8_dn, fp4_nz, (T*)max_dn);
}

template <int M, int N, int N_pad, bool InterleaveExp>
void LaunchTQuantDN_MXFP4_nv_bf16(
    uint16_t* src, uint8_t* fp4_nd, uint8_t* e8_dn, uint8_t* fp4_nz, uint16_t* max_dn, void* stream)
{
    LaunchTQuantDN_MXFP4_nv<bfloat16_t, M, N, N_pad, InterleaveExp>(src, fp4_nd, e8_dn, fp4_nz, max_dn, stream);
}

template <int M, int N, int N_pad, bool InterleaveExp>
void LaunchTQuantDN_MXFP4_nv_fp16(
    uint16_t* src, uint8_t* fp4_nd, uint8_t* e8_dn, uint8_t* fp4_nz, uint16_t* max_dn, void* stream)
{
    LaunchTQuantDN_MXFP4_nv<half, M, N, N_pad, InterleaveExp>(src, fp4_nd, e8_dn, fp4_nz, max_dn, stream);
}

template void LaunchTQuantDN<64, 128, 128>(uint16_t*, int8_t*, uint8_t*, int8_t*, uint8_t*, uint16_t*, void*);
template void LaunchTQuantDN_fp32_nv<128, 128, 128, true>(
    uint32_t*, int8_t*, uint8_t*, int8_t*, uint8_t*, uint32_t*, void*);
template void LaunchTQuantDNValidShapeBF16<512, 48, 64, 24>(uint16_t*, int8_t*, uint8_t*, void*);
template void LaunchTQuantDNValidShapeNVFP16<896, 48, 896, 34>(uint16_t*, int8_t*, uint8_t*, void*);
template void LaunchTQuantDN_MXFP4_fp16<64, 128, 128>(uint16_t*, uint8_t*, uint8_t*, uint8_t*, uint16_t*, void*);
template void LaunchTQuantDN_MXFP4_nv_bf16<128, 128, 128, true>(
    uint16_t*, uint8_t*, uint8_t*, uint8_t*, uint16_t*, void*);
} // namespace TQuantDNTest
