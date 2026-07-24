/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

// A6 (dav-9201) TQUANT BF16 -> HiF4 ST harness.
//
// Calls TQUANT<1, MxQuantAlg::Hif4>(dst, src, &exp, &max, &scaling) — the full
// HiF4 pipeline runs inside TQuant.hpp. After it returns, the UB tiles contain
// the carved sub-regions (max4/8/64, Ea/Eb/Ec, exp_dst, scaling, fp4). This
// kernel dumps every sub-region to a separate GM output for ST verification.
//
// GM<->UB moves use the raw A6 DMA intrinsics (copy_gm_to_ubuf_align_v2 /
// copy_ubuf_to_gm_align_v2) because TLOAD/TSTORE don't compile for A6 yet.
// TASSIGN + TQUANT themselves work fine on A6.

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/common/type.hpp>
#include "acl/acl.h"

using namespace pto;

#define PTO_CEIL(x, y) ((((x) + (y) - 1) / (y)) * (y))

namespace TQuantHif4A6 {

constexpr uint32_t TQUANT_A6_UB_ALIGN_BYTES = 32;
constexpr uint32_t TQUANT_A6_UB_SIZE_BYTES = 384 * 1024;

// ---- A6 raw DMA helpers (TLOAD/TSTORE don't compile for A6) ---------------

PTO_INTERNAL void A6LoadGmToUb(__ubuf__ void* dst, __gm__ void* src, uint32_t lenBytes)
{
    copy_gm_to_ubuf_align_v2(
        (__ubuf__ bfloat16_t*)dst, (__gm__ bfloat16_t*)src, /*sid=*/0,
        /*burst_num=*/1, /*burst_len=*/lenBytes, /*left_pad=*/0, /*right_pad=*/0,
        /*data_select_bit=*/0, /*pre_allocation=*/1, /*l2_cache_ctl=*/0,
        /*burst_src_stride=*/(uint64_t)lenBytes, /*burst_dst_stride=*/(uint64_t)lenBytes,
        /*non_eod_ctrl=*/false);
}

PTO_INTERNAL void A6StoreUbToGm(__gm__ void* dst, __ubuf__ void* src, uint32_t lenBytes)
{
    copy_ubuf_to_gm_align_v2(
        (__gm__ bfloat16_t*)dst, (__ubuf__ bfloat16_t*)src, /*sid=*/0,
        /*burst_num=*/1, /*burst_len=*/lenBytes,
        /*rsw_ctrl=*/false, /*rsw_packet_ctrl=*/false, /*rsw_buffer_size=*/0,
        /*l2_cache_ctl=*/0, /*burst_dst_stride=*/(uint64_t)lenBytes,
        /*burst_src_stride=*/(uint64_t)lenBytes, /*non_eod_ctrl=*/false);
}

// ---- UB layout computation (mirrors TQuant_Hif4_Impl carving) -------------
//
// For [validRows, validCols] BF16 with validCols divisible by 64:
//   src:       validRows * validCols * 2 bytes (BF16)
//   max:       3 sub-regions, each [validRows, gp<k>StaticCols] BF16, 32-B-aligned rows
//   exp:       Ea/Eb/Ec sub-regions + exp_dst tail, uint8
//   scaling:   validRows * (validCols/4) * 2 bytes (BF16, per-4)
//   dst:       validRows * validCols / 2 bytes (packed FP4)

template <int validRows, int validCols>
struct Hif4Layout {
    static constexpr uint32_t ubAlignBytes = 32;
    static constexpr uint32_t totalElem = validRows * validCols;

    // For the CONTINUOUS case, all stage outputs are stored flat-contiguously
    // in UB (no row padding). The _Cont stage functions write sequentially.
    //
    // Max scratch: flat BF16 arrays. No maxGp64 (Ea computed directly).
    static constexpr uint32_t maxGp4Bytes = (totalElem / 4) * sizeof(bfloat16_t);
    static constexpr uint32_t maxGp8Bytes = (totalElem / 8) * sizeof(bfloat16_t);
    static constexpr uint32_t maxTotalBytes = maxGp4Bytes + maxGp8Bytes;

    // Exp: Ea stored as zero-extended BF16 (2B per exponent).
    // Eb stored as packed bits, upsampled 2× for frequency matching with Ec.
    //   Eb: 1 bit per 8-elem group, packed → totalElem/(8*8) bytes, ×2 upsample
    // Ec stored as packed predicate bits (psts PK mode).
    static constexpr uint32_t eaDataBytes = (totalElem / 64) * sizeof(bfloat16_t); // 2B per Ea
    static constexpr uint32_t ebDataBytes = ((totalElem / 8) / 8) * 2;             // packed bits ×2 upsample
    static constexpr uint32_t ecDataBytes = (totalElem / 4) / 8;                   // packed bits → bytes
    static constexpr uint32_t expDstBytes = (totalElem / 64) * 4;
    static constexpr uint32_t expTotalBytes =
        PTO_CEIL(eaDataBytes + ebDataBytes + ecDataBytes + expDstBytes, ubAlignBytes);

    // Tile sizes
    static constexpr uint32_t srcBytes = totalElem * sizeof(bfloat16_t);
    static constexpr uint32_t scalingBytes =
        (totalElem / 2) * sizeof(bfloat16_t);           // half input size: INTLV does 2×, US_B16 does 2×
    static constexpr uint32_t dstBytes = totalElem / 2; // packed FP4 (2 codes/byte)

    // UB offsets (each region 32-B aligned)
    static constexpr uint64_t srcOffset = 0;
    static constexpr uint64_t maxOffset = PTO_CEIL(srcOffset + srcBytes, ubAlignBytes);
    static constexpr uint64_t expOffset = PTO_CEIL(maxOffset + maxTotalBytes, ubAlignBytes);
    static constexpr uint64_t scalingOffset = PTO_CEIL(expOffset + expTotalBytes, ubAlignBytes);
    static constexpr uint64_t dstOffset = PTO_CEIL(scalingOffset + scalingBytes, ubAlignBytes);
    static constexpr uint64_t ubTotal = dstOffset + dstBytes;

    // Sub-region offsets within max tile
    static constexpr uint64_t maxGp4Off = maxOffset;
    static constexpr uint64_t maxGp8Off = maxOffset + maxGp4Bytes;
    static constexpr uint64_t maxGp64Off = maxOffset + maxGp4Bytes + maxGp8Bytes;

    // Sub-region offsets within exp tile
    static constexpr uint64_t eaOff = expOffset;
    static constexpr uint64_t ebOff = expOffset + eaDataBytes;
    static constexpr uint64_t ecOff = expOffset + eaDataBytes + ebDataBytes;
    static constexpr uint64_t expDstOff = expOffset + eaDataBytes + ebDataBytes + ecDataBytes;

    static_assert(ubTotal <= TQUANT_A6_UB_SIZE_BYTES, "HiF4 UB layout exceeds 384 KB");
};

// ---- Kernel --------------------------------------------------------------

template <int validRows, int validCols>
__global__ AICORE void runTQuantHif4A6(
    __gm__ bfloat16_t __in__* src, __gm__ uint8_t __out__* out_max4, __gm__ uint8_t __out__* out_max8,
    __gm__ uint8_t __out__* out_ea, __gm__ uint8_t __out__* out_eb, __gm__ uint8_t __out__* out_ec,
    __gm__ uint8_t __out__* out_exp_dst, __gm__ uint8_t __out__* out_fp4, __gm__ bfloat16_t __out__* out_scale)
{
    using Spec = Hif4Layout<validRows, validCols>;

    // ---- Declare tiles and assign UB offsets ----
    using SrcTile = Tile<TileType::Vec, bfloat16_t, validRows, validCols, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, float4_e1m2x2_t, 1, Spec::dstBytes, BLayout::RowMajor, -1, -1>;
    using ExpTile = Tile<TileType::Vec, uint8_t, 1, Spec::expTotalBytes, BLayout::RowMajor, -1, -1>;
    using MaxTile =
        Tile<TileType::Vec, bfloat16_t, 1, Spec::maxTotalBytes / sizeof(bfloat16_t), BLayout::RowMajor, -1, -1>;
    using ScalingTile =
        Tile<TileType::Vec, bfloat16_t, 1, Spec::scalingBytes / sizeof(bfloat16_t), BLayout::RowMajor, -1, -1>;

    SrcTile srcTile(validRows, validCols);
    DstTile dstTile(1, Spec::dstBytes);
    ExpTile expTile(1, Spec::expTotalBytes);
    MaxTile maxTile(1, Spec::maxTotalBytes / sizeof(bfloat16_t));
    ScalingTile scalingTile(1, Spec::scalingBytes / sizeof(bfloat16_t));

    TASSIGN(srcTile, Spec::srcOffset);
    TASSIGN(maxTile, Spec::maxOffset);
    TASSIGN(expTile, Spec::expOffset);
    TASSIGN(scalingTile, Spec::scalingOffset);
    TASSIGN(dstTile, Spec::dstOffset);

    // ---- MTE2: load src from GM to UB ----
    A6LoadGmToUb(
        reinterpret_cast<__ubuf__ void*>(Spec::srcOffset), reinterpret_cast<__gm__ void*>(src), Spec::srcBytes);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // ---- V: run the full HiF4 pipeline via TQUANT ----
    TQUANT<1, pto::MxQuantAlg::Hif4, DstTile, SrcTile, ExpTile, MaxTile, ScalingTile>(
        dstTile, srcTile, &expTile, &maxTile, &scalingTile);

    // ---- Dump all sub-regions to GM ----
    // V must finish before MTE3 reads the UB tiles.
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    // Max scratch: max4/8 are flat-contiguous in UB.
    A6StoreUbToGm(
        reinterpret_cast<__gm__ void*>(out_max4), reinterpret_cast<__ubuf__ void*>(Spec::maxGp4Off), Spec::maxGp4Bytes);
    A6StoreUbToGm(
        reinterpret_cast<__gm__ void*>(out_max8), reinterpret_cast<__ubuf__ void*>(Spec::maxGp8Off), Spec::maxGp8Bytes);

    // Exponents: Ea/Eb/Ec carved from exp tile.
    A6StoreUbToGm(
        reinterpret_cast<__gm__ void*>(out_ea), reinterpret_cast<__ubuf__ void*>(Spec::eaOff), Spec::eaDataBytes);
    A6StoreUbToGm(
        reinterpret_cast<__gm__ void*>(out_eb), reinterpret_cast<__ubuf__ void*>(Spec::ebOff), Spec::ebDataBytes);
    A6StoreUbToGm(
        reinterpret_cast<__gm__ void*>(out_ec), reinterpret_cast<__ubuf__ void*>(Spec::ecOff), Spec::ecDataBytes);
    A6StoreUbToGm(
        reinterpret_cast<__gm__ void*>(out_exp_dst), reinterpret_cast<__ubuf__ void*>(Spec::expDstOff),
        Spec::expDstBytes);

    // FP4 + scaling
    A6StoreUbToGm(
        reinterpret_cast<__gm__ void*>(out_fp4), reinterpret_cast<__ubuf__ void*>(Spec::dstOffset), Spec::dstBytes);
    A6StoreUbToGm(
        reinterpret_cast<__gm__ void*>(out_scale), reinterpret_cast<__ubuf__ void*>(Spec::scalingOffset),
        Spec::scalingBytes);
}

// ---- Host launch wrapper ----

template <int validRows, int validCols>
void LaunchTQuantHif4A6(
    uint16_t* src, uint16_t* max4, uint16_t* max8, uint8_t* ea, uint8_t* eb, uint8_t* ec, uint8_t* exp_dst,
    uint8_t* fp4, uint16_t* scale, void* stream)
{
    runTQuantHif4A6<validRows, validCols><<<1, nullptr, stream>>>(
        reinterpret_cast<bfloat16_t*>(src), reinterpret_cast<uint8_t*>(max4), reinterpret_cast<uint8_t*>(max8), ea, eb,
        ec, exp_dst, fp4, reinterpret_cast<bfloat16_t*>(scale));
}

} // namespace TQuantHif4A6

// ---- VADD stub probe (basic functionality/connection test) -----------------
// Simple BF16 element-wise add: dst = src0 + src1. Exercises the MTE2 → V → MTE3
// pipeline with raw A6 DMA intrinsics. Kept alongside HiF4 as a sanity check.

namespace TQuantVaddA6 {

template <int validRows, int validCols>
__global__ AICORE void runVaddA6(
    __gm__ bfloat16_t __out__* out, __gm__ bfloat16_t __in__* src0, __gm__ bfloat16_t __in__* src1)
{
    constexpr uint32_t lenBytes = validRows * validCols * sizeof(bfloat16_t);
    constexpr uint32_t alignBytes = 32;
    constexpr uint64_t SRC0_UB = 0x0;
    constexpr uint64_t SRC1_UB = ((SRC0_UB + lenBytes + alignBytes - 1) / alignBytes) * alignBytes;
    constexpr uint64_t DST_UB = ((SRC1_UB + lenBytes + alignBytes - 1) / alignBytes) * alignBytes;

    __ubuf__ bfloat16_t* src0Ub = reinterpret_cast<__ubuf__ bfloat16_t*>(SRC0_UB);
    __ubuf__ bfloat16_t* src1Ub = reinterpret_cast<__ubuf__ bfloat16_t*>(SRC1_UB);
    __ubuf__ bfloat16_t* dstUb = reinterpret_cast<__ubuf__ bfloat16_t*>(DST_UB);

    TQuantHif4A6::A6LoadGmToUb(
        reinterpret_cast<__ubuf__ void*>(src0Ub), reinterpret_cast<__gm__ void*>(src0), lenBytes);
    TQuantHif4A6::A6LoadGmToUb(
        reinterpret_cast<__ubuf__ void*>(src1Ub), reinterpret_cast<__gm__ void*>(src1), lenBytes);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    __VEC_SCOPE__
    {
        constexpr uint32_t elementsPerRepeat = REPEAT_BYTE / sizeof(bfloat16_t);
        constexpr uint32_t totalElements = validRows * validCols;
        constexpr uint32_t repeatCount = totalElements / elementsPerRepeat;
        RegTensor<bfloat16_t> vreg0, vreg1, vregDst;
        MaskReg pregAll = pset_b16(PAT_ALL);
        for (uint16_t i = 0; i < static_cast<uint16_t>(repeatCount); ++i) {
            vlds(vreg0, src0Ub, i * elementsPerRepeat, NORM);
            vlds(vreg1, src1Ub, i * elementsPerRepeat, NORM);
            vadd(vregDst, vreg0, vreg1, pregAll, MODE_ZEROING);
            vsts(vregDst, dstUb, i * elementsPerRepeat, NORM_B16, pregAll);
        }
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TQuantHif4A6::A6StoreUbToGm(reinterpret_cast<__gm__ void*>(out), reinterpret_cast<__ubuf__ void*>(dstUb), lenBytes);
}

template <int validRows, int validCols>
void LaunchVaddA6(uint16_t* dst, uint16_t* src0, uint16_t* src1, void* stream)
{
    runVaddA6<validRows, validCols><<<1, nullptr, stream>>>(
        reinterpret_cast<bfloat16_t*>(dst), reinterpret_cast<bfloat16_t*>(src0), reinterpret_cast<bfloat16_t*>(src1));
}

} // namespace TQuantVaddA6

// Explicit instantiation.
// build.py passes -DTQUANT_KERNEL_HIF4 -DTQUANT_KERNEL_ROWS=N -DTQUANT_KERNEL_COLS=M
// to compile exactly one size for the platform .bin.
// Without any TQUANT_KERNEL_* define (cmake/GTest path), all sizes are instantiated.
#ifdef TQUANT_KERNEL_HIF4
#ifndef TQUANT_KERNEL_ROWS
#define TQUANT_KERNEL_ROWS 128
#endif
#ifndef TQUANT_KERNEL_COLS
#define TQUANT_KERNEL_COLS 128
#endif
template void TQuantHif4A6::LaunchTQuantHif4A6<TQUANT_KERNEL_ROWS, TQUANT_KERNEL_COLS>(
    uint16_t* src, uint16_t* max4, uint16_t* max8, uint8_t* ea, uint8_t* eb, uint8_t* ec, uint8_t* exp_dst,
    uint8_t* fp4, uint16_t* scale, void* stream);
#elif defined(TQUANT_KERNEL_VADD)
template void TQuantVaddA6::LaunchVaddA6<TQUANT_KERNEL_ROWS, TQUANT_KERNEL_COLS>(
    uint16_t* dst, uint16_t* src0, uint16_t* src1, void* stream);
#else
// cmake/GTest path — instantiate all sizes
template void TQuantHif4A6::LaunchTQuantHif4A6<128, 128>(
    uint16_t*, uint16_t*, uint16_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint16_t*, void*);
template void TQuantHif4A6::LaunchTQuantHif4A6<64, 128>(
    uint16_t*, uint16_t*, uint16_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint16_t*, void*);
template void TQuantHif4A6::LaunchTQuantHif4A6<256, 128>(
    uint16_t*, uint16_t*, uint16_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint16_t*, void*);
template void TQuantHif4A6::LaunchTQuantHif4A6<128, 256>(
    uint16_t*, uint16_t*, uint16_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint16_t*, void*);
template void TQuantHif4A6::LaunchTQuantHif4A6<256, 256>(
    uint16_t*, uint16_t*, uint16_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint16_t*, void*);
template void TQuantHif4A6::LaunchTQuantHif4A6<128, 512>(
    uint16_t*, uint16_t*, uint16_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint16_t*, void*);
template void TQuantVaddA6::LaunchVaddA6<128, 128>(uint16_t*, uint16_t*, uint16_t*, void*);
#endif
