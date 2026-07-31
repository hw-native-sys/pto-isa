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
// Runs TQUANT<MxQuantAlg::Hif4> in UB, then dumps each carved sub-region
// (max4/8, Ea/Eb/Ec/exp_dst, scaling, fp4) to a separate GM output for ST.
// GM<->UB uses TLOAD/TSTORE; sub-region dumps use 1D-flat alias store tiles.

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/common/type.hpp>
#include "acl/acl.h"

using namespace pto;

#define PTO_CEIL(x, y) ((((x) + (y) - 1) / (y)) * (y))

namespace TQuantHif4A6 {

constexpr uint32_t TQUANT_A6_UB_ALIGN_BYTES = 32;
constexpr uint32_t TQUANT_A6_UB_SIZE_BYTES = 384 * 1024;

// UB layout: mirrors the carving done by TQuant_Hif4_Impl (_Cont path). All
// regions are flat-contiguous, 32-B aligned.
template <int validRows, int validCols>
struct Hif4Layout {
    static constexpr uint32_t ubAlignBytes = 32;
    static constexpr uint32_t totalElem = validRows * validCols;

    static constexpr uint32_t maxGp4Bytes = (totalElem / 4) * sizeof(bfloat16_t);
    static constexpr uint32_t maxGp8Bytes = (totalElem / 8) * sizeof(bfloat16_t);
    static constexpr uint32_t maxTotalBytes = maxGp4Bytes + maxGp8Bytes;

    static constexpr uint32_t eaDataBytes = (totalElem / 64) * sizeof(bfloat16_t); // Ea: 2B (zero-extended)
    static constexpr uint32_t ebDataBytes = ((totalElem / 8) / 8) * 2;             // Eb: packed bits, ×2 upsample
    static constexpr uint32_t ecDataBytes = (totalElem / 4) / 8;                   // Ec: packed predicate bits
    static constexpr uint32_t expDstBytes = (totalElem / 64) * 4;
    static constexpr uint32_t expTotalBytes =
        PTO_CEIL(eaDataBytes + ebDataBytes + ecDataBytes + expDstBytes, ubAlignBytes);

    static constexpr uint32_t srcBytes = totalElem * sizeof(bfloat16_t);
    static constexpr uint32_t scalingBytes = (totalElem / 2) * sizeof(bfloat16_t); // INTLV 2× × US_B16 2×
    static constexpr uint32_t dstBytes = totalElem / 2;                            // packed FP4 (2/byte)

    static constexpr uint64_t srcOffset = 0;
    static constexpr uint64_t maxOffset = PTO_CEIL(srcOffset + srcBytes, ubAlignBytes);
    static constexpr uint64_t expOffset = PTO_CEIL(maxOffset + maxTotalBytes, ubAlignBytes);
    static constexpr uint64_t scalingOffset = PTO_CEIL(expOffset + expTotalBytes, ubAlignBytes);
    static constexpr uint64_t dstOffset = PTO_CEIL(scalingOffset + scalingBytes, ubAlignBytes);
    static constexpr uint64_t ubTotal = dstOffset + dstBytes;

    static constexpr uint64_t maxGp4Off = maxOffset;
    static constexpr uint64_t maxGp8Off = maxOffset + maxGp4Bytes;
    static constexpr uint64_t maxGp64Off = maxOffset + maxGp4Bytes + maxGp8Bytes;

    static constexpr uint64_t eaOff = expOffset;
    static constexpr uint64_t ebOff = expOffset + eaDataBytes;
    static constexpr uint64_t ecOff = expOffset + eaDataBytes + ebDataBytes;
    static constexpr uint64_t expDstOff = expOffset + eaDataBytes + ebDataBytes + ecDataBytes;

    static_assert(ubTotal <= TQUANT_A6_UB_SIZE_BYTES, "HiF4 UB layout exceeds 384 KB");
};

template <int validRows, int validCols>
__global__ AICORE void runTQuantHif4A6(
    __gm__ bfloat16_t __in__* src, __gm__ uint8_t __out__* out_max4, __gm__ uint8_t __out__* out_max8,
    __gm__ uint8_t __out__* out_ea, __gm__ uint8_t __out__* out_eb, __gm__ uint8_t __out__* out_ec,
    __gm__ uint8_t __out__* out_exp_dst, __gm__ uint8_t __out__* out_fp4, __gm__ bfloat16_t __out__* out_scale)
{
    using Spec = Hif4Layout<validRows, validCols>;

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

    // Globals (1D-flat, contiguous stride).
    using SrcGlobal =
        GlobalTensor<bfloat16_t, Shape<1, 1, 1, validRows, validCols>, pto::Stride<1, 1, 1, validCols, 1>>;
    constexpr uint32_t maxGp4Cnt = Spec::maxGp4Bytes / sizeof(bfloat16_t);
    constexpr uint32_t maxGp8Cnt = Spec::maxGp8Bytes / sizeof(bfloat16_t);
    constexpr uint32_t scalingCnt = Spec::scalingBytes / sizeof(bfloat16_t);
    using Max4Global = GlobalTensor<bfloat16_t, Shape<1, 1, 1, 1, maxGp4Cnt>, pto::Stride<1, 1, 1, maxGp4Cnt, 1>>;
    using Max8Global = GlobalTensor<bfloat16_t, Shape<1, 1, 1, 1, maxGp8Cnt>, pto::Stride<1, 1, 1, maxGp8Cnt, 1>>;
    using ScaleGlobal = GlobalTensor<bfloat16_t, Shape<1, 1, 1, 1, scalingCnt>, pto::Stride<1, 1, 1, scalingCnt, 1>>;
    using EaGlobal =
        GlobalTensor<uint8_t, Shape<1, 1, 1, 1, Spec::eaDataBytes>, pto::Stride<1, 1, 1, Spec::eaDataBytes, 1>>;
    using EbGlobal =
        GlobalTensor<uint8_t, Shape<1, 1, 1, 1, Spec::ebDataBytes>, pto::Stride<1, 1, 1, Spec::ebDataBytes, 1>>;
    using EcGlobal =
        GlobalTensor<uint8_t, Shape<1, 1, 1, 1, Spec::ecDataBytes>, pto::Stride<1, 1, 1, Spec::ecDataBytes, 1>>;
    using ExpDstGlobal =
        GlobalTensor<uint8_t, Shape<1, 1, 1, 1, Spec::expDstBytes>, pto::Stride<1, 1, 1, Spec::expDstBytes, 1>>;
    using Fp4Global = GlobalTensor<uint8_t, Shape<1, 1, 1, 1, Spec::dstBytes>, pto::Stride<1, 1, 1, Spec::dstBytes, 1>>;

    SrcGlobal srcGm(src);
    Max4Global max4Gm(reinterpret_cast<__gm__ bfloat16_t*>(out_max4));
    Max8Global max8Gm(reinterpret_cast<__gm__ bfloat16_t*>(out_max8));
    ScaleGlobal scaleGm(out_scale);
    EaGlobal eaGm(out_ea);
    EbGlobal ebGm(out_eb);
    EcGlobal ecGm(out_ec);
    ExpDstGlobal expDstGm(out_exp_dst);
    Fp4Global fp4Gm(out_fp4);

    // Alias store tiles: 1D-flat NoneBox, TASSIGN'd to each sub-region offset,
    // so TSTORE streams just those bytes (A5 tquant E8StoreTile pattern).
    using Max4StoreTile =
        Tile<TileType::Vec, bfloat16_t, 1, maxGp4Cnt, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    using Max8StoreTile =
        Tile<TileType::Vec, bfloat16_t, 1, maxGp8Cnt, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    using EaStoreTile = Tile<
        TileType::Vec, uint8_t, 1, Spec::eaDataBytes, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    using EbStoreTile = Tile<
        TileType::Vec, uint8_t, 1, Spec::ebDataBytes, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    using EcStoreTile = Tile<
        TileType::Vec, uint8_t, 1, Spec::ecDataBytes, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    using ExpDstStoreTile = Tile<
        TileType::Vec, uint8_t, 1, Spec::expDstBytes, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    using Fp4StoreTile = Tile<
        TileType::Vec, uint8_t, 1, Spec::dstBytes, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;
    using ScaleStoreTile = Tile<
        TileType::Vec, bfloat16_t, 1, scalingCnt, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;

    Max4StoreTile max4StoreTile(1, maxGp4Cnt);
    Max8StoreTile max8StoreTile(1, maxGp8Cnt);
    EaStoreTile eaStoreTile(1, Spec::eaDataBytes);
    EbStoreTile ebStoreTile(1, Spec::ebDataBytes);
    EcStoreTile ecStoreTile(1, Spec::ecDataBytes);
    ExpDstStoreTile expDstStoreTile(1, Spec::expDstBytes);
    Fp4StoreTile fp4StoreTile(1, Spec::dstBytes);
    ScaleStoreTile scaleStoreTile(1, scalingCnt);

    TASSIGN(max4StoreTile, Spec::maxGp4Off);
    TASSIGN(max8StoreTile, Spec::maxGp8Off);
    TASSIGN(eaStoreTile, Spec::eaOff);
    TASSIGN(ebStoreTile, Spec::ebOff);
    TASSIGN(ecStoreTile, Spec::ecOff);
    TASSIGN(expDstStoreTile, Spec::expDstOff);
    TASSIGN(fp4StoreTile, Spec::dstOffset);
    TASSIGN(scaleStoreTile, Spec::scalingOffset);

    TLOAD(srcTile, srcGm);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TQUANT<1, pto::MxQuantAlg::Hif4, DstTile, SrcTile, ExpTile, MaxTile, ScalingTile>(
        dstTile, srcTile, &expTile, &maxTile, &scalingTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(max4Gm, max4StoreTile);
    TSTORE(max8Gm, max8StoreTile);
    TSTORE(eaGm, eaStoreTile);
    TSTORE(ebGm, ebStoreTile);
    TSTORE(ecGm, ecStoreTile);
    TSTORE(expDstGm, expDstStoreTile);
    TSTORE(fp4Gm, fp4StoreTile);
    TSTORE(scaleGm, scaleStoreTile);
}

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
#endif
