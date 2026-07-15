/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TQUANT_HPP
#define TQUANT_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
#include "TReshape.hpp"
#include <type_traits>

namespace pto {

struct NvMxFp8E4M3Spec {
    static constexpr float descaleMultiplier = 1.0f / 448.0f;
    static constexpr uint32_t b16SpecialScaleBits = 0x7F81u;
    static constexpr uint32_t f32SpecialScaleBits = 0x7FC00000u;
};

struct NvMxFp4E2M1Spec {
    static constexpr float descaleMultiplier = 1.0f / 6.0f;
    static constexpr uint32_t b16SpecialScaleBits = 0x7FC0u;
    static constexpr uint32_t f32SpecialScaleBits = 0x7FC00000u;
};

struct OcpMxFp8E4M3Spec {
    static constexpr uint16_t maxExp = 0x0400u;
    static constexpr uint16_t expNan = 0x00FFu;
    static constexpr uint16_t b16Nan = 0x7F81u;
    static constexpr int32_t f32Emax = 8;
};

struct OcpMxFp4E2M1Spec {
    static constexpr uint16_t maxExp = 0x0100u;
    static constexpr uint16_t expNan = 0x00FFu;
    static constexpr uint16_t b16Nan = 0x7FC0u;
};

// Algorithm tags for ExtractB8ExponentAndScalingVL dispatch.
// Each tag uniquely identifies a (quant_type, scale_alg) pair so the VL function
// can constexpr-dispatch to the correct ctx + compute path.
struct OcpF8E4M3Alg {};
struct OcpF4E2M1Alg {};
struct NvF8E4M3Alg {};
struct NvF4E2M1Alg {};

// Helper alias: creates a 1D flat tile from a 2D tile's total element count.
template <typename TileData>
using FlatTile1D = Tile<
    TileType::Vec, typename TileData::DType, tquant_detail::FlatTile1DLayout::rows, TileData::Rows * TileData::Cols,
    BLayout::RowMajor, tquant_detail::FlatTile1DLayout::runtimeValidExtent,
    tquant_detail::FlatTile1DLayout::runtimeValidExtent, SLayout::NoneBox,
    tquant_detail::FlatTile1DLayout::sFractalSize, PadValue::Zero>;

template <typename T, typename U>
PTO_INTERNAL MaskReg TQuantPSetTyped(U dist)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        return pset_b32(dist);
    } else if constexpr (sizeof(T) == sizeof(half)) {
        return pset_b16(dist);
    } else {
        return pset_b8(dist);
    }
}

PTO_INTERNAL void AbsReduceMax_Naive(
    __ubuf__ float* srcPtr, __ubuf__ float* maxPtr, unsigned total_elements_count, unsigned vl_count,
    unsigned elementsPerRepeat, MaskReg& preg_lower32, MaskReg& preg_upper32)
{
    RegTensor<float> vreg_b32;
    vector_s32 vreg_zero;
    vbr(vreg_zero, 0);
    uint32_t elem_count = total_elements_count;
    for (uint16_t i = 0; i < (uint16_t)vl_count; ++i) {
        MaskReg preg = CreatePredicate<float>(elem_count);
        RegTensor<float> vreg_max_0, vreg_max_1;
        vlds(vreg_b32, srcPtr, i * elementsPerRepeat, NORM);
        vabs(vreg_b32, vreg_b32, preg);
        vsel((vector_s32&)vreg_b32, (vector_s32&)vreg_b32, vreg_zero, preg);
        vcmax(vreg_max_0, vreg_b32, preg_lower32);
        vcmax(vreg_max_1, vreg_b32, preg_upper32);
        vsts(vreg_max_0, maxPtr, 2 * i, ONEPT_B32, preg);
        vsts(vreg_max_1, maxPtr + 1, 2 * i, ONEPT_B32, preg);
    }
}

// Assumption: input total size is a multiple of 256 elements
PTO_INTERNAL void AbsReduceMax_f32_opt(
    __ubuf__ float* srcPtr, __ubuf__ float* maxPtr, unsigned vl_count, unsigned elementsPerRepeat,
    unsigned total_elements_count)
{
    vector_f32 vreg_in_1, vreg_in_2, vreg_in_3, vreg_in_4, vreg_max_0, vreg_max_1, vreg_max;
    vector_f32 vreg_dintlv_1, vreg_dintlv_2, vreg_dintlv_3, vreg_dintlv_4, vreg_gp_max;
    vector_f32 vreg_dintlv_out_1, vreg_dintlv_out_2, vreg_dintlv_out_3, vreg_dintlv_out_4;
    vector_align ureg_max;
    uint32_t total_count = total_elements_count;
    MaskReg preg_lower8 = pset_b32(PAT_VL8);
    static constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    for (uint16_t i = 0; i < (uint16_t)vl_count / 4; ++i) {
        MaskReg preg_vl0 = CreatePredicate<float>(total_count);
        MaskReg preg_vl1 = CreatePredicate<float>(total_count);
        MaskReg preg_vl2 = CreatePredicate<float>(total_count);
        MaskReg preg_vl3 = CreatePredicate<float>(total_count);
        vlds(vreg_in_1, vreg_in_2, srcPtr, i * 4 * elementsPerRepeat, DINTLV_B32);
        vlds(vreg_in_3, vreg_in_4, srcPtr + 128, i * 4 * elementsPerRepeat, DINTLV_B32);
        vabs(vreg_in_1, vreg_in_1, preg_vl0);
        vabs(vreg_in_3, vreg_in_3, preg_vl2);
        vdintlv(vreg_dintlv_out_1, vreg_dintlv_out_2, vreg_in_1, vreg_in_3);
        vabs(vreg_in_2, vreg_in_2, preg_vl1);
        vabs(vreg_in_4, vreg_in_4, preg_vl3);
        vdintlv(vreg_dintlv_out_3, vreg_dintlv_out_4, vreg_in_2, vreg_in_4);
        vmax(vreg_max_0, vreg_dintlv_out_1, vreg_dintlv_out_2, preg_vl0);
        vmax(vreg_max_1, vreg_dintlv_out_3, vreg_dintlv_out_4, preg_vl1);
        vmax(vreg_max, vreg_max_0, vreg_max_1, preg_vl0);
        vcgmax(vreg_gp_max, vreg_max, preg_vl0);
        vsts(vreg_gp_max, maxPtr, i * 8, distValue, preg_lower8);
    }
}

// Assumption: input total size is a multiple of 2K elements
PTO_INTERNAL void AbsReduceMax_f32_opt_largesizes(
    __ubuf__ float* srcPtr, __ubuf__ float* maxPtr, unsigned vl_count, unsigned elementsPerRepeat,
    unsigned total_elements_count)
{
    vector_f32 vreg_in_1, vreg_in_2, vreg_in_3, vreg_in_4, vreg_max;
    vector_f32 vreg_gp_max, vreg_dintlv_out_1, vreg_dintlv_out_2;
    vector_align ureg_max;
    uint32_t total_count = total_elements_count;
    MaskReg preg_ALL_B32 = pset_b32(PAT_ALL);
    static constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    for (uint16_t i = 0; i < (uint16_t)vl_count / 32; ++i) {
        for (uint16_t j = 0; j < 8; ++j) { // handling 4 VLs per loop, each VL is 256 B (64 fp32)
            MaskReg preg_vl0 = CreatePredicate<float>(total_count);
            MaskReg preg_vl1 = CreatePredicate<float>(total_count);
            MaskReg preg_vl2 = CreatePredicate<float>(total_count);
            MaskReg preg_vl3 = CreatePredicate<float>(total_count);
            vlds(vreg_in_1, vreg_in_2, srcPtr, (i * 32 + j * 4) * elementsPerRepeat, DINTLV_B32);
            vabs(vreg_in_1, vreg_in_1, preg_vl0);
            vabs(vreg_in_2, vreg_in_2, preg_vl1);
            vlds(
                vreg_in_3, vreg_in_4, srcPtr + 2 * elementsPerRepeat, (i * 32 + j * 4) * elementsPerRepeat, DINTLV_B32);
            vabs(vreg_in_3, vreg_in_3, preg_vl2);
            vabs(vreg_in_4, vreg_in_4, preg_vl3);
            vmax(vreg_in_1, vreg_in_1, vreg_in_2, preg_vl0);
            vmax(vreg_in_3, vreg_in_3, vreg_in_4, preg_vl2);
            vdintlv(vreg_dintlv_out_1, vreg_dintlv_out_2, vreg_in_1, vreg_in_3);
            vmax(vreg_max, vreg_dintlv_out_1, vreg_dintlv_out_2, preg_vl0);
            vcgmax(vreg_gp_max, vreg_max, preg_ALL_B32);
            vstus(ureg_max, 8, vreg_gp_max, maxPtr + 64 * i + 8 * j);
        }
        vstas(ureg_max, maxPtr + 64 * i, 0);
    }
}

// Convert two FP16 vectors to BF16 while explicitly preserving Inf/NaN values.
// The saturating f16->bf16 cast used by the quantization path would otherwise
// silently change NaN/Inf behavior; stage 2 relies on the reduced max having
// bf16 exp==0x7F80 to emit the E8M0 NaN/Inf sentinel.
template <typename SrcHalfVec, typename DstBf16Vec>
PTO_INTERNAL void Fp16ToBf16PreserveSpecial(
    SrcHalfVec& src0, SrcHalfVec& src1, DstBf16Vec& dst0, DstBf16Vec& dst1, MaskReg& preg_vl0, MaskReg& preg_vl1)
{
    constexpr uint16_t kBf16AbsMask = 0x7FFF;
    constexpr uint16_t kFp16ExpMask = 0x7C00;
    constexpr uint16_t kFp16MantissaMask = 0x03FF;
    constexpr uint16_t kFp16InfBits = 0x7C00;
    constexpr uint16_t kBf16InfBits = 0x7F80;
    constexpr uint16_t kBf16NanBits = 0x7FC0;

    RegTensor<uint16_t> vu16_abs_0, vu16_abs_1;
    RegTensor<uint16_t> vu16_exp_0, vu16_exp_1;
    RegTensor<uint16_t> vu16_mantissa_0, vu16_mantissa_1;
    RegTensor<uint16_t> vu16_abs_mask, vu16_exp_mask, vu16_mantissa_mask;
    RegTensor<uint16_t> vu16_bf16_inf, vu16_bf16_nan;
    vector_bool preg_special_0, preg_special_1, preg_nan_0, preg_nan_1, preg_inf_0, preg_inf_1;

    vbr(vu16_abs_mask, kBf16AbsMask);
    vbr(vu16_exp_mask, kFp16ExpMask);
    vbr(vu16_mantissa_mask, kFp16MantissaMask);
    vbr(vu16_bf16_inf, kBf16InfBits);
    vbr(vu16_bf16_nan, kBf16NanBits);
    vand(vu16_abs_0, (vector_u16&)src0, vu16_abs_mask, preg_vl0, MODE_ZEROING);
    vand(vu16_abs_1, (vector_u16&)src1, vu16_abs_mask, preg_vl1, MODE_ZEROING);
    vand(vu16_exp_0, vu16_abs_0, vu16_exp_mask, preg_vl0, MODE_ZEROING);
    vand(vu16_exp_1, vu16_abs_1, vu16_exp_mask, preg_vl1, MODE_ZEROING);
    vand(vu16_mantissa_0, vu16_abs_0, vu16_mantissa_mask, preg_vl0, MODE_ZEROING);
    vand(vu16_mantissa_1, vu16_abs_1, vu16_mantissa_mask, preg_vl1, MODE_ZEROING);
    vcmps_eq(preg_special_0, vu16_exp_0, kFp16ExpMask, preg_vl0);
    vcmps_eq(preg_special_1, vu16_exp_1, kFp16ExpMask, preg_vl1);
    vcmps_ne(preg_nan_0, vu16_mantissa_0, 0, preg_special_0);
    vcmps_ne(preg_nan_1, vu16_mantissa_1, 0, preg_special_1);
    vcmps_eq(preg_inf_0, vu16_abs_0, kFp16InfBits, preg_vl0);
    vcmps_eq(preg_inf_1, vu16_abs_1, kFp16InfBits, preg_vl1);
    vcvt(dst0, src0, preg_vl0, ROUND_Z);
    vcvt(dst1, src1, preg_vl1, ROUND_Z);
    vsel((vector_u16&)dst0, vu16_bf16_inf, (vector_u16&)dst0, preg_inf_0);
    vsel((vector_u16&)dst1, vu16_bf16_inf, (vector_u16&)dst1, preg_inf_1);
    vsel((vector_u16&)dst0, vu16_bf16_nan, (vector_u16&)dst0, preg_nan_0);
    vsel((vector_u16&)dst1, vu16_bf16_nan, (vector_u16&)dst1, preg_nan_1);
}

// Reduce one 256-element DINTLV_B16 window to 8 per-block abs raw maxima.
// OCP follows dynamic_mx_quant_tail_axis_fp8: FP16 is first converted to BF16
// then both FP16/BF16 paths reduce BF16 abs bits. NV/cuBLAS reduces FP16 as
// numeric FP16, so callers can disable that conversion.
template <typename T, bool fp16AsBf16ForMax = true>
PTO_INTERNAL void AbsReduceMax_b16_DintlvWindow(
    __ubuf__ T* srcPtr, uint32_t offset, uint32_t remaining, RegTensor<T>& vb16_max)
{
    RegTensor<T> vb16_in_1, vb16_in_2;
    uint32_t even_count = (remaining + 1) / 2;
    uint32_t odd_count = remaining / 2;
    MaskReg preg_vl0 = CreatePredicate<T>(even_count);
    MaskReg preg_vl1 = CreatePredicate<T>(odd_count);
    vlds(vb16_in_1, vb16_in_2, srcPtr, offset, DINTLV_B16);

    vbr(vu16_bf16_abs_mask, kBf16AbsMask);
    if constexpr (std::is_same<T, half>::value && fp16AsBf16ForMax) {
        vector_bf16 vb16_bf16_1, vb16_bf16_2;
        Fp16ToBf16PreserveSpecial(vb16_in_1, vb16_in_2, vb16_bf16_1, vb16_bf16_2, preg_vl0, preg_vl1);
        vand(vu16_abs_1, (vector_u16&)vb16_bf16_1, vu16_bf16_abs_mask, preg_vl0, MODE_ZEROING);
        vand(vu16_abs_2, (vector_u16&)vb16_bf16_2, vu16_bf16_abs_mask, preg_vl1, MODE_ZEROING);
    } else {
        vand(vu16_abs_1, (vector_u16&)vb16_in_1, vu16_bf16_abs_mask, preg_vl0, MODE_ZEROING);
        vand(vu16_abs_2, (vector_u16&)vb16_in_2, vu16_bf16_abs_mask, preg_vl1, MODE_ZEROING);
    }

    vmax(vu16_abs_1, vu16_abs_1, vu16_abs_2, preg_vl0, MODE_ZEROING);
    vcgmax((vector_u16&)vb16_max, vu16_abs_1, preg_vl0, MODE_ZEROING);
}

// Generic ND path (total_elements_count not a multiple of 2048).
// See npu_skills/pto-isa/instructions/tquant-mxfp8.md for the full rationale
// on why we branch on loop_num and how the vstus/vstas continuation works.
template <typename T, bool fp16AsBf16ForMax = true>
PTO_INTERNAL void AbsReduceMax_b16_ND(
    __ubuf__ T* srcPtr, __ubuf__ T* maxPtr, unsigned vl_count, unsigned total_elem_count)
{
    constexpr uint32_t elements_per_dintlv = 2 * REPEAT_BYTE / sizeof(T); // 256 b16 per DINTLV
    constexpr uint32_t grps_per_dintlv = elements_per_dintlv / 32;        // 8 BF16 abs maxima per iter
    constexpr uint32_t blks_per_vl = REPEAT_BYTE / BLOCK_BYTE_SIZE;
    static constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    uint16_t loop_num = CeilDivision(vl_count, 2);
    RegTensor<T> vb16_max;

    // loop_num==1: single window writes only 16 B of group-maxes. Using
    // vstus+vstas would leave 16 B pending and trip VSTAI. Use predicated
    // vsts directly at maxPtr (always 32-B aligned).
    if (loop_num == 1) {
        uint32_t remaining = (total_elem_count < elements_per_dintlv) ? total_elem_count : elements_per_dintlv;
        uint32_t out_count = CeilDivision(remaining, 32u);
        MaskReg preg_out = CreatePredicate<T>(out_count);
        AbsReduceMax_b16_DintlvWindow(srcPtr, 0u, remaining, vb16_max);
        vsts(vb16_max, maxPtr, 0, distValue, preg_out);
        return;
    }
    // loop_num>=2: stream via vstus, then vstas flushes st_align remainder
    // at the continuation addr (maxPtr + loop_num*grps_per_dintlv).
    // Board: stores within a loop may not be ordered w.r.t. one another;
    // VST_VST forces each iter's vstus to commit before the next iter's.
    vector_align ureg_max;
    for (uint16_t i = 0; i < loop_num; ++i) {
        uint32_t offset = i * elements_per_dintlv;
        uint32_t remaining = (total_elem_count > offset) ? (total_elem_count - offset) : 0;
        if (remaining > elements_per_dintlv)
            remaining = elements_per_dintlv;
        AbsReduceMax_b16_DintlvWindow(srcPtr, offset, remaining, vb16_max);
        vstus(ureg_max, blks_per_vl, vb16_max, maxPtr + i * grps_per_dintlv);
    }
    vstas(ureg_max, maxPtr + loop_num * grps_per_dintlv, 0);
}

// Assumption: input total size is a multiple of 2K elements
// Uses 2 VLs per inner iteration (1 DINTLV + 1 vcgmax + 1 vstus) to avoid
// WAW hazard on the vstus auto-increment scalar register when using 2 vstus per iteration.
template <typename T, bool fp16AsBf16ForMax = true>
PTO_INTERNAL void AbsReduceMax_b16_ND_largesizes(
    __ubuf__ T* srcPtr, __ubuf__ T* maxPtr, unsigned vl_count, unsigned total_elements_count)
{
    constexpr uint16_t kBf16AbsMask = 0x7FFF;
    RegTensor<T> vb16_in_1, vb16_in_2, vb16_max_1;
    RegTensor<uint16_t> vu16_abs_1, vu16_abs_2, vu16_bf16_abs_mask;
    vector_bf16 vb16_bf16_1, vb16_bf16_2;
    vector_align ureg_max;
    uint32_t total_count = total_elements_count;
    constexpr uint32_t grp_size = 32;
    constexpr uint32_t elements_per_vl = REPEAT_BYTE / sizeof(T); // 256 B / 2 B = 128 elements per VL
    constexpr uint32_t grps_per_vl = elements_per_vl / grp_size;  // 128 / 32 = 4 groups per VL
    constexpr uint32_t num_vl_per_inner_loop = 2;                 // 2 VLs per inner loop (1 DINTLV load)
    constexpr uint32_t num_vl_per_outer_loop = 32;
    constexpr uint32_t grps_per_inner_loop = num_vl_per_inner_loop * grps_per_vl; // 2 * 4 = 8 grps per inner loop
    constexpr uint32_t grps_per_outer_loop = num_vl_per_outer_loop * grps_per_vl; // 32 * 4 = 128
    constexpr uint32_t blks_per_vl = REPEAT_BYTE / BLOCK_BYTE_SIZE;               // 8 blocks per VL
    static constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    for (uint16_t i = 0; i < (uint16_t)vl_count / num_vl_per_outer_loop; ++i) {        // 32 VLs per outer loop
        for (uint16_t j = 0; j < num_vl_per_outer_loop / num_vl_per_inner_loop; ++j) { // 2 VLs per inner loop
            MaskReg preg_vl0 = CreatePredicate<T>(total_count);
            MaskReg preg_vl1 = CreatePredicate<T>(total_count);
            uint32_t offset = (i * num_vl_per_outer_loop + j * num_vl_per_inner_loop) * elements_per_vl;
            uint32_t grp_offset = grps_per_outer_loop * i + grps_per_inner_loop * j;
            vlds(vb16_in_1, vb16_in_2, srcPtr, offset, DINTLV_B16); // loads 2 VLs (256 bf16 elements)

            if constexpr (std::is_same<T, half>::value && fp16AsBf16ForMax) {
                Fp16ToBf16PreserveSpecial(vb16_in_1, vb16_in_2, vb16_bf16_1, vb16_bf16_2, preg_vl0, preg_vl1);
                vand(vu16_abs_1, (vector_u16&)vb16_bf16_1, vu16_bf16_abs_mask, preg_vl0, MODE_ZEROING);
                vand(vu16_abs_2, (vector_u16&)vb16_bf16_2, vu16_bf16_abs_mask, preg_vl1, MODE_ZEROING);
            } else {
                vand(vu16_abs_1, (vector_u16&)vb16_in_1, vu16_bf16_abs_mask, preg_vl0, MODE_ZEROING);
                vand(vu16_abs_2, (vector_u16&)vb16_in_2, vu16_bf16_abs_mask, preg_vl1, MODE_ZEROING);
            }

            vmax(vu16_abs_1, vu16_abs_1, vu16_abs_2, preg_vl0, MODE_ZEROING);
            vcgmax((vector_u16&)vb16_max_1, vu16_abs_1, preg_vl0, MODE_ZEROING);
            vstus(ureg_max, blks_per_vl, vb16_max_1, maxPtr + grp_offset);
        }
        vstas(ureg_max, maxPtr + grps_per_outer_loop * i, 0);
    }
}

// 2D version of AbsReduceMax_b16: iterates row-by-row, respecting a physical row
// stride (srcCols) distinct from the valid element count per row (validCols).
// Use when the dynamic valid width differs from the static (padded) tile width so
// rows are NOT contiguous in UB. Assumes pad columns [validCols, srcCols) of the
// source tile have been zero-filled (e.g. by ZeroPadSourceTile) so that pad groups
// produce a zero group-max naturally through vmax/vcgmax.
//
// Max buffer layout: per-row stride = srcCols / 32 (groups per row), matching the
// flattened layout used by the downstream ExtractB8ExponentAndScaling pass.
template <typename T, bool fp16AsBf16ForMax = true>
PTO_INTERNAL void AbsReduceMax_b16_ND_2D(
    __ubuf__ T* srcPtr, __ubuf__ T* maxPtr, unsigned validRows, unsigned validCols, unsigned srcCols)
{
    RegTensor<T> vb16_max_1;
    vector_align ureg_max;
    constexpr uint32_t grp_size = 32;
    constexpr uint32_t elements_per_vl = REPEAT_BYTE / sizeof(T);        // 128
    constexpr uint32_t elements_per_dintlv = 2 * elements_per_vl;        // 256
    constexpr uint32_t grps_per_dintlv = elements_per_dintlv / grp_size; // 8 group maxes per DINTLV
    uint32_t groupsPerRow = srcCols / grp_size;                          // srcCols is always 32-aligned
    uint16_t loop_num_per_row = CeilDivision(srcCols, elements_per_dintlv);
    // Max buffer is packed contiguously across rows (row N's maxes sit right after row N-1's).
    // Stream the stores through a single alignment register with POST_UPDATE so the hardware
    // tracks its own position; a single vstas at the end drains the residual.
    __ubuf__ T* writePtr = maxPtr;
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        uint32_t src_row_off = row * srcCols;
        for (uint16_t i = 0; i < loop_num_per_row; ++i) {
            // Predicates reflect per-DINTLV-register valid element count, computed
            // against the padded srcCols (source pad lanes are zero → safe for max).
            uint32_t col_offset = i * elements_per_dintlv;
            uint32_t remaining = (srcCols > col_offset) ? (srcCols - col_offset) : 0;
            if (remaining > elements_per_dintlv)
                remaining = elements_per_dintlv;
            AbsReduceMax_b16_DintlvWindow<T, fp16AsBf16ForMax>(srcPtr, src_row_off + col_offset, remaining, vb16_max_1);
            // Clamp store width to the groups actually present in this row; writing a
            // full grps_per_dintlv (=8) would overshoot into the next row's max slots
            // when groupsPerRow < 8 (e.g. srcCols=32 → 1 group/row).
            uint32_t grps_written_in_row = (uint32_t)i * grps_per_dintlv;
            uint32_t grps_remaining = (groupsPerRow > grps_written_in_row) ? (groupsPerRow - grps_written_in_row) : 0;
            uint32_t grps_this_iter = (grps_remaining > grps_per_dintlv) ? grps_per_dintlv : grps_remaining;
            vstus(ureg_max, grps_this_iter, vb16_max_1, writePtr, POST_UPDATE);
        }
    }
    vstas(ureg_max, writePtr, 0, POST_UPDATE);
    (void)validCols; // padded source makes validCols implicit; retained for API symmetry
}
// Constants and registers shared by all fp32 OCP exponent/scaling extraction variants.
// Call InitF32OcpQuantCtx() inside a __VEC_SCOPE__ before use.
struct F32OcpQuantCtx {
    vector_s32 vb32_b8_nan, vb32_f32_nan, vb32_b8_emax, vb32_exp_mask, vb32_mantissa_mask, vb32_exp_max;
    vector_s32 vb32_recip_min_scale, vb32_zero;
    vector_bool preg_special, preg_nan, preg_min_scale;
    static constexpr uint32_t kExpMask = 0x7F800000u;
    static constexpr uint32_t kMantissaMask = 0x007FFFFFu;
    static constexpr int32_t kExpMax = 0xFE;
    static constexpr uint32_t kF32Nan = 0x7FC00000u;
    static constexpr uint32_t kRecipMinScale = 0x7F000000u;
    static constexpr int32_t kExpNan = OcpMxFp8E4M3Spec::expNan;
    static constexpr int32_t f32Emax = OcpMxFp8E4M3Spec::f32Emax;
    static constexpr int shr = 23;
};

PTO_INTERNAL void InitF32OcpQuantCtx(F32OcpQuantCtx& ctx)
{
    vbr(ctx.vb32_exp_mask, ctx.kExpMask);
    vbr(ctx.vb32_mantissa_mask, ctx.kMantissaMask);
    vbr(ctx.vb32_b8_nan, ctx.kExpNan);
    vbr(ctx.vb32_f32_nan, ctx.kF32Nan);
    vbr(ctx.vb32_exp_max, ctx.kExpMax);
    vbr(ctx.vb32_b8_emax, ctx.f32Emax);
    vbr(ctx.vb32_recip_min_scale, ctx.kRecipMinScale);
    vbr(ctx.vb32_zero, 0);
}

// Compute exponent and scaling from one VL of max data, using a pre-initialised ctx.
// Outputs: shared_exp written to expPtr (PK4_B32), scaling written to scalingPtr (NORM).
PTO_INTERNAL void ComputeF32OcpExpAndScaling(
    F32OcpQuantCtx& ctx, vector_s32& vb32_shared_exp, vector_s32& vb32_scaling, vector_f32& vb32_max, MaskReg& preg_b32,
    __ubuf__ int32_t* maxPtrRaw, uint32_t loadOff)
{
    vlds((vector_s32&)vb32_max, maxPtrRaw, loadOff, NORM);
    vector_s32 vb32_exponent, vb32_mantissa;
    vand((vector_s32&)vb32_exponent, (vector_s32&)vb32_max, ctx.vb32_exp_mask, preg_b32, MODE_ZEROING);
    vand((vector_s32&)vb32_mantissa, (vector_s32&)vb32_max, ctx.vb32_mantissa_mask, preg_b32, MODE_ZEROING);
    vshrs((vector_s32&)vb32_exponent, (vector_s32&)vb32_exponent, ctx.shr, preg_b32, MODE_ZEROING);
    vsub((vector_u32&)vb32_shared_exp, (vector_u32&)vb32_exponent, (vector_u32&)ctx.vb32_b8_emax, preg_b32);
    vsub((vector_s32&)vb32_scaling, (vector_s32&)ctx.vb32_exp_max, (vector_s32&)vb32_shared_exp, preg_b32);
    vshls((vector_u32&)vb32_scaling, (vector_u32&)vb32_scaling, ctx.shr, preg_b32, MODE_ZEROING);
    vcmps_le(ctx.preg_min_scale, (vector_s32&)vb32_exponent, ctx.f32Emax, preg_b32);
    vsel(vb32_scaling, ctx.vb32_recip_min_scale, vb32_scaling, ctx.preg_min_scale);
    vsel(vb32_shared_exp, ctx.vb32_zero, vb32_shared_exp, ctx.preg_min_scale);
    vcmps_eq(ctx.preg_special, (vector_s32&)vb32_exponent, ctx.kExpNan, preg_b32);
    vcmps_ne(ctx.preg_nan, (vector_s32&)vb32_mantissa, 0, ctx.preg_special);
    vsel(vb32_scaling, ctx.vb32_f32_nan, vb32_scaling, ctx.preg_nan);
    vsel(vb32_shared_exp, ctx.vb32_b8_nan, vb32_shared_exp, ctx.preg_nan);
}

// Constants and registers shared by all b16 OCP exponent/scaling extraction variants.
// Parameterized by OcpFormatSpec (OcpMxFp8E4M3Spec or OcpMxFp4E2M1Spec).
// Call InitB16OcpQuantCtx() inside a __VEC_SCOPE__ before use.
template <typename OcpFormatSpec>
struct B16OcpQuantCtx {
    RegTensor<uint16_t> vu16_max_exp_value, vu16_scale_bias, vu16_exp_nan;
    RegTensor<uint16_t> vu16_nan, vu16_exp_mask, vu16_mantissa_mask;
    vector_bool preg_clamp, preg_special, preg_nan;
    static constexpr uint16_t kExpMask = 0x7F80;
    static constexpr uint16_t kMantissaMask = 0x007F;
    static constexpr uint16_t kExpBias = 0x7F00;
    static constexpr uint16_t kMaxExp = OcpFormatSpec::maxExp;
    static constexpr uint16_t kExpNan = OcpFormatSpec::expNan;
    static constexpr uint16_t kB16Nan = OcpFormatSpec::b16Nan;
    static constexpr int expShift = 7;
};

template <typename OcpFormatSpec>
PTO_INTERNAL void InitB16OcpQuantCtx(B16OcpQuantCtx<OcpFormatSpec>& ctx)
{
    vbr(ctx.vu16_max_exp_value, ctx.kMaxExp);
    vbr(ctx.vu16_scale_bias, ctx.kExpBias);
    vbr(ctx.vu16_exp_nan, ctx.kExpNan);
    vbr(ctx.vu16_nan, ctx.kB16Nan);
    vbr(ctx.vu16_exp_mask, ctx.kExpMask);
    vbr(ctx.vu16_mantissa_mask, ctx.kMantissaMask);
}

// Compute exponent and scaling from one VL of b16 max data, using a pre-initialised ctx.
// Outputs: shared_exp written to expPtr (PK_B16), scaling written to scalingPtr (NORM_B16).
template <typename OcpFormatSpec, typename T>
PTO_INTERNAL void ComputeB16OcpExpAndScaling(
    B16OcpQuantCtx<OcpFormatSpec>& ctx, __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr,
    uint32_t off, uint32_t rem)
{
    __ubuf__ uint16_t* maxPtr_u16 = (__ubuf__ uint16_t*)maxPtr;
    __ubuf__ uint16_t* scalingPtr_u16 = (__ubuf__ uint16_t*)scalingPtr;
    RegTensor<uint16_t> vu16_max_abs, vu16_max_exp, vu16_mantissa;
    RegTensor<uint16_t> vu16_shared_exp, vu16_scale_value, vu16_recip_scale;
    vector_bool preg_b16 = CreatePredicate<T>(rem);
    vlds(vu16_max_abs, maxPtr_u16, off, NORM);
    vand(vu16_max_exp, vu16_max_abs, ctx.vu16_exp_mask, preg_b16, MODE_ZEROING);
    vand(vu16_mantissa, vu16_max_abs, ctx.vu16_mantissa_mask, preg_b16, MODE_ZEROING);
    vcmps_eq(ctx.preg_special, vu16_max_exp, ctx.kExpMask, preg_b16);
    vcmps_ne(ctx.preg_nan, vu16_mantissa, 0, ctx.preg_special);
    vcmps_le(ctx.preg_clamp, vu16_max_exp, ctx.kMaxExp, preg_b16);
    vsel(vu16_max_exp, ctx.vu16_max_exp_value, vu16_max_exp, ctx.preg_clamp);
    vsub(vu16_shared_exp, vu16_max_exp, ctx.vu16_max_exp_value, preg_b16, MODE_ZEROING);
    vshrs(vu16_scale_value, vu16_shared_exp, ctx.expShift, preg_b16, MODE_ZEROING);
    vsel(vu16_scale_value, ctx.vu16_exp_nan, vu16_scale_value, ctx.preg_nan);
    vsts(vu16_scale_value, (__ubuf__ uint16_t*)expPtr, off / sizeof(T), PK_B16, preg_b16);
    vsub(vu16_recip_scale, ctx.vu16_scale_bias, vu16_shared_exp, preg_b16, MODE_ZEROING);
    vsel(vu16_recip_scale, ctx.vu16_nan, vu16_recip_scale, ctx.preg_nan);
    vsts(vu16_recip_scale, scalingPtr_u16, off, NORM_B16, preg_b16);
}

// Shared core for fp32 OCP exponent+scaling extraction. Loads one VL from maxPtr
// at the given element offset, computes shared exponent and reciprocal scaling,
// and stores exponent (PK4_B32) and scaling (NORM_B32). Used by both the
// loop-based ExtractB8ExponentAndScaling<float> and the DN-mode VL variant.
PTO_INTERNAL void ExtractF32OcpExponentAndScalingCore(
    __ubuf__ float* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ float* scalingPtr, uint32_t off, uint32_t elemCount)
{
    static constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    F32OcpQuantCtx ctx;
    InitF32OcpQuantCtx(ctx);
    uint32_t preg_cols_b32 = elemCount;
    uint32_t preg_cols_b8 = elemCount * 4;
    MaskReg preg_b32 = CreatePredicate<float>(preg_cols_b32);
    MaskReg preg_b8 = CreatePredicate<uint8_t>(preg_cols_b8);
    vector_f32 vb32_max;
    vector_s32 vb32_shared_exp, vb32_scaling;
    ComputeF32OcpExpAndScaling(ctx, vb32_shared_exp, vb32_scaling, vb32_max, preg_b32, (__ubuf__ int32_t*)maxPtr, off);
    vsts((vector_s32&)vb32_shared_exp, ((__ubuf__ int32_t*)expPtr), off / 4, PK4_B32, preg_b8);
    vsts((vector_s32&)vb32_scaling, ((__ubuf__ int32_t*)scalingPtr), off, distValue, preg_b32);
}

// Shared unrolled scaling store: interleaves scaling with itself and writes two
// NORM_B32 halves. Used by both OCP and NV unrolled f32 extraction loops.
PTO_INTERNAL void StoreScalingUnrolled(
    __ubuf__ float* scalingPtr, vector_s32& vb32_scaling, uint16_t iter, unsigned elementsPerRepeat,
    uint32_t scaling_elem_count)
{
    vector_s32 vb32_scaling_0, vb32_scaling_1;
    vintlv(vb32_scaling_0, vb32_scaling_1, vb32_scaling, vb32_scaling);
    uint32_t baseOff0 = 2 * iter * elementsPerRepeat;
    uint32_t baseOff1 = baseOff0 + elementsPerRepeat;
    uint32_t rem0 = scaling_elem_count > baseOff0 ? scaling_elem_count - baseOff0 : 0;
    uint32_t rem1 = scaling_elem_count > baseOff1 ? scaling_elem_count - baseOff1 : 0;
    MaskReg preg_sc0 = CreatePredicate<float>(rem0);
    vsts(
        (vector_s32&)vb32_scaling_0, ((__ubuf__ int32_t*)scalingPtr), 2 * iter * elementsPerRepeat, NORM_B32, preg_sc0);
    if (rem1 > 0) {
        MaskReg preg_sc1 = CreatePredicate<float>(rem1);
        vsts(
            (vector_s32&)vb32_scaling_1, ((__ubuf__ int32_t*)scalingPtr + 64), 2 * iter * elementsPerRepeat, NORM_B32,
            preg_sc1);
    }
}

// Unrolled variant: interleaves scaling stores for higher throughput.
PTO_INTERNAL void ExtractB8ExponentAndScalingUnrolled(
    __ubuf__ float* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ float* scalingPtr, unsigned exp_max_loop_count,
    unsigned total_elements_count, unsigned elementsPerRepeat)
{
    F32OcpQuantCtx ctx;
    InitF32OcpQuantCtx(ctx);
    uint32_t total_count = total_elements_count;
    uint32_t scaling_elem_count = total_elements_count * 2;
    vector_f32 vb32_max;
    vector_s32 vb32_shared_exp, vb32_scaling;
    for (uint16_t i = 0; i < (uint16_t)exp_max_loop_count; ++i) {
        MaskReg preg_b32 = CreatePredicate<float>(total_count);
        ComputeF32OcpExpAndScaling(
            ctx, vb32_shared_exp, vb32_scaling, vb32_max, preg_b32, (__ubuf__ int32_t*)maxPtr, i * elementsPerRepeat);
        vsts((vector_s32&)vb32_shared_exp, ((__ubuf__ int32_t*)expPtr), i * elementsPerRepeat / 4, PK4_B32, preg_b32);
        StoreScalingUnrolled(scalingPtr, vb32_scaling, i, elementsPerRepeat, scaling_elem_count);
    }
}

// Assumptions: Input is float, data is continuous and 1D, and the usual assumptions about M (divisible by 64)
// Computing scalar focus and exponent for F32 -> b8 e4m3 quantization
template <bool unroll = false>
PTO_INTERNAL void ExtractB8ExponentAndScaling(
    __ubuf__ float* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ float* scalingPtr, unsigned exp_max_loop_count,
    unsigned total_elements_count, unsigned elementsPerRepeat)
{
    if constexpr (unroll) {
        ExtractB8ExponentAndScalingUnrolled(
            maxPtr, expPtr, scalingPtr, exp_max_loop_count, total_elements_count, elementsPerRepeat);
    } else {
        for (uint16_t i = 0; i < (uint16_t)exp_max_loop_count; ++i) {
            ExtractF32OcpExponentAndScalingCore(
                maxPtr, expPtr, scalingPtr, i * elementsPerRepeat, total_elements_count);
        }
    }
}

// NVIDIA MX scale algorithm:
// shared_exp = ceil(log2(max_abs * descaleMultiplier)) + fp32_bias, with exact-power cases kept
// unrounded. The stored reciprocal scale remains 2^(254 - shared_exp).
// Constants and registers shared by all fp32 NV exponent/scaling extraction variants.
// Call InitF32NvQuantCtx() inside a __VEC_SCOPE__ before use.
template <typename NvFormatSpec>
struct F32NvQuantCtx {
    vector_s32 vb32_exp_mask, vb32_mantissa_mask, vb32_exp_max, vb32_exp_nan;
    vector_s32 vb32_special_scale, vb32_min_rcp;
    vector_bool preg_exp_ff, preg_inf, preg_nan;
    vector_bool preg_round_normal, preg_round_subnormal, preg_round_up;
    vector_bool preg_exp_gt_zero, preg_exp_lt_max, preg_exp_eq_zero;
    vector_bool preg_mant_gt_zero, preg_mant_eq_zero, preg_mant_gt_half_subnormal;
    static constexpr float descaleMultiplier = NvFormatSpec::descaleMultiplier;
    static constexpr int shr = 23;
};

template <typename NvFormatSpec>
PTO_INTERNAL void InitF32NvQuantCtx(F32NvQuantCtx<NvFormatSpec>& ctx)
{
    vbr(ctx.vb32_exp_mask, 0x7F800000);
    vbr(ctx.vb32_mantissa_mask, 0x007FFFFF);
    vbr(ctx.vb32_exp_nan, 0xFF);
    vbr(ctx.vb32_exp_max, 0xFE);
    vbr(ctx.vb32_special_scale, NvFormatSpec::f32SpecialScaleBits);
    vbr(ctx.vb32_min_rcp, 0x00400000);
}

// Compute NV exponent and scaling from one VL of max data, using a pre-initialised ctx.
// Outputs: vb32_shared_exp and vb32_scaling in f32 registers (caller handles store).
template <typename NvFormatSpec>
PTO_INTERNAL void ComputeF32NvExpAndScaling(
    F32NvQuantCtx<NvFormatSpec>& ctx, vector_s32& vb32_shared_exp, vector_s32& vb32_scaling, vector_f32& vb32_max,
    MaskReg& preg_b32, __ubuf__ int32_t* maxPtrRaw, uint32_t loadOff)
{
    vector_s32 vb32_exponent, vb32_mantissa, vb32_shared_exp_inc;
    vlds((vector_s32&)vb32_max, maxPtrRaw, loadOff, NORM);
    vmuls(vb32_max, vb32_max, ctx.descaleMultiplier, preg_b32, MODE_ZEROING);
    vand(vb32_exponent, (vector_s32&)vb32_max, ctx.vb32_exp_mask, preg_b32, MODE_ZEROING);
    vand(vb32_mantissa, (vector_s32&)vb32_max, ctx.vb32_mantissa_mask, preg_b32, MODE_ZEROING);
    vshrs(vb32_exponent, vb32_exponent, ctx.shr, preg_b32, MODE_ZEROING);
    vb32_shared_exp = vb32_exponent;
    vadds(vb32_shared_exp_inc, vb32_shared_exp, 1, preg_b32, MODE_ZEROING);
    vcmps_ne(ctx.preg_mant_gt_zero, vb32_mantissa, 0, preg_b32);
    vcmps_gt(ctx.preg_exp_gt_zero, vb32_exponent, 0, preg_b32);
    vcmps_lt(ctx.preg_exp_lt_max, vb32_exponent, 0xFE, preg_b32);
    vcmps_eq(ctx.preg_exp_eq_zero, vb32_exponent, 0, preg_b32);
    pand(ctx.preg_round_normal, ctx.preg_mant_gt_zero, ctx.preg_exp_gt_zero, preg_b32);
    pand(ctx.preg_round_normal, ctx.preg_round_normal, ctx.preg_exp_lt_max, preg_b32);
    vcmps_gt(ctx.preg_mant_gt_half_subnormal, vb32_mantissa, 0x00400000, preg_b32);
    pand(ctx.preg_round_subnormal, ctx.preg_mant_gt_half_subnormal, ctx.preg_exp_eq_zero, preg_b32);
    por(ctx.preg_round_up, ctx.preg_round_normal, ctx.preg_round_subnormal, preg_b32);
    vsel(vb32_shared_exp, vb32_shared_exp_inc, vb32_shared_exp, ctx.preg_round_up);
    vsub(vb32_scaling, ctx.vb32_exp_max, vb32_shared_exp, preg_b32);
    vshls((vector_u32&)vb32_scaling, (vector_u32&)vb32_scaling, ctx.shr, preg_b32, MODE_ZEROING);
    vcmps_eq(ctx.preg_exp_ff, vb32_exponent, 0xFF, preg_b32);
    pnot(ctx.preg_mant_eq_zero, ctx.preg_mant_gt_zero, preg_b32);
    pand(ctx.preg_inf, ctx.preg_exp_ff, ctx.preg_mant_eq_zero, preg_b32);
    pand(ctx.preg_nan, ctx.preg_exp_ff, ctx.preg_mant_gt_zero, preg_b32);
    vsel(vb32_scaling, ctx.vb32_min_rcp, vb32_scaling, ctx.preg_inf);
    vsel(vb32_shared_exp, ctx.vb32_exp_max, vb32_shared_exp, ctx.preg_inf);
    vsel(vb32_scaling, ctx.vb32_special_scale, vb32_scaling, ctx.preg_nan);
    vsel(vb32_shared_exp, ctx.vb32_exp_nan, vb32_shared_exp, ctx.preg_nan);
}

// VL-level NV f32 exponent+scaling extraction: creates ctx, computes, stores.
template <typename NvFormatSpec>
PTO_INTERNAL void ExtractF32NvExponentAndScalingCore(
    __ubuf__ float* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ float* scalingPtr, uint32_t off, uint32_t elemCount)
{
    static constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    F32NvQuantCtx<NvFormatSpec> ctx;
    InitF32NvQuantCtx(ctx);
    uint32_t preg_cols_b32 = elemCount;
    uint32_t preg_cols_b8 = elemCount * 4;
    MaskReg preg_b32 = CreatePredicate<float>(preg_cols_b32);
    MaskReg preg_b8 = CreatePredicate<uint8_t>(preg_cols_b8);
    vector_f32 vb32_max;
    vector_s32 vb32_shared_exp, vb32_scaling;
    ComputeF32NvExpAndScaling<NvFormatSpec>(
        ctx, vb32_shared_exp, vb32_scaling, vb32_max, preg_b32, (__ubuf__ int32_t*)maxPtr, off);
    vsts((vector_s32&)vb32_shared_exp, ((__ubuf__ int32_t*)expPtr), off / 4, PK4_B32, preg_b8);
    vsts((vector_s32&)vb32_scaling, ((__ubuf__ int32_t*)scalingPtr), off, distValue, preg_b32);
}

template <typename NvFormatSpec, bool unroll = false>
PTO_INTERNAL void ExtractNVExponentAndScalingF32(
    __ubuf__ float* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ float* scalingPtr, unsigned exp_max_loop_count,
    unsigned total_elements_count, unsigned elementsPerRepeat)
{
    static constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    F32NvQuantCtx<NvFormatSpec> ctx;
    InitF32NvQuantCtx(ctx);
    uint32_t total_count = total_elements_count;
    uint32_t scaling_elem_count = total_elements_count * 2;
    vector_f32 vb32_max;
    vector_s32 vb32_shared_exp, vb32_scaling;
    for (uint16_t i = 0; i < (uint16_t)exp_max_loop_count; ++i) {
        vector_bool preg_b32 = CreatePredicate<float>(total_count);
        ComputeF32NvExpAndScaling<NvFormatSpec>(
            ctx, vb32_shared_exp, vb32_scaling, vb32_max, preg_b32, (__ubuf__ int32_t*)maxPtr, i * elementsPerRepeat);
        vsts((vector_s32&)vb32_shared_exp, ((__ubuf__ int32_t*)expPtr), i * elementsPerRepeat / 4, PK4_B32, preg_b32);
        if constexpr (unroll) {
            StoreScalingUnrolled(scalingPtr, vb32_scaling, i, elementsPerRepeat, scaling_elem_count);
        } else {
            vsts(vb32_scaling, ((__ubuf__ int32_t*)scalingPtr), i * elementsPerRepeat, distValue, preg_b32);
        }
    }
}

template <bool unroll = false>
PTO_INTERNAL void ExtractB8ExponentAndScalingNV(
    __ubuf__ float* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ float* scalingPtr, unsigned exp_max_loop_count,
    unsigned total_elements_count, unsigned elementsPerRepeat)
{
    ExtractNVExponentAndScalingF32<NvMxFp8E4M3Spec, unroll>(
        maxPtr, expPtr, scalingPtr, exp_max_loop_count, total_elements_count, elementsPerRepeat);
}

// B16 (BF16/FP16) -> FP8/FP4 shared-exponent + BF16 reciprocal scaling for MXFP8/MXFP4.
// Standalone VL-level entry: creates ctx, inits, and calls compute.
template <typename T, typename OcpFormatSpec>
PTO_INTERNAL void ExtractMxOcpExponentAndScalingVL(
    __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, uint32_t off, uint32_t rem)
{
    B16OcpQuantCtx<OcpFormatSpec> ctx;
    InitB16OcpQuantCtx(ctx);
    ComputeB16OcpExpAndScaling<OcpFormatSpec, T>(ctx, maxPtr, expPtr, scalingPtr, off, rem);
}

template <typename T, typename OcpFormatSpec>
PTO_INTERNAL void ExtractMxOcpExponentAndScaling(
    __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, unsigned exp_max_loop_count,
    unsigned total_elements_count)
{
    constexpr uint32_t elementsPerVL = REPEAT_BYTE / sizeof(T);
    B16OcpQuantCtx<OcpFormatSpec> ctx;
    InitB16OcpQuantCtx(ctx);
    for (uint16_t i = 0; i < (uint16_t)exp_max_loop_count; ++i) {
        uint32_t off = i * elementsPerVL;
        uint32_t rem = (total_elements_count > off) ? (total_elements_count - off) : 0;
        if (rem > elementsPerVL)
            rem = elementsPerVL;
        ComputeB16OcpExpAndScaling<OcpFormatSpec, T>(ctx, maxPtr, expPtr, scalingPtr, off, rem);
    }
}

// Forward declarations needed by the VL-level dispatcher before their definitions.
template <typename NvFormatSpec>
PTO_INTERNAL void ExtractF32NvExponentAndScalingCore(
    __ubuf__ float* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ float* scalingPtr, uint32_t off, uint32_t rem);

template <typename NvFormatSpec, typename T>
PTO_INTERNAL void ExtractB16NvExponentAndScalingCore(
    __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, uint32_t off, uint32_t rem);

// Unified VL-level exponent+scaling extraction, dispatched by algorithm tag.
// Alg selects the (quant_type, scale_alg) pair; T is the source type (checked by callers).
// Creates the appropriate ctx, inits it, and calls the matching compute function.
template <typename Alg, typename T>
PTO_INTERNAL void ExtractB8ExponentAndScalingVL(
    __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, uint32_t off, uint32_t rem)
{
    if constexpr (std::is_same<Alg, OcpF8E4M3Alg>::value) {
        if constexpr (std::is_same<T, float>::value) {
            ExtractF32OcpExponentAndScalingCore(maxPtr, expPtr, scalingPtr, off, rem);
        } else {
            ExtractMxOcpExponentAndScalingVL<T, OcpMxFp8E4M3Spec>(maxPtr, expPtr, scalingPtr, off, rem);
        }
    } else if constexpr (std::is_same<Alg, OcpF4E2M1Alg>::value) {
        ExtractMxOcpExponentAndScalingVL<T, OcpMxFp4E2M1Spec>(maxPtr, expPtr, scalingPtr, off, rem);
    } else if constexpr (std::is_same<Alg, NvF8E4M3Alg>::value) {
        if constexpr (std::is_same<T, float>::value) {
            ExtractF32NvExponentAndScalingCore<NvMxFp8E4M3Spec>(maxPtr, expPtr, scalingPtr, off, rem);
        } else {
            ExtractB16NvExponentAndScalingCore<NvMxFp8E4M3Spec, T>(maxPtr, expPtr, scalingPtr, off, rem);
        }
    } else if constexpr (std::is_same<Alg, NvF4E2M1Alg>::value) {
        if constexpr (std::is_same<T, float>::value) {
            ExtractF32NvExponentAndScalingCore<NvMxFp4E2M1Spec>(maxPtr, expPtr, scalingPtr, off, rem);
        } else {
            ExtractB16NvExponentAndScalingCore<NvMxFp4E2M1Spec, T>(maxPtr, expPtr, scalingPtr, off, rem);
        }
    }
}

// B16 (BF16/FP16) -> FP8 shared-exponent + scaling for MXFP8 (OCP MX spec).
// OCP MX fixes the block scale to E8M0 (bias 127), so shared_exp must be on
// the bias-127 axis. BF16 is already bias-127 (b8_emax=8, exp_max_val=0xFE).
// FP16 is bias-15; we fold the +112 rebias into the constants (b8_emax=-104,
// exp_max_val=0x8E) so a single vsub yields the correct bias-127 result.
// Other format-specific constants (shr, exp mask, NaN/subnorm, clamp) are
// picked at compile time via T.
template <typename T>
PTO_INTERNAL void ExtractB8ExponentAndScaling(
    __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, unsigned exp_max_loop_count,
    unsigned total_elements_count)
{
    ExtractMxOcpExponentAndScaling<T, OcpMxFp8E4M3Spec>(
        maxPtr, expPtr, scalingPtr, exp_max_loop_count, total_elements_count);
}
// NVIDIA MX scale algorithm for b16 source:
//   shared_exp = ceil(log2(max_abs * descaleMultiplier))
//   scaling    = (0xFE - shared_exp) << 7  (BF16 reciprocal scale bits)
// Constants and registers shared by all b16 NV exponent/scaling extraction variants.
// Call InitB16NvQuantCtx() inside a __VEC_SCOPE__ before use.
template <typename NvFormatSpec>
struct B16NvQuantCtx {
    vector_s32 vb32_exp_mask, vb32_mantissa_mask, vb32_exp_max, vb32_exp_nan;
    vector_s32 vb32_special_scale, vb32_min_rcp;
    vector_bool preg_exp_ff, preg_inf, preg_nan;
    vector_bool preg_round_normal, preg_round_subnormal, preg_round_up;
    vector_bool preg_exp_gt_zero, preg_exp_lt_max, preg_exp_eq_zero;
    vector_bool preg_mant_gt_zero, preg_mant_eq_zero, preg_mant_gt_half_subnormal;
    static constexpr float descaleMultiplier = NvFormatSpec::descaleMultiplier;
    static constexpr int f32ExpShift = 23;
    static constexpr int b16ExpShift = 7;
};

template <typename NvFormatSpec>
PTO_INTERNAL void InitB16NvQuantCtx(B16NvQuantCtx<NvFormatSpec>& ctx)
{
    vbr(ctx.vb32_exp_mask, 0x7F800000);
    vbr(ctx.vb32_mantissa_mask, 0x007FFFFF);
    vbr(ctx.vb32_exp_nan, 0xFF);
    vbr(ctx.vb32_exp_max, 0xFE);
    vbr(ctx.vb32_special_scale, NvFormatSpec::b16SpecialScaleBits);
    vbr(ctx.vb32_min_rcp, 0x0040);
}

// Compute NV exponent and scaling from one VL of b16 max data, using a pre-initialised ctx.
// Loads b16 (UNPK_B16), upcasts to f32, applies descale, computes shared_exp with
// ceiling rounding, then stores exp (PK4_B32) and scaling (PK_B32, lower 16 bits).
template <typename NvFormatSpec, typename T>
PTO_INTERNAL void ComputeB16NvExpAndScaling(
    B16NvQuantCtx<NvFormatSpec>& ctx, __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr,
    uint32_t off, uint32_t rem)
{
    uint32_t chunkCountB16 = rem * 2;
    uint32_t chunkCountB32 = rem;
    MaskReg preg_b16 = CreatePredicate<T>(chunkCountB16);
    MaskReg preg_b32 = CreatePredicate<float>(chunkCountB32);
    RegTensor<T> vb16_max;
    vector_f32 vb32_max;
    vector_s32 vb32_exponent, vb32_mantissa, vb32_shared_exp, vb32_shared_exp_inc, vb32_bf16_scale_bits;
    vlds(vb16_max, maxPtr, off, UNPK_B16);
    vcvt(vb32_max, vb16_max, preg_b16, PART_EVEN);
    vmuls(vb32_max, vb32_max, ctx.descaleMultiplier, preg_b32, MODE_ZEROING);
    vand(vb32_exponent, (vector_s32&)vb32_max, ctx.vb32_exp_mask, preg_b32, MODE_ZEROING);
    vand(vb32_mantissa, (vector_s32&)vb32_max, ctx.vb32_mantissa_mask, preg_b32, MODE_ZEROING);
    vshrs(vb32_exponent, vb32_exponent, ctx.f32ExpShift, preg_b32, MODE_ZEROING);
    vb32_shared_exp = vb32_exponent;
    vadds(vb32_shared_exp_inc, vb32_shared_exp, 1, preg_b32, MODE_ZEROING);
    vcmps_ne(ctx.preg_mant_gt_zero, vb32_mantissa, 0, preg_b32);
    vcmps_gt(ctx.preg_exp_gt_zero, vb32_exponent, 0, preg_b32);
    vcmps_lt(ctx.preg_exp_lt_max, vb32_exponent, 0xFE, preg_b32);
    vcmps_eq(ctx.preg_exp_eq_zero, vb32_exponent, 0, preg_b32);
    pand(ctx.preg_round_normal, ctx.preg_mant_gt_zero, ctx.preg_exp_gt_zero, preg_b32);
    pand(ctx.preg_round_normal, ctx.preg_round_normal, ctx.preg_exp_lt_max, preg_b32);
    vcmps_gt(ctx.preg_mant_gt_half_subnormal, vb32_mantissa, 0x00400000, preg_b32);
    pand(ctx.preg_round_subnormal, ctx.preg_mant_gt_half_subnormal, ctx.preg_exp_eq_zero, preg_b32);
    por(ctx.preg_round_up, ctx.preg_round_normal, ctx.preg_round_subnormal, preg_b32);
    vsel(vb32_shared_exp, vb32_shared_exp_inc, vb32_shared_exp, ctx.preg_round_up);
    vsub(vb32_bf16_scale_bits, ctx.vb32_exp_max, vb32_shared_exp, preg_b32);
    vshls(
        (vector_u32&)vb32_bf16_scale_bits, (vector_u32&)vb32_bf16_scale_bits, ctx.b16ExpShift, preg_b32, MODE_ZEROING);
    vcmps_eq(ctx.preg_exp_ff, vb32_exponent, 0xFF, preg_b32);
    pnot(ctx.preg_mant_eq_zero, ctx.preg_mant_gt_zero, preg_b32);
    pand(ctx.preg_inf, ctx.preg_exp_ff, ctx.preg_mant_eq_zero, preg_b32);
    pand(ctx.preg_nan, ctx.preg_exp_ff, ctx.preg_mant_gt_zero, preg_b32);
    vsel(vb32_bf16_scale_bits, ctx.vb32_min_rcp, vb32_bf16_scale_bits, ctx.preg_inf);
    vsel(vb32_shared_exp, ctx.vb32_exp_max, vb32_shared_exp, ctx.preg_inf);
    vsel(vb32_bf16_scale_bits, ctx.vb32_special_scale, vb32_bf16_scale_bits, ctx.preg_nan);
    vsel(vb32_shared_exp, ctx.vb32_exp_nan, vb32_shared_exp, ctx.preg_nan);
    vsts(vb32_shared_exp, ((__ubuf__ int32_t*)expPtr), off / 4, PK4_B32, preg_b32);
    vsts((vector_u16&)vb32_bf16_scale_bits, (__ubuf__ uint16_t*)scalingPtr, off, PK_B32, preg_b32);
}

// VL-level NV b16 exponent+scaling extraction: creates ctx, computes, stores.
template <typename NvFormatSpec, typename T>
PTO_INTERNAL void ExtractB16NvExponentAndScalingCore(
    __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, uint32_t off, uint32_t rem)
{
    B16NvQuantCtx<NvFormatSpec> ctx;
    InitB16NvQuantCtx(ctx);
    ComputeB16NvExpAndScaling<NvFormatSpec, T>(ctx, maxPtr, expPtr, scalingPtr, off, rem);
}

template <typename T, typename NvFormatSpec>
PTO_INTERNAL void ExtractNVExponentAndScalingB16(
    __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, unsigned total_elements_count)
{
    B16NvQuantCtx<NvFormatSpec> ctx;
    InitB16NvQuantCtx(ctx);
    constexpr uint32_t elementsPerChunk = REPEAT_BYTE / sizeof(float);
    uint16_t loopCount = CeilDivision(total_elements_count, elementsPerChunk);
    for (uint16_t i = 0; i < loopCount; ++i) {
        uint32_t off = i * elementsPerChunk;
        uint32_t rem = (total_elements_count > off) ? (total_elements_count - off) : 0;
        if (rem > elementsPerChunk)
            rem = elementsPerChunk;
        ComputeB16NvExpAndScaling<NvFormatSpec, T>(ctx, maxPtr, expPtr, scalingPtr, off, rem);
    }
}

template <typename T>
PTO_INTERNAL void ExtractB8ExponentAndScalingNV(
    __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, unsigned /*exp_max_loop_count*/,
    unsigned total_elements_count)
{
    ExtractNVExponentAndScalingB16<T, NvMxFp8E4M3Spec>(maxPtr, expPtr, scalingPtr, total_elements_count);
}

template <typename T>
PTO_INTERNAL void ExtractE2M1ExponentAndScalingVL(
    __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, uint32_t off, uint32_t rem)
{
    static_assert(std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value,
                  "ExtractE2M1ExponentAndScalingVL: T must be bfloat16_t or half");
    constexpr uint16_t kBf16ExpMask = 0x7F80;
    constexpr uint16_t kBf16MantissaMask = 0x007F;
    constexpr uint16_t kFp4E2M1MaxExp = 0x0100;
    constexpr uint16_t kBf16ExpBias = 0x7F00;
    constexpr uint16_t kFp4Nan = 0x00FF;
    constexpr uint16_t kBf16Nan = 0x7FC0;

    __ubuf__ uint16_t *maxPtr_u16 = (__ubuf__ uint16_t *)maxPtr;
    __ubuf__ uint16_t *scalingPtr_u16 = (__ubuf__ uint16_t *)scalingPtr;
    RegTensor<uint16_t> vu16_max_abs, vu16_max_exp, vu16_mantissa;
    RegTensor<uint16_t> vu16_shared_exp, vu16_scale_value, vu16_recip_scale;
    RegTensor<uint16_t> vu16_max_exp_value, vu16_scale_bias, vu16_fp4_nan;
    RegTensor<uint16_t> vu16_nan, vu16_exp_mask, vu16_mantissa_mask;
    vector_bool preg_clamp, preg_special, preg_nan;
    vector_bool preg_b16 = CreatePredicate<T>(rem);

    vbr(vu16_max_exp_value, kFp4E2M1MaxExp);
    vbr(vu16_scale_bias, kBf16ExpBias);
    vbr(vu16_fp4_nan, kFp4Nan);
    vbr(vu16_nan, kBf16Nan);
    vbr(vu16_exp_mask, kBf16ExpMask);
    vbr(vu16_mantissa_mask, kBf16MantissaMask);

    vlds(vu16_max_abs, maxPtr_u16, off, NORM);
    vand(vu16_max_exp, vu16_max_abs, vu16_exp_mask, preg_b16, MODE_ZEROING);
    vand(vu16_mantissa, vu16_max_abs, vu16_mantissa_mask, preg_b16, MODE_ZEROING);
    vcmps_eq(preg_special, vu16_max_exp, kBf16ExpMask, preg_b16);
    vcmps_ne(preg_nan, vu16_mantissa, 0, preg_special);
    vcmps_le(preg_clamp, vu16_max_exp, kFp4E2M1MaxExp, preg_b16);
    vsel(vu16_max_exp, vu16_max_exp_value, vu16_max_exp, preg_clamp);

    vsub(vu16_shared_exp, vu16_max_exp, vu16_max_exp_value, preg_b16, MODE_ZEROING);
    vshrs(vu16_scale_value, vu16_shared_exp, 7, preg_b16, MODE_ZEROING);
    vsel(vu16_scale_value, vu16_fp4_nan, vu16_scale_value, preg_nan);
    vsts(vu16_scale_value, (__ubuf__ uint16_t *)expPtr, off / sizeof(T), PK_B16, preg_b16);

    vsub(vu16_recip_scale, vu16_scale_bias, vu16_shared_exp, preg_b16, MODE_ZEROING);
    vsel(vu16_recip_scale, vu16_nan, vu16_recip_scale, preg_nan);
    vsts(vu16_recip_scale, scalingPtr_u16, off, NORM_B16, preg_b16);
}

template <typename T>
PTO_INTERNAL void ExtractE2M1ExponentAndScaling(
    __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, unsigned exp_max_loop_count,
    unsigned total_elements_count)
{
    ExtractMxOcpExponentAndScaling<T, OcpMxFp4E2M1Spec>(
        maxPtr, expPtr, scalingPtr, exp_max_loop_count, total_elements_count);
}

template <typename T>
PTO_INTERNAL void ExtractE2M1ExponentAndScalingNV(
    __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, unsigned /*exp_max_loop_count*/,
    unsigned total_elements_count)
{
    ExtractNVExponentAndScalingB16<T, NvMxFp4E2M1Spec>(maxPtr, expPtr, scalingPtr, total_elements_count);
}

// 2D variant of ExtractB8ExponentAndScaling for the padded (validCols != srcCols) path.
// Iterates per row, processing only the groups backing valid columns. Max, exp and
// scaling buffers share a packed per-row layout (row r's first group at row * groupsPerRow).
// Only safe when groupsPerRow * sizeof(T) is a multiple of 32 B (i.e. srcCols >= 512 for
// B16) so that each per-row NORM load/store address is 32-byte aligned. Callers must
// gate on that condition.
template <typename T>
PTO_INTERNAL void ExtractB8ExponentAndScaling_2D(
    __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, unsigned validRows, unsigned validCols,
    unsigned srcCols)
{
    static_assert(std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value,
                  "ExtractB8ExponentAndScaling_2D: T must be bfloat16_t or half");
    constexpr uint32_t elementsPerVL = REPEAT_BYTE / sizeof(T); // 128 group-maxes per VL

    uint32_t groupsPerRow = srcCols / 32; // srcCols is 32-aligned
    uint32_t validGroupsPerRow = CeilDivision((uint32_t)validCols, 32u);
    uint16_t loopsPerRow = CeilDivision(validGroupsPerRow, elementsPerVL);
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        uint32_t rowOff = row * groupsPerRow; // group-indexed offset into packed buffers
        for (uint16_t i = 0; i < loopsPerRow; ++i) {
            uint32_t off = i * elementsPerVL;
            uint32_t rem = (validGroupsPerRow > off) ? (validGroupsPerRow - off) : 0;
            if (rem > elementsPerVL)
                rem = elementsPerVL;
            ExtractB8ExponentAndScalingVL<OcpF8E4M3Alg, T>(
                maxPtr + rowOff, expPtr + rowOff, scalingPtr + rowOff, off, rem);
        }
    }
}

// FP32 -> FP8
PTO_INTERNAL void CalcQuantizedFP8Values(
    __ubuf__ float* srcPtr, __ubuf__ float* scalingPtr, __ubuf__ uint8_t* dstPtr, unsigned vl_count,
    unsigned elementsPerRepeat, unsigned total_elements_count, MaskReg& preg_lower32, MaskReg& preg_upper32)
{
    vector_f32 vb32_scaling_0, vb32_scaling_1, vb32_in, vb32_out_1, vb32_out_2, vb32_out;
    vector_f8e4m3 vb8_out;
    uint32_t elem_count = total_elements_count;
    MaskReg preg_ALL = pset_b32(PAT_ALL);
    for (uint16_t i = 0; i < (uint16_t)vl_count; ++i) {
        MaskReg preg = CreatePredicate<float>(elem_count);
        vlds(vb32_scaling_0, scalingPtr, 2 * i, BRC_B32);
        vlds(vb32_scaling_1, scalingPtr + 1, 2 * i, BRC_B32);
        vlds(vb32_in, srcPtr, i * elementsPerRepeat, NORM);
        vmul(vb32_out_1, vb32_in, vb32_scaling_0, preg_lower32, MODE_ZEROING);
        vmul(vb32_out_2, vb32_in, vb32_scaling_1, preg_upper32, MODE_ZEROING);
        vor(vb32_out, vb32_out_1, vb32_out_2, preg_ALL);
        vcvt((vector_f8e4m3&)vb8_out, (vector_f32&)vb32_out, preg, ROUND_R, RS_ENABLE, PART_P0);
        vsts((vector_u8&)vb8_out, (__ubuf__ uint8_t*)dstPtr, i * elementsPerRepeat, PK4_B32, preg);
    }
}

PTO_INTERNAL void CalcQuantizedFP8Values_Unroll2(
    __ubuf__ float* srcPtr, __ubuf__ float* scalingPtr, __ubuf__ uint8_t* dstPtr, unsigned vl_count,
    unsigned elementsPerRepeat, unsigned total_elements_count)
{
    vector_f32 vb32_scaling, vb32_in_even, vb32_in_odd, vb32_out_1, vb32_out_2, vb32_out;
    vector_f8e4m3 vb8_out_P0, vb8_out_P1, vb8_out;
    uint32_t elem_count = total_elements_count;
    MaskReg preg_ALL = pset_b32(PAT_ALL);
    MaskReg preg_ALL_b8 = pset_b8(PAT_ALL);
    for (uint16_t i = 0; i < (uint16_t)vl_count / 2; ++i) {
        vlds(vb32_scaling, scalingPtr, 8 * i, E2B_B32);
        vlds(vb32_in_even, vb32_in_odd, srcPtr, 2 * i * elementsPerRepeat, DINTLV_B32);
        vmul(vb32_out_1, vb32_in_even, vb32_scaling, preg_ALL, MODE_ZEROING);
        vmul(vb32_out_2, vb32_in_odd, vb32_scaling, preg_ALL, MODE_ZEROING);
        vcvt((vector_f8e4m3&)vb8_out_P0, (vector_f32&)vb32_out_1, preg_ALL, ROUND_R, RS_ENABLE, PART_P0);
        vcvt((vector_f8e4m3&)vb8_out_P1, (vector_f32&)vb32_out_2, preg_ALL, ROUND_R, RS_ENABLE, PART_P1);
        vor(vb8_out, vb8_out_P0, vb8_out_P1, preg_ALL_b8);
        vsts((vector_u16&)vb8_out, (__ubuf__ uint16_t*)dstPtr, i * elementsPerRepeat, PK_B32, preg_ALL);
    }
}

// FP16-specific scaling+multiply for one FP8 window. The scaling buffer stores
// bf16 reciprocal scales even for fp16 input; widen to fp32 and multiply in fp32
// so the mantissa precision is preserved before downcast to fp8.
PTO_INTERNAL void ApplyHalfScalingToFP8Window(
    vector_f32& vb32_cvt_1, vector_f32& vb32_cvt_2, vector_f32& vb32_cvt_3, vector_f32& vb32_cvt_4,
    RegTensor<half>& vb16_in_1, RegTensor<half>& vb16_in_2, __ubuf__ half* scalingPtr, uint16_t i, MaskReg& preg_b16_1,
    MaskReg& preg_b16_2, MaskReg& preg_f32_1_even, MaskReg& preg_f32_1_odd, MaskReg& preg_f32_2_even,
    MaskReg& preg_f32_2_odd)
{
    vector_bf16 vb16_scaling_bf16;
    vector_f32 vb32_scaling;
    MaskReg preg_all_b16 = pset_b16(PAT_ALL);
    vlds((vector_u16&)vb16_scaling_bf16, (__ubuf__ uint16_t*)scalingPtr, 8 * i, E2B_B16);
    vcvt(vb32_scaling, vb16_scaling_bf16, preg_all_b16, PART_EVEN);
    vcvt(vb32_cvt_1, vb16_in_1, preg_b16_1, PART_EVEN);
    vcvt(vb32_cvt_2, vb16_in_1, preg_b16_1, PART_ODD);
    vcvt(vb32_cvt_3, vb16_in_2, preg_b16_2, PART_EVEN);
    vcvt(vb32_cvt_4, vb16_in_2, preg_b16_2, PART_ODD);
    vmul(vb32_cvt_1, vb32_cvt_1, vb32_scaling, preg_f32_1_even, MODE_ZEROING);
    vmul(vb32_cvt_2, vb32_cvt_2, vb32_scaling, preg_f32_1_odd, MODE_ZEROING);
    vmul(vb32_cvt_3, vb32_cvt_3, vb32_scaling, preg_f32_2_even, MODE_ZEROING);
    vmul(vb32_cvt_4, vb32_cvt_4, vb32_scaling, preg_f32_2_odd, MODE_ZEROING);
}

// B16 (BF16/FP16) -> FP8. FP16 uses BF16 reciprocal scale, matching dynamic_mx_quant:
// convert input and BF16 scale to fp32, multiply in fp32, then downcast to fp8.
// Quantize one 256-element DINTLV_B16 window to FP8: scale via broadcast of
// 8 per-group scaling values, upcast b16->fp32 (EVEN/ODD), downcast fp32->fp8
// (PART_P0-P3 pack mod-4 bytes), OR-combine, and store.
template <typename T>
PTO_INTERNAL void CalcQuantizedFP8Values_B16_Window(
    __ubuf__ T* srcPtr, __ubuf__ T* scalingPtr, __ubuf__ uint8_t* dstPtr, uint16_t i, uint32_t offset_b16,
    uint32_t remaining)
{
    constexpr uint32_t elementsPerVL_b8 = REPEAT_BYTE / sizeof(uint8_t);
    RegTensor<T> vb16_scaling, vb16_in_1, vb16_in_2, vb16_out_1, vb16_out_2;
    vector_f32 vb32_cvt_1, vb32_cvt_2, vb32_cvt_3, vb32_cvt_4;
    vector_f8e4m3 vb8_or1, vb8_or2, vb8_out, vb8_p0, vb8_p1, vb8_p2, vb8_p3;
    uint32_t even_count = (remaining + 1) / 2;
    uint32_t odd_count = remaining / 2;
    uint32_t b8_count = remaining;
    MaskReg preg_b16_1 = CreatePredicate<T>(even_count);
    MaskReg preg_b16_2 = CreatePredicate<T>(odd_count);
    MaskReg preg_b8 = CreatePredicate<uint8_t>(b8_count);
    vlds((vector_u16 &)vb16_scaling, (__ubuf__ uint16_t *)scalingPtr, 8 * i, E2B_B16);
    vlds(vb16_in_1, vb16_in_2, srcPtr, offset_b16, DINTLV_B16);
    if constexpr (std::is_same<T, half>::value) {
        ApplyHalfScalingToFP8Window(
            vb32_cvt_1, vb32_cvt_2, vb32_cvt_3, vb32_cvt_4, vb16_in_1, vb16_in_2, (__ubuf__ half*)scalingPtr, i,
            preg_b16_1, preg_b16_2, preg_f32_1_even, preg_f32_1_odd, preg_f32_2_even, preg_f32_2_odd);
    } else {
        vlds((vector_u16&)vb16_scaling, (__ubuf__ uint16_t*)scalingPtr, 8 * i, E2B_B16);
        vmul(vb16_out_1, vb16_in_1, vb16_scaling, preg_b16_1, MODE_ZEROING);
        vmul(vb16_out_2, vb16_in_2, vb16_scaling, preg_b16_2, MODE_ZEROING);
        vcvt(vb32_cvt_1, vb16_out_1, preg_b16_1, PART_EVEN);
        vcvt(vb32_cvt_2, vb16_out_1, preg_b16_1, PART_ODD);
        vcvt(vb32_cvt_3, vb16_out_2, preg_b16_2, PART_EVEN);
        vcvt(vb32_cvt_4, vb16_out_2, preg_b16_2, PART_ODD);
    }
    // fp32->fp8 P0..P3 writes to bytes 0..3 of each 32-bit slot; pair with mod-4 index.
    vcvt(vb8_p0, vb32_cvt_1, preg_b16_1, ROUND_R, RS_ENABLE, PART_P0);
    vcvt(vb8_p1, vb32_cvt_3, preg_b16_2, ROUND_R, RS_ENABLE, PART_P1);
    vcvt(vb8_p2, vb32_cvt_2, preg_b16_1, ROUND_R, RS_ENABLE, PART_P2);
    vcvt(vb8_p3, vb32_cvt_4, preg_b16_2, ROUND_R, RS_ENABLE, PART_P3);
    vor(vb8_or1, vb8_p0, vb8_p1, preg_b8);
    vor(vb8_or2, vb8_p2, vb8_p3, preg_b8);
    vor(vb8_out, vb8_or1, vb8_or2, preg_b8);
    vsts((vector_u8&)vb8_out, (__ubuf__ uint8_t*)dstPtr, i * elementsPerVL_b8, NORM_B8, preg_b8);
}

// B16 (BF16/FP16) -> FP8. 2 VLs per iter (one DINTLV_B16 load). Ceil-div on
// iter count so a partial final window is still covered; per-window predicates
// clamp to the exact remaining element count.
template <typename T>
PTO_INTERNAL void CalcQuantizedFP8Values(
    __ubuf__ T* srcPtr, __ubuf__ T* scalingPtr, __ubuf__ uint8_t* dstPtr, unsigned total_elements_count)
{
    static_assert(std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value,
                  "CalcQuantizedFP8Values B16: T must be bfloat16_t or half");
    constexpr uint32_t elementsPerVL_b16 = REPEAT_BYTE / sizeof(T);
    constexpr uint32_t elementsPerDintlv = 2 * elementsPerVL_b16;
    uint32_t vl_count = CeilDivision(total_elements_count, elementsPerVL_b16);
    uint16_t quant_iters = (uint16_t)CeilDivision(vl_count, static_cast<uint32_t>(2u));
    for (uint16_t i = 0; i < quant_iters; ++i) {
        uint32_t offset_b16 = i * elementsPerDintlv;
        uint32_t remaining = (total_elements_count > offset_b16) ? (total_elements_count - offset_b16) : 0;
        if (remaining > elementsPerDintlv)
            remaining = elementsPerDintlv;
        CalcQuantizedFP8Values_B16_Window<T>(srcPtr, scalingPtr, dstPtr, i, offset_b16, remaining);
    }
}

// 2D variant of CalcQuantizedFP8Values for the padded (validCols != srcCols) path.
// Iterates per row using srcCols as src/dst stride (elements) and groupsPerRow as the
// packed scaling-buffer stride. Processes only validCols elements per row; pad-col dst
// bytes are not written (TSTORE trims them via GM shape). Alignment requirements:
//   - scalingPtr + row * groupsPerRow must be 16 B-aligned for E2B_B16 load
//     (groupsPerRow * sizeof(T) % 16 == 0, i.e. srcCols % 256 == 0)
//   - srcPtr  + row * srcCols must be 32 B-aligned for DINTLV_B16 (srcCols % 16 == 0)
//   - dstPtr  + row * srcCols must be 32 B-aligned for NORM_B8 (srcCols % 32 == 0, always true)
// Callers must gate on the srcCols %% 256 == 0 condition.
template <typename T>
PTO_INTERNAL void CalcQuantizedFP8Values_2D(
    __ubuf__ T* srcPtr, __ubuf__ T* scalingPtr, __ubuf__ uint8_t* dstPtr, unsigned validRows, unsigned validCols,
    unsigned srcCols)
{
    constexpr uint32_t elementsPerVL_b16 = REPEAT_BYTE / sizeof(T); // 128
    constexpr uint32_t elementsPerDintlv = 2 * elementsPerVL_b16;   // 256
    uint32_t groupsPerRow = srcCols / 32;
    uint16_t loopsPerRow = CeilDivision((uint32_t)validCols, elementsPerDintlv);
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        uint32_t srcRowOff = row * srcCols;        // T-indexed
        uint32_t dstRowOff = row * srcCols;        // uint8_t-indexed (1 byte per elem)
        uint32_t scaleRowOff = row * groupsPerRow; // T-indexed, packed scaling layout
        for (uint16_t i = 0; i < loopsPerRow; ++i) {
            uint32_t colOff = i * elementsPerDintlv;
            uint32_t remaining = (validCols > colOff) ? (validCols - colOff) : 0;
            if (remaining > elementsPerDintlv)
                remaining = elementsPerDintlv;
            CalcQuantizedFP8Values_B16_Window<T>(
                srcPtr + srcRowOff, scalingPtr + scaleRowOff, dstPtr + dstRowOff, i, colOff, remaining);
        }
    }
}

PTO_INTERNAL void CalcE2M1SignedCodeI32(vector_s32& signedCode, vector_f32 scaled, MaskReg& preg_f32)
{
    constexpr uint32_t kInfBits = 0x7F800000;
    vector_u32 vu32_abs_bits, vu32_exp, vu32_tmp;
    vector_bool preg_sign, preg_nan;

    vshrs(vu32_tmp, (vector_u32&)scaled, F32::signBitOffset, preg_f32, MODE_ZEROING);
    vcmps_ne(preg_sign, vu32_tmp, (uint32_t)0, preg_f32);
    vshls(vu32_abs_bits, (vector_u32&)scaled, F32::signClearShift, preg_f32, MODE_ZEROING);
    vshrs(vu32_abs_bits, vu32_abs_bits, F32::signClearShift, preg_f32, MODE_ZEROING);
    vcmps_gt(preg_nan, vu32_abs_bits, F32::positiveInfBits, preg_f32);

    vshrs(vu32_exp, vu32_abs_bits, (int16_t)23, preg_f32, MODE_ZEROING);
    vmaxs(vu32_exp, vu32_exp, (uint32_t)127, preg_f32, MODE_ZEROING);
    vmins(vu32_exp, vu32_exp, (uint32_t)129, preg_f32, MODE_ZEROING);

    vadds((vector_s32&)vu32_tmp, (vector_s32&)vu32_exp, E2M1::magicRoundingExponentOffset, preg_f32, MODE_ZEROING);
    vshls(vu32_tmp, vu32_tmp, F32::exponentOffset, preg_f32, MODE_ZEROING);
    vadd(scaled, (vector_f32&)vu32_abs_bits, (vector_f32&)vu32_tmp, preg_f32, MODE_ZEROING);
    vsub(vu32_abs_bits, (vector_u32&)scaled, vu32_tmp, preg_f32);

    vadds((vector_s32&)vu32_exp, (vector_s32&)vu32_exp, F32::negativeExponentBias, preg_f32, MODE_ZEROING);
    vshls(vu32_exp, vu32_exp, E2M1::magnitudeCodeShift, preg_f32, MODE_ZEROING);
    vadd(vu32_abs_bits, vu32_abs_bits, vu32_exp, preg_f32, MODE_ZEROING);
    vmins(vu32_abs_bits, vu32_abs_bits, (uint32_t)7, preg_f32, MODE_ZEROING);

    vadds(signedCode, (vector_s32&)vu32_abs_bits, E2M1::negativeCodeOffset, preg_f32, MODE_ZEROING);
    vsel(signedCode, signedCode, (vector_s32&)vu32_abs_bits, preg_sign);

    vsel(signedCode, (vector_s32&)vu32_abs_bits, signedCode, preg_nan);
}

PTO_INTERNAL void PackE2M1SignedCodeBytes(
    vector_u8& packedBytes, vector_s32 evenCode, vector_s32 oddCode, vector_u8& packIndex, MaskReg& preg_f32)
{
    vector_u32 vu32_even, vu32_odd;

    vshls(vu32_even, (vector_u32&)evenCode, Pack::lowCodeShift, preg_f32, MODE_ZEROING);
    vshrs(vu32_even, vu32_even, Pack::lowCodeShift, preg_f32, MODE_ZEROING);
    vshls(vu32_odd, (vector_u32&)oddCode, Pack::lowCodeShift, preg_f32, MODE_ZEROING);
    vshrs(vu32_odd, vu32_odd, Pack::highCodeShift, preg_f32, MODE_ZEROING);
    vor(vu32_even, vu32_even, vu32_odd, preg_f32, MODE_ZEROING);
    vselr(packedBytes, (vector_u8&)vu32_even, packIndex);
}

PTO_INTERNAL void SaturateBf16NaNToPosInf(vector_u16& value, MaskReg& preg_b16)
{
    constexpr uint16_t kBf16AbsMask = 0x7FFF;
    constexpr uint16_t kBf16Inf = 0x7F80;
    vector_u16 v_abs, v_abs_mask, v_inf;
    vector_bool preg_nan;

    vbr(v_abs_mask, kBf16AbsMask);
    vbr(v_inf, kBf16Inf);
    vand(v_abs, value, v_abs_mask, preg_b16, MODE_ZEROING);
    vcmps_gt(preg_nan, v_abs, kBf16Inf, preg_b16);
    vsel(value, v_inf, value, preg_nan);
}

PTO_INTERNAL void CalcQuantizedFP4E2M1Values_Half_Window(
    __ubuf__ half* srcPtr, __ubuf__ half* scalingPtr, __ubuf__ uint8_t* dstPtr, uint16_t window, vector_u8& packIndex)
{
    constexpr uint32_t kElementsPerWindow = 256;
    constexpr uint32_t kPackedBytesPerWindow = kElementsPerWindow / 2;
    constexpr uint32_t kB16LanesPerReg = REPEAT_BYTE / sizeof(half);
    constexpr uint32_t kF32LanesPerReg = REPEAT_BYTE / sizeof(float);
    uint32_t b16LanesPerReg = kB16LanesPerReg;
    uint32_t f32LanesPerReg = kF32LanesPerReg;
    uint32_t packedBytesPerWindow = kPackedBytesPerWindow;
    MaskReg preg_b16 = CreatePredicate<half>(b16LanesPerReg);
    MaskReg preg_f32 = CreatePredicate<float>(f32LanesPerReg);
    MaskReg preg_b8 = CreatePredicate<uint8_t>(packedBytesPerWindow);
    MaskReg preg_all_b16 = pset_b16(PAT_ALL);
    RegTensor<half> v_input_0, v_input_1;
    vector_bf16 v_scaling_bf16;
    vector_f32 v_scaling_f32, v_mod_even, v_mod_odd;
    vector_s32 v_even_code, v_odd_code;
    vector_u8 v_pair01, v_pair23, v_output, v_scratch;

    vlds(v_input_0, v_input_1, srcPtr, window * kElementsPerWindow, DINTLV_B16);
    vlds((vector_u16&)v_scaling_bf16, (__ubuf__ uint16_t*)scalingPtr, 8 * window, E2B_B16);
    vcvt(v_scaling_f32, v_scaling_bf16, preg_all_b16, PART_EVEN);

    vcvt(v_mod_even, v_input_0, preg_b16, PART_EVEN);
    vcvt(v_mod_odd, v_input_1, preg_b16, PART_EVEN);
    vmul(v_mod_even, v_mod_even, v_scaling_f32, preg_f32, MODE_ZEROING);
    vmul(v_mod_odd, v_mod_odd, v_scaling_f32, preg_f32, MODE_ZEROING);
    CalcE2M1SignedCodeI32(v_even_code, v_mod_even, preg_f32);
    CalcE2M1SignedCodeI32(v_odd_code, v_mod_odd, preg_f32);
    PackE2M1SignedCodeBytes(v_pair01, v_even_code, v_odd_code, packIndex, preg_f32);

    vcvt(v_mod_even, v_input_0, preg_b16, PART_ODD);
    vcvt(v_mod_odd, v_input_1, preg_b16, PART_ODD);
    vmul(v_mod_even, v_mod_even, v_scaling_f32, preg_f32, MODE_ZEROING);
    vmul(v_mod_odd, v_mod_odd, v_scaling_f32, preg_f32, MODE_ZEROING);
    CalcE2M1SignedCodeI32(v_even_code, v_mod_even, preg_f32);
    CalcE2M1SignedCodeI32(v_odd_code, v_mod_odd, preg_f32);
    PackE2M1SignedCodeBytes(v_pair23, v_even_code, v_odd_code, packIndex, preg_f32);

    vintlv(
        (RegTensor<uint8_t>&)v_output, (RegTensor<uint8_t>&)v_scratch, (RegTensor<uint8_t>&)v_pair01,
        (RegTensor<uint8_t>&)v_pair23);
    vsts((RegTensor<uint8_t>&)v_output, (__ubuf__ uint8_t*)dstPtr, window * kPackedBytesPerWindow, NORM_B8, preg_b8);
}

PTO_INTERNAL void CalcQuantizedFP4E2M1Values_Half_Tail(
    __ubuf__ half* srcTailPtr, __ubuf__ half* scalingTailPtr, __ubuf__ uint8_t* dstWritePtr, uint32_t tailGroups,
    vector_u8& v_idx)
{
    constexpr uint32_t kGroupSize = 32;
    constexpr uint32_t kPackedBytesPerGroup = kGroupSize / 2;
    uint32_t groupSize = kGroupSize;
    uint32_t packedBytesPerGroup = kPackedBytesPerGroup;
    MaskReg preg_b16 = CreatePredicate<half>(groupSize);
    MaskReg preg_f32 = CreatePredicate<float>(packedBytesPerGroup);
    MaskReg preg_all_b16 = pset_b16(PAT_ALL);
    MaskReg preg_idx = pset_b8(PAT_ALL);

    vector_u8 v_idx;
    vci((RegTensor<int8_t> &)v_idx, (int8_t)0, INC_ORDER);
    vmuls((RegTensor<int16_t> &)v_idx, (RegTensor<int16_t> &)v_idx, (int16_t)4, preg_idx);

    uint32_t windowCount = totalGroups / 8;
    for (uint16_t window = 0; window < (uint16_t)windowCount; ++window) {
        CalcQuantizedFP4E2M1Values_Half_Window(srcPtr, scalingPtr, dstPtr, window, v_idx);
    }

    uint32_t tailGroups = totalGroups - windowCount * 8;
    if (tailGroups == 0) {
        return;
    }

    UnalignReg ureg_out;
    __ubuf__ half *srcTailPtr = srcPtr + windowCount * 256;
    __ubuf__ half *scalingTailPtr = scalingPtr + windowCount * 8;
    __ubuf__ uint8_t *dstWritePtr = dstPtr + windowCount * 128;
    for (uint16_t group = 0; group < (uint16_t)tailGroups; ++group) {
        RegTensor<half> v_input;
        vector_bf16 v_scaling_bf16;
        vector_f32 v_scaling_f32, v_even, v_odd;
        vector_s32 v_even_code, v_odd_code;
        vector_u8 v_output;

        vlds(v_input, srcTailPtr, group * kGroupSize, NORM);
        vcvt(v_even, v_input, preg_b16, PART_EVEN);
        vcvt(v_odd, v_input, preg_b16, PART_ODD);
        vlds((vector_u16&)v_scaling_bf16, (__ubuf__ uint16_t*)scalingTailPtr, group, BRC_B16);
        vcvt(v_scaling_f32, v_scaling_bf16, preg_all_b16, PART_EVEN);
        vmul(v_even, v_even, v_scaling_f32, preg_f32, MODE_ZEROING);
        vmul(v_odd, v_odd, v_scaling_f32, preg_f32, MODE_ZEROING);
        CalcE2M1SignedCodeI32(v_even_code, v_even, preg_f32);
        CalcE2M1SignedCodeI32(v_odd_code, v_odd, preg_f32);
        PackE2M1SignedCodeBytes(v_output, v_even_code, v_odd_code, v_idx, preg_f32);
        mem_bar(VST_VST);
        vstus(ureg_out, kPackedBytesPerGroup, (RegTensor<uint8_t>&)v_output, dstWritePtr, POST_UPDATE);
    }
    vstas(ureg_out, dstWritePtr, 0, POST_UPDATE);
}

PTO_INTERNAL void CalcQuantizedFP4E2M1Values_Half(
    __ubuf__ half* srcPtr, __ubuf__ half* scalingPtr, __ubuf__ uint8_t* dstPtr, uint32_t totalGroups)
{
    constexpr uint32_t kGroupSize = 32;
    constexpr uint32_t kPackedBytesPerGroup = kGroupSize / 2;
    uint32_t groupSize = kGroupSize;
    uint32_t packedBytesPerGroupForPred = kPackedBytesPerGroup;
    MaskReg preg_b16 = CreatePredicate<half>(groupSize);
    MaskReg preg_f32 = CreatePredicate<float>(packedBytesPerGroupForPred);
    MaskReg preg_all_b16 = pset_b16(PAT_ALL);
    MaskReg preg_idx = pset_b8(PAT_ALL);

    vector_u8 v_idx;
    vci((RegTensor<int8_t>&)v_idx, (int8_t)0, INC_ORDER);
    vmuls((RegTensor<int16_t>&)v_idx, (RegTensor<int16_t>&)v_idx, (int16_t)4, preg_idx);

    uint32_t windowCount = totalGroups / 8;
    for (uint16_t window = 0; window < (uint16_t)windowCount; ++window) {
        CalcQuantizedFP4E2M1Values_Half_Window(srcPtr, scalingPtr, dstPtr, window, v_idx);
    }

    uint32_t tailGroups = totalGroups - windowCount * 8;
    if (tailGroups == 0) {
        return;
    }
    CalcQuantizedFP4E2M1Values_Half_Tail(
        srcPtr + windowCount * 256, scalingPtr + windowCount * 8, dstPtr + windowCount * 128, tailGroups, v_idx);
}

// Quantize the tail groups (1..7) of a BF16->MXFP4 E2M1 tile after the full
// 8-group windows have been processed.
PTO_INTERNAL void CalcQuantizedFP4E2M1Values_Bf16_Tail(
    __ubuf__ bfloat16_t* srcTailPtr, __ubuf__ bfloat16_t* scalingTailPtr, __ubuf__ uint8_t* dstWritePtr,
    uint32_t tailGroups, MaskReg& preg_b16_group, vector_u8& v_idx)
{
    constexpr uint32_t kGroupSize = 32;
    constexpr uint32_t kPackedBytesPerGroup = kGroupSize / 2;
    UnalignReg ureg_out;
    for (uint32_t group = 0; group < tailGroups; ++group) {
        vector_bf16 v_input;
        vector_bf16 v_scale;
        vector_bf16 v_scaled;
        vector_f4e2m1x2 v_output_p0, v_output;

        vlds(v_input, srcTailPtr, group * kGroupSize, NORM);
        vlds((vector_u16&)v_scale, (__ubuf__ uint16_t*)scalingTailPtr, group, BRC_B16);
        vmul(v_scaled, v_input, v_scale, preg_b16_group, MODE_ZEROING);
        SaturateBf16NaNToPosInf((vector_u16&)v_scaled, preg_b16_group);
        vcvt(v_output_p0, v_scaled, preg_b16_group, ROUND_R, PART_P0);
        vselr((RegTensor<uint8_t>&)v_output, (RegTensor<uint8_t>&)v_output_p0, (RegTensor<uint8_t>&)v_idx);
        mem_bar(VST_VST);
        vstus(ureg_out, kPackedBytesPerGroup, (RegTensor<uint8_t>&)v_output, dstWritePtr, POST_UPDATE);
    }
    vstas(ureg_out, dstWritePtr, 0, POST_UPDATE);
}

PTO_INTERNAL void CalcQuantizedFP4E2M1Values_Bf16(
    __ubuf__ bfloat16_t* srcPtr, __ubuf__ bfloat16_t* scalingPtr, __ubuf__ uint8_t* dstPtr, uint32_t totalGroups)
{
    constexpr uint32_t kGroupSize = 32;
    constexpr uint32_t kPackedBytesPerGroup = kGroupSize / 2;
    constexpr uint32_t kGroupsPerWindow = 8;
    constexpr uint32_t kElementsPerWindow = kGroupSize * kGroupsPerWindow;
    constexpr uint32_t kPackedBytesPerWindow = kElementsPerWindow / 2;
    constexpr uint32_t kPackedBytesPerHalfWindow = kPackedBytesPerWindow / 2;
    uint32_t groupSize = kGroupSize;
    uint32_t packedBytesPerGroup = kPackedBytesPerGroup;
    MaskReg preg_b16_window = pset_b16(PAT_ALL);
    MaskReg preg_b16_group = CreatePredicate<bfloat16_t>(groupSize);
    MaskReg preg_idx = pset_b8(PAT_ALL);

    vector_u8 v_idx;
    vci((RegTensor<int8_t>&)v_idx, (int8_t)0, INC_ORDER);
    vmuls((RegTensor<int16_t>&)v_idx, (RegTensor<int16_t>&)v_idx, (int16_t)4, preg_idx);

    uint32_t windowCount = totalGroups / kGroupsPerWindow;
    for (uint32_t window = 0; window < windowCount; ++window) {
        vector_bf16 v_input_0, v_input_1;
        vector_bf16 v_intlv_0, v_intlv_1;
        vector_bf16 v_scale;
        vector_f4e2m1x2 v_output_0, v_output_1;

        vlds(v_input_0, v_input_1, srcPtr, window * kElementsPerWindow, DINTLV_B16);
        vlds((vector_u16&)v_scale, (__ubuf__ uint16_t*)scalingPtr, window * kGroupsPerWindow, E2B_B16);
        vmul(v_input_0, v_input_0, v_scale, preg_b16_window, MODE_ZEROING);
        vmul(v_input_1, v_input_1, v_scale, preg_b16_window, MODE_ZEROING);
        SaturateBf16NaNToPosInf((vector_u16&)v_input_0, preg_b16_window);
        SaturateBf16NaNToPosInf((vector_u16&)v_input_1, preg_b16_window);
        vintlv(v_intlv_0, v_intlv_1, v_input_0, v_input_1);
        vcvt(v_output_0, v_intlv_0, preg_b16_window, ROUND_R, PART_P0);
        vcvt(v_output_1, v_intlv_1, preg_b16_window, ROUND_R, PART_P0);
        vsts((RegTensor<uint8_t>&)v_output_0, dstPtr, window * kPackedBytesPerWindow, PK4_B32, preg_b16_window);
        vsts(
            (RegTensor<uint8_t>&)v_output_1, dstPtr, window * kPackedBytesPerWindow + kPackedBytesPerHalfWindow,
            PK4_B32, preg_b16_window);
    }

    uint32_t tailGroups = totalGroups - windowCount * kGroupsPerWindow;
    if (tailGroups == 0) {
        return;
    }

// FP32 -> MXFP8 quantization: shared exponent extraction + FP8 conversion.
template <QuantScaleAlg scale_alg, unsigned StaticRows, unsigned StaticCols>
PTO_INTERNAL void TQuant_MXFP8_F32_Quantize(
    __ubuf__ float* srcPtr, __ubuf__ uint8_t* expPtr, __ubuf__ uint8_t* dstPtr, __ubuf__ float* maxPtr,
    __ubuf__ float* scalingPtr, uint16_t vl_count, unsigned exp_loop_count, uint32_t numGroups,
    unsigned elementsPerRepeat, uint32_t total_elements_count, MaskReg& preg_lower32, MaskReg& preg_upper32)
{
    constexpr bool canUnroll = (StaticRows * StaticCols > 1024) && (StaticRows * StaticCols % 256 == 0);
    if constexpr (canUnroll) {
        if (total_elements_count % 256 == 0) {
            if constexpr (scale_alg == QuantScaleAlg::NV)
                ExtractB8ExponentAndScalingNV<true>(
                    maxPtr, expPtr, scalingPtr, exp_loop_count, numGroups, elementsPerRepeat);
            else
                ExtractB8ExponentAndScaling<true>(
                    maxPtr, expPtr, scalingPtr, exp_loop_count, numGroups, elementsPerRepeat);
            mem_bar(VST_VLD);
            CalcQuantizedFP8Values_Unroll2(
                srcPtr, scalingPtr, dstPtr, vl_count, elementsPerRepeat, total_elements_count);
            return;
        }
    }
    if constexpr (scale_alg == QuantScaleAlg::NV)
        ExtractB8ExponentAndScalingNV<false>(maxPtr, expPtr, scalingPtr, exp_loop_count, numGroups, elementsPerRepeat);
    else
        ExtractB8ExponentAndScaling<false>(maxPtr, expPtr, scalingPtr, exp_loop_count, numGroups, elementsPerRepeat);
    mem_bar(VST_VLD);
    CalcQuantizedFP8Values(
        srcPtr, scalingPtr, dstPtr, vl_count, elementsPerRepeat, total_elements_count, preg_lower32, preg_upper32);
}

// FP32 -> MXFP8 quantization: AbsReduceMax + ExponentScaling + FP8 conversion.
template <QuantScaleAlg scale_alg, unsigned StaticRows, unsigned StaticCols>
PTO_INTERNAL void TQuant_MXFP8_F32(
    __ubuf__ float* srcPtr, __ubuf__ uint8_t* expPtr, __ubuf__ uint8_t* dstPtr, __ubuf__ float* maxPtr,
    __ubuf__ float* scalingPtr, uint16_t vl_count, unsigned exp_loop_count, uint32_t numGroups,
    unsigned elementsPerRepeat, uint32_t total_elements_count, unsigned validRows, unsigned validCols)
{
    MaskReg preg_lower32 = pset_b32(PAT_VL32), preg_upper32, preg_ALL = pset_b32(PAT_ALL);
    pxor(preg_upper32, preg_ALL, preg_lower32, preg_ALL);
    __ubuf__ float* maxPtr_backup = maxPtr;
    if (validRows * validCols <= 1024)
        AbsReduceMax_Naive(
            srcPtr, maxPtr, total_elements_count, vl_count, elementsPerRepeat, preg_lower32, preg_upper32);
    else {
        uint32_t aligned_total = (total_elements_count / 256) * 256;
        uint32_t tail_total = total_elements_count - aligned_total;
        if (aligned_total > 0) {
            uint16_t aligned_vl_count = aligned_total / elementsPerRepeat;
            if (aligned_total % 2048 == 0)
                AbsReduceMax_f32_opt_largesizes(srcPtr, maxPtr, aligned_vl_count, elementsPerRepeat, aligned_total);
            else
                AbsReduceMax_f32_opt(srcPtr, maxPtr, aligned_vl_count, elementsPerRepeat, aligned_total);
        }
        if (tail_total > 0) {
            uint32_t aligned_groups = aligned_total / 32;
            uint16_t tail_vl_count = CeilDivision(tail_total, elementsPerRepeat);
            AbsReduceMax_Naive(
                srcPtr + aligned_total, maxPtr + aligned_groups, tail_total, tail_vl_count, elementsPerRepeat,
                preg_lower32, preg_upper32);
        }
    }
    mem_bar(VST_VLD);
    maxPtr = maxPtr_backup;
    TQuant_MXFP8_F32_Quantize<scale_alg, StaticRows, StaticCols>(
        srcPtr, expPtr, dstPtr, maxPtr, scalingPtr, vl_count, exp_loop_count, numGroups, elementsPerRepeat,
        total_elements_count, preg_lower32, preg_upper32);
}

template <QuantScaleAlg scale_alg, typename T>
PTO_INTERNAL void ReduceMxB16AbsMaxFlat(
    __ubuf__ T* srcPtr, __ubuf__ T* maxPtr, uint16_t vl_count, uint32_t total_elements_count)
{
    constexpr uint32_t elementsPerVL = REPEAT_BYTE / sizeof(T);
    constexpr uint32_t elementsPerLargeLoop = 32 * elementsPerVL;
    const bool useLargeSizePath = (total_elements_count % elementsPerLargeLoop == 0);

    if constexpr (scale_alg == QuantScaleAlg::NV && std::is_same<T, half>::value) {
        if (useLargeSizePath)
            AbsReduceMax_b16_ND_largesizes<T, false>(srcPtr, maxPtr, vl_count, total_elements_count);
        else
            AbsReduceMax_b16_ND<T, false>(srcPtr, maxPtr, vl_count, total_elements_count);
    } else {
        if (useLargeSizePath)
            AbsReduceMax_b16_ND_largesizes(srcPtr, maxPtr, vl_count, total_elements_count);
        else
            AbsReduceMax_b16_ND(srcPtr, maxPtr, vl_count, total_elements_count);
    }
}

// =====================================================================================
// 2D-strided MXFP8 (B16) — PyPTO-facing 2D layout.
//
// PyPTO passes a 2D `exp` tile of shape [validRows, expColStride] (e.g. [2, 32] for
// real shape [2, 64] BF16 with srcCols=128). The kernel writes E8M0 bytes at byte
// offset `row * expColStride` so PyPTO can address row r at the static row stride —
// it can no longer rely on the previous packed-flat layout when there are tail
// columns. `max` and `scaling` are internal scratch. The active 2D path packs them
// tightly as [validRows, ceil(validCols / 32)] and uses unaligned stores/loads for
// those scratch rows; the strided helpers below are kept for the aligned layout.
// =====================================================================================

// Pad a tightly-packed max write pointer up to BLOCK_BYTE alignment and close the
// unaligned store sequence.
template <typename T>
PTO_INTERNAL void PadMaxWriteToAlignment(
    RegTensor<T>& vb16_max, vector_align& ureg_max, __ubuf__ T*& writePtr, uint32_t groupsWritten)
{
    uint32_t alignGroups = BLOCK_BYTE_SIZE / sizeof(T);
    uint32_t paddedGroups = CeilDivision(groupsWritten, alignGroups) * alignGroups;
    uint32_t padCount = paddedGroups - groupsWritten;
    if (padCount > 0)
        vstus(ureg_max, padCount, vb16_max, writePtr, POST_UPDATE);
    vstas(ureg_max, writePtr, 0, POST_UPDATE);
}

// Packed 2D AbsReduceMax: rows are read with srcCols stride, but max is written
// tightly as [validRows, validGroupsPerRow]. This keeps max/scaling scratch sized
// by the real tail-axis shape instead of the static tile tail.
template <QuantScaleAlg scale_alg, typename T>
PTO_INTERNAL void AbsReduceMax_b16_ND_2D_Packed(
    __ubuf__ T* srcPtr, __ubuf__ T* maxPtr, unsigned validRows, unsigned validCols, unsigned srcCols)
{
    RegTensor<T> vb16_max;
    vector_align ureg_max;
    constexpr uint32_t grp_size = 32;
    constexpr uint32_t elements_per_vl = REPEAT_BYTE / sizeof(T);
    constexpr uint32_t elements_per_dintlv = 2 * elements_per_vl;
    constexpr uint32_t grps_per_dintlv = elements_per_dintlv / grp_size;
    uint32_t groupsPerRow = CeilDivision((uint32_t)validCols, grp_size);
    uint32_t elemsPerRow = groupsPerRow * grp_size;
    uint16_t loopNumPerRow = (uint16_t)CeilDivision(elemsPerRow, elements_per_dintlv);
    if (loopNumPerRow == 1) {
        __ubuf__ T* writePtr = maxPtr;
        for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
            uint32_t srcRowOff = (uint32_t)row * srcCols;
            if constexpr (scale_alg == QuantScaleAlg::NV && std::is_same<T, half>::value)
                AbsReduceMax_b16_DintlvWindow<T, false>(srcPtr, srcRowOff, elemsPerRow, vb16_max);
            else
                AbsReduceMax_b16_DintlvWindow<T>(srcPtr, srcRowOff, elemsPerRow, vb16_max);
            uint32_t outCount = groupsPerRow;
            vstus(ureg_max, outCount, vb16_max, writePtr, POST_UPDATE);
        }
        PadMaxWriteToAlignment<T>(vb16_max, ureg_max, writePtr, (uint32_t)validRows * groupsPerRow);
        return;
    }
    __ubuf__ T* writePtr = maxPtr;
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        uint32_t srcRowOff = (uint32_t)row * srcCols;
        for (uint16_t i = 0; i < loopNumPerRow; ++i) {
            uint32_t colOff = (uint32_t)i * elements_per_dintlv;
            uint32_t remaining = (elemsPerRow > colOff) ? (elemsPerRow - colOff) : 0;
            if (remaining > elements_per_dintlv)
                remaining = elements_per_dintlv;
            if constexpr (scale_alg == QuantScaleAlg::NV && std::is_same<T, half>::value)
                AbsReduceMax_b16_DintlvWindow<T, false>(srcPtr, srcRowOff + colOff, remaining, vb16_max);
            else
                AbsReduceMax_b16_DintlvWindow<T>(srcPtr, srcRowOff + colOff, remaining, vb16_max);

            uint32_t grpsWritten = (uint32_t)i * grps_per_dintlv;
            uint32_t grpsRemaining = (groupsPerRow > grpsWritten) ? (groupsPerRow - grpsWritten) : 0;
            uint32_t grpsThisIter = (grpsRemaining > grps_per_dintlv) ? grps_per_dintlv : grpsRemaining;
            vstus(ureg_max, grpsThisIter, vb16_max, writePtr, POST_UPDATE);
        }
    }
    vstas(ureg_max, writePtr, 0, POST_UPDATE);
}

// OCP per-group exponent/scaling context: broadcast constants and masks used by
// ComputeMxOcpExpScaling.
template <typename OcpFormatSpec>
struct OcpExpScalingCtx {
    RegTensor<uint16_t> vu16_max_exp_value;
    RegTensor<uint16_t> vu16_scale_bias;
    RegTensor<uint16_t> vu16_exp_nan;
    RegTensor<uint16_t> vu16_nan;
    RegTensor<uint16_t> vu16_exp_mask;
    RegTensor<uint16_t> vu16_mantissa_mask;
};

// Broadcast OCP-specific constants for exponent extraction.
template <typename OcpFormatSpec>
PTO_INTERNAL void InitOcpExpScalingCtx(OcpExpScalingCtx<OcpFormatSpec>& ctx)
{
    constexpr uint16_t kBf16ExpMask = 0x7F80;
    constexpr uint16_t kBf16MantissaMask = 0x007F;
    constexpr uint16_t kMxMaxExp = OcpFormatSpec::maxExp;
    constexpr uint16_t kBf16ExpBias = 0x7F00;
    constexpr uint16_t kMxExpNan = OcpFormatSpec::expNan;
    constexpr uint16_t kMxB16Nan = OcpFormatSpec::b16Nan;

    vbr(ctx.vu16_max_exp_value, kMxMaxExp);
    vbr(ctx.vu16_scale_bias, kBf16ExpBias);
    vbr(ctx.vu16_exp_nan, kMxExpNan);
    vbr(ctx.vu16_nan, kMxB16Nan);
    vbr(ctx.vu16_exp_mask, kBf16ExpMask);
    vbr(ctx.vu16_mantissa_mask, kBf16MantissaMask);
}

// Per-group OCP exponent/scaling computation from a broadcast-loaded bf16 max.
template <typename OcpFormatSpec>
PTO_INTERNAL void ComputeMxOcpExpScaling(
    RegTensor<uint16_t>& vu16_exp_value, RegTensor<uint16_t>& vu16_recip_scale, RegTensor<uint16_t>& vu16_max_abs,
    MaskReg& preg_one_b16, OcpExpScalingCtx<OcpFormatSpec>& ctx)
{
    constexpr uint16_t kBf16ExpMask = 0x7F80;
    constexpr uint16_t kMxMaxExp = OcpFormatSpec::maxExp;
    RegTensor<uint16_t> vu16_max_exp, vu16_mantissa, vu16_shared_exp;
    vector_bool preg_clamp, preg_special, preg_nan;
    vand(vu16_max_exp, vu16_max_abs, ctx.vu16_exp_mask, preg_one_b16, MODE_ZEROING);
    vand(vu16_mantissa, vu16_max_abs, ctx.vu16_mantissa_mask, preg_one_b16, MODE_ZEROING);
    vcmps_eq(preg_special, vu16_max_exp, kBf16ExpMask, preg_one_b16);
    vcmps_ne(preg_nan, vu16_mantissa, 0, preg_special);
    vcmps_le(preg_clamp, vu16_max_exp, kMxMaxExp, preg_one_b16);
    vsel(vu16_max_exp, ctx.vu16_max_exp_value, vu16_max_exp, preg_clamp);

    vsub(vu16_shared_exp, vu16_max_exp, ctx.vu16_max_exp_value, preg_one_b16, MODE_ZEROING);
    vshrs(vu16_exp_value, vu16_shared_exp, 7, preg_one_b16, MODE_ZEROING);
    vsel(vu16_exp_value, ctx.vu16_exp_nan, vu16_exp_value, preg_nan);

    vsub(vu16_recip_scale, ctx.vu16_scale_bias, vu16_shared_exp, preg_one_b16, MODE_ZEROING);
    vsel(vu16_recip_scale, ctx.vu16_nan, vu16_recip_scale, preg_nan);
}

// Packed 2D exponent/scaling extraction. max/scaling use tight group layout:
//   row * validGroupsPerRow + group
// exp keeps the caller-visible 2D row stride:
//   row * expColStride + group
template <typename T, typename OcpFormatSpec>
PTO_INTERNAL void ExtractMxOcpExponentAndScaling_2D_Packed(
    __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, unsigned validRows,
    unsigned validGroupsPerRow, unsigned expColStride)
{
    RegTensor<uint16_t> vu16_max_abs;
    RegTensor<uint16_t> vu16_scale_value, vu16_recip_scale;
    uint32_t one = 1;
    MaskReg preg_one_b16 = CreatePredicate<T>(one);
    UnalignReg ureg_exp;
    UnalignReg ureg_scaling;
    OcpExpScalingCtx<OcpFormatSpec> ctx;

    InitOcpExpScalingCtx<OcpFormatSpec>(ctx);
    __ubuf__ uint16_t* scaleWritePtr = (__ubuf__ uint16_t*)scalingPtr;
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        __ubuf__ uint8_t* expWritePtr = expPtr + (uint32_t)row * expColStride;
        __ubuf__ T* maxReadPtr = maxPtr + (uint32_t)row * validGroupsPerRow;
        for (uint16_t group = 0; group < (uint16_t)validGroupsPerRow; ++group) {
            vlds(vu16_max_abs, (__ubuf__ uint16_t*)(maxReadPtr + group), 0, BRC_B16);
            ComputeMxOcpExpScaling<OcpFormatSpec>(vu16_scale_value, vu16_recip_scale, vu16_max_abs, preg_one_b16, ctx);
            vstus(ureg_exp, 1, (RegTensor<uint8_t>&)vu16_scale_value, expWritePtr, POST_UPDATE);
            vstus(ureg_scaling, 1, vu16_recip_scale, scaleWritePtr, POST_UPDATE);
        }
        vstas(ureg_exp, expWritePtr, 0, POST_UPDATE);
    }
    vstas(ureg_scaling, scaleWritePtr, 0, POST_UPDATE);
}

// Broadcast constants for the B16 NV exponent/scaling path.
template <typename NvFormatSpec>
PTO_INTERNAL void InitB16NvExpScalingCtx(
    vector_s32& vb32_exp_mask, vector_s32& vb32_mantissa_mask, vector_s32& vb32_exp_nan, vector_s32& vb32_bf16_nan,
    vector_s32& vb32_bf16_min_rcp, vector_s32& vb32_exp_max, uint32_t b16SpecialScaleBits)
{
    vbr(vb32_exp_mask, 0x7F800000);
    vbr(vb32_mantissa_mask, 0x007FFFFF);
    vbr(vb32_exp_nan, 0xFF);
    vbr(vb32_bf16_nan, b16SpecialScaleBits);
    vbr(vb32_bf16_min_rcp, 0x0040);
    vbr(vb32_exp_max, 0xFE);
}

// NV per-group B16 exponent+scaling from a broadcast-loaded max: bf16->f32 widen,
// descale, ceil-round the shared exponent, then bf16-packed scaling + Inf/NaN
// sentinels. Outputs vb32_shared_exp and vb32_bf16_scale_bits (caller stores).
template <typename T, typename NvFormatSpec>
PTO_INTERNAL void ComputeB16NvExpScaling(
    vector_s32& vb32_shared_exp, vector_s32& vb32_bf16_scale_bits, RegTensor<T>& vb16_max, MaskReg& preg_b16,
    MaskReg& preg_b32, vector_s32& vb32_exp_mask, vector_s32& vb32_mantissa_mask, vector_s32& vb32_exp_max,
    vector_s32& vb32_exp_nan, vector_s32& vb32_bf16_nan, vector_s32& vb32_bf16_min_rcp, float descaleMultiplier)
{
    constexpr int f32ExpShift = 23;
    constexpr int bf16ExpShift = 7;
    vector_f32 vb32_max;
    vector_s32 vb32_exponent, vb32_mantissa, vb32_shared_exp_inc;
    vector_bool preg_exp_ff, preg_inf, preg_nan, preg_round_normal, preg_round_subnormal, preg_round_up;
    vector_bool preg_exp_gt_zero, preg_exp_lt_max, preg_exp_eq_zero, preg_mant_gt_zero, preg_mant_eq_zero;
    vector_bool preg_mant_gt_half_subnormal;
    vcvt(vb32_max, vb16_max, preg_b16, PART_EVEN);
    vmuls(vb32_max, vb32_max, descaleMultiplier, preg_b32, MODE_ZEROING);
    vand(vb32_exponent, (vector_s32&)vb32_max, vb32_exp_mask, preg_b32, MODE_ZEROING);
    vand(vb32_mantissa, (vector_s32&)vb32_max, vb32_mantissa_mask, preg_b32, MODE_ZEROING);
    vshrs(vb32_exponent, vb32_exponent, f32ExpShift, preg_b32, MODE_ZEROING);
    vb32_shared_exp = vb32_exponent;
    vadds(vb32_shared_exp_inc, vb32_shared_exp, 1, preg_b32, MODE_ZEROING);
    vcmps_ne(preg_mant_gt_zero, vb32_mantissa, 0, preg_b32);
    vcmps_gt(preg_exp_gt_zero, vb32_exponent, 0, preg_b32);
    vcmps_lt(preg_exp_lt_max, vb32_exponent, 0xFE, preg_b32);
    vcmps_eq(preg_exp_eq_zero, vb32_exponent, 0, preg_b32);
    pand(preg_round_normal, preg_mant_gt_zero, preg_exp_gt_zero, preg_b32);
    pand(preg_round_normal, preg_round_normal, preg_exp_lt_max, preg_b32);
    vcmps_gt(preg_mant_gt_half_subnormal, vb32_mantissa, 0x00400000, preg_b32);
    pand(preg_round_subnormal, preg_mant_gt_half_subnormal, preg_exp_eq_zero, preg_b32);
    por(preg_round_up, preg_round_normal, preg_round_subnormal, preg_b32);
    vsel(vb32_shared_exp, vb32_shared_exp_inc, vb32_shared_exp, preg_round_up);
    vsub(vb32_bf16_scale_bits, vb32_exp_max, vb32_shared_exp, preg_b32);
    vshls((vector_u32&)vb32_bf16_scale_bits, (vector_u32&)vb32_bf16_scale_bits, bf16ExpShift, preg_b32, MODE_ZEROING);
    vcmps_eq(preg_exp_ff, vb32_exponent, 0xFF, preg_b32);
    pnot(preg_mant_eq_zero, preg_mant_gt_zero, preg_b32);
    pand(preg_inf, preg_exp_ff, preg_mant_eq_zero, preg_b32);
    pand(preg_nan, preg_exp_ff, preg_mant_gt_zero, preg_b32);
    vsel(vb32_bf16_scale_bits, vb32_bf16_min_rcp, vb32_bf16_scale_bits, preg_inf);
    vsel(vb32_shared_exp, vb32_exp_max, vb32_shared_exp, preg_inf);
    vsel(vb32_bf16_scale_bits, vb32_bf16_nan, vb32_bf16_scale_bits, preg_nan);
    vsel(vb32_shared_exp, vb32_exp_nan, vb32_shared_exp, preg_nan);
}

template <typename T, typename NvFormatSpec>
PTO_INTERNAL void ExtractNVExponentAndScalingB16_2D_Packed(
    __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, unsigned validRows,
    unsigned validGroupsPerRow, unsigned expColStride)
{
    constexpr float descaleMultiplier = NvFormatSpec::descaleMultiplier;
    constexpr uint32_t b16SpecialScaleBits = NvFormatSpec::b16SpecialScaleBits;
    vector_s32 vb32_exp_nan, vb32_bf16_nan, vb32_bf16_min_rcp, vb32_exp_mask, vb32_mantissa_mask, vb32_exp_max;
    InitB16NvExpScalingCtx<NvFormatSpec>(
        vb32_exp_mask, vb32_mantissa_mask, vb32_exp_nan, vb32_bf16_nan, vb32_bf16_min_rcp, vb32_exp_max,
        b16SpecialScaleBits);
    uint32_t oneB16 = 1;
    uint32_t oneB32 = 1;
    MaskReg preg_b16 = CreatePredicate<T>(oneB16);
    MaskReg preg_b32 = CreatePredicate<float>(oneB32);
    UnalignReg ureg_exp;
    UnalignReg ureg_scaling;
    __ubuf__ uint16_t* scaleWritePtr = (__ubuf__ uint16_t*)scalingPtr;
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        __ubuf__ T* maxReadPtr = maxPtr + (uint32_t)row * validGroupsPerRow;
        __ubuf__ uint8_t* expWritePtr = expPtr + (uint32_t)row * expColStride;
        for (uint16_t group = 0; group < (uint16_t)validGroupsPerRow; ++group) {
            RegTensor<T> vb16_max;
            vector_s32 vb32_shared_exp, vb32_bf16_scale_bits;
            vlds(vb16_max, maxReadPtr + group, 0, BRC_B16);
            ComputeB16NvExpScaling<T, NvFormatSpec>(
                vb32_shared_exp, vb32_bf16_scale_bits, vb16_max, preg_b16, preg_b32, vb32_exp_mask, vb32_mantissa_mask,
                vb32_exp_max, vb32_exp_nan, vb32_bf16_nan, vb32_bf16_min_rcp, descaleMultiplier);
            vstus(ureg_exp, 1, (RegTensor<uint8_t>&)vb32_shared_exp, expWritePtr, POST_UPDATE);
            vstus(ureg_scaling, 1, (vector_u16&)vb32_bf16_scale_bits, scaleWritePtr, POST_UPDATE);
        }
        vstas(ureg_exp, expWritePtr, 0, POST_UPDATE);
    }
    vstas(ureg_scaling, scaleWritePtr, 0, POST_UPDATE);
}

template <typename T, typename OcpFormatSpec, typename NvFormatSpec, QuantScaleAlg scale_alg>
PTO_INTERNAL void ExtractMxExponentAndScaling_2D_Packed(
    __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, unsigned validRows,
    unsigned validGroupsPerRow, unsigned expColStride)
{
    if constexpr (scale_alg == QuantScaleAlg::NV) {
        ExtractNVExponentAndScalingB16_2D_Packed<T, NvFormatSpec>(
            maxPtr, expPtr, scalingPtr, validRows, validGroupsPerRow, expColStride);
    } else {
        ExtractMxOcpExponentAndScaling_2D_Packed<T, OcpFormatSpec>(
            maxPtr, expPtr, scalingPtr, validRows, validGroupsPerRow, expColStride);
    }
}

template <QuantScaleAlg scale_alg, typename T>
PTO_INTERNAL void ExtractB8ExponentAndScaling_2D_Packed(
    __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, unsigned validRows,
    unsigned validGroupsPerRow, unsigned expColStride)
{
    ExtractMxExponentAndScaling_2D_Packed<T, OcpMxFp8E4M3Spec, NvMxFp8E4M3Spec, scale_alg>(
        maxPtr, expPtr, scalingPtr, validRows, validGroupsPerRow, expColStride);
}

template <QuantScaleAlg scale_alg, typename T>
PTO_INTERNAL void ExtractE2M1ExponentAndScaling_2D_Packed(
    __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, unsigned validRows,
    unsigned validGroupsPerRow, unsigned expColStride)
{
    ExtractMxExponentAndScaling_2D_Packed<T, OcpMxFp4E2M1Spec, NvMxFp4E2M1Spec, scale_alg>(
        maxPtr, expPtr, scalingPtr, validRows, validGroupsPerRow, expColStride);
}

template <typename T>
PTO_INTERNAL void BuildPackedScaleWindow(__ubuf__ T* packedScalingPtr, __ubuf__ T* scaleTmpPtr, unsigned groupCount)
{
    RegTensor<T> v_scale;
    UnalignReg ureg_scale;
    __ubuf__ T* writePtr = scaleTmpPtr;
    for (uint16_t group = 0; group < (uint16_t)groupCount; ++group) {
        vlds(v_scale, packedScalingPtr + group, 0, BRC_B16);
        vstus(ureg_scale, 1, v_scale, writePtr, POST_UPDATE);
    }
    vstas(ureg_scale, writePtr, 0, POST_UPDATE);
}

// Packed 2D quantization. scaling is read from tight
// [validRows, groupsPerRow] layout, copied to an aligned
// 8-group temp in maxTmpPtr, then the existing DINTLV/P0-P3
// FP8 packer is reused for the row window.
template <typename T>
PTO_INTERNAL void CalcQuantizedFP8Values_2D_Packed(
    __ubuf__ T* srcPtr, __ubuf__ T* scalingPtr, __ubuf__ uint8_t* dstPtr, __ubuf__ T* maxTmpPtr,
    __ubuf__ uint8_t* expPtr, unsigned validRows, unsigned validCols, unsigned srcCols, unsigned expColStride)
{
    constexpr uint32_t kGroupSize = 32;
    constexpr uint32_t kGroupsPerWindow = 8;
    uint32_t groupsPerRow = CeilDivision((uint32_t)validCols, kGroupSize);
    uint32_t totalGroups = (uint32_t)validRows * groupsPerRow;
    uint16_t windowsPerRow = (uint16_t)CeilDivision(groupsPerRow, kGroupsPerWindow);
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        __ubuf__ T* srcRow = srcPtr + (uint32_t)row * srcCols;
        __ubuf__ uint8_t* dstRow = dstPtr + (uint32_t)row * srcCols;
        __ubuf__ T* scaleRow = scalingPtr + (uint32_t)row * groupsPerRow;
        __ubuf__ T* scaleTmpPtr = maxTmpPtr;
        if (totalGroups < kGroupsPerWindow) {
            scaleTmpPtr = (__ubuf__ T*)(expPtr + (uint32_t)row * expColStride + 16);
        }
        for (uint16_t window = 0; window < windowsPerRow; ++window) {
            uint32_t groupBase = (uint32_t)window * kGroupsPerWindow;
            uint32_t groupsThisWindow = groupsPerRow - groupBase;
            if (groupsThisWindow > kGroupsPerWindow)
                groupsThisWindow = kGroupsPerWindow;
            BuildPackedScaleWindow(scaleRow + groupBase, scaleTmpPtr, groupsThisWindow);
            mem_bar(VST_VLD);
            uint32_t colOff = groupBase * kGroupSize;
            uint32_t remaining = groupsThisWindow * kGroupSize;
            CalcQuantizedFP8Values_B16_Window<T>(srcRow, scaleTmpPtr, dstRow + colOff, 0, colOff, remaining);
        }
    }
}

// Initialise the broadcast f32 exponent/scaling constants.
// NV and OCP share most masks; only the special-case
// sentinels differ. Kept small to stay under the 50-line
// body limit for the per-algorithm compute helpers below.
template <QuantScaleAlg scale_alg>
PTO_INTERNAL void InitF32ExpScalingCtx(
    vector_s32& vb32_exp_mask, vector_s32& vb32_mantissa_mask, vector_s32& vb32_b8_nan, vector_s32& vb32_f32_nan,
    vector_s32& vb32_b8_emax, vector_s32& vb32_exp_max, vector_s32& vb32_recip_min_scale, vector_s32& vb32_b8_exp_fe,
    vector_s32& vb32_f32_min_rcp, vector_s32& vb32_zero)
{
    vbr(vb32_exp_mask, 0x7F800000);
    vbr(vb32_mantissa_mask, 0x007FFFFF);
    vbr(vb32_b8_nan, 0xFF);
    vbr(vb32_zero, 0);
    if constexpr (scale_alg == QuantScaleAlg::NV) {
        vbr(vb32_b8_exp_fe, 0xFE);
        vbr(vb32_f32_nan, NvMxFp8E4M3Spec::f32SpecialScaleBits);
        vbr(vb32_f32_min_rcp, 0x00400000);
    } else {
        vbr(vb32_f32_nan, 0x7FC00000);
        vbr(vb32_b8_emax, OcpMxFp8E4M3Spec::f32Emax);
        vbr(vb32_exp_max, 0xFE);
        vbr(vb32_recip_min_scale, 0x7F000000);
    }
}

// NV per-group f32 exponent+scaling: ceil(log2(max*descale))
// rounding, then Inf/NaN sentinels. Mirrors
// ComputeF32NvExpAndScaling but takes the already
// broadcast-loaded vb32_max (the 2D_Packed path loads via
// BRC_B32).
template <typename NvFormatSpec>
PTO_INTERNAL void ComputeF32ExpScalingNV(
    vector_s32& vb32_shared_exp, vector_s32& vb32_scaling, vector_f32& vb32_max, MaskReg& preg_b32,
    vector_s32& vb32_exp_mask, vector_s32& vb32_mantissa_mask, vector_s32& vb32_b8_exp_fe, vector_s32& vb32_b8_nan,
    vector_s32& vb32_f32_nan, vector_s32& vb32_f32_min_rcp)
{
    constexpr int shr = 23;
    vector_s32 vb32_exponent, vb32_mantissa, vb32_shared_exp_inc;
    vector_bool preg_exp_ff, preg_inf, preg_nan, preg_round_normal, preg_round_subnormal, preg_round_up;
    vector_bool preg_exp_gt_zero, preg_exp_lt_max, preg_exp_eq_zero, preg_mant_gt_zero, preg_mant_eq_zero;
    vector_bool preg_mant_gt_half_subnormal;
    vmuls(vb32_max, vb32_max, NvFormatSpec::descaleMultiplier, preg_b32, MODE_ZEROING);
    vand(vb32_exponent, (vector_s32&)vb32_max, vb32_exp_mask, preg_b32, MODE_ZEROING);
    vand(vb32_mantissa, (vector_s32&)vb32_max, vb32_mantissa_mask, preg_b32, MODE_ZEROING);
    vshrs(vb32_exponent, vb32_exponent, shr, preg_b32, MODE_ZEROING);
    vb32_shared_exp = vb32_exponent;
    vadds(vb32_shared_exp_inc, vb32_shared_exp, 1, preg_b32, MODE_ZEROING);
    vcmps_ne(preg_mant_gt_zero, vb32_mantissa, 0, preg_b32);
    vcmps_gt(preg_exp_gt_zero, vb32_exponent, 0, preg_b32);
    vcmps_lt(preg_exp_lt_max, vb32_exponent, 0xFE, preg_b32);
    vcmps_eq(preg_exp_eq_zero, vb32_exponent, 0, preg_b32);
    pand(preg_round_normal, preg_mant_gt_zero, preg_exp_gt_zero, preg_b32);
    pand(preg_round_normal, preg_round_normal, preg_exp_lt_max, preg_b32);
    vcmps_gt(preg_mant_gt_half_subnormal, vb32_mantissa, 0x00400000, preg_b32);
    pand(preg_round_subnormal, preg_mant_gt_half_subnormal, preg_exp_eq_zero, preg_b32);
    por(preg_round_up, preg_round_normal, preg_round_subnormal, preg_b32);
    vsel(vb32_shared_exp, vb32_shared_exp_inc, vb32_shared_exp, preg_round_up);
    vsub(vb32_scaling, vb32_b8_exp_fe, vb32_shared_exp, preg_b32);
    vshls((vector_u32&)vb32_scaling, (vector_u32&)vb32_scaling, shr, preg_b32, MODE_ZEROING);
    vcmps_eq(preg_exp_ff, vb32_exponent, 0xFF, preg_b32);
    pnot(preg_mant_eq_zero, preg_mant_gt_zero, preg_b32);
    pand(preg_inf, preg_exp_ff, preg_mant_eq_zero, preg_b32);
    pand(preg_nan, preg_exp_ff, preg_mant_gt_zero, preg_b32);
    vsel(vb32_scaling, vb32_f32_min_rcp, vb32_scaling, preg_inf);
    vsel(vb32_shared_exp, vb32_b8_exp_fe, vb32_shared_exp, preg_inf);
    vsel(vb32_scaling, vb32_f32_nan, vb32_scaling, preg_nan);
    vsel(vb32_shared_exp, vb32_b8_nan, vb32_shared_exp, preg_nan);
}

// OCP per-group f32 exponent+scaling: shared_exp = exp -
// emax, subnormal clamp, NaN sentinel. Takes the already
// broadcast-loaded vb32_max.
PTO_INTERNAL void ComputeF32ExpScalingOCP(
    vector_s32& vb32_shared_exp, vector_s32& vb32_scaling, vector_f32& vb32_max, MaskReg& preg_b32,
    vector_s32& vb32_exp_mask, vector_s32& vb32_mantissa_mask, vector_s32& vb32_b8_emax, vector_s32& vb32_exp_max,
    vector_s32& vb32_recip_min_scale, vector_s32& vb32_f32_nan, vector_s32& vb32_b8_nan, vector_s32& vb32_zero)
{
    constexpr int shr = 23;
    constexpr int32_t f32Emax = OcpMxFp8E4M3Spec::f32Emax;
    vector_s32 vb32_exponent, vb32_mantissa;
    vector_bool preg_special, preg_nan, preg_min_scale;
    vand((vector_s32&)vb32_exponent, (vector_s32&)vb32_max, vb32_exp_mask, preg_b32, MODE_ZEROING);
    vand((vector_s32&)vb32_mantissa, (vector_s32&)vb32_max, vb32_mantissa_mask, preg_b32, MODE_ZEROING);
    vshrs((vector_s32&)vb32_exponent, (vector_s32&)vb32_exponent, shr, preg_b32, MODE_ZEROING);
    vsub((vector_u32&)vb32_shared_exp, (vector_u32&)vb32_exponent, (vector_u32&)vb32_b8_emax, preg_b32);
    vsub((vector_s32&)vb32_scaling, (vector_s32&)vb32_exp_max, (vector_s32&)vb32_shared_exp, preg_b32);
    vshls((vector_u32&)vb32_scaling, (vector_u32&)vb32_scaling, shr, preg_b32, MODE_ZEROING);
    vcmps_le(preg_min_scale, (vector_s32&)vb32_exponent, f32Emax, preg_b32);
    vsel(vb32_scaling, vb32_recip_min_scale, vb32_scaling, preg_min_scale);
    vsel(vb32_shared_exp, vb32_zero, vb32_shared_exp, preg_min_scale);
    vcmps_eq(preg_special, (vector_s32&)vb32_exponent, 0xFF, preg_b32);
    vcmps_ne(preg_nan, (vector_s32&)vb32_mantissa, 0, preg_special);
    vsel(vb32_scaling, vb32_f32_nan, vb32_scaling, preg_nan);
    vsel(vb32_shared_exp, vb32_b8_nan, vb32_shared_exp, preg_nan);
}

template <QuantScaleAlg scale_alg>
PTO_INTERNAL void ExtractB8ExponentAndScalingF32_2D_Packed(
    __ubuf__ float* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ float* scalingPtr, unsigned validRows,
    unsigned validGroupsPerRow, unsigned expColStride)
{
    vector_s32 vb32_exp_mask, vb32_mantissa_mask, vb32_b8_nan, vb32_f32_nan, vb32_b8_emax, vb32_exp_max;
    vector_s32 vb32_recip_min_scale, vb32_b8_exp_fe, vb32_f32_min_rcp, vb32_zero;
    InitF32ExpScalingCtx<scale_alg>(
        vb32_exp_mask, vb32_mantissa_mask, vb32_b8_nan, vb32_f32_nan, vb32_b8_emax, vb32_exp_max, vb32_recip_min_scale,
        vb32_b8_exp_fe, vb32_f32_min_rcp, vb32_zero);
    uint32_t one = 1;
    MaskReg preg_b32 = CreatePredicate<float>(one);
    UnalignReg ureg_exp;
    UnalignReg ureg_scaling;
    __ubuf__ float* scaleWritePtr = scalingPtr;
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        __ubuf__ float* maxReadPtr = maxPtr + (uint32_t)row * validGroupsPerRow;
        __ubuf__ uint8_t* expWritePtr = expPtr + (uint32_t)row * expColStride;
        for (uint16_t group = 0; group < (uint16_t)validGroupsPerRow; ++group) {
            vector_f32 vb32_max;
            vector_s32 vb32_shared_exp, vb32_scaling;
            vlds((vector_s32&)vb32_max, (__ubuf__ int32_t*)(maxReadPtr + group), 0, BRC_B32);
            if constexpr (scale_alg == QuantScaleAlg::NV)
                ComputeF32ExpScalingNV<NvMxFp8E4M3Spec>(
                    vb32_shared_exp, vb32_scaling, vb32_max, preg_b32, vb32_exp_mask, vb32_mantissa_mask,
                    vb32_b8_exp_fe, vb32_b8_nan, vb32_f32_nan, vb32_f32_min_rcp);
            else
                ComputeF32ExpScalingOCP(
                    vb32_shared_exp, vb32_scaling, vb32_max, preg_b32, vb32_exp_mask, vb32_mantissa_mask, vb32_b8_emax,
                    vb32_exp_max, vb32_recip_min_scale, vb32_f32_nan, vb32_b8_nan, vb32_zero);
            vstus(ureg_exp, 1, (RegTensor<uint8_t>&)vb32_shared_exp, expWritePtr, POST_UPDATE);
            vstus(ureg_scaling, 1, (vector_f32&)vb32_scaling, scaleWritePtr, POST_UPDATE);
        }
        vstas(ureg_exp, expWritePtr, 0, POST_UPDATE);
    }
    vstas(ureg_scaling, scaleWritePtr, 0, POST_UPDATE);
}

template <QuantScaleAlg scale_alg>
PTO_INTERNAL void TQuant_MXFP8_F32_2D(
    __ubuf__ float* srcPtr, __ubuf__ uint8_t* expPtr, __ubuf__ uint8_t* dstPtr, __ubuf__ float* maxPtr,
    __ubuf__ float* scalingPtr, unsigned validRows, unsigned validCols, unsigned srcCols, unsigned expColStride)
{
    constexpr uint32_t kGroupSize = 32;
    constexpr uint32_t elementsPerRepeat = REPEAT_BYTE / sizeof(float);
    uint32_t groupsPerRow = CeilDivision((uint32_t)validCols, kGroupSize);
    uint16_t vlCountPerRow = (uint16_t)CeilDivision((uint32_t)validCols, elementsPerRepeat);
    MaskReg preg_lower32 = pset_b32(PAT_VL32), preg_upper32, preg_ALL = pset_b32(PAT_ALL);
    pxor(preg_upper32, preg_ALL, preg_lower32, preg_ALL);

    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        AbsReduceMax_Naive(
            srcPtr + (uint32_t)row * srcCols, maxPtr + (uint32_t)row * groupsPerRow, validCols, vlCountPerRow,
            elementsPerRepeat, preg_lower32, preg_upper32);
    }
    mem_bar(VST_VLD);
    ExtractB8ExponentAndScalingF32_2D_Packed<scale_alg>(
        maxPtr, expPtr, scalingPtr, validRows, groupsPerRow, expColStride);
    mem_bar(VST_VLD);
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        CalcQuantizedFP8Values(
            srcPtr + (uint32_t)row * srcCols, scalingPtr + (uint32_t)row * groupsPerRow,
            dstPtr + (uint32_t)row * srcCols, vlCountPerRow, elementsPerRepeat, validCols, preg_lower32, preg_upper32);
    }
}

// 2D-strided AbsReduceMax: per row run a fresh 1D reducer,
// writing groupsPerRow group-maxes packed into row stride
// `maxRowGroupStride` (T elements). Source pad columns must
// be zero so per-row max over the full padded srcCols is
// correct. Caller guarantees `maxRowGroupStride * sizeof(T)
// % 32 == 0`.
template <QuantScaleAlg scale_alg, typename T>
PTO_INTERNAL void AbsReduceMax_b16_ND_2D_Strided(
    __ubuf__ T* srcPtr, __ubuf__ T* maxPtr, unsigned validRows, unsigned srcCols, unsigned maxRowGroupStride)
{
    constexpr uint32_t elemPerVL = REPEAT_BYTE / sizeof(T);
    uint16_t vlPerRow = (uint16_t)CeilDivision(srcCols, elemPerVL);
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        if constexpr (scale_alg == QuantScaleAlg::NV && std::is_same<T, half>::value)
            AbsReduceMax_b16_ND<T, false>(srcPtr + row * srcCols, maxPtr + row * maxRowGroupStride, vlPerRow, srcCols);
        else
            AbsReduceMax_b16_ND<T>(srcPtr + row * srcCols, maxPtr + row * maxRowGroupStride, vlPerRow, srcCols);
    }
}

template <QuantScaleAlg scale_alg, typename T>
PTO_INTERNAL void ExtractB8ExponentAndScaling_2D_Strided(
    __ubuf__ T* maxPtr, __ubuf__ uint8_t* expPtr, __ubuf__ T* scalingPtr, unsigned validRows, unsigned srcCols,
    unsigned maxRowGroupStride, unsigned expColStride, unsigned scalingRowGroupStride)
{
    constexpr uint32_t elementsPerVL = REPEAT_BYTE / sizeof(T);
    uint32_t groupsPerRow = srcCols / 32;
    uint16_t loopsPerRow = (uint16_t)CeilDivision(groupsPerRow, elementsPerVL);
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        __ubuf__ T* maxRow = maxPtr + row * maxRowGroupStride;
        __ubuf__ uint8_t* expRow = expPtr + row * expColStride;
        __ubuf__ T* scalingRow = scalingPtr + row * scalingRowGroupStride;
        if constexpr (scale_alg == QuantScaleAlg::NV) {
            ExtractB8ExponentAndScalingNV(maxRow, expRow, scalingRow, loopsPerRow, groupsPerRow);
        } else {
            for (uint16_t i = 0; i < loopsPerRow; ++i) {
                uint32_t off = i * elementsPerVL;
                uint32_t rem = (groupsPerRow > off) ? (groupsPerRow - off) : 0;
                if (rem > elementsPerVL)
                    rem = elementsPerVL;
                ExtractB8ExponentAndScalingVL<OcpF8E4M3Alg, T>(maxRow, expRow, scalingRow, off, rem);
            }
        }
    }
}

template <typename T>
PTO_INTERNAL void CalcQuantizedFP8Values_2D_Strided(
    __ubuf__ T* srcPtr, __ubuf__ T* scalingPtr, __ubuf__ uint8_t* dstPtr, unsigned validRows, unsigned srcCols,
    unsigned scalingRowGroupStride)
{
    constexpr uint32_t elementsPerVL_b16 = REPEAT_BYTE / sizeof(T);
    constexpr uint32_t elementsPerDintlv = 2 * elementsPerVL_b16;
    uint16_t loopsPerRow = (uint16_t)CeilDivision(srcCols, elementsPerDintlv);
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        uint32_t srcRowOff = row * srcCols;
        uint32_t dstRowOff = row * srcCols;
        uint32_t scaleRowOff = row * scalingRowGroupStride;
        for (uint16_t i = 0; i < loopsPerRow; ++i) {
            uint32_t colOff = i * elementsPerDintlv;
            uint32_t remaining = (srcCols > colOff) ? (srcCols - colOff) : 0;
            if (remaining > elementsPerDintlv)
                remaining = elementsPerDintlv;
            CalcQuantizedFP8Values_B16_Window<T>(
                srcPtr + srcRowOff, scalingPtr + scaleRowOff, dstPtr + dstRowOff, i, colOff, remaining);
        }
    }
}

// 2D-strided B16 → MXFP8 entry: produces exp in PyPTO's 2D
// layout. `expColStride` = TileDataExp::Cols (bytes per row
// in the user-visible exp tile). Caller (TQUANT_IMPL)
// zero-pads source pad columns and guarantees max/scaling
// scratch buffers are sized for `validRows *
// paddedGroupsPerRow` B16 elements.
template <QuantScaleAlg scale_alg, typename T>
PTO_INTERNAL void TQuant_MXFP8_B16_2D(
    __ubuf__ T* srcPtr, __ubuf__ uint8_t* expPtr, __ubuf__ uint8_t* dstPtr, __ubuf__ T* maxPtr, __ubuf__ T* scalingPtr,
    unsigned validRows, unsigned validCols, unsigned srcCols, unsigned expColStride)
{
    uint32_t groupsPerRow = CeilDivision((uint32_t)validCols, 32u);

    AbsReduceMax_b16_ND_2D_Packed<scale_alg, T>(srcPtr, maxPtr, validRows, validCols, srcCols);
    mem_bar(VST_VLD);
    ExtractB8ExponentAndScaling_2D_Packed<scale_alg, T>(
        maxPtr, expPtr, scalingPtr, validRows, groupsPerRow, expColStride);
    mem_bar(VST_VLD);
    CalcQuantizedFP8Values_2D_Packed<T>(
        srcPtr, scalingPtr, dstPtr, maxPtr, expPtr, validRows, validCols, srcCols, expColStride);
}

// B16 (BF16/FP16) -> MXFP8 quantization: AbsReduceMax +
// ExponentScaling + FP8 conversion. When validCols ==
// srcCols (static == dynamic width), the source tile is
// contiguous in UB so the flat 1D reducer applies. Otherwise
// rows are padded to srcCols (ZeroPadSourceTile) and we
// dispatch the 2D per-row reducer that honors the row
// stride. The 2D Extract/Calc passes are only used when
// srcCols % 512 == 0 (NORM 32 B / E2B_B16 16 B alignment),
// else we fall back to the flat Extract/Calc over the
// zero-padded buffer (pad lanes are zero so the result is
// exact; TSTORE trims pad cols via the GM shape).
// 2D-strided B16 -> MXFP8 path: per-row max, then choose aligned 2D or flat
// extract/quantize depending on srcCols alignment and scale algorithm.
template <QuantScaleAlg scale_alg, typename T>
PTO_INTERNAL void TQuant_MXFP8_B16_2D(
    __ubuf__ T* srcPtr, __ubuf__ uint8_t* expPtr, __ubuf__ uint8_t* dstPtr, __ubuf__ T* maxPtr, __ubuf__ T* scalingPtr,
    unsigned exp_loop_count, uint32_t numGroups, uint32_t total_elements_count, unsigned validRows, unsigned validCols,
    unsigned srcCols)
{
    if constexpr (scale_alg == QuantScaleAlg::NV && std::is_same<T, half>::value)
        AbsReduceMax_b16_ND_2D<T, false>(srcPtr, maxPtr, validRows, validCols, srcCols);
    else
        AbsReduceMax_b16_ND_2D(srcPtr, maxPtr, validRows, validCols, srcCols);
    mem_bar(VST_VLD);
    if constexpr (scale_alg == QuantScaleAlg::NV) {
        ExtractB8ExponentAndScalingNV(maxPtr, expPtr, scalingPtr, exp_loop_count, numGroups);
        mem_bar(VST_VLD);
        CalcQuantizedFP8Values(srcPtr, scalingPtr, dstPtr, total_elements_count);
    } else if (srcCols % 512 == 0) {
        ExtractB8ExponentAndScaling_2D<T>(maxPtr, expPtr, scalingPtr, validRows, validCols, srcCols);
        mem_bar(VST_VLD);
        CalcQuantizedFP8Values_2D<T>(srcPtr, scalingPtr, dstPtr, validRows, validCols, srcCols);
    } else {
        ExtractB8ExponentAndScaling(maxPtr, expPtr, scalingPtr, exp_loop_count, numGroups);
        mem_bar(VST_VLD);
        CalcQuantizedFP8Values(srcPtr, scalingPtr, dstPtr, total_elements_count);
    }
}

template <QuantScaleAlg scale_alg, typename T>
PTO_INTERNAL void TQuant_MXFP8_B16(
    __ubuf__ T* srcPtr, __ubuf__ uint8_t* expPtr, __ubuf__ uint8_t* dstPtr, __ubuf__ T* maxPtr, __ubuf__ T* scalingPtr,
    uint16_t vl_count, unsigned exp_loop_count, uint32_t numGroups, uint32_t total_elements_count, unsigned validRows,
    unsigned validCols, unsigned srcCols)
{
    __ubuf__ T* maxPtr_backup = maxPtr;
    if (validCols == srcCols) {
        ReduceMxB16AbsMaxFlat<scale_alg>(srcPtr, maxPtr, vl_count, total_elements_count);
        // Board: add VST_VST alongside VST_VLD/VV_ALL. Sim orders stores implicitly,
        // board does not — missing VST_VST lets Phase-3 E2B_B16 read stale scaling.
        mem_bar(VST_VLD);
        maxPtr = maxPtr_backup;
        if constexpr (scale_alg == QuantScaleAlg::NV)
            ExtractB8ExponentAndScalingNV(maxPtr, expPtr, scalingPtr, exp_loop_count, numGroups);
        else
            ExtractB8ExponentAndScaling(maxPtr, expPtr, scalingPtr, exp_loop_count, numGroups);
        mem_bar(VST_VLD);
        CalcQuantizedFP8Values(srcPtr, scalingPtr, dstPtr, total_elements_count);
    } else {
        TQuant_MXFP8_B16_2D<scale_alg, T>(
            srcPtr, expPtr, dstPtr, maxPtr_backup, scalingPtr, exp_loop_count, numGroups, total_elements_count,
            validRows, validCols, srcCols);
    }
}

template <QuantScaleAlg scale_alg, typename T>
PTO_INTERNAL void TQuant_MXFP4_E2M1_B16(
    __ubuf__ T* srcPtr, __ubuf__ uint8_t* expPtr, __ubuf__ uint8_t* dstPtr, __ubuf__ T* maxPtr, __ubuf__ T* scalingPtr,
    uint16_t vl_count, unsigned exp_loop_count, uint32_t numGroups, uint32_t total_elements_count, unsigned validCols,
    unsigned srcCols)
{
    (void)validCols;
    (void)srcCols;
    __ubuf__ T* maxPtr_backup = maxPtr;
    ReduceMxB16AbsMaxFlat<scale_alg>(srcPtr, maxPtr, vl_count, total_elements_count);
    mem_bar(VST_VLD);
    maxPtr = maxPtr_backup;
    ExtractE2M1ExponentAndScaling(maxPtr, expPtr, scalingPtr, exp_loop_count, numGroups);
    mem_bar(VST_VLD);
    if constexpr (std::is_same<T, half>::value)
        CalcQuantizedFP4E2M1Values_Half(srcPtr, scalingPtr, dstPtr, numGroups);
    else
        CalcQuantizedFP4E2M1Values_Bf16(srcPtr, scalingPtr, dstPtr, numGroups);
}

template <typename T>
PTO_INTERNAL void CalcQuantizedFP4E2M1Values_2D_Packed(
    __ubuf__ T* srcPtr, __ubuf__ T* scalingPtr, __ubuf__ uint8_t* dstPtr, __ubuf__ T* scaleTmpPtr,
    __ubuf__ uint8_t* expPtr, unsigned validRows, unsigned validCols, unsigned srcCols, unsigned expColStride)
{
    constexpr uint32_t kGroupSize = 32;
    constexpr uint32_t kGroupsPerWindow = 8;
    constexpr uint32_t kPackedBytesPerGroup = kGroupSize / 2;
    uint32_t groupsPerRow = CeilDivision((uint32_t)validCols, kGroupSize);
    uint32_t totalGroups = (uint32_t)validRows * groupsPerRow;
    uint16_t windowsPerRow = (uint16_t)CeilDivision(groupsPerRow, kGroupsPerWindow);
    for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
        __ubuf__ T* srcRow = srcPtr + (uint32_t)row * srcCols;
        __ubuf__ T* scaleRow = scalingPtr + (uint32_t)row * groupsPerRow;
        __ubuf__ uint8_t* dstRow = dstPtr + (uint32_t)row * (srcCols / 2);
        __ubuf__ T* rowScaleTmp = scaleTmpPtr;
        (void)totalGroups;
        (void)expPtr;
        (void)expColStride;
        for (uint16_t window = 0; window < windowsPerRow; ++window) {
            uint32_t groupBase = (uint32_t)window * kGroupsPerWindow;
            uint32_t groupsThisWindow = groupsPerRow - groupBase;
            if (groupsThisWindow > kGroupsPerWindow)
                groupsThisWindow = kGroupsPerWindow;
            BuildPackedScaleWindow(scaleRow + groupBase, rowScaleTmp, groupsThisWindow);
            mem_bar(VST_VLD);
            if constexpr (std::is_same<T, half>::value) {
                CalcQuantizedFP4E2M1Values_Half(
                    srcRow + groupBase * kGroupSize, rowScaleTmp, dstRow + groupBase * kPackedBytesPerGroup,
                    groupsThisWindow);
            } else {
                CalcQuantizedFP4E2M1Values_Bf16(
                    srcRow + groupBase * kGroupSize, rowScaleTmp, dstRow + groupBase * kPackedBytesPerGroup,
                    groupsThisWindow);
            }
        }
    }
}

template <QuantScaleAlg scale_alg, typename T>
PTO_INTERNAL void TQuant_MXFP4_E2M1_B16_2D(
    __ubuf__ T* srcPtr, __ubuf__ uint8_t* expPtr, __ubuf__ uint8_t* dstPtr, __ubuf__ T* maxPtr, __ubuf__ T* scalingPtr,
    unsigned validRows, unsigned validCols, unsigned srcCols, unsigned expColStride)
{
    uint32_t groupsPerRow = CeilDivision((uint32_t)validCols, 32u);

    AbsReduceMax_b16_ND_2D_Packed<scale_alg, T>(srcPtr, maxPtr, validRows, validCols, srcCols);
    mem_bar(VST_VLD);
    ExtractE2M1ExponentAndScaling_2D_Packed<scale_alg, T>(
        maxPtr, expPtr, scalingPtr, validRows, groupsPerRow, expColStride);
    mem_bar(VST_VLD);
    CalcQuantizedFP4E2M1Values_2D_Packed<T>(
        srcPtr, scalingPtr, dstPtr, maxPtr, expPtr, validRows, validCols, srcCols, expColStride);
}

// Zero-pad columns [validCols, StaticCols) of a 16-bit
// source tile at VL-aligned offsets (full-VL vlds -> vsel ->
// vsts). Sub-VL stores at non-VL-aligned offsets are
// unreliable on some hardware revisions. Requires StaticCols
// | elemPerVL. Must be called from inside a __VEC_SCOPE__.
template <typename T, unsigned StaticCols>
PTO_INTERNAL void ZeroPadColumns_VLAligned(__ubuf__ T* srcPtr, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elemPerVL = REPEAT_BYTE / sizeof(T);
    static_assert(
        elemPerVL % StaticCols == 0, "StaticCols must evenly divide "
                                     "elements-per-VL for VL-aligned padding");
    constexpr unsigned rowsPerVL = elemPerVL / StaticCols;

    MaskReg pg_all = TQuantPSetTyped<T>(PAT_ALL);

    // Build a periodic predicate: bit p is set iff (p %
    // StaticCols) < validCols. Row 0 contributes positions
    // [0, validCols).
    uint32_t vc = (uint32_t)validCols;
    MaskReg preg_valid = CreatePredicate<T>(vc);
    for (uint16_t r = 1; r < (uint16_t)rowsPerVL; ++r) {
        uint32_t rangeStart = (uint32_t)(r * StaticCols);
        uint32_t rangeEnd = rangeStart + (uint32_t)validCols;
        MaskReg p_end = CreatePredicate<T>(rangeEnd);
        MaskReg p_start = CreatePredicate<T>(rangeStart);
        MaskReg p_row;
        pnot(p_row, p_start, p_end);
        por(preg_valid, preg_valid, p_row, pg_all);
    }

    RegTensor<T> vreg_zero;
    vdup(vreg_zero, (T)0, pg_all, MODE_ZEROING);

    // Write-only: store zeros at padding positions without reading the source.
    // Avoids RMW on MTE2-written UB data which can race on hardware.
    MaskReg preg_pad;
    pxor(preg_pad, pg_all, preg_valid, pg_all);

    uint32_t totalElems = (uint32_t)(validRows * StaticCols);
    uint16_t vlCount = CeilDivision(totalElems, (unsigned)elemPerVL);

    for (uint16_t vi = 0; vi < vlCount; ++vi) {
        vsts(vreg_zero, srcPtr, vi * elemPerVL, NORM_B16, preg_pad);
    }
}

// Fallback zero-padding using vstus/vstas for cases where
// StaticCols doesn't divide VL. Must be called from inside a
// __VEC_SCOPE__.
template <typename T, unsigned StaticCols>
PTO_INTERNAL void ZeroPadColumns_Unaligned(__ubuf__ T* srcPtr, unsigned validRows, unsigned validCols)
{
    constexpr unsigned padElemPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned padCols = StaticCols - validCols;
    uint16_t padRepeatTimes = CeilDivision(padCols, padElemPerRepeat);
    RegTensor<T> vreg_zero;
    UnalignReg ureg_pad;
    MaskReg pg_all = TQuantPSetTyped<T>(PAT_ALL);
    vdup(vreg_zero, (T)0, pg_all, MODE_ZEROING);
    for (uint16_t i = 0; i < (uint16_t)(validRows); ++i) {
        uint32_t cols = (uint32_t)(padCols);
        __ubuf__ T* pdst = srcPtr + i * StaticCols + validCols;
        for (uint16_t j = 0; j < padRepeatTimes; ++j) {
            uint32_t sreg = cols > padElemPerRepeat ? padElemPerRepeat : cols;
            vstus(ureg_pad, sreg, vreg_zero, pdst, POST_UPDATE);
            cols -= padElemPerRepeat;
        }
        vstas(ureg_pad, pdst, 0, POST_UPDATE);
    }
}

// Zero-pad source tile columns for non-float types.
// Dispatches between VL-aligned (full-VL vlds/vsel/vsts) and
// unaligned (vstus/vstas) paths based on tile geometry. Must
// be called from inside a __VEC_SCOPE__.
template <typename T, unsigned StaticCols>
PTO_INTERNAL void ZeroPadSourceTile(__ubuf__ T* srcPtr, unsigned validRows, unsigned validCols)
{
    if constexpr (!std::is_same<T, float>::value) {
        if (validCols < StaticCols) {
            constexpr unsigned elemPerVL = REPEAT_BYTE / sizeof(T);
            if constexpr (elemPerVL % StaticCols == 0)
                ZeroPadColumns_VLAligned<T, StaticCols>(srcPtr, validRows, validCols);
            else
                ZeroPadColumns_Unaligned<T, StaticCols>(srcPtr, validRows, validCols);
        }
    }
}

// TQUANT: FP32/BF16/FP16 -> MXFP8 (e4m3) quantization.
// Exp2DStrided keeps a 2D exp tile's row stride while
// max/scaling stay compact.
template <
    QuantScaleAlg scale_alg, bool Exp2DStrided = false, typename TileDataOut = void, typename TileDataSrc = void,
    typename TileDataExp = void, typename TileDataMax = void, typename TileDataScaling = void>
__tf__ PTO_INTERNAL void TQuant_MXFP8_Impl(
    typename TileDataOut::TileDType __out__ dst, typename TileDataExp::TileDType __out__ exp,
    typename TileDataMax::TileDType __out__ max, typename TileDataScaling::TileDType __out__ scaling,
    typename TileDataSrc::TileDType __in__ src, unsigned validRows, unsigned validCols)
{
    using OutT = typename TileDataOut::DType;
    using ExpT = typename TileDataExp::DType;
    using T = typename TileDataSrc::DType;
    __ubuf__ ExpT* expPtr = (__ubuf__ ExpT*)__cce_get_tile_ptr(exp);
    __ubuf__ T* maxPtr = (__ubuf__ T*)__cce_get_tile_ptr(max);
    __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    __ubuf__ T* scalingPtr = (__ubuf__ T*)__cce_get_tile_ptr(scaling);
    __ubuf__ OutT* dstPtr = (__ubuf__ OutT*)__cce_get_tile_ptr(dst);

    set_ctrl(static_cast<uint64_t>(1) << 50);
    __VEC_SCOPE__
    {
        ZeroPadSourceTile<T, TileDataSrc::Cols>(srcPtr, validRows, validCols);
        mem_bar(VST_VLD);

        constexpr unsigned elemPerVL = REPEAT_BYTE / sizeof(T);
        uint32_t totalElems = validRows * (unsigned)TileDataSrc::Cols;
        uint16_t vlCount = CeilDivision(totalElems, elemPerVL);
        uint32_t numGroups = totalElems / 32;
        unsigned expLoopCount = CeilDivision(numGroups, elemPerVL);
        if constexpr (Exp2DStrided) {
            if constexpr (std::is_same<T, float>::value) {
                TQuant_MXFP8_F32_2D<scale_alg>(
                    srcPtr, (__ubuf__ uint8_t*)expPtr, (__ubuf__ uint8_t*)dstPtr, maxPtr, scalingPtr, validRows,
                    validCols, (unsigned)TileDataSrc::Cols, (unsigned)TileDataExp::Cols);
            } else {
                TQuant_MXFP8_B16_2D<scale_alg, T>(
                    srcPtr, (__ubuf__ uint8_t*)expPtr, (__ubuf__ uint8_t*)dstPtr, maxPtr, scalingPtr, validRows,
                    validCols, (unsigned)TileDataSrc::Cols, (unsigned)TileDataExp::Cols);
            }
        } else if constexpr (std::is_same<T, float>::value) {
            TQuant_MXFP8_F32<scale_alg, TileDataSrc::Rows, TileDataSrc::Cols>(
                srcPtr, (__ubuf__ uint8_t*)expPtr, (__ubuf__ uint8_t*)dstPtr, maxPtr, scalingPtr, vlCount, expLoopCount,
                numGroups, elemPerVL, totalElems, validRows, validCols);
        } else {
            TQuant_MXFP8_B16<scale_alg>(
                srcPtr, (__ubuf__ uint8_t*)expPtr, (__ubuf__ uint8_t*)dstPtr, maxPtr, scalingPtr, vlCount, expLoopCount,
                numGroups, totalElems, validRows, validCols, (unsigned)TileDataSrc::Cols);
        }
    }
}

template <
    QuantScaleAlg scale_alg, bool Exp2DStrided = false, typename TileDataOut = void, typename TileDataSrc = void,
    typename TileDataExp = void, typename TileDataMax = void, typename TileDataScaling = void>
__tf__ PTO_INTERNAL void TQuant_MXFP4_E2M1_Impl(
    typename TileDataOut::TileDType __out__ dst, typename TileDataExp::TileDType __out__ exp,
    typename TileDataMax::TileDType __out__ max, typename TileDataScaling::TileDType __out__ scaling,
    typename TileDataSrc::TileDType __in__ src, unsigned validRows, unsigned validCols)
{
    using T = typename TileDataSrc::DType;
    using ExpT = typename TileDataExp::DType;
    using OutT = typename TileDataOut::DType;
    __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    __ubuf__ ExpT* expPtr = (__ubuf__ ExpT*)__cce_get_tile_ptr(exp);
    __ubuf__ OutT* dstPtr = (__ubuf__ OutT*)__cce_get_tile_ptr(dst);
    __ubuf__ T* maxPtr = (__ubuf__ T*)__cce_get_tile_ptr(max);
    __ubuf__ T* scalingPtr = (__ubuf__ T*)__cce_get_tile_ptr(scaling);

    set_ctrl(static_cast<uint64_t>(1) << 50);
    __VEC_SCOPE__
    {
        ZeroPadSourceTile<T, TileDataSrc::Cols>(srcPtr, validRows, validCols);
        mem_bar(VST_VLD);

        constexpr unsigned elemPerVL = REPEAT_BYTE / sizeof(T);
        uint32_t totalElems = validRows * (unsigned)TileDataSrc::Cols;
        uint16_t vlCount = CeilDivision(totalElems, elemPerVL);
        uint32_t numGroups = totalElems / 32;
        unsigned expLoopCount = CeilDivision(numGroups, elemPerVL);
        if constexpr (Exp2DStrided) {
            TQuant_MXFP4_E2M1_B16_2D<scale_alg>(
                srcPtr, (__ubuf__ uint8_t*)expPtr, (__ubuf__ uint8_t*)dstPtr, maxPtr, scalingPtr, validRows, validCols,
                (unsigned)TileDataSrc::Cols, (unsigned)TileDataExp::Cols);
        } else {
            TQuant_MXFP4_E2M1_B16<scale_alg>(
                srcPtr, (__ubuf__ uint8_t*)expPtr, (__ubuf__ uint8_t*)dstPtr, maxPtr, scalingPtr, vlCount, expLoopCount,
                numGroups, totalElems, validCols, (unsigned)TileDataSrc::Cols);
        }
    }
}

template <typename TileDataOut, typename TileDataSrc, typename TileDataPara>
__tf__ PTO_INTERNAL void TQuant_Int8Sym(
    typename TileDataOut::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src,
    typename TileDataPara::TileDType __in__ scale, unsigned validRows, unsigned validCols)
{
    using T = typename TileDataSrc::DType;  // fp32
    using S = typename TileDataPara::DType; // fp32
    using U = typename TileDataOut::DType;  // int8
    __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    __ubuf__ U* dstPtr = (__ubuf__ U*)__cce_get_tile_ptr(dst);
    __ubuf__ S* scalePtr = (__ubuf__ S*)__cce_get_tile_ptr(scale);
    uint16_t repeatTimes = CeilDivision(validCols, ELE_CNT_B32);
    __VEC_SCOPE__
    {
        RegTensor<float> v_input, v_scale;
        RegTensor<int32_t> v_s32;
        RegTensor<half> vb16;
        RegTensor<int8_t> v_output_s8;
        for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
            uint32_t sreg = validCols;
            for (uint16_t idx = 0; idx < repeatTimes; ++idx) {
                MaskReg preg_b32 = CreatePredicate<float>(sreg);
                vlds(v_scale, scalePtr, row,
                     BRC_B32); // broadcast row scaling
                vlds(v_input, srcPtr, ELE_CNT_B32 * idx + row * TileDataSrc::Cols, NORM);
                vmul(v_input, v_input, v_scale, preg_b32, MODE_ZEROING);
                // Round once at fp32 (s32 round-trip) then exact fp32->fp16->s8.
                vcvt(v_s32, v_input, preg_b32, ROUND_R, RS_ENABLE);
                vcvt(v_input, v_s32, preg_b32, ROUND_R);
                vcvt(vb16, v_input, preg_b32, ROUND_R, RS_ENABLE, PART_EVEN);
                vcvt(v_output_s8, vb16, preg_b32, ROUND_R, RS_ENABLE, PART_EVEN);
                vsts(v_output_s8, dstPtr, ELE_CNT_B32 * idx + row * TileDataOut::Cols, PK4_B32, preg_b32);
            }
        }
    }
}

// TQUANT: fp32 -> u8 conversion, Int8Asym
template <typename TileDataOut, typename TileDataSrc, typename TileDataPara>
__tf__ PTO_INTERNAL void TQuant_Int8Asym(
    typename TileDataOut::TileDType __out__ dst, typename TileDataSrc::TileDType __in__ src,
    typename TileDataPara::TileDType __in__ scale, typename TileDataPara::TileDType __in__ offset, unsigned validRows,
    unsigned validCols)
{
    using T = typename TileDataSrc::DType;  // fp32
    using U = typename TileDataOut::DType;  // uint8
    using S = typename TileDataPara::DType; // fp32
    __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    __ubuf__ U* dstPtr = (__ubuf__ U*)__cce_get_tile_ptr(dst);
    __ubuf__ S* scalePtr = (__ubuf__ S*)__cce_get_tile_ptr(scale);
    __ubuf__ S* offsetPtr = (__ubuf__ S*)__cce_get_tile_ptr(offset);
    uint16_t repeatTimes = CeilDivision(validCols, ELE_CNT_B32);
    __VEC_SCOPE__
    {
        RegTensor<float> vb32_scale, vb32_input, vb32_offset;
        RegTensor<int32_t> vb32_int;
        RegTensor<half> vb16_output;
        RegTensor<uint8_t> vb8_output;
        for (uint16_t row = 0; row < (uint16_t)validRows; ++row) {
            uint32_t sreg = validCols;
            for (uint16_t idx = 0; idx < repeatTimes; ++idx) {
                MaskReg preg_b32 = CreatePredicate<float>(sreg);
                vlds(vb32_scale, scalePtr, row,
                     BRC_B32); // broadcast row scaling
                vlds(vb32_offset, offsetPtr, row,
                     BRC_B32); // broadcast row offset
                vlds(vb32_input, srcPtr, ELE_CNT_B32 * idx + row * TileDataSrc::Cols, NORM);
                vmul(vb32_input, vb32_input, vb32_scale, preg_b32, MODE_ZEROING);
                vadd(vb32_input, vb32_input, vb32_offset, preg_b32, MODE_ZEROING);
                // Round once at fp32 (s32 round-trip) then exact fp32->fp16->u8.
                vcvt(vb32_int, vb32_input, preg_b32, ROUND_R, RS_ENABLE);
                vcvt(vb32_input, vb32_int, preg_b32, ROUND_R);
                vcvt(vb16_output, vb32_input, preg_b32, ROUND_R, RS_ENABLE, PART_EVEN);
                vcvt(vb8_output, vb16_output, preg_b32, ROUND_R, RS_ENABLE, PART_EVEN);
                vsts(vb8_output, dstPtr, ELE_CNT_B32 * idx + row * TileDataOut::Cols, PK4_B32, preg_b32);
            }
        }
    }
}

// Missing stuff and TODOS:
// 1) Dynamic vs static predicate implementation
// 2) Testing on board to assure this case does not fail
// 3) Loop peeling is more efficient than using vbr, but just
// get the correctness first then use loop peeling
// Assumptions:
// 1) validRows is divisible by 32 (grpSize)
template <typename T, uint32_t StaticCols>
PTO_INTERNAL void AbsReduceMax_DN(__ubuf__ T* srcPtr, __ubuf__ T* maxPtr, unsigned validRows, unsigned validCols)
{
    constexpr uint32_t grpSize = 32;
    constexpr uint32_t elementsPerVL = REPEAT_BYTE / sizeof(T);
    uint32_t num_vls_per_row = CeilDivision(validCols, elementsPerVL);
    uint32_t num_grps_per_col = CeilDivision(validRows, grpSize);
    constexpr uint32_t num_vls_inner_loop = 4;
    uint32_t inner_loop_iters = CeilDivision(grpSize, num_vls_inner_loop);
    static constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    // vabs expects a floating-point vector type. bf16
    // registers are represented as vector_f16 on this
    // hardware, so cast accordingly; fp32 uses vector_f32.
    using AbsVecType = std::conditional_t<std::is_same<T, float>::value, vector_f32, vector_f16>;
    RegTensor<T> vreg_0, vreg_1, vreg_2, vreg_3;
    RegTensor<T> vreg_max, vreg_max_0, vreg_max_1, vreg_max_2, vreg_max_3;
    uint32_t preg_cols = validCols;
    for (uint32_t i = 0; i < num_vls_per_row; ++i) {
        uint32_t vl_start = i * elementsPerVL;
        MaskReg preg = CreatePredicate<T>(preg_cols);
        for (uint32_t j = 0; j < num_grps_per_col; ++j) {
            vbr(vreg_max_0, (T)0);
            vbr(vreg_max_1, (T)0);
            vbr(vreg_max_2, (T)0);
            vbr(vreg_max_3, (T)0);
            uint32_t grp_start = j * grpSize * StaticCols;
            for (uint32_t k = 0; k < inner_loop_iters; ++k) {
                uint32_t inner_start = k * num_vls_inner_loop * StaticCols;
                uint32_t offset = vl_start + grp_start + inner_start;
                vlds(vreg_0, srcPtr + offset, 0, NORM);
                vlds(vreg_1, srcPtr + offset, 1 * StaticCols, NORM);
                vlds(vreg_2, srcPtr + offset, 2 * StaticCols, NORM);
                vlds(vreg_3, srcPtr + offset, 3 * StaticCols, NORM);
                vabs((AbsVecType&)vreg_0, (AbsVecType&)vreg_0, preg);
                vabs((AbsVecType&)vreg_1, (AbsVecType&)vreg_1, preg);
                vabs((AbsVecType&)vreg_2, (AbsVecType&)vreg_2, preg);
                vabs((AbsVecType&)vreg_3, (AbsVecType&)vreg_3, preg);
                vmax(vreg_max_0, vreg_0, vreg_max_0, preg, MODE_ZEROING);
                vmax(vreg_max_1, vreg_1, vreg_max_1, preg, MODE_ZEROING);
                vmax(vreg_max_2, vreg_2, vreg_max_2, preg, MODE_ZEROING);
                vmax(vreg_max_3, vreg_3, vreg_max_3, preg, MODE_ZEROING);
            }
            vmax(vreg_max_0, vreg_max_0, vreg_max_1, preg, MODE_ZEROING);
            vmax(vreg_max_2, vreg_max_2, vreg_max_3, preg, MODE_ZEROING);
            vmax(vreg_max, vreg_max_0, vreg_max_2, preg, MODE_ZEROING);
            vsts(vreg_max, maxPtr, j * StaticCols + i * elementsPerVL, distValue, preg);
        }
    }
}

// fp16 & bf16
template <typename T, uint32_t StaticCols>
PTO_INTERNAL void calcQuantizedFP8Values_DN_B16(
    __ubuf__ T* srcPtr, __ubuf__ T* scalingPtr, __ubuf__ uint8_t* dstPtr, unsigned validRows, unsigned validCols)
{
    constexpr uint32_t grpSize = 32;
    constexpr uint32_t b16ElementsPerVL = REPEAT_BYTE / sizeof(T); // B16 elements per VL
    uint32_t num_vls_per_row = CeilDivision((uint32_t)validCols, b16ElementsPerVL);
    uint32_t num_grps_per_col = CeilDivision((uint32_t)validRows, grpSize);
    RegTensor<T> vb16_scaling, vb16_input;
    vector_f32 vb32_scaling_even, vb32_scaling_odd;
    vector_f32 vb32_input_even, vb32_input_odd;
    vector_f8e4m3 vb8_p0, vb8_p1, vb8_out;
    uint32_t preg_cols_b16 = validCols;
    uint32_t preg_cols_b8 = validCols * 2;
    uint32_t preg_cols_b32 = validCols;
    for (uint32_t i = 0; i < num_vls_per_row; ++i) {
        uint32_t vl_start = i * b16ElementsPerVL;
        MaskReg preg_b16 = CreatePredicate<bfloat16_t>(preg_cols_b16);
        MaskReg preg_b8 = CreatePredicate<uint8_t>(preg_cols_b8);
        MaskReg preg_b32 = CreatePredicate<float>(preg_cols_b32);
        for (uint32_t j = 0; j < num_grps_per_col; ++j) {
            uint32_t row_base = j * grpSize;
            vlds(vb16_scaling, scalingPtr, vl_start + j * StaticCols, NORM);
            vcvt(vb32_scaling_even, vb16_scaling, preg_b16, PART_EVEN);
            vcvt(vb32_scaling_odd, vb16_scaling, preg_b16, PART_ODD);
            for (uint32_t k = 0; k < grpSize; ++k) {
                uint32_t r = row_base + k;
                vlds(vb16_input, srcPtr, r * StaticCols + vl_start, NORM);
                vcvt(vb32_input_even, vb16_input, preg_b16, PART_EVEN);
                vcvt(vb32_input_odd, vb16_input, preg_b16, PART_ODD);
                vmul(vb32_input_even, vb32_input_even, vb32_scaling_even, preg_b32, MODE_ZEROING);
                vmul(vb32_input_odd, vb32_input_odd, vb32_scaling_odd, preg_b32, MODE_ZEROING);
                vcvt(vb8_p0, vb32_input_even, preg_b32, ROUND_R, RS_ENABLE, PART_P0);
                vcvt(vb8_p1, vb32_input_odd, preg_b32, ROUND_R, RS_ENABLE, PART_P1);
                vor(vb8_out, vb8_p0, vb8_p1, preg_b8);
                uint32_t dst_byte_offset = r * StaticCols + vl_start;
                vsts((vector_u32&)vb8_out, (__ubuf__ uint32_t*)dstPtr, dst_byte_offset / 4, PK_B32, preg_b8);
            }
        }
    }
}

// fp32
template <uint32_t StaticCols>
PTO_INTERNAL void calcQuantizedFP8Values_DN_float(
    __ubuf__ float* srcPtr, __ubuf__ float* scalingPtr, __ubuf__ uint8_t* dstPtr, unsigned validRows,
    unsigned validCols)
{
    constexpr uint32_t grpSize = 32;
    constexpr uint32_t b32ElementsPerVL = REPEAT_BYTE / sizeof(float); // B32 elements per VL
    uint32_t num_vls_per_row = CeilDivision((uint32_t)validCols, b32ElementsPerVL);
    uint32_t num_grps_per_col = CeilDivision((uint32_t)validRows, grpSize);
    RegTensor<float> vf32_scaling, vf32_input;
    vector_f8e4m3 vb8_out;
    for (uint32_t i = 0; i < num_vls_per_row; ++i) {
        uint32_t vl_start = i * b32ElementsPerVL;
        uint32_t preg_cols_b32 = validCols;
        uint32_t preg_cols_b8 = validCols * 4;
        MaskReg preg_b32 = CreatePredicate<float>(preg_cols_b32);
        MaskReg preg_b8 = CreatePredicate<uint8_t>(preg_cols_b8);
        for (uint32_t j = 0; j < num_grps_per_col; ++j) {
            uint32_t row_base = j * grpSize;
            vlds(vf32_scaling, scalingPtr, vl_start + j * StaticCols, NORM);
            for (uint32_t k = 0; k < grpSize; ++k) {
                uint32_t r = row_base + k;
                vlds(vf32_input, srcPtr, r * StaticCols + vl_start, NORM);
                vmul(vf32_input, vf32_input, vf32_scaling, preg_b32, MODE_ZEROING);
                vcvt(vb8_out, vf32_input, preg_b32, ROUND_R, RS_ENABLE, PART_P0);
                vsts((vector_u8&)vb8_out, dstPtr, r * StaticCols + vl_start, PK4_B32, preg_b8);
            }
        }
    }
}

// MXFP4 E2M1 DN stage 3 for BF16. Mirrors
// calcQuantizedFP8Values_DN_B16 but emits FP4: bf16 multiply
// (bf16 rounding) -> NaN->+Inf -> vcvt bf16->fp4 (PART_P0)
// -> PK4_B32 store (128 bf16 -> 64 packed bytes, 2
// nibbles/byte).
template <uint32_t StaticCols>
PTO_INTERNAL void calcQuantizedFP4E2M1Values_DN_Bf16(
    __ubuf__ bfloat16_t* srcPtr, __ubuf__ bfloat16_t* scalingPtr, __ubuf__ uint8_t* dstPtr, unsigned validRows,
    unsigned validCols)
{
    constexpr uint32_t grpSize = 32;
    constexpr uint32_t b16ElementsPerVL = REPEAT_BYTE / sizeof(bfloat16_t);
    uint32_t num_vls_per_row = CeilDivision((uint32_t)validCols, b16ElementsPerVL);
    uint32_t num_grps_per_col = CeilDivision((uint32_t)validRows, grpSize);
    RegTensor<bfloat16_t> vb16_scaling, vb16_input;
    vector_bf16 vb16_scaled;
    vector_f4e2m1x2 vfp4;
    uint32_t preg_cols_b16 = validCols;
    for (uint32_t i = 0; i < num_vls_per_row; ++i) {
        uint32_t vl_start = i * b16ElementsPerVL;
        MaskReg preg_b16 = CreatePredicate<bfloat16_t>(preg_cols_b16);
        for (uint32_t j = 0; j < num_grps_per_col; ++j) {
            uint32_t row_base = j * grpSize;
            vlds(vb16_scaling, scalingPtr, vl_start + j * StaticCols, NORM);
            for (uint32_t k = 0; k < grpSize; ++k) {
                uint32_t r = row_base + k;
                vlds(vb16_input, srcPtr, r * StaticCols + vl_start, NORM);
                vmul(vb16_scaled, (vector_bf16&)vb16_input, (vector_bf16&)vb16_scaling, preg_b16, MODE_ZEROING);
                SaturateBf16NaNToPosInf((vector_u16&)vb16_scaled, preg_b16);
                vcvt(vfp4, vb16_scaled, preg_b16, ROUND_R, PART_P0);
                uint32_t dst_byte_offset = r * StaticCols + vl_start;
                vsts((RegTensor<uint8_t>&)vfp4, dstPtr, dst_byte_offset / 2, PK4_B32, preg_b16);
            }
        }
    }
}

// Per-row FP16->MXFP4 E2M1 DN quantization. The fp16 input is widened to fp32,
// multiplied by the pre-widened EVEN/ODD fp32 scaling, rounded back to bf16,
// NaN-saturated, and packed to FP4.
template <uint32_t StaticCols>
PTO_INTERNAL void calcQuantizedFP4E2M1Values_DN_Fp16_Row(
    __ubuf__ half* srcPtr, __ubuf__ uint8_t* dstPtr, vector_f32& vf32_scaling_even, vector_f32& vf32_scaling_odd,
    uint32_t r, uint32_t vl_start, MaskReg& preg_b16, MaskReg& preg_b32)
{
    RegTensor<half> vf16_input;
    vector_bf16 vb16_even, vb16_odd, vb16_merged;
    vector_f32 vf32_even, vf32_odd;
    vector_f4e2m1x2 vfp4;

    vlds(vf16_input, srcPtr, r * StaticCols + vl_start, NORM);
    vcvt(vf32_even, (vector_f16&)vf16_input, preg_b16, PART_EVEN);
    vcvt(vf32_odd, (vector_f16&)vf16_input, preg_b16, PART_ODD);
    vmul(vf32_even, vf32_even, vf32_scaling_even, preg_b32, MODE_ZEROING);
    vmul(vf32_odd, vf32_odd, vf32_scaling_odd, preg_b32, MODE_ZEROING);
    vcvt(vb16_even, vf32_even, preg_b32, ROUND_R, RS_ENABLE, PART_EVEN);
    vcvt(vb16_odd, vf32_odd, preg_b32, ROUND_R, RS_ENABLE, PART_ODD);
    vor(vb16_merged, vb16_even, vb16_odd, preg_b16, MODE_ZEROING);
    SaturateBf16NaNToPosInf((vector_u16&)vb16_merged, preg_b16);
    vcvt(vfp4, vb16_merged, preg_b16, ROUND_R, PART_P0);
    uint32_t dst_byte_offset = r * StaticCols + vl_start;
    vsts((RegTensor<uint8_t>&)vfp4, dstPtr, dst_byte_offset / 2, PK4_B32, preg_b16);
}

// MXFP4 E2M1 DN stage 3 for FP16. fp16 must be scaled in
// fp32 to keep mantissa precision (user requirement), and
// there is no fp32->fp4 vcvt, so the path is fp16->fp32
// (EVEN/ODD) -> fp32*scaling_fp32 -> fp32->bf16 (ROUND_R)
// via vintlv merge -> NaN->+Inf -> vcvt bf16->fp4 (PART_P0)
// -> PK4_B32. Scaling is stored as bf16; it is widened to
// fp32 once per group (hoisted).
template <uint32_t StaticCols>
PTO_INTERNAL void calcQuantizedFP4E2M1Values_DN_Fp16(
    __ubuf__ half* srcPtr, __ubuf__ half* scalingPtr, __ubuf__ uint8_t* dstPtr, unsigned validRows, unsigned validCols)
{
    constexpr uint32_t grpSize = 32;
    constexpr uint32_t b16ElementsPerVL = REPEAT_BYTE / sizeof(half);
    uint32_t num_vls_per_row = CeilDivision((uint32_t)validCols, b16ElementsPerVL);
    uint32_t num_grps_per_col = CeilDivision((uint32_t)validRows, grpSize);
    vector_bf16 vb16_scaling;
    vector_f32 vf32_scaling_even, vf32_scaling_odd;
    uint32_t preg_cols_b16 = validCols;
    uint32_t preg_cols_b32 = validCols;
    for (uint32_t i = 0; i < num_vls_per_row; ++i) {
        uint32_t vl_start = i * b16ElementsPerVL;
        MaskReg preg_b16 = CreatePredicate<half>(preg_cols_b16);
        MaskReg preg_b32 = CreatePredicate<float>(preg_cols_b32);
        for (uint32_t j = 0; j < num_grps_per_col; ++j) {
            uint32_t row_base = j * grpSize;
            // Scaling bits are bf16 (stage 2 writes bf16) but the tile is typed
            // half when T=half; load as bf16 via a u16 cast (matches non-DN FP16).
            vlds((vector_u16&)vb16_scaling, (__ubuf__ uint16_t*)(scalingPtr + vl_start + j * StaticCols), 0, NORM);
            vcvt(vf32_scaling_even, vb16_scaling, preg_b16, PART_EVEN);
            vcvt(vf32_scaling_odd, vb16_scaling, preg_b16, PART_ODD);
            for (uint32_t k = 0; k < grpSize; ++k) {
                uint32_t r = row_base + k;
                calcQuantizedFP4E2M1Values_DN_Fp16_Row<StaticCols>(
                    srcPtr, dstPtr, vf32_scaling_even, vf32_scaling_odd, r, vl_start, preg_b16, preg_b32);
            }
        }
    }
}

template <QuantScaleAlg scale_alg, typename T, unsigned StaticCols>
PTO_INTERNAL void TQuant_MXFP8_DN(
    __ubuf__ T* srcPtr, __ubuf__ uint8_t* expPtr, __ubuf__ uint8_t* dstPtr, __ubuf__ T* maxPtr, __ubuf__ T* scalingPtr,
    unsigned validRows, unsigned validCols)
{
    constexpr uint32_t grpSize = 32;
    constexpr uint32_t elementsPerVL = REPEAT_BYTE / sizeof(T);
    AbsReduceMax_DN<T, StaticCols>(srcPtr, maxPtr, validRows, validCols);
    mem_bar(VST_VLD);
    // DN-aware 2D extraction: each row-group j owns one row
    // of width StaticCols in the max/scaling/exp tiles.
    // Process each row as a 1D VL sequence so all offsets
    // stay aligned while preserving the 2D tile shape.
    uint32_t num_grps_per_col = CeilDivision((uint32_t)validRows, grpSize);
    uint32_t num_vls_per_row = CeilDivision((uint32_t)validCols, elementsPerVL);
    for (uint32_t j = 0; j < num_grps_per_col; ++j) {
        __ubuf__ T* maxRowPtr = maxPtr + j * StaticCols;
        __ubuf__ uint8_t* expRowPtr = expPtr + j * StaticCols;
        __ubuf__ T* scalingRowPtr = scalingPtr + j * StaticCols;
        for (uint32_t i = 0; i < num_vls_per_row; ++i) {
            uint32_t off = i * elementsPerVL;
            uint32_t rem = (validCols > off) ? (validCols - off) : 0;
            if (rem > elementsPerVL)
                rem = elementsPerVL;
            ExtractB8ExponentAndScalingVL<OcpF8E4M3Alg, T>(maxRowPtr, expRowPtr, scalingRowPtr, off, rem);
        }
    }
    mem_bar(VST_VLD);
    if constexpr (std::is_same<T, float>::value)
        calcQuantizedFP8Values_DN_float<StaticCols>(srcPtr, scalingPtr, dstPtr, validRows, validCols);
    else
        calcQuantizedFP8Values_DN_B16<T, StaticCols>(srcPtr, scalingPtr, dstPtr, validRows, validCols);
}

template <
    QuantScaleAlg scale_alg, typename TileDataOut, typename TileDataSrc, typename TileDataExp, typename TileDataMax,
    typename TileDataScaling>
__tf__ PTO_INTERNAL void TQuant_MXFP8_Impl_DN(
    typename TileDataOut::TileDType __out__ dst, typename TileDataExp::TileDType __out__ exp,
    typename TileDataMax::TileDType __out__ max, typename TileDataScaling::TileDType __out__ scaling,
    typename TileDataSrc::TileDType __in__ src, unsigned validRows, unsigned validCols)
{
    using T = typename TileDataSrc::DType;
    using ExpT = typename TileDataExp::DType;
    using OutT = typename TileDataOut::DType;
    __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    __ubuf__ ExpT* expPtr = (__ubuf__ ExpT*)__cce_get_tile_ptr(exp);
    __ubuf__ OutT* dstPtr = (__ubuf__ OutT*)__cce_get_tile_ptr(dst);
    __ubuf__ T* maxPtr = (__ubuf__ T*)__cce_get_tile_ptr(max);
    __ubuf__ T* scalingPtr = (__ubuf__ T*)__cce_get_tile_ptr(scaling);
    // expDn was dropped: the caller owns the E8 reshape via
    // TMOV(e8DnTile, exp).

    set_ctrl(static_cast<uint64_t>(1) << 50);
    __VEC_SCOPE__
    {
        ZeroPadSourceTile<T, TileDataSrc::Cols>(srcPtr, validRows, validCols);
        mem_bar(VST_VLD);
        TQuant_MXFP8_DN<scale_alg, T, TileDataSrc::Cols>(
            srcPtr, (__ubuf__ uint8_t*)expPtr, (__ubuf__ uint8_t*)dstPtr, maxPtr, scalingPtr, validRows, validCols);
    }
}

// MXFP4 E2M1 DN: same 3-stage shape as MXFP8 DN but OCP E2M1
// exponent extraction (OcpF4E2M1Alg -> maxExp 0x0100) and
// FP4 stage-3 quantize. scaling is bf16.
template <QuantScaleAlg scale_alg, typename T, unsigned StaticCols>
PTO_INTERNAL void TQuant_MXFP4_E2M1_DN(
    __ubuf__ T* srcPtr, __ubuf__ uint8_t* expPtr, __ubuf__ uint8_t* dstPtr, __ubuf__ T* maxPtr, __ubuf__ T* scalingPtr,
    unsigned validRows, unsigned validCols)
{
    constexpr uint32_t grpSize = 32;
    constexpr uint32_t elementsPerVL = REPEAT_BYTE / sizeof(T);
    AbsReduceMax_DN<T, StaticCols>(srcPtr, maxPtr, validRows, validCols);
    mem_bar(VST_VLD);
    uint32_t num_grps_per_col = CeilDivision((uint32_t)validRows, grpSize);
    uint32_t num_vls_per_row = CeilDivision((uint32_t)validCols, elementsPerVL);
    for (uint32_t j = 0; j < num_grps_per_col; ++j) {
        __ubuf__ T* maxRowPtr = maxPtr + j * StaticCols;
        __ubuf__ uint8_t* expRowPtr = expPtr + j * StaticCols;
        __ubuf__ T* scalingRowPtr = scalingPtr + j * StaticCols;
        for (uint32_t i = 0; i < num_vls_per_row; ++i) {
            uint32_t off = i * elementsPerVL;
            uint32_t rem = (validCols > off) ? (validCols - off) : 0;
            if (rem > elementsPerVL)
                rem = elementsPerVL;
            ExtractB8ExponentAndScalingVL<OcpF4E2M1Alg, T>(maxRowPtr, expRowPtr, scalingRowPtr, off, rem);
        }
    }
    mem_bar(VST_VLD);
    if constexpr (std::is_same<T, bfloat16_t>::value)
        calcQuantizedFP4E2M1Values_DN_Bf16<StaticCols>(srcPtr, scalingPtr, dstPtr, validRows, validCols);
    else
        calcQuantizedFP4E2M1Values_DN_Fp16<StaticCols>(srcPtr, scalingPtr, dstPtr, validRows, validCols);
}

template <
    QuantScaleAlg scale_alg, typename TileDataOut, typename TileDataSrc, typename TileDataExp, typename TileDataMax,
    typename TileDataScaling>
__tf__ PTO_INTERNAL void TQuant_MXFP4_E2M1_Impl_DN(
    typename TileDataOut::TileDType __out__ dst, typename TileDataExp::TileDType __out__ exp,
    typename TileDataMax::TileDType __out__ max, typename TileDataScaling::TileDType __out__ scaling,
    typename TileDataSrc::TileDType __in__ src, unsigned validRows, unsigned validCols)
{
    using T = typename TileDataSrc::DType;
    using ExpT = typename TileDataExp::DType;
    using OutT = typename TileDataOut::DType;
    __ubuf__ T* srcPtr = (__ubuf__ T*)__cce_get_tile_ptr(src);
    __ubuf__ ExpT* expPtr = (__ubuf__ ExpT*)__cce_get_tile_ptr(exp);
    __ubuf__ OutT* dstPtr = (__ubuf__ OutT*)__cce_get_tile_ptr(dst);
    __ubuf__ T* maxPtr = (__ubuf__ T*)__cce_get_tile_ptr(max);
    __ubuf__ T* scalingPtr = (__ubuf__ T*)__cce_get_tile_ptr(scaling);
    // expDn was dropped from this signature: the caller owns
    // the E8 reshape via TMOV(e8DnTile, exp) so TQUANT only
    // writes the exponent into exp.

    set_ctrl(static_cast<uint64_t>(1) << 50);
    __VEC_SCOPE__
    {
        ZeroPadSourceTile<T, TileDataSrc::Cols>(srcPtr, validRows, validCols);
        mem_bar(VST_VLD);
        TQuant_MXFP4_E2M1_DN<scale_alg, T, TileDataSrc::Cols>(
            srcPtr, (__ubuf__ uint8_t*)expPtr, (__ubuf__ uint8_t*)dstPtr, maxPtr, scalingPtr, validRows, validCols);
    }
}
template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara>
PTO_INTERNAL void TQUANT_IMPL(TileDataOut& dst, TileDataSrc& src, TileDataPara& scale, TileDataPara* offset = nullptr)
{
    using T = typename TileDataSrc::DType;
    static_assert(std::is_same<T, float32_t>::value, "Fix: Input has to be float 32");

    if constexpr (quant_type == QuantType::INT8_SYM) {
        using U = typename TileDataOut::DType;
        static_assert(
            std::is_same<U, int8_t>::value, "Fix: Quant INT8 sym: Out data type "
                                            "has to be int8");
        TQuant_Int8Sym<TileDataOut, TileDataSrc, TileDataPara>(
            dst.data(), src.data(), scale.data(), src.GetValidRow(), src.GetValidCol());
    } else if constexpr (quant_type == QuantType::INT8_ASYM) {
        using U = typename TileDataOut::DType;
        static_assert(
            std::is_same<U, uint8_t>::value, "Fix: Quant INT8 asym: Out data type "
                                             "has to be uint8");
        TQuant_Int8Asym<TileDataOut, TileDataSrc, TileDataPara>(
            dst.data(), src.data(), scale.data(), offset->data(), src.GetValidRow(), src.GetValidCol());
    }
}

// Tmp-aware overload to keep the INT8 SYM/ASYM interface
// identical to A2/A3. A5 broadcasts scale/offset natively
// (vlds BRC_B32) and needs no scratch tile, so tmp is
// unused.
template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara, typename TileDataTmp>
PTO_INTERNAL void TQUANT_IMPL(
    TileDataOut& dst, TileDataSrc& src, TileDataPara& scale, [[maybe_unused]] TileDataTmp& tmp,
    TileDataPara* offset = nullptr)
{
    TQUANT_IMPL<quant_type, TileDataOut, TileDataSrc, TileDataPara>(dst, src, scale, offset);
}

// TQUANT Interface for FP32/BF16/FP16->MXFP8 (ND mode)
// E8M0, max, and scaling tiles may be passed as 2D; TQUANT
// reshapes them to 1D internally.

// Single validation point for MX TQUANT type constraints. Called from the two
// public TQUANT_IMPL entry points; no other MX implementation function should
// repeat these static_asserts.
template <QuantType quant_type, typename T, typename OutT>
PTO_INTERNAL constexpr void CheckTQuantMxTypes()
{
    static_assert(
        quant_type == QuantType::MXFP8 || quant_type == QuantType::MXFP4_E2M1,
        "Fix: TQUANT MX supports MXFP8/MXFP4_E2M1 only.");
    if constexpr (quant_type == QuantType::MXFP8) {
        static_assert(
            std::is_same<T, float32_t>::value || std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value,
            "Fix: MXFP8 input has to be float32, bfloat16, or float16 (half)");
    } else { // MXFP4_E2M1
        static_assert(
            std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value,
            "Fix: MXFP4_E2M1 input has to be float16 (half) or bfloat16");
        static_assert(std::is_same<OutT, float4_e2m1x2_t>::value, "Fix: MXFP4_E2M1 output has to be float4_e2m1x2_t");
    }
}

template <
    QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataExp, typename TileDataMax,
    typename TileDataScaling>
PTO_INTERNAL void TQUANT_IMPL(
    TileDataOut& dst, TileDataSrc& src, TileDataExp* exp, TileDataMax* max, TileDataScaling* scaling);

template <
    QuantType quant_type, QuantScaleAlg scale_alg, typename TileDataOut, typename TileDataSrc, typename TileDataExp,
    typename TileDataMax, typename TileDataScaling>
PTO_INTERNAL void TQUANT_IMPL(
    TileDataOut& dst, TileDataSrc& src, TileDataExp* exp, TileDataMax* max, TileDataScaling* scaling);

template <
    QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataExp, typename TileDataMax,
    typename TileDataScaling>
PTO_INTERNAL void TQUANT_IMPL(
    TileDataOut& dst, TileDataSrc& src, TileDataExp* exp, TileDataMax* max, TileDataScaling* scaling)
{
    TQUANT_IMPL<quant_type, QuantScaleAlg::OCP, TileDataOut, TileDataSrc, TileDataExp, TileDataMax, TileDataScaling>(
        dst, src, exp, max, scaling);
}

template <
    QuantType quant_type, QuantScaleAlg scale_alg, typename TileDataOut, typename TileDataSrc, typename TileDataExp,
    typename TileDataMax, typename TileDataScaling>
PTO_INTERNAL void TQUANT_IMPL(
    TileDataOut& dst, TileDataSrc& src, TileDataExp* exp, TileDataMax* max, TileDataScaling* scaling)
{
    using T = typename TileDataSrc::DType;
    static_assert(quant_type == QuantType::MXFP8 || quant_type == QuantType::MXFP4_E2M1,
                  "Fix: MX quant overload supports MXFP8/MXFP4_E2M1.");
    if constexpr (quant_type == QuantType::MXFP8) {
        static_assert(
            std::is_same<T, float32_t>::value || std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value,
            "Fix: MXFP8 input has to be float32, bfloat16, or float16 (half)");
    } else {
        static_assert(std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value,
                      "Fix: MXFP4_E2M1 input has to be float16 (half) or bfloat16");
        static_assert(std::is_same<typename TileDataOut::DType, float4_e2m1x2_t>::value,
                      "Fix: MXFP4_E2M1 output has to be float4_e2m1x2_t");
    }
    // Create 1D flat views — TQuant operates on flattened buffers internally.
    constexpr int expN = TileDataExp::Rows * TileDataExp::Cols;
    FlatTile1D<TileDataExp> flatExp(1, expN);
    TRESHAPE_IMPL(flatExp, *exp);
    constexpr int maxN = TileDataMax::Rows * TileDataMax::Cols;
    FlatTile1D<TileDataMax> flatMax(1, maxN);
    TRESHAPE_IMPL(flatMax, *max);
    constexpr int scalN = TileDataScaling::Rows * TileDataScaling::Cols;
    FlatTile1D<TileDataScaling> flatScaling(1, scalN);
    TRESHAPE_IMPL(flatScaling, *scaling);
    if constexpr (exp2D) {
        // Pass exp as-is (2D) so the kernel can write at
        // `row * TileDataExp::Cols`.
        if constexpr (quant_type == QuantType::MXFP8) {
            TQuant_MXFP8_Impl<
                scale_alg, true, TileDataOut, TileDataSrc, TileDataExp, FlatTile1D<TileDataMax>,
                FlatTile1D<TileDataScaling>>(
                dst.data(), exp->data(), flatMax.data(), flatScaling.data(), src.data(), src.GetValidRow(),
                src.GetValidCol());
        } else {
            TQuant_MXFP4_E2M1_Impl<
                scale_alg, true, TileDataOut, TileDataSrc, TileDataExp, FlatTile1D<TileDataMax>,
                FlatTile1D<TileDataScaling>>(
                dst.data(), exp->data(), flatMax.data(), flatScaling.data(), src.data(), src.GetValidRow(),
                src.GetValidCol());
        }
    } else {
        constexpr int expN = TileDataExp::Rows * TileDataExp::Cols;
        FlatTile1D<TileDataExp> flatExp(1, expN);
        TRESHAPE_IMPL(flatExp, *exp);
        if constexpr (quant_type == QuantType::MXFP8) {
            TQuant_MXFP8_Impl<
                scale_alg, false, TileDataOut, TileDataSrc, FlatTile1D<TileDataExp>, FlatTile1D<TileDataMax>,
                FlatTile1D<TileDataScaling>>(
                dst.data(), flatExp.data(), flatMax.data(), flatScaling.data(), src.data(), src.GetValidRow(),
                src.GetValidCol());
        } else {
            TQuant_MXFP4_E2M1_Impl<
                scale_alg, false, TileDataOut, TileDataSrc, FlatTile1D<TileDataExp>, FlatTile1D<TileDataMax>,
                FlatTile1D<TileDataScaling>>(
                dst.data(), flatExp.data(), flatMax.data(), flatScaling.data(), src.data(), src.GetValidRow(),
                src.GetValidCol());
        }
        TRESHAPE_IMPL(*exp, flatExp);
    }
}

// Generic grp_axis + mx_alg dispatch (5-tile). grp_axis=0
// (DN) calls the *_Impl_DN pipeline, grp_axis=1 (ND) calls
// the scale-alg ND pipeline. The DN caller reshapes the
// exponent via TMOV(e8DnTile, exp) after TQUANT. All
// validation lives here (impl layer), not in the
// pto_instr.hpp wrapper. The single MxQuantAlg tag is
// decoded inline (no helper functions) to a (QuantType,
// QuantScaleAlg) pair.
template <
    int grp_axis, MxQuantAlg mx_alg, typename TileDataOut, typename TileDataSrc, typename TileDataExp,
    typename TileDataMax, typename TileDataScaling>
PTO_INTERNAL void TQUANT_IMPL(
    TileDataOut& dst, TileDataSrc& src, TileDataExp* exp, TileDataMax* max, TileDataScaling* scaling)
{
    using T = typename TileDataSrc::DType;
    using OutT = typename TileDataOut::DType;
    constexpr bool isFp8 = (mx_alg == MxQuantAlg::OcpMxFp8E4M3 || mx_alg == MxQuantAlg::NvMxFp8E4M3);
    constexpr bool isNv = (mx_alg == MxQuantAlg::NvMxFp8E4M3 || mx_alg == MxQuantAlg::NvMxFp4E2M1);
    constexpr QuantType quant_type = isFp8 ? QuantType::MXFP8 : QuantType::MXFP4_E2M1;
    constexpr QuantScaleAlg scale_alg = isNv ? QuantScaleAlg::NV : QuantScaleAlg::OCP;
    CheckTQuantMxTypes<quant_type, T, OutT>();
    if constexpr (grp_axis == 0) {
        if constexpr (quant_type == QuantType::MXFP8) {
            TQuant_MXFP8_Impl_DN<scale_alg, TileDataOut, TileDataSrc, TileDataExp, TileDataMax, TileDataScaling>(
                dst.data(), exp->data(), max->data(), scaling->data(), src.data(), src.GetValidRow(),
                src.GetValidCol());
        } else {
            TQuant_MXFP4_E2M1_Impl_DN<scale_alg, TileDataOut, TileDataSrc, TileDataExp, TileDataMax, TileDataScaling>(
                dst.data(), exp->data(), max->data(), scaling->data(), src.data(), src.GetValidRow(),
                src.GetValidCol());
        }
    } else {
        static_assert(grp_axis == 1, "Fix: grp_axis must be 0 (DN) or 1 (ND).");
        TQUANT_IMPL<quant_type, scale_alg, TileDataOut, TileDataSrc, TileDataExp, TileDataMax, TileDataScaling>(
            dst, src, exp, max, scaling);
    }
    // Reshape exp back to user's original tile shape. Max and scaling are scratch buffers.
    TRESHAPE_IMPL(*exp, flatExp);
}
} // namespace pto
#endif // TQUANT_HPP
