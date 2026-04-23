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

enum class QuantType
{
    MXFP8,
    INT8_SYM,
    INT8_ASYM
};

// Helper alias: creates a 1D flat tile from a 2D tile's total element count.
template <typename TileData>
using FlatTile1D = Tile<TileType::Vec, typename TileData::DType, 1, TileData::Rows * TileData::Cols, BLayout::RowMajor,
                        -1, -1, SLayout::NoneBox, 512, PadValue::Zero>;

PTO_INTERNAL void AbsReduceMax_Naive(__ubuf__ float *srcPtr, __ubuf__ float *maxPtr, unsigned total_elements_count,
                                     unsigned vl_count, unsigned elementsPerRepeat, MaskReg &preg_lower32,
                                     MaskReg &preg_upper32)
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
        vsel((vector_s32 &)vreg_b32, (vector_s32 &)vreg_b32, vreg_zero, preg);
        vcmax(vreg_max_0, vreg_b32, preg_lower32);
        vcmax(vreg_max_1, vreg_b32, preg_upper32);
        vsts(vreg_max_0, maxPtr, 2 * i, ONEPT_B32, preg);
        vsts(vreg_max_1, maxPtr + 1, 2 * i, ONEPT_B32, preg);
    }
}

// Assumption: input total size is a multiple of 256 elements
PTO_INTERNAL void AbsReduceMax_f32_opt(__ubuf__ float *srcPtr, __ubuf__ float *maxPtr, unsigned vl_count,
                                       unsigned elementsPerRepeat, unsigned total_elements_count)
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
PTO_INTERNAL void AbsReduceMax_f32_opt_largesizes(__ubuf__ float *srcPtr, __ubuf__ float *maxPtr, unsigned vl_count,
                                                  unsigned elementsPerRepeat, unsigned total_elements_count)
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
            vlds(vreg_in_3, vreg_in_4, srcPtr + 2 * elementsPerRepeat, (i * 32 + j * 4) * elementsPerRepeat,
                 DINTLV_B32);
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

// Abs-reduce-max over one 256-element DINTLV_B16 window. Loads 2 VLs,
// abs-es, pairwise-maxes, then vcgmax packs 8 group-maxes into vb16_max.
// `remaining` clamps to the number of valid source elements in this window.
template <typename T>
PTO_INTERNAL void AbsReduceMax_b16_DintlvWindow(__ubuf__ T *srcPtr, uint32_t offset, uint32_t remaining,
                                                RegTensor<T> &vb16_max)
{
    RegTensor<T> vb16_in_1, vb16_in_2;
    uint32_t even_count = (remaining + 1) / 2;
    uint32_t odd_count = remaining / 2;
    MaskReg preg_vl0 = CreatePredicate<T>(even_count);
    MaskReg preg_vl1 = CreatePredicate<T>(odd_count);
    vlds(vb16_in_1, vb16_in_2, srcPtr, offset, DINTLV_B16);
    vabs((vector_f16 &)vb16_in_1, (vector_f16 &)vb16_in_1, preg_vl0);
    vabs((vector_f16 &)vb16_in_2, (vector_f16 &)vb16_in_2, preg_vl1);
    vmax(vb16_in_1, vb16_in_1, vb16_in_2, preg_vl0);
    vcgmax((vector_f16 &)vb16_max, (vector_f16 &)vb16_in_1, preg_vl0);
}

// Generic ND path (total_elements_count not a multiple of 2048).
// See npu_skills/pto-isa/instructions/tquant-mxfp8.md for the full rationale
// on why we branch on loop_num and how the vstus/vstas continuation works.
template <typename T>
PTO_INTERNAL void AbsReduceMax_b16_ND(__ubuf__ T *srcPtr, __ubuf__ T *maxPtr, unsigned vl_count,
                                      unsigned total_elem_count)
{
    constexpr uint32_t elements_per_dintlv = 2 * REPEAT_BYTE / sizeof(T); // 256 b16 per DINTLV
    constexpr uint32_t grps_per_dintlv = elements_per_dintlv / 32;        // 8 group-maxes per iter
    constexpr uint32_t blks_per_vl = REPEAT_BYTE / BLOCK_SIZE;
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
        mem_bar(VST_VST);
    }
    vstas(ureg_max, maxPtr + loop_num * grps_per_dintlv, 0);
    mem_bar(VST_VST);
}

// Assumption: input total size is a multiple of 2K elements
// Uses 2 VLs per inner iteration (1 DINTLV + 1 vcgmax + 1 vstus) to avoid
// WAW hazard on the vstus auto-increment scalar register when using 2 vstus per iteration.
template <typename T>
PTO_INTERNAL void AbsReduceMax_b16_ND_largesizes(__ubuf__ T *srcPtr, __ubuf__ T *maxPtr, unsigned vl_count,
                                                 unsigned total_elements_count)
{
    vector_bf16 vb16_in_1, vb16_in_2, vb16_max_1;
    vector_align ureg_max;
    uint32_t total_count = total_elements_count;
    constexpr uint32_t grp_size = 32;
    constexpr uint32_t elements_per_vl = REPEAT_BYTE / sizeof(T); // 256 B / 2 B = 128 elements per VL
    constexpr uint32_t grps_per_vl = elements_per_vl / grp_size;  // 128 / 32 = 4 groups per VL
    constexpr uint32_t num_vl_per_inner_loop = 2;                 // 2 VLs per inner loop (1 DINTLV load)
    constexpr uint32_t num_vl_per_outer_loop = 32;
    constexpr uint32_t grps_per_inner_loop = num_vl_per_inner_loop * grps_per_vl; // 2 * 4 = 8 grps per inner loop
    constexpr uint32_t grps_per_outer_loop = num_vl_per_outer_loop * grps_per_vl; // 32 * 4 = 128
    constexpr uint32_t blks_per_vl = REPEAT_BYTE / BLOCK_SIZE;                    // 8 blocks per VL
    static constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    for (uint16_t i = 0; i < (uint16_t)vl_count / num_vl_per_outer_loop; ++i) {        // 32 VLs per outer loop
        for (uint16_t j = 0; j < num_vl_per_outer_loop / num_vl_per_inner_loop; ++j) { // 2 VLs per inner loop
            MaskReg preg_vl0 = CreatePredicate<T>(total_count);
            MaskReg preg_vl1 = CreatePredicate<T>(total_count);
            uint32_t offset = (i * num_vl_per_outer_loop + j * num_vl_per_inner_loop) * elements_per_vl;
            uint32_t grp_offset = grps_per_outer_loop * i + grps_per_inner_loop * j;
            vlds(vb16_in_1, vb16_in_2, srcPtr, offset, DINTLV_B16); // loads 2 VLs (256 bf16 elements)
            vabs((vector_f16 &)vb16_in_1, (vector_f16 &)vb16_in_1, preg_vl0);
            vabs((vector_f16 &)vb16_in_2, (vector_f16 &)vb16_in_2, preg_vl1);
            vmax(vb16_in_1, vb16_in_1, vb16_in_2, preg_vl0);
            vcgmax((vector_f16 &)vb16_max_1, (vector_f16 &)vb16_in_1, preg_vl0); // 8 group maxes per 2 VLs
            vstus(ureg_max, blks_per_vl, vb16_max_1, maxPtr + grp_offset);
        }
        vstas(ureg_max, maxPtr + grps_per_outer_loop * i, 0);
    }
}

// Computing scalar focus and exponent for F32 -> b8 e4m3 quantization
template <bool unroll = false>
PTO_INTERNAL void ExtractB8ExponentAndScaling(__ubuf__ float *maxPtr, __ubuf__ uint8_t *expPtr,
                                              __ubuf__ float *scalingPtr, unsigned exp_max_loop_count,
                                              unsigned total_elements_count, unsigned elementsPerRepeat)
{
    static constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<float, DistVST::DIST_NORM>())>();
    vector_f32 vb32_max;
    vector_s32 vb32_exponent, vb32_shared_exp, vb32_scaling, vb32_nan, vb32_subnorm;
    vector_s32 vb32_b8_shared_exp, vb32_b8_nan, vb32_b8_emax, vb32_exp_mask, vb32_exp_max;
    constexpr int shr = 23;
    vbr(vb32_exp_mask, 0x7F800000);
    vbr(vb32_b8_nan, 0xFF);
    vbr(vb32_subnorm, 0x7F800000);
    vbr(vb32_exp_max, 0xFE);
    vbr(vb32_exponent, 0x7F800000);
    vbr(vb32_b8_emax, 8); // Max exponent for e4m3 is 8
    vector_bool preg_inf;
    uint32_t total_count = total_elements_count;
    uint32_t scaling_elem_count = total_elements_count * 2;
    for (uint16_t i = 0; i < (uint16_t)exp_max_loop_count; ++i) {
        vector_bool preg_b32 = CreatePredicate<float>(total_count);
        vlds((vector_s32 &)vb32_max, (__ubuf__ int32_t *)maxPtr, i * elementsPerRepeat, NORM);
        vand((vector_s32 &)vb32_exponent, (vector_s32 &)vb32_max, vb32_exp_mask, preg_b32, MODE_ZEROING);
        vshrs((vector_s32 &)vb32_exponent, (vector_s32 &)vb32_exponent, shr, preg_b32, MODE_ZEROING);
        vsub((vector_u32 &)vb32_shared_exp, (vector_u32 &)vb32_exponent, (vector_u32 &)vb32_b8_emax, preg_b32);
        vsub((vector_s32 &)vb32_scaling, (vector_s32 &)vb32_exp_max, (vector_s32 &)vb32_shared_exp, preg_b32);
        vshls((vector_u32 &)vb32_scaling, (vector_u32 &)vb32_scaling, shr, preg_b32, MODE_ZEROING);

        vcmps_ne(preg_inf, (vector_s32 &)vb32_exponent, 0xFF, preg_b32);
        vsel(vb32_scaling, vb32_scaling, vb32_b8_nan, preg_inf);
        vsel(vb32_shared_exp, vb32_shared_exp, vb32_b8_nan, preg_inf);
        vcmps_ge(preg_inf, (vector_s32 &)vb32_scaling, -127, preg_b32);
        vsel(vb32_scaling, vb32_scaling, vb32_subnorm, preg_inf);
        vsel(vb32_shared_exp, vb32_shared_exp, vb32_subnorm, preg_inf);
        vsts((vector_s32 &)vb32_shared_exp, ((__ubuf__ int32_t *)expPtr), i * elementsPerRepeat / 4, PK4_B32, preg_b32);
        if constexpr (unroll) {
            vector_s32 vb32_scaling_0, vb32_scaling_1;
            vintlv(vb32_scaling_0, vb32_scaling_1, vb32_scaling, vb32_scaling);
            MaskReg preg_scaling_0 = CreatePredicate<float>(scaling_elem_count);
            MaskReg preg_scaling_1 = CreatePredicate<float>(scaling_elem_count);
            vsts((vector_s32 &)vb32_scaling_0, ((__ubuf__ int32_t *)scalingPtr), 2 * i * elementsPerRepeat, NORM_B32,
                 preg_scaling_0);
            vsts((vector_s32 &)vb32_scaling_1, ((__ubuf__ int32_t *)scalingPtr + 64), 2 * i * elementsPerRepeat,
                 NORM_B32, preg_scaling_1);

        } else
            vsts((vector_s32 &)vb32_scaling, ((__ubuf__ int32_t *)scalingPtr), i * elementsPerRepeat, distValue,
                 preg_b32);
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
PTO_INTERNAL void ExtractB8ExponentAndScaling(__ubuf__ T *maxPtr, __ubuf__ uint8_t *expPtr, __ubuf__ T *scalingPtr,
                                              unsigned exp_max_loop_count, unsigned total_elements_count)
{
    static_assert(std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value,
                  "ExtractB8ExponentAndScaling B16: T must be bfloat16_t or half");
    static constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    constexpr bool is_bf16 = std::is_same<T, bfloat16_t>::value;
    constexpr int shr = is_bf16 ? 7 : 10;
    constexpr int16_t exp_mask_val = is_bf16 ? 0x7F80 : 0x7C00;
    constexpr int16_t nan_check = is_bf16 ? 0xFF : 0x1F;
    // FP16 constants pre-shifted by -112 to fold E8M0 rebias (see function header).
    constexpr int16_t exp_max_val = is_bf16 ? 0xFE : 0x8E;
    constexpr int16_t b8_emax_val = is_bf16 ? 8 : -104;
    constexpr int16_t subnorm_val = is_bf16 ? 0x7F80 : 0x7C00;
    constexpr int16_t clamp_val = is_bf16 ? -127 : -15;
    RegTensor<T> vb16_max;
    vector_s16 vb16_exponent, vb16_shared_exp, vb16_scaling, vb16_nan, vb16_subnorm;
    vector_s16 vb16_b8_shared_exp, vb16_b8_nan, vb16_b8_emax, vb16_exp_mask, vb16_exp_max;
    vbr(vb16_exp_mask, exp_mask_val);
    vbr(vb16_b8_nan, 0xFF);
    vbr(vb16_subnorm, subnorm_val);
    vbr(vb16_exp_max, exp_max_val);
    vbr(vb16_exponent, exp_mask_val);
    vbr(vb16_b8_emax, b8_emax_val);
    vector_bool preg_inf;
    constexpr uint32_t elementsPerVL = REPEAT_BYTE / sizeof(T);
    uint32_t total_count = total_elements_count;
    for (uint16_t i = 0; i < (uint16_t)exp_max_loop_count; ++i) {
        vector_bool preg_b16 = CreatePredicate<T>(total_count);
        vlds(vb16_max, maxPtr, i * elementsPerVL, NORM);
        // biased exponent
        vand((vector_s16 &)vb16_exponent, (vector_s16 &)vb16_max, vb16_exp_mask, preg_b16, MODE_ZEROING);
        vshrs((vector_s16 &)vb16_exponent, (vector_s16 &)vb16_exponent, shr, preg_b16, MODE_ZEROING);
        vsub((vector_s16 &)vb16_shared_exp, (vector_s16 &)vb16_exponent, (vector_s16 &)vb16_b8_emax, preg_b16);
        // scaling = 1 / shared_exponent
        vsub((vector_s16 &)vb16_scaling, (vector_s16 &)vb16_exp_max, (vector_s16 &)vb16_shared_exp, preg_b16);
        vshls((vector_s16 &)vb16_scaling, (vector_s16 &)vb16_scaling, shr, preg_b16, MODE_ZEROING);
        // NaN / Inf / subnormal clamping
        vcmps_ne(preg_inf, (vector_s16 &)vb16_exponent, nan_check, preg_b16);
        vsel(vb16_scaling, vb16_scaling, vb16_b8_nan, preg_inf);
        vsel(vb16_shared_exp, vb16_shared_exp, vb16_b8_nan, preg_inf);
        vcmps_ge(preg_inf, (vector_s16 &)vb16_scaling, clamp_val, preg_b16);
        vsel(vb16_scaling, vb16_scaling, vb16_subnorm, preg_inf);
        vsel(vb16_shared_exp, vb16_shared_exp, vb16_subnorm, preg_inf);

        vsts((vector_s16 &)vb16_shared_exp, ((__ubuf__ int16_t *)expPtr), i * elementsPerVL / sizeof(T), PK_B16,
             preg_b16);
        // Board: enforce store ordering between expPtr and scalingPtr writes,
        // and between successive loop iterations, so Phase 3's E2B_B16 load
        // of scaling is guaranteed to see every per-group value.
        mem_bar(VST_VST);
        vsts((vector_s16 &)vb16_scaling, ((__ubuf__ int16_t *)scalingPtr), i * elementsPerVL, distValue, preg_b16);
        mem_bar(VST_VST);
    }
}

// FP32 -> FP8
PTO_INTERNAL void CalcQuantizedFP8Values(__ubuf__ float *srcPtr, __ubuf__ float *scalingPtr, __ubuf__ uint8_t *dstPtr,
                                         unsigned vl_count, unsigned elementsPerRepeat, unsigned total_elements_count,
                                         MaskReg &preg_lower32, MaskReg &preg_upper32)
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
        vcvt((vector_f8e4m3 &)vb8_out, (vector_f32 &)vb32_out, preg, ROUND_R, RS_ENABLE, PART_P0);
        vsts((vector_u8 &)vb8_out, (__ubuf__ uint8_t *)dstPtr, i * elementsPerRepeat, PK4_B32, preg);
    }
}

PTO_INTERNAL void CalcQuantizedFP8Values_Unroll2(__ubuf__ float *srcPtr, __ubuf__ float *scalingPtr,
                                                 __ubuf__ uint8_t *dstPtr, unsigned vl_count,
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
        vcvt((vector_f8e4m3 &)vb8_out_P0, (vector_f32 &)vb32_out_1, preg_ALL, ROUND_R, RS_ENABLE, PART_P0);
        vcvt((vector_f8e4m3 &)vb8_out_P1, (vector_f32 &)vb32_out_2, preg_ALL, ROUND_R, RS_ENABLE, PART_P1);
        vor(vb8_out, vb8_out_P0, vb8_out_P1, preg_ALL_b8);
        vsts((vector_u16 &)vb8_out, (__ubuf__ uint16_t *)dstPtr, i * elementsPerRepeat, PK_B32, preg_ALL);
    }
}

// B16 (BF16/FP16) -> FP8. No direct b16->e4m3; convert up to fp32 then down.
// RegTensor<T> dispatches the correct vector type (vector_bf16 or vector_f16).
// Quantize one 256-element DINTLV_B16 window to FP8: scale via broadcast of
// 8 per-group scaling values, upcast b16->fp32 (EVEN/ODD), downcast fp32->fp8
// (PART_P0-P3 pack mod-4 bytes), OR-combine, and store.
template <typename T>
PTO_INTERNAL void CalcQuantizedFP8Values_B16_Window(__ubuf__ T *srcPtr, __ubuf__ T *scalingPtr,
                                                    __ubuf__ uint8_t *dstPtr, uint16_t i, uint32_t offset_b16,
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
    vmul(vb16_out_1, vb16_in_1, vb16_scaling, preg_b16_1, MODE_ZEROING);
    vmul(vb16_out_2, vb16_in_2, vb16_scaling, preg_b16_2, MODE_ZEROING);
    // b16->fp32 EVEN/ODD splits each 128-lane reg into 2x64 fp32 (mod-4: 0,2,1,3).
    vcvt(vb32_cvt_1, vb16_out_1, preg_b16_1, PART_EVEN);
    vcvt(vb32_cvt_2, vb16_out_1, preg_b16_1, PART_ODD);
    vcvt(vb32_cvt_3, vb16_out_2, preg_b16_2, PART_EVEN);
    vcvt(vb32_cvt_4, vb16_out_2, preg_b16_2, PART_ODD);
    // fp32->fp8 P0..P3 writes to bytes 0..3 of each 32-bit slot; pair with mod-4 index.
    vcvt(vb8_p0, vb32_cvt_1, preg_b16_1, ROUND_R, RS_ENABLE, PART_P0);
    vcvt(vb8_p1, vb32_cvt_3, preg_b16_2, ROUND_R, RS_ENABLE, PART_P1);
    vcvt(vb8_p2, vb32_cvt_2, preg_b16_1, ROUND_R, RS_ENABLE, PART_P2);
    vcvt(vb8_p3, vb32_cvt_4, preg_b16_2, ROUND_R, RS_ENABLE, PART_P3);
    vor(vb8_or1, vb8_p0, vb8_p1, preg_b8);
    vor(vb8_or2, vb8_p2, vb8_p3, preg_b8);
    vor(vb8_out, vb8_or1, vb8_or2, preg_b8);
    vsts((vector_u8 &)vb8_out, (__ubuf__ uint8_t *)dstPtr, i * elementsPerVL_b8, NORM_B8, preg_b8);
    mem_bar(VST_VLD);
    mem_bar(VST_VST);
}

// B16 (BF16/FP16) -> FP8. 2 VLs per iter (one DINTLV_B16 load). Ceil-div on
// iter count so a partial final window is still covered; per-window predicates
// clamp to the exact remaining element count.
template <typename T>
PTO_INTERNAL void CalcQuantizedFP8Values(__ubuf__ T *srcPtr, __ubuf__ T *scalingPtr, __ubuf__ uint8_t *dstPtr,
                                         unsigned total_elements_count)
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

// FP32 -> MXFP8 quantization: AbsReduceMax + ExponentScaling + FP8 conversion.
template <unsigned StaticRows, unsigned StaticCols>
PTO_INTERNAL void TQuant_MXFP8_F32(__ubuf__ float *srcPtr, __ubuf__ uint8_t *expPtr, __ubuf__ uint8_t *dstPtr,
                                   __ubuf__ float *maxPtr, __ubuf__ float *scalingPtr, uint16_t vl_count,
                                   unsigned exp_loop_count, uint32_t numGroups, unsigned elementsPerRepeat,
                                   uint32_t total_elements_count, unsigned validRows, unsigned validCols)
{
    MaskReg preg_lower32 = pset_b32(PAT_VL32), preg_upper32, preg_ALL = pset_b32(PAT_ALL);
    pxor(preg_upper32, preg_ALL, preg_lower32, preg_ALL);
    __ubuf__ float *maxPtr_backup = maxPtr;
    if (validRows * validCols <= 1024)
        AbsReduceMax_Naive(srcPtr, maxPtr, total_elements_count, vl_count, elementsPerRepeat, preg_lower32,
                           preg_upper32);
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
            AbsReduceMax_Naive(srcPtr + aligned_total, maxPtr + aligned_groups, tail_total, tail_vl_count,
                               elementsPerRepeat, preg_lower32, preg_upper32);
        }
    }
    mem_bar(VST_VST);
    mem_bar(VST_VLD);
    mem_bar(VV_ALL);
    maxPtr = maxPtr_backup;
    constexpr bool unroll = (StaticRows * StaticCols > 1024) && (StaticRows * StaticCols % 256 == 0);
    ExtractB8ExponentAndScaling<unroll>(maxPtr, expPtr, scalingPtr, exp_loop_count, numGroups, elementsPerRepeat);
    mem_bar(VST_VST);
    mem_bar(VST_VLD);
    mem_bar(VV_ALL);
    if constexpr (unroll)
        CalcQuantizedFP8Values_Unroll2(srcPtr, scalingPtr, dstPtr, vl_count, elementsPerRepeat, total_elements_count);
    else
        CalcQuantizedFP8Values(srcPtr, scalingPtr, dstPtr, vl_count, elementsPerRepeat, total_elements_count,
                               preg_lower32, preg_upper32);
}

// B16 (BF16/FP16) -> MXFP8 quantization: AbsReduceMax + ExponentScaling + FP8 conversion.
template <typename T>
PTO_INTERNAL void TQuant_MXFP8_B16(__ubuf__ T *srcPtr, __ubuf__ uint8_t *expPtr, __ubuf__ uint8_t *dstPtr,
                                   __ubuf__ T *maxPtr, __ubuf__ T *scalingPtr, uint16_t vl_count,
                                   unsigned exp_loop_count, uint32_t numGroups, uint32_t total_elements_count)
{
    __ubuf__ T *maxPtr_backup = maxPtr;
    // AbsReduceMax operates on raw 16-bit bit patterns (abs/max are valid on
    // positive bf16/fp16 u16 encodings alike), so we cast to bf16* and share
    // the same implementation for both b16 formats.
    __ubuf__ bfloat16_t *srcPtr_b16 = (__ubuf__ bfloat16_t *)srcPtr;
    __ubuf__ bfloat16_t *maxPtr_b16 = (__ubuf__ bfloat16_t *)maxPtr;
    if (total_elements_count % 2048 == 0)
        AbsReduceMax_b16_ND_largesizes(srcPtr_b16, maxPtr_b16, vl_count, total_elements_count);
    else
        AbsReduceMax_b16_ND(srcPtr_b16, maxPtr_b16, vl_count, total_elements_count);
    // Board: add VST_VST alongside VST_VLD/VV_ALL. Sim orders stores implicitly,
    // board does not — missing VST_VST lets Phase-3 E2B_B16 read stale scaling.
    mem_bar(VST_VST);
    mem_bar(VST_VLD);
    mem_bar(VV_ALL);
    maxPtr = maxPtr_backup;
    ExtractB8ExponentAndScaling(maxPtr, expPtr, scalingPtr, exp_loop_count, numGroups);
    mem_bar(VST_VST);
    mem_bar(VST_VLD);
    mem_bar(VV_ALL);
    CalcQuantizedFP8Values(srcPtr, scalingPtr, dstPtr, total_elements_count);
}

// Zero-pad columns [validCols, StaticCols) of a 16-bit source tile at VL-aligned
// offsets (full-VL vlds -> vsel -> vsts). Sub-VL stores at non-VL-aligned offsets
// are unreliable on some hardware revisions. Requires StaticCols | elemPerVL.
// Must be called from inside a __VEC_SCOPE__.
template <typename T, unsigned StaticCols>
PTO_INTERNAL void ZeroPadColumns_VLAligned(__ubuf__ T *srcPtr, unsigned validRows, unsigned validCols)
{
    constexpr unsigned elemPerVL = REPEAT_BYTE / sizeof(T);
    static_assert(elemPerVL % StaticCols == 0, "StaticCols must evenly divide elements-per-VL for VL-aligned padding");
    constexpr unsigned rowsPerVL = elemPerVL / StaticCols;

    MaskReg pg_all = PSetTyped<T>(PAT_ALL);

    // Build a periodic predicate: bit p is set iff (p % StaticCols) < validCols.
    // Row 0 contributes positions [0, validCols).
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
        // Board: ensure successive pad stores commit in order so AbsReduceMax
        // cannot observe a partially padded tile.
        mem_bar(VST_VST);
    }
    mem_bar(VST_VST);
    mem_bar(VST_VLD);
    mem_bar(VV_ALL);
}

// Fallback zero-padding using vstus/vstas for cases where StaticCols doesn't divide VL.
// Must be called from inside a __VEC_SCOPE__.
template <typename T, unsigned StaticCols>
PTO_INTERNAL void ZeroPadColumns_Unaligned(__ubuf__ T *srcPtr, unsigned validRows, unsigned validCols)
{
    constexpr unsigned padElemPerRepeat = REPEAT_BYTE / sizeof(T);
    unsigned padCols = StaticCols - validCols;
    uint16_t padRepeatTimes = CeilDivision(padCols, padElemPerRepeat);
    RegTensor<T> vreg_zero;
    UnalignReg ureg_pad;
    MaskReg pg_all = PSetTyped<T>(PAT_ALL);
    vdup(vreg_zero, (T)0, pg_all, MODE_ZEROING);
    for (uint16_t i = 0; i < (uint16_t)(validRows); ++i) {
        uint32_t cols = (uint32_t)(padCols);
        __ubuf__ T *pdst = srcPtr + i * StaticCols + validCols;
        for (uint16_t j = 0; j < padRepeatTimes; ++j) {
            uint32_t sreg = cols > padElemPerRepeat ? padElemPerRepeat : cols;
            vstus(ureg_pad, sreg, vreg_zero, pdst, POST_UPDATE);
            cols -= padElemPerRepeat;
        }
        vstas(ureg_pad, pdst, 0, POST_UPDATE);
    }
    mem_bar(VST_VST);
    mem_bar(VST_VLD);
    mem_bar(VV_ALL);
}

// Zero-pad source tile columns for non-float types. Dispatches between VL-aligned
// (full-VL vlds/vsel/vsts) and unaligned (vstus/vstas) paths based on tile geometry.
// Must be called from inside a __VEC_SCOPE__.
template <typename T, unsigned StaticCols>
PTO_INTERNAL void ZeroPadSourceTile(__ubuf__ T *srcPtr, unsigned validRows, unsigned validCols)
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

// TQuant: FP32/BF16/FP16 -> MXFP8 (e4m3) quantization, ND mode only.
template <typename TileDataOut, typename TileDataSrc, typename TileDataExp, typename TileDataMax,
          typename TileDataScaling>
__tf__ PTO_INTERNAL void TQuant_MXFP8_Impl(typename TileDataOut::TileDType __out__ dst,
                                           typename TileDataExp::TileDType __out__ exp,
                                           typename TileDataMax::TileDType __out__ max,
                                           typename TileDataScaling::TileDType __out__ scaling,
                                           typename TileDataSrc::TileDType __in__ src, unsigned validRows,
                                           unsigned validCols)
{
    using T = typename TileDataSrc::DType;
    using ExpT = typename TileDataExp::DType;
    using OutT = typename TileDataOut::DType;
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ ExpT *expPtr = (__ubuf__ ExpT *)__cce_get_tile_ptr(exp);
    __ubuf__ OutT *dstPtr = (__ubuf__ OutT *)__cce_get_tile_ptr(dst);
    __ubuf__ T *maxPtr = (__ubuf__ T *)__cce_get_tile_ptr(max);
    __ubuf__ T *scalingPtr = (__ubuf__ T *)__cce_get_tile_ptr(scaling);

    set_ctrl(static_cast<uint64_t>(1) << 50);
    __VEC_SCOPE__
    {
        ZeroPadSourceTile<T, TileDataSrc::Cols>(srcPtr, validRows, validCols);

        constexpr unsigned elemPerVL = REPEAT_BYTE / sizeof(T);
        uint32_t totalElems = validRows * (unsigned)TileDataSrc::Cols;
        uint16_t vlCount = CeilDivision(totalElems, elemPerVL);
        uint32_t numGroups = totalElems / 32;
        unsigned expLoopCount = CeilDivision(numGroups, elemPerVL);
        if constexpr (std::is_same<T, float>::value)
            TQuant_MXFP8_F32<TileDataSrc::Rows, TileDataSrc::Cols>(
                srcPtr, (__ubuf__ uint8_t *)expPtr, (__ubuf__ uint8_t *)dstPtr, maxPtr, scalingPtr, vlCount,
                expLoopCount, numGroups, elemPerVL, totalElems, validRows, validCols);
        else
            TQuant_MXFP8_B16(srcPtr, (__ubuf__ uint8_t *)expPtr, (__ubuf__ uint8_t *)dstPtr, maxPtr, scalingPtr,
                             vlCount, expLoopCount, numGroups, totalElems);
    }
}

template <typename TileDataOut, typename TileDataSrc, typename TileDataPara>
__tf__ PTO_INTERNAL void TQuant_Int8Sym(typename TileDataOut::TileDType __out__ dst,
                                        typename TileDataSrc::TileDType __in__ src,
                                        typename TileDataPara::TileDType __in__ scale, unsigned validRows,
                                        unsigned validCols)
{
    using T = typename TileDataSrc::DType;  // fp32
    using S = typename TileDataPara::DType; // fp32
    using U = typename TileDataOut::DType;  // int8
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ U *dstPtr = (__ubuf__ U *)__cce_get_tile_ptr(dst);
    __ubuf__ S *scalePtr = (__ubuf__ S *)__cce_get_tile_ptr(scale);
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
                vlds(v_scale, scalePtr, row, BRC_B32); // broadcast row scaling
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

// TQuant: fp32 -> u8 conversion, Int8Asym
template <typename TileDataOut, typename TileDataSrc, typename TileDataPara>
__tf__ PTO_INTERNAL void TQuant_Int8Asym(typename TileDataOut::TileDType __out__ dst,
                                         typename TileDataSrc::TileDType __in__ src,
                                         typename TileDataPara::TileDType __in__ scale,
                                         typename TileDataPara::TileDType __in__ offset, unsigned validRows,
                                         unsigned validCols)
{
    using T = typename TileDataSrc::DType;  // fp32
    using U = typename TileDataOut::DType;  // uint8
    using S = typename TileDataPara::DType; // fp32
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ U *dstPtr = (__ubuf__ U *)__cce_get_tile_ptr(dst);
    __ubuf__ S *scalePtr = (__ubuf__ S *)__cce_get_tile_ptr(scale);
    __ubuf__ S *offsetPtr = (__ubuf__ S *)__cce_get_tile_ptr(offset);
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
                vlds(vb32_scale, scalePtr, row, BRC_B32);   // broadcast row scaling
                vlds(vb32_offset, offsetPtr, row, BRC_B32); // broadcast row offset
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

// TQuant Interface for FP32/FP16/BF16->INT4/8/16
template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataPara>
PTO_INTERNAL void TQUANT_IMPL(TileDataOut &dst, TileDataSrc &src, TileDataPara &scale, TileDataPara *offset = nullptr)
{
    using T = typename TileDataSrc::DType;
    static_assert(std::is_same<T, float32_t>::value, "Fix: Input has to be float 32");

    if constexpr (quant_type == QuantType::INT8_SYM) {
        using U = typename TileDataOut::DType;
        static_assert(std::is_same<U, int8_t>::value, "Fix: Quant INT8 sym: Out data type has to be int8");
        TQuant_Int8Sym<TileDataOut, TileDataSrc, TileDataPara>(dst.data(), src.data(), scale.data(), src.GetValidRow(),
                                                               src.GetValidCol());
    } else if constexpr (quant_type == QuantType::INT8_ASYM) {
        using U = typename TileDataOut::DType;
        static_assert(std::is_same<U, uint8_t>::value, "Fix: Quant INT8 asym: Out data type has to be uint8");
        TQuant_Int8Asym<TileDataOut, TileDataSrc, TileDataPara>(dst.data(), src.data(), scale.data(), offset->data(),
                                                                src.GetValidRow(), src.GetValidCol());
    }
}

// TQuant Interface for FP32/BF16/FP16->MXFP8 (ND mode)
// E8M0, max, and scaling tiles may be passed as 2D; TQuant reshapes them to 1D internally.
template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataExp, typename TileDataMax,
          typename TileDataScaling>
PTO_INTERNAL void TQUANT_IMPL(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                              TileDataScaling *scaling)
{
    using T = typename TileDataSrc::DType;
    static_assert(
        std::is_same<T, float32_t>::value || std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value,
        "Fix: Input has to be float32, bfloat16, or float16 (half)");
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
    TQuant_MXFP8_Impl<TileDataOut, TileDataSrc, FlatTile1D<TileDataExp>, FlatTile1D<TileDataMax>,
                      FlatTile1D<TileDataScaling>>(dst.data(), flatExp.data(), flatMax.data(), flatScaling.data(),
                                                   src.data(), src.GetValidRow(), src.GetValidCol());
    // Reshape exp back to user's original tile shape. Max and scaling are scratch buffers.
    TRESHAPE_IMPL(*exp, flatExp);
}
} // namespace pto
#endif // TQUANT_HPP