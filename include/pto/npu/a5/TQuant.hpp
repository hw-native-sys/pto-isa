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

// Assumption: input total size is NOT a multiple of 256 elements
template <typename T>
PTO_INTERNAL void AbsReduceMax_b16_ND(__ubuf__ T *srcPtr, __ubuf__ T *maxPtr, unsigned vl_count,
                                      unsigned total_elem_count)
{
    vector_bf16 vb16_in_1, vb16_in_2, vb16_max_1;
    vector_align ureg_max;
    constexpr uint32_t grp_size = 32;
    constexpr uint32_t elements_per_vl = REPEAT_BYTE / sizeof(T);
    constexpr uint32_t elements_per_dintlv = 2 * elements_per_vl; // 256 bf16 per DINTLV load
    constexpr uint32_t blks_per_vl = REPEAT_BYTE / BLOCK_SIZE;
    uint16_t loop_num = CeilDivision(vl_count, 2);
    for (uint16_t i = 0; i < (uint16_t)loop_num; ++i) {
        // DINTLV_B16 deinterleaves 256 elements into even/odd registers.
        // Predicates must reflect per-register valid count (half of remaining),
        // NOT sequential 128-subtraction which assumes linear VL consumption.
        uint32_t offset = i * elements_per_dintlv;
        uint32_t remaining = (total_elem_count > offset) ? (total_elem_count - offset) : 0;
        if (remaining > elements_per_dintlv)
            remaining = elements_per_dintlv;
        uint32_t even_count = (remaining + 1) / 2;
        uint32_t odd_count = remaining / 2;
        MaskReg preg_vl0 = CreatePredicate<T>(even_count);
        MaskReg preg_vl1 = CreatePredicate<T>(odd_count);
        vlds(vb16_in_1, vb16_in_2, srcPtr, offset, DINTLV_B16); // loads 2 VLs (256 bf16 elements)
        vabs((vector_f16 &)vb16_in_1, (vector_f16 &)vb16_in_1, preg_vl0);
        vabs((vector_f16 &)vb16_in_2, (vector_f16 &)vb16_in_2, preg_vl1);
        vmax(vb16_in_1, vb16_in_1, vb16_in_2, preg_vl0);
        vcgmax((vector_f16 &)vb16_max_1, (vector_f16 &)vb16_in_1, preg_vl0); // 8 group maxes per 2 VLs
        // Use vstus (alignment-safe store) instead of vsts to avoid 32-byte alignment issues
        // when i*8 elements (i*16 bytes for bf16) is not a multiple of 32 bytes.
        vstus(ureg_max, blks_per_vl, vb16_max_1, maxPtr + i * 8);
    }
    vstas(ureg_max, maxPtr, 0);
}

// Assumption: input total size is a multiple of 256 elements
template <typename T>
PTO_INTERNAL void AbsReduceMax_b16_ND_opt(__ubuf__ T *srcPtr, __ubuf__ T *maxPtr, unsigned vl_count,
                                          unsigned total_elem_count)
{
    vector_bf16 vb16_in_1, vb16_in_2, vb16_max_1;
    vector_align ureg_max;
    uint32_t total_count = total_elem_count;
    constexpr uint32_t grp_size = 32;
    constexpr uint32_t elements_per_vl = REPEAT_BYTE / sizeof(T);
    uint16_t loop_num = CeilDivision(vl_count, 2);
    uint32_t num_st = CeilDivision(total_count, grp_size);
    MaskReg preg_lower8 = pset_b16(PAT_VL8);
    static constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    for (uint16_t i = 0; i < (uint16_t)loop_num; ++i) { // 32 VLs per outer loop
        MaskReg preg_vl0 = CreatePredicate<T>(total_count);
        MaskReg preg_vl1 = CreatePredicate<T>(total_count);
        vlds(vb16_in_1, vb16_in_2, srcPtr, 2 * i * elements_per_vl, DINTLV_B16); // loads 2 VLs (256 bf16 elements)
        vabs((vector_f16 &)vb16_in_1, (vector_f16 &)vb16_in_1, preg_vl0);
        vabs((vector_f16 &)vb16_in_2, (vector_f16 &)vb16_in_2, preg_vl1);
        vmax(vb16_in_1, vb16_in_1, vb16_in_2, preg_vl0);
        vcgmax((vector_f16 &)vb16_max_1, (vector_f16 &)vb16_in_1, preg_vl0); // 8 group maxes per 2 VLs
        vsts(vb16_max_1, maxPtr, i * 8, distValue, preg_lower8);
    }
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

// Computing scalar focus and exponent for B16 (BF16/FP16) -> b8 e4m3 quantization.
// Compile-time constants are selected based on the source data type T (bfloat16_t or half).
//   BF16: shr=7,  exp_mask=0x7F80, nan_check=0xFF,  exp_max=0xFE, subnorm=0x7F80, clamp=-127
//   FP16: shr=10, exp_mask=0x7C00, nan_check=0x1F,  exp_max=0x1E, subnorm=0x7C00, clamp=-15
// E8M0 uses bias 127. For BF16 (bias 127) emax_e8m0=8. For FP16 (bias 15) we subtract
// the bias difference: emax_e8m0 = 8 - (127 - 15) = -104, so E8M0 = biased_fp16 + 104.
// Scaling base: scaling = (exp_max + 8) - exponent, independent of E8M0 bias correction.
template <typename T>
PTO_INTERNAL void ExtractB8ExponentAndScaling(__ubuf__ T *maxPtr, __ubuf__ uint8_t *expPtr, __ubuf__ T *scalingPtr,
                                              unsigned exp_max_loop_count, unsigned total_elements_count)
{
    static_assert(std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value,
                  "ExtractB8ExponentAndScaling B16: T must be bfloat16_t or half");
    static constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    // Compile-time format-specific constants
    constexpr bool is_bf16 = std::is_same<T, bfloat16_t>::value;
    constexpr int shr = is_bf16 ? 7 : 10;                       // mantissa bits
    constexpr int16_t exp_mask_val = is_bf16 ? 0x7F80 : 0x7C00; // exponent field mask
    constexpr int16_t nan_check = is_bf16 ? 0xFF : 0x1F;        // all-ones exponent (NaN/Inf)
    constexpr int16_t exp_max_val = is_bf16 ? 0xFE : 0x1E;      // max non-Inf biased exponent
    constexpr int16_t subnorm_val = is_bf16 ? 0x7F80 : 0x7C00;  // +Inf sentinel for clamping
    constexpr int16_t clamp_val = is_bf16 ? -127 : -15;         // negative bias (clamping threshold)

    constexpr int16_t emax_e8m0 = is_bf16 ? 8 : (8 - 112);
    constexpr int16_t scaling_base = exp_max_val + 8;
    RegTensor<T> vb16_max;
    vector_s16 vb16_exponent, vb16_shared_exp, vb16_scaling, vb16_nan, vb16_subnorm;
    vector_s16 vb16_b8_shared_exp, vb16_b8_nan, vb16_b8_emax, vb16_exp_mask, vb16_scaling_base;
    vbr(vb16_exp_mask, exp_mask_val);
    vbr(vb16_b8_nan, 0xFF);
    vbr(vb16_subnorm, subnorm_val);
    vbr(vb16_scaling_base, scaling_base);
    vbr(vb16_exponent, exp_mask_val);
    vbr(vb16_b8_emax, emax_e8m0);
    vector_bool preg_inf;
    constexpr uint32_t elementsPerVL = REPEAT_BYTE / sizeof(T);
    uint32_t total_count = total_elements_count;
    for (uint16_t i = 0; i < (uint16_t)exp_max_loop_count; ++i) {
        vector_bool preg_b16 = CreatePredicate<T>(total_count);
        vlds(vb16_max, maxPtr, i * elementsPerVL, NORM);
        // Getting biased exponent
        vand((vector_s16 &)vb16_exponent, (vector_s16 &)vb16_max, vb16_exp_mask, preg_b16, MODE_ZEROING);
        vshrs((vector_s16 &)vb16_exponent, (vector_s16 &)vb16_exponent, shr, preg_b16, MODE_ZEROING);
        // E8M0: shared_exp = exponent - emax_e8m0 (bias-corrected for FP16)
        vsub((vector_s16 &)vb16_shared_exp, (vector_s16 &)vb16_exponent, (vector_s16 &)vb16_b8_emax, preg_b16);
        // Scaling: scaling = (exp_max + 8) - exponent (always uses raw emax=8, unaffected by E8M0 bias)
        vsub((vector_s16 &)vb16_scaling, (vector_s16 &)vb16_scaling_base, (vector_s16 &)vb16_exponent, preg_b16);
        vshls((vector_s16 &)vb16_scaling, (vector_s16 &)vb16_scaling, shr, preg_b16, MODE_ZEROING);
        // Handling special cases for NaN and Inf
        vcmps_ne(preg_inf, (vector_s16 &)vb16_exponent, nan_check, preg_b16);
        vsel(vb16_scaling, vb16_scaling, vb16_b8_nan, preg_inf);
        vsel(vb16_shared_exp, vb16_shared_exp, vb16_b8_nan, preg_inf);
        vcmps_ge(preg_inf, (vector_s16 &)vb16_scaling, clamp_val, preg_b16);
        vsel(vb16_scaling, vb16_scaling, vb16_subnorm, preg_inf);
        vsel(vb16_shared_exp, vb16_shared_exp, vb16_subnorm, preg_inf);

        vsts((vector_s16 &)vb16_shared_exp, ((__ubuf__ int16_t *)expPtr), i * elementsPerVL / sizeof(T), PK_B16,
             preg_b16);
        vsts((vector_s16 &)vb16_scaling, ((__ubuf__ int16_t *)scalingPtr), i * elementsPerVL, distValue, preg_b16);
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

// B16 (BF16/FP16) -> FP8 (No direct b16->e4m3 support; convert up to fp32 then down to fp8)
// Uses RegTensor<T> to dispatch the correct vector type (vector_bf16 or vector_f16).
template <typename T>
PTO_INTERNAL void CalcQuantizedFP8Values(__ubuf__ T *srcPtr, __ubuf__ T *scalingPtr, __ubuf__ uint8_t *dstPtr,
                                         unsigned total_elements_count)
{
    static_assert(std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value,
                  "CalcQuantizedFP8Values B16: T must be bfloat16_t or half");
    RegTensor<T> vb16_scaling, vb16_in_1, vb16_in_2, vb16_out_1, vb16_out_2;
    vector_f32 vb32_cvt_1, vb32_cvt_2, vb32_cvt_3, vb32_cvt_4;
    vector_f8e4m3 vb8_or1, vb8_or2, vb8_out, vb8_p0, vb8_p1, vb8_p2, vb8_p3;
    constexpr uint32_t elementsPerVL_b16 = REPEAT_BYTE / sizeof(T);
    constexpr uint32_t elementsPerVL_b8 = REPEAT_BYTE / sizeof(uint8_t);
    constexpr uint32_t elementsPerDintlv = 2 * elementsPerVL_b16; // 256 bf16 per DINTLV load
    uint32_t vl_count = CeilDivision(total_elements_count, elementsPerVL_b16);
    for (uint16_t i = 0; i < (uint16_t)vl_count / 2; ++i) {
        // DINTLV_B16 deinterleaves 256 elements into even/odd registers.
        // Predicates must reflect per-register valid count (half of remaining),
        // NOT sequential 128-subtraction which assumes linear VL consumption.
        uint32_t offset_b16 = i * elementsPerDintlv;
        uint32_t remaining = (total_elements_count > offset_b16) ? (total_elements_count - offset_b16) : 0;
        if (remaining > elementsPerDintlv)
            remaining = elementsPerDintlv;
        uint32_t even_count = (remaining + 1) / 2;
        uint32_t odd_count = remaining / 2;
        MaskReg preg_b16_1 = CreatePredicate<T>(even_count);
        MaskReg preg_b16_2 = CreatePredicate<T>(odd_count);
        uint32_t remaining_b8 = remaining; // 1 bf16 → 1 FP8 byte
        MaskReg preg_b8 = CreatePredicate<uint8_t>(remaining_b8);
        vlds((vector_u16 &)vb16_scaling, (__ubuf__ uint16_t *)scalingPtr, 8 * i, E2B_B16);
        vlds(vb16_in_1, vb16_in_2, srcPtr, offset_b16, DINTLV_B16);
        vmul(vb16_out_1, vb16_in_1, vb16_scaling, preg_b16_1, MODE_ZEROING);
        vmul(vb16_out_2, vb16_in_2, vb16_scaling, preg_b16_2, MODE_ZEROING);
        // b16->fp32: EVEN/ODD split each 128-element b16 register into 2x64 fp32
        vcvt(vb32_cvt_1, vb16_out_1, preg_b16_1, PART_EVEN); // indices mod 4 = 0: [b0, b4, b8, ...]
        vcvt(vb32_cvt_2, vb16_out_1, preg_b16_1, PART_ODD);  // indices mod 4 = 2: [b2, b6, b10, ...]
        vcvt(vb32_cvt_3, vb16_out_2, preg_b16_2, PART_EVEN); // indices mod 4 = 1: [b1, b5, b9, ...]
        vcvt(vb32_cvt_4, vb16_out_2, preg_b16_2, PART_ODD);  // indices mod 4 = 3: [b3, b7, b11, ...]
        // fp32 -> fp8: P0-P3 place fp8 bytes at byte 0-3 of each 32-bit slot
        // Must map: P0=mod0, P1=mod1, P2=mod2, P3=mod3 for correct sequential output
        vcvt(vb8_p0, vb32_cvt_1, preg_b16_1, ROUND_R, RS_ENABLE, PART_P0); // mod 0 → byte 0
        vcvt(vb8_p1, vb32_cvt_3, preg_b16_2, ROUND_R, RS_ENABLE, PART_P1); // mod 1 → byte 1
        vcvt(vb8_p2, vb32_cvt_2, preg_b16_1, ROUND_R, RS_ENABLE, PART_P2); // mod 2 → byte 2
        vcvt(vb8_p3, vb32_cvt_4, preg_b16_2, ROUND_R, RS_ENABLE, PART_P3); // mod 3 → byte 3

        vor(vb8_or1, vb8_p0, vb8_p1, preg_b8);
        vor(vb8_or2, vb8_p2, vb8_p3, preg_b8);
        vor(vb8_out, vb8_or1, vb8_or2, preg_b8);
        vsts((vector_u8 &)vb8_out, (__ubuf__ uint8_t *)dstPtr, i * elementsPerVL_b8, NORM_B8, preg_b8);
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
    mem_bar(VST_VLD);
    maxPtr = maxPtr_backup;
    constexpr bool unroll = (StaticRows * StaticCols > 1024) && (StaticRows * StaticCols % 256 == 0);
    ExtractB8ExponentAndScaling<unroll>(maxPtr, expPtr, scalingPtr, exp_loop_count, numGroups, elementsPerRepeat);
    mem_bar(VST_VLD);
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
    // AbsReduceMax functions use vector_bf16 intrinsics; cast pointers for fp16 compatibility
    // (both types are 16-bit; abs/max operate on raw bit patterns identically for exponent extraction)
    __ubuf__ bfloat16_t *srcPtr_b16 = (__ubuf__ bfloat16_t *)srcPtr;
    __ubuf__ bfloat16_t *maxPtr_b16 = (__ubuf__ bfloat16_t *)maxPtr;
    if (total_elements_count % 2048 == 0)
        AbsReduceMax_b16_ND_largesizes(srcPtr_b16, maxPtr_b16, vl_count, total_elements_count);
    else if (total_elements_count % 256 == 0)
        AbsReduceMax_b16_ND_opt(srcPtr_b16, maxPtr_b16, vl_count, total_elements_count);
    else
        AbsReduceMax_b16_ND(srcPtr_b16, maxPtr_b16, vl_count, total_elements_count);
    mem_bar(VST_VLD);
    maxPtr = maxPtr_backup;
    ExtractB8ExponentAndScaling(maxPtr, expPtr, scalingPtr, exp_loop_count, numGroups);
    mem_bar(VST_VLD);
    CalcQuantizedFP8Values(srcPtr, scalingPtr, dstPtr, total_elements_count);
}

// Zero-pad columns [validCols, StaticCols) in each row of a 16-bit source tile.
// Uses full-VL vlds → vsel → vsts at VL-aligned offsets so that every UB
// store is 256-byte-aligned.  Sub-VL stores (vstus/vstas/predicated vsts at
// non-VL-aligned offsets) are unreliable on some hardware revisions.
// Requires StaticCols to evenly divide elements-per-VL.
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

    RegTensor<T> vreg_data;
    RegTensor<T> vreg_zero;
    vdup(vreg_zero, (T)0, pg_all, MODE_ZEROING);

    uint32_t totalElems = (uint32_t)(validRows * StaticCols);
    uint16_t vlCount = CeilDivision(totalElems, (unsigned)elemPerVL);

    for (uint16_t vi = 0; vi < vlCount; ++vi) {
        vlds(vreg_data, srcPtr, vi * elemPerVL, NORM);
        vsel((vector_s16 &)vreg_data, (vector_s16 &)vreg_data, (vector_s16 &)vreg_zero, preg_valid);
        vsts(vreg_data, srcPtr, vi * elemPerVL, NORM_B16, pg_all);
    }
    mem_bar(VST_VLD);
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
    mem_bar(VST_VLD);
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
                // Round at fp32 precision to avoid double-rounding:
                // fp32 -> s32 (round once) -> fp32 -> fp16 -> s8 (all exact for small ints)
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
                // Round at fp32 precision to avoid double-rounding:
                // fp32 -> s32 (round once) -> fp32 -> fp16 -> u8 (all exact for small ints)
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