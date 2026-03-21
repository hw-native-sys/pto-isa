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
#include <type_traits>

namespace pto {

enum class QuantType
{
    MXFP8,
    INT8_SYM,
    INT8_ASYM
};

enum class VecStoreMode
{
    ND,
    NZ
};

PTO_INTERNAL void AbsReduceMax_Naive(__ubuf__ float *srcPtr, __ubuf__ float *maxPtr, unsigned total_elements_count,
                                     unsigned vl_count, unsigned elementsPerRepeat, MaskReg &preg_lower32,
                                     MaskReg &preg_upper32)
{
    RegTensor<float> vreg_b32;
    uint32_t elem_count = total_elements_count;
    // assumption: input total num of elements is a multiple of 64
    for (uint16_t i = 0; i < (uint16_t)vl_count; ++i) {
        MaskReg preg = CreatePredicate<float>(elem_count);
        RegTensor<float> vreg_max_0, vreg_max_1;
        vlds(vreg_b32, srcPtr, i * elementsPerRepeat, NORM);
        vabs(vreg_b32, vreg_b32, preg);
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
    vector_f32 vreg_in_1, vreg_in_2, vreg_in_3, vreg_in_4, vreg_max_0, vreg_max_1, vreg_max;
    vector_f32 vreg_dintlv_1, vreg_dintlv_2, vreg_dintlv_3, vreg_dintlv_4, vreg_gp_max;
    vector_f32 vreg_dintlv_out_1, vreg_dintlv_out_2, vreg_dintlv_out_3, vreg_dintlv_out_4;
    vector_bf16 vreg_max_bf16;
    vector_align ureg_max;
    uint32_t total_count = total_elements_count;
    MaskReg preg_lower1 = pset_b16(PAT_VL16);
    MaskReg preg_ALL_B16 = pset_b16(PAT_ALL);
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
            vlds(vreg_in_3, vreg_in_4, srcPtr + 128, (i * 32 + j * 4) * elementsPerRepeat, DINTLV_B32);
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
    for (uint16_t i = 0; i < (uint16_t)exp_max_loop_count; ++i) {
        vector_bool preg_b32 = CreatePredicate<float>(total_count);
        vlds((vector_s32 &)vb32_max, (__ubuf__ int32_t *)maxPtr, i * elementsPerRepeat, NORM);
        // Getting biased exponent
        vand((vector_s32 &)vb32_exponent, (vector_s32 &)vb32_max, vb32_exp_mask, preg_b32, MODE_ZEROING);
        vshrs((vector_s32 &)vb32_exponent, (vector_s32 &)vb32_exponent, shr, preg_b32, MODE_ZEROING);
        // Offsetting exponent to get shared exponent for fp8 e4m3
        vsub((vector_u32 &)vb32_shared_exp, (vector_u32 &)vb32_exponent, (vector_u32 &)vb32_b8_emax, preg_b32);

        // calculating scaling factor = 1/shared_exponent
        vsub((vector_s32 &)vb32_scaling, (vector_s32 &)vb32_exp_max, (vector_s32 &)vb32_shared_exp, preg_b32);
        vshls((vector_u32 &)vb32_scaling, (vector_u32 &)vb32_scaling, shr, preg_b32, MODE_ZEROING);

        // Handling special cases for NaN and Subnormal
        vcmps_ne(preg_inf, (vector_s32 &)vb32_exponent, 0xFF, preg_b32);
        vsel(vb32_scaling, vb32_scaling, vb32_b8_nan, preg_inf);
        vsel(vb32_shared_exp, vb32_shared_exp, vb32_b8_nan, preg_inf);

        // clamp to scale_bits range
        vcmps_ge(preg_inf, (vector_s32 &)vb32_scaling, -127, preg_b32);
        vsel(vb32_scaling, vb32_scaling, vb32_subnorm, preg_inf);
        vsel(vb32_shared_exp, vb32_shared_exp, vb32_subnorm, preg_inf);

        vsts((vector_s32 &)vb32_shared_exp, ((__ubuf__ int32_t *)expPtr), i * elementsPerRepeat / 4, PK4_B32, preg_b32);
        if constexpr (unroll) {
            vector_s32 vb32_scaling_0, vb32_scaling_1;
            vintlv(vb32_scaling_0, vb32_scaling_1, vb32_scaling, vb32_scaling);
            vsts((vector_s32 &)vb32_scaling_0, ((__ubuf__ int32_t *)scalingPtr), 2 * i * elementsPerRepeat, NORM_B32,
                 preg_b32);
            vsts((vector_s32 &)vb32_scaling_1, ((__ubuf__ int32_t *)scalingPtr + 64), 2 * i * elementsPerRepeat,
                 NORM_B32, preg_b32);

        } else
            vsts((vector_s32 &)vb32_scaling, ((__ubuf__ int32_t *)scalingPtr), i * elementsPerRepeat, distValue,
                 preg_b32);
    }
}

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
        vcvt((vector_f8e4m3 &)vb8_out, (vector_f32 &)vb32_out, preg_ALL, ROUND_R, RS_ENABLE, PART_P0);
        vsts((vector_u8 &)vb8_out, (__ubuf__ uint8_t *)dstPtr, i * elementsPerRepeat, PK4_B32, preg_ALL);
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

PTO_INTERNAL void ReorderB8IndicesZZ(__ubuf__ uint8_t *E8m0ZZPtr, __ubuf__ uint8_t *e8m0Ptr,
                                     __ubuf__ uint8_t *vgather_idx_ptr, unsigned b8_exp_count)
{
    vector_u16 vb16_e8m0, vb16_vgather_idx, vb16_e8m0_zz;
    __ubuf__ uint16_t *idxPtr_u16 = (__ubuf__ uint16_t *)vgather_idx_ptr;
    __ubuf__ uint16_t *expPtr_u16 = (__ubuf__ uint16_t *)e8m0Ptr;
    __ubuf__ uint16_t *zzPtr_u16 = (__ubuf__ uint16_t *)E8m0ZZPtr;
    unsigned b16_exp_count = CeilDivision(b8_exp_count, 2);
    unsigned loop_count = CeilDivision(b16_exp_count, REPEAT_BYTE / sizeof(uint16_t));
    for (uint16_t i = 0; i < (uint16_t)loop_count; ++i) {
        MaskReg preg_b16 = CreatePredicate<uint16_t>(b16_exp_count);
        vlds(vb16_vgather_idx, idxPtr_u16, 128, NORM, POST_UPDATE);
        vgather2(vb16_e8m0_zz, expPtr_u16, vb16_vgather_idx, preg_b16);
        vsts(vb16_e8m0_zz, zzPtr_u16, 128, NORM_B16, preg_b16, POST_UPDATE);
    }
}

// TQuant: fp32 -> mxed fp8(e4m3) quantization, supports ND and NZ store modes
template <VecStoreMode store_mode, typename TileDataOut, typename TileDataSrc, typename TileDataExp,
          typename TileDataMax, typename TileDataIdx>
__tf__ PTO_INTERNAL void TQuant_MXFP8(
    typename TileDataOut::TileDType __out__ dst, typename TileDataExp::TileDType __out__ exp,
    typename TileDataMax::TileDType __out__ max, typename TileDataSrc::TileDType __out__ scaling,
    typename TileDataExp::TileDType __out__ exp_zz, typename TileDataIdx::TileDType __in__ vgather_idx,
    typename TileDataSrc::TileDType __in__ src, unsigned validRows, unsigned validCols)
{
    using T = typename TileDataSrc::DType; // fp32
    using U = typename TileDataExp::DType; // f8e8m0
    using V = typename TileDataOut::DType; // f8e4m3
    using I = typename TileDataIdx::DType; // uint16
    __ubuf__ T *srcPtr = (__ubuf__ T *)__cce_get_tile_ptr(src);
    __ubuf__ U *expPtr = (__ubuf__ U *)__cce_get_tile_ptr(exp);
    __ubuf__ U *expZZPtr = (__ubuf__ U *)__cce_get_tile_ptr(exp_zz);
    __ubuf__ V *dstPtr = (__ubuf__ V *)__cce_get_tile_ptr(dst);
    __ubuf__ T *maxPtr = (__ubuf__ T *)__cce_get_tile_ptr(max);
    __ubuf__ T *maxPtr_backup = (__ubuf__ T *)__cce_get_tile_ptr(max);
    __ubuf__ T *scalingPtr = (__ubuf__ T *)__cce_get_tile_ptr(scaling);
    __ubuf__ I *gatherIdxPtr = (__ubuf__ I *)__cce_get_tile_ptr(vgather_idx);
    set_ctrl(static_cast<uint64_t>(1) << 50); // set SPR.CTRL[50] to 1, to allow data clipping into MAX_NORM
    __VEC_SCOPE__
    {
        constexpr unsigned elementsPerRepeat = REPEAT_BYTE / sizeof(T);
        unsigned numRepeatPerRow = CeilDivision(validCols, elementsPerRepeat);
        unsigned exp_max_loop_count = CeilDivision(validRows * TileDataSrc::Cols, 32 * elementsPerRepeat);
        MaskReg preg_lower32 = pset_b32(PAT_VL32);
        MaskReg preg_upper32;
        MaskReg preg_ALL = pset_b32(PAT_ALL);
        pxor(preg_upper32, preg_ALL, preg_lower32, preg_ALL);
        uint16_t vl_count = CeilDivision(validRows * TileDataSrc::Cols, elementsPerRepeat);
        uint32_t total_elements_count = validRows * TileDataSrc::Cols;
        // if total valid size less than 1024, don't unroll
        if (validRows * validCols <= 1024) {
            AbsReduceMax_Naive(srcPtr, maxPtr, total_elements_count, vl_count, elementsPerRepeat, preg_lower32,
                               preg_upper32);
        } else if ((validRows * validCols) % 2048 == 0) {
            AbsReduceMax_f32_opt_largesizes(srcPtr, maxPtr, vl_count, elementsPerRepeat, total_elements_count);
        } else
            AbsReduceMax_f32_opt(srcPtr, maxPtr, vl_count, elementsPerRepeat, total_elements_count);
        mem_bar(VST_VLD);
        maxPtr = maxPtr_backup; // reset maxPtr
        constexpr unsigned total_static_size = TileDataSrc::Rows * TileDataSrc::Cols;
        constexpr bool unroll_condition = (total_static_size > 1024) && (total_static_size % 256 == 0);
        ExtractB8ExponentAndScaling<unroll_condition>(maxPtr, expPtr, scalingPtr, exp_max_loop_count,
                                                      total_elements_count, elementsPerRepeat);
        mem_bar(VST_VLD);
        if constexpr (unroll_condition)
            CalcQuantizedFP8Values_Unroll2(srcPtr, scalingPtr, (__ubuf__ uint8_t *)dstPtr, vl_count, elementsPerRepeat,
                                           total_elements_count);
        else
            CalcQuantizedFP8Values(srcPtr, scalingPtr, (__ubuf__ uint8_t *)dstPtr, vl_count, elementsPerRepeat,
                                   total_elements_count, preg_lower32, preg_upper32);
        unsigned b8_exp_count = CeilDivision(TileDataSrc::Rows * TileDataSrc::Cols, 32);
        if constexpr (store_mode == VecStoreMode::NZ)
            ReorderB8IndicesZZ(expZZPtr, expPtr, (__ubuf__ uint8_t *)gatherIdxPtr, b8_exp_count);
    }
} // namespace pto

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

// TQuant Interface for FP32/FP16/BF16->MXFP8/4 (ND mode)
template <QuantType quant_type, typename TileDataOut, typename TileDataSrc, typename TileDataExp, typename TileDataMax>
PTO_INTERNAL void TQUANT_IMPL(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                              TileDataSrc *scaling)
{
    using T = typename TileDataSrc::DType;
    static_assert(std::is_same<T, float32_t>::value, "Fix: Input has to be float 32");

    TQuant_MXFP8<VecStoreMode::ND, TileDataOut, TileDataSrc, TileDataExp, TileDataMax, TileDataExp>(
        dst.data(), exp->data(), max->data(), scaling->data(), exp->data(), exp->data(), src.data(), src.GetValidRow(),
        src.GetValidCol());
}

// TQuant Interface for FP32/FP16/BF16->MXFP8/4 (NZ mode, with ZZ exponent reordering)
template <QuantType quant_type, VecStoreMode store_mode, typename TileDataOut, typename TileDataSrc,
          typename TileDataExp, typename TileDataMax, typename TileDataIdx>
PTO_INTERNAL void TQUANT_IMPL(TileDataOut &dst, TileDataSrc &src, TileDataExp *exp, TileDataMax *max,
                              TileDataSrc *scaling, TileDataExp *exp_zz, TileDataIdx *vgather_idx)
{
    using T = typename TileDataSrc::DType;
    static_assert(std::is_same<T, float32_t>::value, "Fix: Input has to be float 32");

    TQuant_MXFP8<store_mode, TileDataOut, TileDataSrc, TileDataExp, TileDataMax, TileDataIdx>(
        dst.data(), exp->data(), max->data(), scaling->data(), exp_zz->data(), vgather_idx->data(), src.data(),
        src.GetValidRow(), src.GetValidCol());
}
} // namespace pto
#endif // TQUANT_HPP