/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TDIV_INT_SOFT_HPP
#define TDIV_INT_SOFT_HPP

/**
 * @file DivIntSoft.hpp
 * @brief Soft division implementations for kirinDev0000 (__NPU_ARCH__ == 5101).
 *
 * Contains two categories of soft division:
 * 1. Integer soft division (u16/s16): kirinDev0000 hardware does not support integer
 *    division. The bisheng compiler provides soft vdiv overloads for integer types only
 *    for __NPU_ARCH__ == 5102/5161 (a5/a6 variants), guarded by macros that exclude 5101
 *    even though __DAV_L510__ is defined. The algorithms here are adapted from the
 *    bisheng compiler's __clang_cce_vector_intrinsics.h. All required building-block
 *    intrinsics (vmull via __DAV_L510__, vfcvt, pintlv_b32, vmul, vcvt, etc.) are
 *    available on 5101.
 * 2. Half-precision IEEE 754 division via float32: the native DivIEEE754HalfImpl
 *    produces 1-bit-accurate results on kirinDev0000. The wrapper here converts half
 *    operands to float32, calls the bit-exact DivIEEE754FloatImpl, then converts back.
 *    Since float32 can represent all half values losslessly, the only rounding occurs at
 *    the final float->half step, which is the correct behavior.
 */

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/a5/common.hpp>
#include "pto/npu/a5/custom/Div754.hpp"

namespace pto {

// ============================================================================
// vdiv soft implement for uint16_t
// Algorithm: convert u16 to f32, estimate 1/y via hardware f32 vdiv, scale by
// 2^16, convert back to s32, multiply by x (32-bit), deinterleave to u16
// quotient, then 2 iterations of quotient/remainder refinement.
// ============================================================================
AICORE inline void DivSoftIntImpl(vector_u16& dst, vector_u16 src0, vector_u16 src1, vector_bool mask)
{
    // x/0 = 0xFFFF
    vector_bool zero_mask, non_zero_mask, ori_mask;
    ori_mask = mask;
    vcmps_eq(zero_mask, src1, 0, mask);
    vector_u16 q_zero;
    vdup(q_zero, 0xFFFF, mask, MODE_ZEROING);
    pnot(non_zero_mask, zero_mask, mask);
    mask = non_zero_mask;

    // Since we need to do interleave, we temply use PAT_ALL.
    vector_bool pg = pset_b16(PAT_ALL);
    vector_f32 v_one_f32, v_zero_f32;
    vector_u16 v_zero_u16;
    vdup(v_one_f32, (float)1.0, pg, MODE_ZEROING);
    vdup(v_zero_u16, (uint16_t)0, pg, MODE_ZEROING);
    vdup(v_zero_f32, (float)0.0, pg, MODE_ZEROING);

    // convert u16 to f32
    vector_f32 vy_lower_f32, vy_higher_f32;
    vector_u16 vy_lower_u16, vy_higher_u16;
    vintlv(vy_lower_u16, vy_higher_u16, src1, v_zero_u16);
    vcvt(vy_lower_f32, (vector_s32&)vy_lower_u16, pg, ROUND_F);
    vcvt(vy_higher_f32, (vector_s32&)vy_higher_u16, pg, ROUND_F);

    // Initial estimate of inv(y).
    vector_f32 vy_rec_lower, vy_rec_higher;
    vdiv(vy_rec_lower, v_one_f32, vy_lower_f32, pg, MODE_ZEROING);
    vdiv(vy_rec_higher, v_one_f32, vy_higher_f32, pg, MODE_ZEROING);
    vector_f32 vy_scale_lower, vy_scale_higher;
    vector_u32 v_const;
    // Since f32 can fully cover the numerical range of u16,
    // we use 2^16 for the initial value calculation.
    vdup(v_const, 0x47800000U, pg, MODE_ZEROING);
    vmul(vy_scale_lower, vy_rec_lower, (vector_f32&)v_const, pg, MODE_ZEROING);
    vmul(vy_scale_higher, vy_rec_higher, (vector_f32&)v_const, pg, MODE_ZEROING);

    // trick: no need to do f32->u16, we can fully reuse s32 to do vmull(u16).
    vector_s32 v_lower_s32, v_higher_s32;
    vcvt(v_lower_s32, vy_scale_lower, pg, ROUND_F, RS_DISABLE);
    vcvt(v_higher_s32, vy_scale_higher, pg, ROUND_F, RS_DISABLE);

    // Quotient/remainder estimate.
    // vmull(u16)
    vector_u32 q_tmp_lower, q_tmp_higher;
    vector_u16 vx_lower_u16, vx_higher_u16;
    vector_u16 q_tmp, q_lower;
    vintlv(vx_lower_u16, vx_higher_u16, src0, v_zero_u16);
    vmul(q_tmp_lower, (vector_u32&)v_lower_s32, (vector_u32&)vx_lower_u16, pg, MODE_ZEROING);
    vmul(q_tmp_higher, (vector_u32&)v_higher_s32, (vector_u32&)vx_higher_u16, pg, MODE_ZEROING);
    vdintlv(q_lower, q_tmp, (vector_u16&)q_tmp_lower, (vector_u16&)q_tmp_higher);

    vector_u16 yq_tmp, r_tmp;
    vmul(yq_tmp, q_tmp, src1, mask, MODE_ZEROING);
    vsub(r_tmp, src0, yq_tmp, mask, MODE_ZEROING);

    // Two times of quotient/remainder refinement.
    vector_bool preg_1;
    vcmp_ge(preg_1, r_tmp, src1, mask);
    vsub(r_tmp, r_tmp, src1, preg_1, MODE_MERGING);
    vadds(q_tmp, q_tmp, 1, preg_1, MODE_MERGING);
    vcmp_ge(preg_1, r_tmp, src1, mask);
    vsub(r_tmp, r_tmp, src1, preg_1, MODE_MERGING);
    vadds(q_tmp, q_tmp, 1, preg_1, MODE_MERGING);

    vsel(q_tmp, q_zero, q_tmp, zero_mask);
    // MODE_ZEROING: dst = q_tmp
    dst = q_tmp;
}

// ============================================================================
// vdiv soft implement for int16_t
// Algorithm: strip sign, compute |x|/|y| via u16 soft div, then restore sign.
// ============================================================================
AICORE inline void DivSoftIntImpl(vector_s16& dst, vector_s16 src0, vector_s16 src1, vector_bool mask)
{
    // x/0 = -1
    vector_bool zero_mask, non_zero_mask, ori_mask, neg_x_mask;
    ori_mask = mask;
    vcmps_eq(zero_mask, src1, 0, mask);
    vector_s16 q_zero, neg_q_zero, pos_q_zero;
    vdup(q_zero, 0x7FFF, mask, MODE_ZEROING);
    vdup(neg_q_zero, 0x8000, mask, MODE_ZEROING);
    vcmps_lt(neg_x_mask, src0, 0, mask);
    vsel(q_zero, neg_q_zero, q_zero, neg_x_mask);
    pnot(non_zero_mask, zero_mask, mask);
    mask = non_zero_mask;

    // strip sign
    vector_u16 abs_x;
    vector_u16 abs_y;
    vabs((vector_s16&)abs_x, src0, mask, MODE_ZEROING);
    vabs((vector_s16&)abs_y, src1, mask, MODE_ZEROING);
    vector_s16 x_xor_y;
    vxor(x_xor_y, src0, src1, mask, MODE_ZEROING);
    vector_bool p_pos;
    vcmps_ge(p_pos, x_xor_y, 0, mask);

    // reuse vdiv u16
    vector_u16 dst_tmp;
    DivSoftIntImpl(dst_tmp, abs_x, abs_y, mask);

    // handle sign
    vector_s16 neg_q;
    vneg(neg_q, (vector_s16)dst_tmp, mask, MODE_ZEROING);
    vector_s16 q;
    vsel(q, (vector_s16)dst_tmp, neg_q, p_pos);

    vsel(q, q_zero, q, zero_mask);
    // MODE_ZEROING: dst = q
    dst = q;
}

// ============================================================================
// Half-precision IEEE 754 division via float32
// The native DivIEEE754HalfImpl produces 1-bit-accurate results on kirinDev0000.
// This wrapper converts half operands to float32, calls the bit-exact
// DivIEEE754FloatImpl, then converts back to half. Since float32 can represent
// all half values losslessly, the only rounding occurs at the final float->half
// step, which is the correct behavior.
// ============================================================================
template <typename U>
PTO_INTERNAL void DivIEEE754HalfViaFloatImpl(
    RegTensor<half>& dst, RegTensor<half>& src0, RegTensor<half>& src1, MaskReg& mask)
{
    // Convert b16 predicate to b32 predicates for even/odd float elements.
    // A half register holds 128 elements; a float register holds 64.
    // punpack with LOWER extracts even-positioned bits (for PART_EVEN),
    // HIGHER extracts odd-positioned bits (for PART_ODD).
    MaskReg mask_b32_even, mask_b32_odd;
    punpack(mask_b32_even, mask, LOWER);
    punpack(mask_b32_odd, mask, HIGHER);

    // Convert half -> float (each half reg -> two float regs: even & odd parts)
    RegTensor<float> src0_f32_even, src0_f32_odd;
    RegTensor<float> src1_f32_even, src1_f32_odd;
    vfcvt(src0_f32_even, src0, PART_EVEN);
    vfcvt(src0_f32_odd, src0, PART_ODD);
    vfcvt(src1_f32_even, src1, PART_EVEN);
    vfcvt(src1_f32_odd, src1, PART_ODD);

    // Float32 IEEE 754 high-precision division (bit-exact)
    RegTensor<float> dst_f32_even, dst_f32_odd;
    DivIEEE754FloatImpl<float, RegTensor<float> >(dst_f32_even, src0_f32_even, src1_f32_even, mask_b32_even);
    DivIEEE754FloatImpl<float, RegTensor<float> >(dst_f32_odd, src0_f32_odd, src1_f32_odd, mask_b32_odd);

    // Convert float -> half (each pair of float regs -> one half reg)
    // PART_EVEN places 64 results at even half positions (odd positions = 0)
    // PART_ODD  places 64 results at odd half positions (even positions = 0)
    // vor combines them into the final half register.
    RegTensor<half> dst_even, dst_odd;
    vcvt(dst_even, dst_f32_even, mask_b32_even, ROUND_R, RS_DISABLE, PART_EVEN);
    vcvt(dst_odd, dst_f32_odd, mask_b32_odd, ROUND_R, RS_DISABLE, PART_ODD);
    vor(dst, dst_even, dst_odd, mask);
}

} // namespace pto
#endif // TDIV_INT_SOFT_HPP
