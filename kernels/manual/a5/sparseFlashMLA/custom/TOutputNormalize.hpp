/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SPARSE_FLASH_MLA_CUSTOM_TOUTPUTNORMALIZE_HPP_
#define SPARSE_FLASH_MLA_CUSTOM_TOUTPUTNORMALIZE_HPP_

#include "common.hpp"

#if defined(__CCE_AICORE__)
namespace flash_mla_bf16 {
namespace custom {
namespace detail {
PTO_INTERNAL void outputNormalizeImpl(
    __ubuf__ float* maximum, __ubuf__ float* sum, __ubuf__ float* sink, __ubuf__ float* lse,
    __ubuf__ float* multipliers)
{
    __VEC_SCOPE__
    {
        const auto rows = pset_b32(PAT_VL32);
        vector_f32 zero, one, positiveInfinity;
        vector_u32 positiveInfinityBits, negativeInfinityBits;
        vbr(zero, 0.0f);
        vbr(one, 1.0f);
        vbr(positiveInfinityBits, 0x7f800000u);
        vbr(negativeInfinityBits, 0xff800000u);
        vbr((vector_u32&)positiveInfinity, 0x7f800000u);
        // NORM reads 256 B; only the first 32 lanes contain live row state.
        vector_f32 m, s, safeSum, logsum, bias, difference, e, denominator, weight, multiplier;
        MaskReg nonempty;
        vlds(m, maximum, 0, NORM);
        vlds(s, sum, 0, NORM);
        vlds(bias, sink, 0, NORM);
        vcmp_gt(nonempty, s, zero, rows);
        vsel(safeSum, s, one, nonempty);
        vln(logsum, safeSum, rows, MODE_ZEROING);
        vadd(logsum, logsum, m, rows, MODE_ZEROING);
        vsel(logsum, logsum, zero, nonempty);
        // Substitute finite sink operands, then apply exact infinity weights.
        MaskReg positiveSink, negativeSink;
        vcmp_eq(positiveSink, (vector_u32&)bias, positiveInfinityBits, rows);
        vcmp_eq(negativeSink, (vector_u32&)bias, negativeInfinityBits, rows);
        vsel(bias, zero, bias, positiveSink);
        vsel(bias, zero, bias, negativeSink);
        // sigmoid(LSE-sink), using exp(-abs(x)) to avoid overflow.
        vsub(difference, logsum, bias, rows, MODE_ZEROING);
        vabs(e, difference, rows, MODE_ZEROING);
        vmuls(e, e, -1.0f, rows, MODE_ZEROING);
        vexp(e, e, rows, MODE_ZEROING);
        vadd(denominator, one, e, rows, MODE_ZEROING);
        vdiv(weight, one, denominator, rows, MODE_ZEROING);
        MaskReg negative;
        vcmp_lt(negative, difference, zero, rows);
        vmul(e, e, weight, rows, MODE_ZEROING);
        vsel(weight, e, weight, negative);
        vsel(weight, zero, weight, positiveSink);
        vsel(weight, one, weight, negativeSink);
        vdiv(multiplier, weight, safeSum, rows, MODE_ZEROING);
        vsel(multiplier, multiplier, zero, nonempty);
        vsel(logsum, logsum, positiveInfinity, nonempty);
        // The sink load's readable padding overlaps the first LSE slots.
        mem_bar(VLD_VST);
        vsts(logsum, lse, 0, NORM_B32, rows);
        vsts(multiplier, multipliers, 0, NORM_B32, rows);
        mem_bar(VST_VLD);
    }
}

} // namespace detail

// Public instruction memory operands are tiles. The caller owns UB allocation
// and cross-pipe handoffs. Each instruction issues one VF launch.
// Wait events follow PTO conventions.
template <typename... WaitEvents>
PTO_INST RecordEvent TOutputNormalize(
    RowStateTile& lse, RowStateTile& multipliers, RowStateTile& maximum, RowStateTile& sum, RowStateTile& sink,
    WaitEvents&... events)
{
    WaitAllEvents(events...);
    detail::outputNormalizeImpl(
        (__ubuf__ float*)maximum.data(), (__ubuf__ float*)sum.data(), (__ubuf__ float*)sink.data(),
        (__ubuf__ float*)lse.data(), (__ubuf__ float*)multipliers.data());
    return {};
}

} // namespace custom
} // namespace flash_mla_bf16
#endif // __CCE_AICORE__
#endif // SPARSE_FLASH_MLA_CUSTOM_TOUTPUTNORMALIZE_HPP_
