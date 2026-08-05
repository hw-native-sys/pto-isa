/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef INT64_COMMON_HPP
#define INT64_COMMON_HPP

#include <pto/npu/a5/common.hpp>

namespace pto {

enum class Int64Op { Add, Sub, Mul, Shl, Shr, Max, Min };

PTO_INTERNAL void Int64AddRegs(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& mask)
{
    MaskReg carry, carryOut;
    vaddc(carry, dstLow, lhsLow, rhsLow, mask);
    vaddcs(carryOut, dstHigh, lhsHigh, rhsHigh, carry, mask);
}

PTO_INTERNAL void Int64SubRegs(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& mask)
{
    MaskReg carry, carryOut;
    vsubc(carry, dstLow, lhsLow, rhsLow, mask);
    vsubcs(carryOut, dstHigh, lhsHigh, rhsHigh, carry, mask);
}

PTO_INTERNAL void Int64MulRegs(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& mask)
{
    vmull((vector_u32&)dstLow, (vector_u32&)dstHigh, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
    vmula(dstHigh, lhsLow, rhsHigh, mask, MODE_ZEROING);
    vmula(dstHigh, lhsHigh, rhsLow, mask, MODE_ZEROING);
}

PTO_INTERNAL void Int64SelectRegs(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow,
    vector_s32& rhsHigh, MaskReg& mask)
{
    vsel(dstLow, lhsLow, rhsLow, mask);
    vsel(dstHigh, lhsHigh, rhsHigh, mask);
}

PTO_INTERNAL void Int64CompareEqRegs(
    MaskReg& dst, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow, vector_s32& rhsHigh, MaskReg& mask)
{
    MaskReg lowEq, highEq;
    vcmp_eq(lowEq, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
    vcmp_eq(highEq, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);
    pand(dst, lowEq, highEq, mask);
}

PTO_INTERNAL void Int64CompareGeURegs(
    MaskReg& dst, vector_s32& lhsLow, vector_s32& lhsHigh, vector_s32& rhsLow, vector_s32& rhsHigh, MaskReg& mask)
{
    MaskReg highEq, lowGe, highGe;
    vcmp_eq(highEq, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);
    vcmp_ge(lowGe, (vector_u32&)lhsLow, (vector_u32&)rhsLow, mask);
    vcmp_ge(highGe, (vector_u32&)lhsHigh, (vector_u32&)rhsHigh, mask);
    psel(dst, lowGe, highGe, highEq);
}

PTO_INTERNAL void Int64AbsRegs(
    vector_s32& dstLow, vector_s32& dstHigh, vector_s32& srcLow, vector_s32& srcHigh, MaskReg& mask)
{
    vector_s32 zero, negLow, negHigh;
    vbr(zero, 0);
    MaskReg negative, carry, carryOut;
    vcmp_lt(negative, srcHigh, zero, mask);
    vsubc(carry, negLow, zero, srcLow, negative);
    vsubcs(carryOut, negHigh, zero, srcHigh, carry, negative);
    vsel(dstLow, negLow, srcLow, negative);
    vsel(dstHigh, negHigh, srcHigh, negative);
}

PTO_INTERNAL void Int64NotRegs(vector_s32& low, vector_s32& high, MaskReg& mask)
{
    vnot((vector_u32&)low, (vector_u32&)low, mask, MODE_ZEROING);
    vnot((vector_u32&)high, (vector_u32&)high, mask, MODE_ZEROING);
}

PTO_INTERNAL void Int64ToFloat(vector_f32& dst, vector_s32& srcLow, vector_s32& srcHigh)
{
    vector_s64 even, odd;
    vector_f32 evenFloat, oddFloat, dummy;
    vintlv((vector_s32&)even, (vector_s32&)odd, srcLow, srcHigh);
    MaskReg allMask = pset_b32(PAT_ALL);
    vcvt(evenFloat, even, allMask, ROUND_R, PART_EVEN);
    vcvt(oddFloat, odd, allMask, ROUND_R, PART_EVEN);
    vdintlv(dst, dummy, evenFloat, oddFloat);
}

PTO_INTERNAL void FloatToInt64(vector_s32& dstLow, vector_s32& dstHigh, vector_f32& src)
{
    vector_f32 evenFloat, oddFloat;
    vector_s64 even, odd;
    vintlv(evenFloat, oddFloat, src, src);
    MaskReg allMask = pset_b32(PAT_ALL);
    vcvt(even, evenFloat, allMask, ROUND_Z, RS_DISABLE, PART_EVEN);
    vcvt(odd, oddFloat, allMask, ROUND_Z, RS_DISABLE, PART_EVEN);
    vdintlv(dstLow, dstHigh, (vector_s32&)even, (vector_s32&)odd);
}

PTO_INTERNAL void Int64DivFloatPreprocess(vector_u32& reciprocalBits, vector_f32& divisorFloat, MaskReg& mask)
{
    vector_f32 one;
    vbr(one, 1.0f);
    vdiv(one, one, divisorFloat, mask, MODE_ZEROING);
    vadds(reciprocalBits, (vector_u32&)one, 0x1ffffffeU, mask);
}

PTO_INTERNAL void Int64DuplicateRegs(vector_s32& low, vector_s32& high, uint32_t lowScalar, uint32_t highScalar)
{
    vbr((vector_u32&)low, lowScalar);
    vbr((vector_u32&)high, highScalar);
}

PTO_INTERNAL void Int64DivSign(
    MaskReg& sameSign, vector_s32& lhsHigh, vector_s32& rhsHigh, vector_s32& zero, MaskReg& mask)
{
    MaskReg lhsNonNegative, rhsNonNegative;
    vcmp_ge(lhsNonNegative, lhsHigh, zero, mask);
    vcmp_ge(rhsNonNegative, rhsHigh, zero, mask);
    pxor(sameSign, lhsNonNegative, rhsNonNegative, mask);
    pnot(sameSign, sameSign, mask);
}

} // namespace pto

#endif
