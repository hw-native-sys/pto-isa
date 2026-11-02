/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TDIV754_HPP
#define TDIV754_HPP
#include "pto/common/arch_cce_intrinsic.hpp"

namespace pto {
template <typename T, typename U>
PTO_INTERNAL void DivDiffCompensationFloatImpl(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& mask)
{
    constexpr uint32_t infNanBound = 0xff800000u;
    constexpr uint32_t signBit = 0x80000000u;
    constexpr uint32_t exponentMask = 0x7f800000u;
    constexpr int32_t exponentBias = 127;
    constexpr int32_t precisionThreshold = -64;

    RegTensor<T> negZero;
    RegTensor<T> rawDst;
    RegTensor<T> r, z, y;
    RegTensor<T> rPre, rNext, zPre, zNext;
    RegTensor<uint32_t> absResultBits;
    RegTensor<uint32_t> exponentMaskReg;
    RegTensor<uint32_t> src0ExponentBits;
    RegTensor<int32_t> src0Exponent;
    RegTensor<int32_t> exponentBiasReg;
    RegTensor<int32_t> scaleExponent;
    RegTensor<int32_t> thresholdReg;
    RegTensor<int32_t> zeroReg;
    RegTensor<int32_t> scaleBiasedExponent;
    RegTensor<uint32_t> scaleBits;
    RegTensor<T> scale;
    RegTensor<T> scaleOne;
    RegTensor<T> scaledSrc0;
    RegTensor<T> scaledSrc1;

    MaskReg zeroResultMask;
    MaskReg specialResultMask;
    MaskReg needScaleMask;
    MaskReg compareMask;
    MaskReg allMask = pset_b8(PAT_ALL);

    // Keep the hardware result for Zero/Inf/NaN lanes and only compensate finite non-zero results.
    vdup((vector_u32&)negZero, signBit, allMask, MODE_ZEROING);
    vdiv(z, srcReg0, srcReg1, mask, MODE_ZEROING);
    rawDst = z;
    vor(absResultBits, (vector_u32&)z, (vector_u32&)negZero, mask, MODE_ZEROING);
    vcmps_eq(zeroResultMask, z, 0.0f, mask);
    vcmps_ge(specialResultMask, absResultBits, infNanBound, mask);
    por(specialResultMask, specialResultMask, zeroResultMask, mask);

    // Scale very small dividends before residual calculation. Scaling both operands preserves the quotient.
    vdup(exponentMaskReg, exponentMask, mask, MODE_ZEROING);
    vand(src0ExponentBits, (RegTensor<uint32_t>&)srcReg0, exponentMaskReg, mask, MODE_ZEROING);
    vshrs(src0ExponentBits, src0ExponentBits, (int16_t)23, mask, MODE_ZEROING);
    vdup(exponentBiasReg, exponentBias, mask, MODE_ZEROING);
    vsub(src0Exponent, (RegTensor<int32_t>&)src0ExponentBits, exponentBiasReg, mask, MODE_ZEROING);
    vcmps_lt(needScaleMask, src0Exponent, precisionThreshold, mask);

    vdup(thresholdReg, precisionThreshold, mask, MODE_ZEROING);
    vdup(zeroReg, 0, mask, MODE_ZEROING);
    vsub(scaleExponent, thresholdReg, src0Exponent, mask, MODE_ZEROING);
    vmax(scaleExponent, scaleExponent, zeroReg, mask, MODE_ZEROING);
    vadds(scaleBiasedExponent, scaleExponent, exponentBias, needScaleMask, MODE_ZEROING);
    pto_vshls(scaleBits, (RegTensor<uint32_t>&)scaleBiasedExponent, (int32_t)23, needScaleMask);
    vdup(scaleOne, 1.0f, mask, MODE_ZEROING);
    vsel(scale, (RegTensor<T>&)scaleBits, scaleOne, needScaleMask);
    vmul(scaledSrc0, srcReg0, scale, mask, MODE_ZEROING);
    vmul(scaledSrc1, srcReg1, scale, mask, MODE_ZEROING);

    // Difference compensation: choose the quotient with the smallest residual from q and q +/- 1 ulp.
    vmuls(y, scaledSrc1, -1.0f, mask, MODE_ZEROING);
    r = scaledSrc0;
    vmula(r, z, y, mask, MODE_ZEROING);
    vadds((vector_s32&)zPre, (vector_s32&)z, -1, mask, MODE_ZEROING);
    vadds((vector_s32&)zNext, (vector_s32&)z, 1, mask, MODE_ZEROING);

    rPre = scaledSrc0;
    rNext = scaledSrc0;
    vmula(rPre, zPre, y, mask, MODE_ZEROING);
    vmula(rNext, zNext, y, mask, MODE_ZEROING);

    vabs(r, r, mask, MODE_ZEROING);
    vabs(rPre, rPre, mask, MODE_ZEROING);
    vabs(rNext, rNext, mask, MODE_ZEROING);
    vcmp_lt(compareMask, r, rPre, mask);
    vsel(r, r, rPre, compareMask);
    vsel(z, z, zPre, compareMask);
    vcmp_lt(compareMask, rNext, r, mask);
    vsel(z, zNext, z, compareMask);

    vsel(dstReg, rawDst, z, specialResultMask);
}

template <typename T, typename U>
PTO_INTERNAL void DivPrecisionImpl(U& dstReg, U& srcReg0, U& srcReg1, MaskReg& activeMask, MaskReg& correctMask)
{
    RegTensor<T> rawDst;
    RegTensor<T> r, z, y;
    RegTensor<T> rPre, rNext, zPre, zNext;

    MaskReg cmpMaskReg;

    // VDIV runs on every active lane so Zero/Inf keep the hardware division result.
    vdiv(z, srcReg0, srcReg1, activeMask, MODE_ZEROING);
    rawDst = z;

    // Apply +/-1 ulp correction only to lanes selected by the caller. NaN is canonicalized later.
    vmuls(y, srcReg1, -1.0f, correctMask, MODE_ZEROING);
    r = srcReg0;
    vmula(r, z, y, correctMask, MODE_ZEROING);

    vadds((vector_s32&)zPre, (vector_s32&)z, -1, correctMask, MODE_ZEROING);
    vadds((vector_s32&)zNext, (vector_s32&)z, 1, correctMask, MODE_ZEROING);

    rPre = srcReg0;
    rNext = srcReg0;
    vmula(rPre, zPre, y, correctMask, MODE_ZEROING);
    vmula(rNext, zNext, y, correctMask, MODE_ZEROING);

    vabs(r, r, correctMask, MODE_ZEROING);
    vabs(rPre, rPre, correctMask, MODE_ZEROING);
    vabs(rNext, rNext, correctMask, MODE_ZEROING);
    vcmp_lt(cmpMaskReg, r, rPre, correctMask);
    vsel(r, r, rPre, cmpMaskReg);
    vsel(z, z, zPre, cmpMaskReg);

    vcmp_lt(cmpMaskReg, rNext, r, correctMask);
    vsel(z, zNext, z, cmpMaskReg);

    // Correctable lanes use the compensated quotient; other active lanes keep raw VDIV.
    vsel(dstReg, z, rawDst, correctMask);
}

template <typename T, typename U>
PTO_INTERNAL void DivIEEE754FloatImpl(
    RegTensor<float>& dst, RegTensor<float>& src0, RegTensor<float>& src1, MaskReg& mask)
{
    // Bit masks for extracting IEEE 754 components from float32
    constexpr uint32_t exponentExtractor = 0x807FFFFF;  // Mask bits [30:23] - 8-bit exponent
    constexpr uint32_t signExtractor = 0x80000000;      // Mask bit 31 - sign bit
    constexpr uint32_t exponentNormalizer = 0x3F800000; // 1.0f reference (bias=127)
    constexpr uint32_t F32_INF = 0x7f800000;            // +Infinity: sign=0, exp=0xFF, mant=0

    FloatUnion subnormalThreshold;
    subnormalThreshold.i = 0x007FFFFF; // Threshold for subnormal (denormal) detection: 2^23 - 1

    FloatUnion nan;
    nan.i = 0x7fc00000; // NaN: sign=0, exp=0xFF, mant!=0

    FloatUnion min_denormal;
    min_denormal.i = 0x1; // Minimum denormal value detection (smallest positive float32 = 2^-149)

    // Scaling factor for denormal normalization: 2^23 shifts denormals into normal range.
    FloatUnion normalizeScaleEnlarge;
    normalizeScaleEnlarge.i = 0x4B000000; // 2^23

    RegTensor<float> maxSubnormal;
    RegTensor<uint32_t> tmp0;
    RegTensor<int32_t> tmp1;
    RegTensor<uint32_t> tmp2;

    RegTensor<float> src0Abs;
    RegTensor<float> src0Subnormal;
    RegTensor<float> src0Norm;
    RegTensor<float> src0All;
    RegTensor<float> src0AbsNorm;

    RegTensor<float> src1Abs;
    RegTensor<float> src1Subnormal;
    RegTensor<float> src1Norm;
    RegTensor<float> src1All;
    RegTensor<float> src1AbsNorm;

    MaskReg mask0;
    MaskReg maskSrc0Subnormal;
    MaskReg maskSrc1Subnormal;
    MaskReg maskTmp;
    MaskReg maskNan;      // divisor or dividend 0
    MaskReg maskInf;      // divisor or dividend inf
    MaskReg maskSrc0Zero; // dividend 0
    MaskReg maskSrc1Zero; // divisor 0
    MaskReg maskValid;
    MaskReg maskNorm;

    RegTensor<uint32_t> src0Exponent;
    RegTensor<uint32_t> src1Exponent;

    RegTensor<float> z1;
    RegTensor<float> z2;
    RegTensor<int32_t> normalizeAdjust;
    RegTensor<int32_t> scale;
    RegTensor<uint32_t> dstSign;

    // ========== Implementation: SIMD-optimized IEEE 754 float32 division ==========
    // subnormal threshold
    vdup(maxSubnormal, subnormalThreshold.f, mask, MODE_ZEROING);
    // Extract absolute values of operands
    vabs(src0Abs, src0, mask, MODE_ZEROING); // Absolute value of dividend
    vabs(src1Abs, src1, mask, MODE_ZEROING); // Absolute value of divisor

    // ========== Detect Infinity Values ==========
    // Create mask for positions where dividend is ±Infinity
    vdup(tmp0, F32_INF, mask, MODE_ZEROING);
    vcmp_eq(maskInf, (RegTensor<uint32_t>&)src0Abs, tmp0, mask);
    // Create mask for positions where divisor is ±Infinity
    vcmp_eq(maskTmp, (RegTensor<uint32_t>&)src1Abs, tmp0, mask);
    // Combine: valid computation only where neither is Infinity
    por(maskValid, maskInf, maskTmp, mask);

    // ========== Detect Zero Values ==========
    // Create mask for positions where dividend is zero
    vdup(tmp0, 0, mask, MODE_ZEROING);
    vcmp_eq(maskSrc0Zero, (RegTensor<uint32_t>&)src0Abs, tmp0, mask);
    // Merge zero checks into invalid mask
    por(maskValid, maskValid, maskSrc0Zero, mask);
    vcmp_eq(maskSrc1Zero, (RegTensor<uint32_t>&)src1Abs, tmp0, mask);
    // Merge zero divisor into invalid mask
    por(maskValid, maskValid, maskSrc1Zero, mask);
    pnot(maskValid, maskValid, mask);

    // ========== Normalize Subnormal Numbers (Denormals) ==========
    // Encode the post-division compensation in a vector so the subnormal predicates
    // do not have to stay live across DivPrecisionImpl.
    vdup(normalizeAdjust, 0, mask, MODE_ZEROING);

    // get positions of subnormal numbers in dividend
    vcmp_eq(maskSrc0Subnormal, src0Abs, maxSubnormal, mask);
    // Scale subnormals up to normal range (multiply by 2^23)
    vmuls(src0Subnormal, src0, normalizeScaleEnlarge.f, maskSrc0Subnormal, MODE_ZEROING);
    vdup(tmp1, -23, maskSrc0Subnormal, MODE_ZEROING);
    vsel(normalizeAdjust, tmp1, normalizeAdjust, maskSrc0Subnormal);

    // Detect subnormal elements in divisor
    vcmp_lt(maskSrc1Subnormal, src1Abs, maxSubnormal, mask);
    vmuls(src1Subnormal, src1, normalizeScaleEnlarge.f, maskSrc1Subnormal, MODE_ZEROING);
    vadds(tmp1, normalizeAdjust, 23, maskSrc1Subnormal, MODE_ZEROING);
    vsel(normalizeAdjust, tmp1, normalizeAdjust, maskSrc1Subnormal);

    // Merge normalized subnormals with normal values. Both subnormal predicates die here.
    vsel(src0All, src0Subnormal, src0, maskSrc0Subnormal);
    vsel(src1All, src1Subnormal, src1, maskSrc1Subnormal);

    // ========== Standardize Exponent Bits ==========
    // zero out the exponent bits 00000000
    vdup(tmp0, exponentExtractor, mask, MODE_ZEROING);
    vand((RegTensor<uint32_t>&)src0Norm, (RegTensor<uint32_t>&)src0All, tmp0, maskValid);
    vand((RegTensor<uint32_t>&)src1Norm, (RegTensor<uint32_t>&)src1All, tmp0, maskValid);

    // Set exponent bits to biased 127 (01111111) - standard normalized form
    vdup(tmp0, exponentNormalizer, mask, MODE_ZEROING);
    vadd((RegTensor<uint32_t>&)src0Norm, (RegTensor<uint32_t>&)src0Norm, tmp0, maskValid);
    vadd((RegTensor<uint32_t>&)src1Norm, (RegTensor<uint32_t>&)src1Norm, tmp0, maskValid);
    // Merge back with mantissa-only values
    vsel(src0Norm, src0Norm, src0All, maskValid);
    vsel(src1Norm, src1Norm, src1All, maskValid);

    // mask controls the tile range; maskValid controls which lanes receive precision correction.
    // NaN lanes are canonicalized at the end of this function, so their temporary value is discarded.
    DivPrecisionImpl<float, U>(dst, src0Norm, src1Norm, mask, maskValid);

    // Compare normalized operands after the precision division to avoid keeping maskNorm alive across the call.
    vabs(src0AbsNorm, src0Norm, maskValid);
    vabs(src1AbsNorm, src1Norm, maskValid);
    vcmp_le(maskNorm, src0AbsNorm, src1AbsNorm, maskValid);

    // Apply 2^normalizeAdjust. The subnormal predicates are no longer needed here.
    vadds(tmp1, normalizeAdjust, 127, mask, MODE_ZEROING);
    pto_vshls(tmp1, tmp1, (int32_t)23, mask);
    vmul(dst, dst, (RegTensor<float>&)tmp1, mask, MODE_ZEROING);

    // preserve sign for error handling section below
    vdup(tmp0, signExtractor, mask, MODE_ZEROING);
    vand((RegTensor<uint32_t>&)dstSign, (RegTensor<uint32_t>&)dst, tmp0, mask);

    // ===========================================================
    // exponent operation
    // ===========================================================
    // extract the exponent section 0 11..11 00..00
    vdup(tmp0, F32_INF, mask, MODE_ZEROING);
    vand(src0Exponent, (RegTensor<uint32_t>&)src0All, tmp0, mask);
    vand(src1Exponent, (RegTensor<uint32_t>&)src1All, tmp0, mask);

    // exponent subtraction (effectively fp number division)
    vshrs(src0Exponent, src0Exponent, (int16_t)23, mask);
    vshrs(src1Exponent, src1Exponent, (int16_t)23, mask);
    vsub(scale, (RegTensor<int32_t>&)src0Exponent, (RegTensor<int32_t>&)src1Exponent, mask);
    vadds(scale, scale, 127, mask);
    // ===========================================================
    // exception handling
    // ===========================================================
    // overflow (exponent over 255) underflow (exponent under 0) detection // FP32:1S + 8E + 23M
    vdup(tmp1, -23, mask, MODE_ZEROING);
    // True if underflow/overflow
    vcmp_eq(mask0, scale, (RegTensor<int32_t>&)tmp1, mask);
    pand(mask0, mask0, maskValid, mask);
    vdup(tmp0, min_denormal.i, mask0, MODE_ZEROING);
    vadd((RegTensor<uint32_t>&)z1, (RegTensor<uint32_t>&)dstSign, tmp0, mask0);
    vdup(tmp2, static_cast<uint32_t>(0), mask0, MODE_ZEROING);
    vadd((RegTensor<uint32_t>&)z2, (RegTensor<uint32_t>&)dstSign, tmp2, mask0);
    vsel(z1, z2, z1, maskNorm);
    vsel(dst, z1, dst, mask0);
    pnot(mask0, mask0, mask);
    pand(maskValid, mask0, maskValid, mask);
    vcmp_lt(mask0, scale, (RegTensor<int32_t>&)tmp1, mask);
    // set overflown/underflown result to infinity
    pand(mask0, mask0, maskValid, mask);
    vdup(tmp0, 0, mask, MODE_ZEROING); // set to 0
    vadd((RegTensor<uint32_t>&)z1, (RegTensor<uint32_t>&)dstSign, tmp0, mask0);
    vsel(dst, z1, dst, mask0);
    pnot(mask0, mask0, mask);
    pand(maskValid, mask0, maskValid, mask);

    vdup(tmp0, 255, mask, MODE_ZEROING);
    vcmp_eq(mask0, scale, (RegTensor<int32_t>&)tmp0, mask);
    pand(mask0, mask0, maskValid, mask);
    vdup(tmp1, 1, mask0, MODE_ZEROING);
    vsub(tmp1, scale, tmp1, mask0);
    vsel(scale, tmp1, scale, mask0);
    vmuls(z1, dst, 2, mask0, MODE_ZEROING);
    vsel(dst, z1, dst, mask0);

    vcmp_gt(mask0, scale, (RegTensor<int32_t>&)tmp0, mask);
    pand(mask0, mask0, maskValid, mask);
    vdup(tmp0, F32_INF, mask, MODE_ZEROING); // set to infinity
    vadd((RegTensor<uint32_t>&)z1, (RegTensor<uint32_t>&)dstSign, tmp0, mask0);
    vsel(dst, z1, dst, mask0);
    pnot(mask0, mask0, mask);
    pand(maskValid, mask0, maskValid, mask);

    vdup(tmp0, 0, maskValid, MODE_ZEROING);
    vcmp_gt(mask0, scale, (RegTensor<int32_t>&)tmp0, maskValid);
    pto_vshls(tmp1, scale, (int32_t)23, mask0);
    vmul(z1, dst, (RegTensor<float>&)tmp1, mask0);
    vsel(dst, z1, dst, mask0);

    pnot(mask0, mask0, maskValid);
    vdup(tmp0, 4194304, mask0, MODE_ZEROING); // set 0x0040 0000
    vabs(scale, scale, mask0);
    pto_vshr(scale, (RegTensor<int32_t>&)tmp0, scale, mask0);
    vmul(z1, dst, (RegTensor<float>&)scale, mask0);
    vsel(dst, z1, dst, mask0);

    // get the position of nan
    vdup(tmp0, nan.i, mask, MODE_ZEROING);
    vcmp_ne(maskNan, src0Abs, src0Abs, mask);
    vcmp_ne(maskTmp, src1Abs, src1Abs, mask);
    por(maskNan, maskNan, maskTmp, mask);
    // set output with nan input to nan
    vsel(dst, (RegTensor<float>&)tmp0, dst, maskNan);
}

template <typename T, typename U>
PTO_INTERNAL void DivIEEE754HalfImpl(RegTensor<half>& dst, RegTensor<half>& src0, RegTensor<half>& src1, MaskReg& mask)
{
    constexpr uint16_t exponentExtractor = 0x83FF;
    constexpr uint16_t signExtractor = 0x8000;
    constexpr uint16_t exponentNormalizer = 0x3C00;
    constexpr uint16_t F16_INF = 0x7C00;

    HalfUnion subnormalThreshold;
    subnormalThreshold.i = 0x03FF;

    HalfUnion nan;
    nan.i = 0x7E00;
    HalfUnion min_denormal;
    min_denormal.i = 0x1;

    HalfUnion normalizeScaleEnlarge;
    normalizeScaleEnlarge.i = 0x6400; // 2^10

    RegTensor<half> maxSubnormal;
    RegTensor<uint16_t> tmp0;
    RegTensor<int16_t> tmp1;
    RegTensor<uint16_t> tmp2;

    RegTensor<half> src0Abs;
    RegTensor<half> src0Subnormal;
    RegTensor<half> src0Norm;
    RegTensor<half> src0All;
    RegTensor<half> src0AbsNorm;

    RegTensor<half> src1Abs;
    RegTensor<half> src1Subnormal;
    RegTensor<half> src1Norm;
    RegTensor<half> src1All;
    RegTensor<half> src1AbsNorm;

    MaskReg mask0;
    MaskReg maskSrc0Subnormal;
    MaskReg maskSrc1Subnormal;
    MaskReg maskTmp;
    MaskReg maskNan;      // divisor or dividend 0
    MaskReg maskInf;      // divisor or dividend inf
    MaskReg maskSrc0Zero; // dividend 0
    MaskReg maskSrc1Zero; // divisor 0
    MaskReg maskValid;
    MaskReg maskNorm;

    RegTensor<uint16_t> src0Exponent;
    RegTensor<uint16_t> src1Exponent;

    RegTensor<half> z1;
    RegTensor<half> z2;
    RegTensor<int16_t> normalizeAdjust;
    RegTensor<int16_t> scale;
    RegTensor<uint16_t> dstSign;

    // subnormal threshold
    vdup(maxSubnormal, subnormalThreshold.f, mask, MODE_ZEROING);

    // ===========================================================
    // acquiring valid numbers (no inf, no 0)
    // ===========================================================
    vabs(src0Abs, src0, mask);
    vabs(src1Abs, src1, mask);

    // get positions of inf values
    vdup(tmp0, F16_INF, mask, MODE_ZEROING);
    vcmp_eq(maskInf, (RegTensor<uint16_t>&)src0Abs, tmp0, mask);
    vcmp_eq(maskTmp, (RegTensor<uint16_t>&)src1Abs, tmp0, mask);
    por(maskValid, maskInf, maskTmp, mask);
    // get positions of 0 divisor or dividend
    vdup(tmp0, 0, mask, MODE_ZEROING);
    vcmp_eq(maskSrc0Zero, (RegTensor<uint16_t>&)src0Abs, tmp0, mask);
    // merge for positions of invalid numbers
    por(maskValid, maskValid, maskSrc0Zero, mask);
    vcmp_eq(maskSrc1Zero, (RegTensor<uint16_t>&)src1Abs, tmp0, mask);
    // negating for positions of valid numbers
    por(maskValid, maskValid, maskSrc1Zero, mask);
    pnot(maskValid, maskValid, mask);

    // Encode the post-division compensation in a vector so the subnormal predicates
    // do not have to stay live across vdiv.
    vdup(normalizeAdjust, 0, mask, MODE_ZEROING);

    // normalize subnormal elements of src0
    // get positions of subnormal numbers in dividend
    vcmp_lt(maskSrc0Subnormal, src0Abs, maxSubnormal, mask);
    // normalizatoin
    vmuls(src0Subnormal, src0, normalizeScaleEnlarge.f, maskSrc0Subnormal, MODE_ZEROING);
    vdup(tmp1, -10, maskSrc0Subnormal, MODE_ZEROING);
    vsel(normalizeAdjust, tmp1, normalizeAdjust, maskSrc0Subnormal);

    // normalize subnormal elements of src1
    vcmp_lt(maskSrc1Subnormal, src1Abs, maxSubnormal, mask);
    vmuls(src1Subnormal, src1, normalizeScaleEnlarge.f, maskSrc1Subnormal, MODE_ZEROING);
    vadds(tmp1, normalizeAdjust, 10, maskSrc1Subnormal, MODE_ZEROING);
    vsel(normalizeAdjust, tmp1, normalizeAdjust, maskSrc1Subnormal);

    // Merge normalized subnormals with normal values. Both subnormal predicates die here.
    vsel(src0All, src0Subnormal, src0, maskSrc0Subnormal);
    vsel(src1All, src1Subnormal, src1, maskSrc1Subnormal);

    // standardized the exponent bits of src0 vand src1
    // zero out the exponent bits 00000000
    vdup(tmp0, exponentExtractor, mask, MODE_ZEROING);
    vand((RegTensor<uint16_t>&)src0Norm, (RegTensor<uint16_t>&)src0All, tmp0, maskValid);
    vand((RegTensor<uint16_t>&)src1Norm, (RegTensor<uint16_t>&)src1All, tmp0, maskValid);
    // set the exponent bits to 01111111
    vdup(tmp0, exponentNormalizer, mask, MODE_ZEROING);
    vadd((RegTensor<uint16_t>&)src0Norm, (RegTensor<uint16_t>&)src0Norm, tmp0, maskValid);
    vadd((RegTensor<uint16_t>&)src1Norm, (RegTensor<uint16_t>&)src1Norm, tmp0, maskValid);
    vsel(src0Norm, src0Norm, src0All, maskValid);
    vsel(src1Norm, src1Norm, src1All, maskValid);

    vdiv(dst, src0Norm, src1Norm, mask, MODE_ZEROING);

    // Compare normalized operands after vdiv to avoid keeping maskNorm live across the call.
    vabs(src0AbsNorm, src0Norm, maskValid);
    vabs(src1AbsNorm, src1Norm, maskValid);
    vcmp_le(maskNorm, src0AbsNorm, src1AbsNorm, maskValid);

    // Apply 2^normalizeAdjust. The subnormal predicates are no longer needed here.
    vadds(tmp1, normalizeAdjust, 15, mask, MODE_ZEROING);
    pto_vshls(tmp1, tmp1, (int16_t)10, mask);
    vmul(dst, dst, (RegTensor<half>&)tmp1, mask, MODE_ZEROING);

    // preserve sign for error handling section below
    vdup(tmp0, signExtractor, mask, MODE_ZEROING);
    vand((RegTensor<uint16_t>&)dstSign, (RegTensor<uint16_t>&)dst, tmp0, mask);

    // ===========================================================
    // exponent operation
    // ===========================================================
    // extract the exponent section 0 11..11 00..00
    vdup(tmp0, F16_INF, mask, MODE_ZEROING);
    vand(src0Exponent, (RegTensor<uint16_t>&)src0All, tmp0, mask);
    vand(src1Exponent, (RegTensor<uint16_t>&)src1All, tmp0, mask);

    // exponent subtraction (effectively fp number division)
    vshrs(src0Exponent, src0Exponent, (int16_t)10, mask);
    vshrs(src1Exponent, src1Exponent, (int16_t)10, mask);
    vsub(scale, (RegTensor<int16_t>&)src0Exponent, (RegTensor<int16_t>&)src1Exponent, mask);
    vadds(scale, scale, 15, mask);
    // ===========================================================
    // exception handling
    // ===========================================================
    // overflow (exponent over 31) underflow (exponent under -9) detection // FP16:1S + 5E + 9M
    vdup(tmp1, -9, mask, MODE_ZEROING);
    vcmp_eq(mask0, scale, (RegTensor<int16_t>&)tmp1, mask);
    pand(mask0, mask0, maskValid, mask);
    vdup(tmp0, min_denormal.i, mask0, MODE_ZEROING);
    vadd((RegTensor<uint16_t>&)z1, (RegTensor<uint16_t>&)dstSign, tmp0, mask0);
    vdup(tmp2, static_cast<uint16_t>(0), mask0, MODE_ZEROING);
    vadd((RegTensor<uint16_t>&)z2, (RegTensor<uint16_t>&)dstSign, tmp2, mask0);
    vsel(z1, z2, z1, maskNorm);
    vsel(dst, z1, dst, mask0);
    pnot(mask0, mask0, mask);
    pand(maskValid, mask0, maskValid, mask);
    // True if underflow/overflow
    vcmp_lt(mask0, scale, (RegTensor<int16_t>&)tmp1, mask);
    // set overflown/underflown result to infinity
    pand(mask0, mask0, maskValid, mask);
    vdup(tmp0, 0, mask, MODE_ZEROING); // set to 0
    vadd((RegTensor<uint16_t>&)z1, (RegTensor<uint16_t>&)dstSign, tmp0, mask0);
    vsel(dst, z1, dst, mask0);
    pnot(mask0, mask0, mask);
    pand(maskValid, mask0, maskValid, mask);

    vdup(tmp0, 31, mask, MODE_ZEROING);
    vcmp_eq(mask0, scale, (RegTensor<int16_t>&)tmp0, mask);
    pand(mask0, mask0, maskValid, mask);
    vdup(tmp1, 1, mask0, MODE_ZEROING);
    vsub(tmp1, scale, tmp1, mask0);
    vsel(scale, tmp1, scale, mask0);
    vmuls(z1, dst, 2, mask0, MODE_ZEROING);
    vsel(dst, z1, dst, mask0);

    vcmp_gt(mask0, scale, (RegTensor<int16_t>&)tmp0, mask);
    pand(mask0, mask0, maskValid, mask);
    vdup(tmp0, F16_INF, mask, MODE_ZEROING); // set to infinity
    vadd((RegTensor<uint16_t>&)z1, (RegTensor<uint16_t>&)dstSign, tmp0, mask0);
    vsel(dst, z1, dst, mask0);
    pnot(mask0, mask0, mask);
    pand(maskValid, mask0, maskValid, mask);

    vdup(tmp0, 0, maskValid, MODE_ZEROING);
    vcmp_gt(mask0, scale, (RegTensor<int16_t>&)tmp0, maskValid);
    pto_vshls(tmp1, scale, (int16_t)10, mask0);
    vmul(z1, dst, (RegTensor<half>&)tmp1, mask0);
    vsel(dst, z1, dst, mask0);

    pnot(mask0, mask0, maskValid);
    vdup(tmp0, 512, mask0, MODE_ZEROING); // set 0x0200
    vabs(scale, scale, mask0);
    pto_vshr(scale, (RegTensor<int16_t>&)tmp0, scale, mask0);
    vmul(z1, dst, (RegTensor<half>&)scale, mask0);
    vsel(dst, z1, dst, mask0);

    // get the position of nan
    vdup(tmp0, nan.i, mask, MODE_ZEROING);
    vcmp_ne(maskNan, src0Abs, src0Abs, mask);
    vcmp_ne(maskTmp, src1Abs, src1Abs, mask);
    por(maskNan, maskNan, maskTmp, mask);
    // set output with nan input to nan
    vsel(dst, (RegTensor<half>&)tmp0, dst, maskNan);
}
} // namespace pto
#endif // TINSERT_CUSTOM_HPP
