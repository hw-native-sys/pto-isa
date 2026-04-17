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

namespace pto {
template <typename T, typename U>
PTO_INTERNAL void DivPrecisionImpl(U &dstReg, U &srcReg0, U &srcReg1, MaskReg &mask)
{
    constexpr uint32_t infNanBound = 0xff800000u;
    constexpr uint32_t signBitNum = 0x80000000u;
    RegTensor<T> regNegZero;
    RegTensor<T> tmpDst;
    RegTensor<T> r, z, y;
    RegTensor<uint32_t> infNan;

    MaskReg cmpMaskReg;
    MaskReg infNanCmp;
    MaskReg zeroCmp;
    MaskReg preg;
    preg = pset_b8(PAT_ALL);
    vdup((vector_u32 &)regNegZero, signBitNum, preg, MODE_ZEROING);
    vdiv(z, srcReg0, srcReg1, mask, MODE_ZEROING);
    vor(infNan, (vector_u32 &)z, (vector_u32 &)regNegZero, mask, MODE_ZEROING);
    tmpDst = z;
    vcmps_eq(zeroCmp, z, 0.0f, mask);
    vcmps_ge(infNanCmp, infNan, infNanBound, mask);
    por(infNanCmp, infNanCmp, zeroCmp, mask);

    vmuls(y, srcReg1, -1.0f, mask, MODE_ZEROING);
    r = srcReg0;
    vmula(r, z, y, mask, MODE_ZEROING);
    RegTensor<T> rPre, rNext, zPre, zNext;

    vadds((vector_s32 &)zPre, (vector_s32 &)z, -1, mask, MODE_ZEROING);
    vadds((vector_s32 &)zNext, (vector_s32 &)z, 1, mask, MODE_ZEROING);

    rPre = srcReg0;
    rNext = srcReg0;

    vmula(rPre, zPre, y, mask, MODE_ZEROING);
    vmula(rNext, zNext, y, mask, MODE_ZEROING);

    vabs(r, r, mask, MODE_ZEROING);
    vabs(rPre, rPre, mask, MODE_ZEROING);
    vabs(rNext, rNext, mask, MODE_ZEROING);
    vcmp_lt(cmpMaskReg, r, rPre, mask);
    vsel(r, r, rPre, cmpMaskReg);
    vsel(z, z, zPre, cmpMaskReg);

    vcmp_lt(cmpMaskReg, rNext, r, mask);
    vsel(z, zNext, z, cmpMaskReg);
    vsel(dstReg, tmpDst, z, infNanCmp);
}

template <typename T, typename U>
PTO_INTERNAL void DivIEEE754FloatImpl(RegTensor<float> &dst, RegTensor<float> &src0, RegTensor<float> &src1,
                                      MaskReg &mask)
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

    // Scaling factors for denormal normalization:
    // normalizeScaleEnlarge = 2^23: shifts denormals into normal range
    // normalizeScaleReduce = 2^-23: inverse operation for denormal result
    FloatUnion normalizeScaleEnlarge;
    normalizeScaleEnlarge.i = 0x4B000000; // 2^23
    FloatUnion normalizeScaleReduce;
    normalizeScaleReduce.i = 0x34000000; // 2^-23

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
    MaskReg maskSrc0Normal;
    MaskReg maskSrc0Subnormal;
    MaskReg maskSrc1Normal;
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
    RegTensor<int32_t> scale;
    RegTensor<uint32_t> dstExponent;
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
    vcmp_eq(maskInf, (RegTensor<uint32_t> &)src0Abs, tmp0, mask);
    // Create mask for positions where divisor is ±Infinity
    vcmp_eq(maskTmp, (RegTensor<uint32_t> &)src1Abs, tmp0, mask);
    // Combine: valid computation only where neither is Infinity
    por(maskValid, maskInf, maskTmp, mask);

    // ========== Detect Zero Values ==========
    // Create mask for positions where dividend is zero
    vdup(tmp0, 0, mask, MODE_ZEROING);
    vcmp_eq(maskSrc0Zero, (RegTensor<uint32_t> &)src0Abs, tmp0, mask);
    // Merge zero checks into invalid mask
    por(maskValid, maskValid, maskSrc0Zero, mask);
    vcmp_eq(maskSrc1Zero, (RegTensor<uint32_t> &)src1Abs, tmp0, mask);
    // Merge zero divisor into invalid mask
    por(maskValid, maskValid, maskSrc1Zero, mask);
    pnot(maskValid, maskValid, mask);

    // ========== Normalize Subnormal Numbers (Denormals) ==========
    // get positions of subnormal numbers in dividend
    vcmp_eq(maskSrc0Subnormal, src0Abs, maxSubnormal, mask);
    // Invert to get normal positions
    pnot(maskSrc0Normal, maskSrc0Subnormal, mask);
    // Scale subnormals up to normal range (multiply by 2^23)
    vmuls(src0Subnormal, src0, normalizeScaleEnlarge.f, maskSrc0Subnormal, MODE_ZEROING);

    // Detect subnormal elements in divisor
    vcmp_lt(maskSrc1Subnormal, src1Abs, maxSubnormal, mask);
    pnot(maskSrc1Normal, maskSrc1Subnormal, mask);
    vmuls(src1Subnormal, src1, normalizeScaleEnlarge.f, maskSrc1Subnormal, MODE_ZEROING);

    // Merge normalized subnormals with normal values
    vsel(src0All, src0, src0Subnormal, maskSrc0Normal);
    vsel(src1All, src1, src1Subnormal, maskSrc1Normal);

    // ========== Standardize Exponent Bits ==========
    // zero out the exponent bits 00000000
    vdup(tmp0, exponentExtractor, mask, MODE_ZEROING);
    vand((RegTensor<uint32_t> &)src0Norm, (RegTensor<uint32_t> &)src0All, tmp0, maskValid);
    vand((RegTensor<uint32_t> &)src1Norm, (RegTensor<uint32_t> &)src1All, tmp0, maskValid);

    // Set exponent bits to biased 127 (01111111) - standard normalized form
    vdup(tmp0, exponentNormalizer, mask, MODE_ZEROING);
    vadd((RegTensor<uint32_t> &)src0Norm, (RegTensor<uint32_t> &)src0Norm, tmp0, maskValid);
    vadd((RegTensor<uint32_t> &)src1Norm, (RegTensor<uint32_t> &)src1Norm, tmp0, maskValid);
    // Merge back with mantissa-only values
    vsel(src0Norm, src0Norm, src0All, maskValid);
    vsel(src1Norm, src1Norm, src1All, maskValid);
    // Extract absolute values again after exponent manipulation
    vabs(src0AbsNorm, src0Norm, maskValid);
    vabs(src1AbsNorm, src1Norm, maskValid);
    vcmp_le(maskNorm, src0AbsNorm, src1AbsNorm, maskValid);

    DivPrecisionImpl<float, U>(dst, src0Norm, src1Norm, mask);

    // subnormal dividend, normal divisor
    pand(mask0, maskSrc0Subnormal, maskSrc1Normal, mask);
    // normalization compensation
    vmuls(z1, dst, normalizeScaleReduce.f, mask0, MODE_ZEROING);
    vsel(dst, z1, dst, mask0);

    // normal dividend, subnormal divisor
    pand(mask0, maskSrc0Normal, maskSrc1Subnormal, mask);
    // normalization compensation
    vmuls(z1, dst, normalizeScaleEnlarge.f, mask0, MODE_ZEROING);
    // merge the compensated result
    vsel(dst, z1, dst, mask0);

    // preserve sign for error handling section below
    vdup(tmp0, signExtractor, mask, MODE_ZEROING);
    vand((RegTensor<uint32_t> &)dstSign, (RegTensor<uint32_t> &)dst, tmp0, mask);

    // ===========================================================
    // exponent operation
    // ===========================================================
    // extract the exponent section 0 11..11 00..00
    vdup(tmp0, F32_INF, mask, MODE_ZEROING);
    vand(src0Exponent, (RegTensor<uint32_t> &)src0All, tmp0, mask);
    vand(src1Exponent, (RegTensor<uint32_t> &)src1All, tmp0, mask);
    vand(dstExponent, (RegTensor<uint32_t> &)dst, tmp0, mask);

    // exponent subtraction (effectively fp number division)
    vshrs(src0Exponent, src0Exponent, (int16_t)23, mask);
    vshrs(src1Exponent, src1Exponent, (int16_t)23, mask);
    vshrs(dstExponent, dstExponent, (int16_t)23, mask);
    vsub(scale, (RegTensor<int32_t> &)src0Exponent, (RegTensor<int32_t> &)src1Exponent, mask);
    vadds(scale, scale, 127, mask);
    // ===========================================================
    // exception handling
    // ===========================================================
    // overflow (exponent over 255) underflow (exponent under 0) detection // FP32:1S + 8E + 23M
    vdup(tmp1, -23, mask, MODE_ZEROING);
    // True if underflow/overflow
    vcmp_eq(mask0, scale, (RegTensor<int32_t> &)tmp1, mask);
    pand(mask0, mask0, maskValid, mask);
    vdup(tmp0, min_denormal.i, mask0, MODE_ZEROING);
    vadd((RegTensor<uint32_t> &)z1, (RegTensor<uint32_t> &)dstSign, tmp0, mask0);
    vdup(tmp2, static_cast<uint32_t>(0), mask0, MODE_ZEROING);
    vadd((RegTensor<uint32_t> &)z2, (RegTensor<uint32_t> &)dstSign, tmp2, mask0);
    vsel(z1, z2, z1, maskNorm);
    vsel(dst, z1, dst, mask0);
    pnot(mask0, mask0, mask);
    pand(maskValid, mask0, maskValid, mask);
    vcmp_lt(mask0, scale, (RegTensor<int32_t> &)tmp1, mask);
    // set overflown/underflown result to infinity
    pand(mask0, mask0, maskValid, mask);
    vdup(tmp0, 0, mask, MODE_ZEROING); // set to 0
    vadd((RegTensor<uint32_t> &)z1, (RegTensor<uint32_t> &)dstSign, tmp0, mask0);
    vsel(dst, z1, dst, mask0);
    pnot(mask0, mask0, mask);
    pand(maskValid, mask0, maskValid, mask);

    vdup(tmp0, 255, mask, MODE_ZEROING);
    vcmp_eq(mask0, scale, (RegTensor<int32_t> &)tmp0, mask);
    pand(mask0, mask0, maskValid, mask);
    vdup(tmp1, 1, mask0, MODE_ZEROING);
    vsub(tmp1, scale, tmp1, mask0);
    vsel(scale, tmp1, scale, mask0);
    vmuls(z1, dst, 2, mask0, MODE_ZEROING);
    vsel(dst, z1, dst, mask0);

    vcmp_gt(mask0, scale, (RegTensor<int32_t> &)tmp0, mask);
    pand(mask0, mask0, maskValid, mask);
    vdup(tmp0, F32_INF, mask, MODE_ZEROING); // set to infinity
    vadd((RegTensor<uint32_t> &)z1, (RegTensor<uint32_t> &)dstSign, tmp0, mask0);
    vsel(dst, z1, dst, mask0);
    pnot(mask0, mask0, mask);
    pand(maskValid, mask0, maskValid, mask);

    vdup(tmp0, 0, maskValid, MODE_ZEROING);
    vcmp_gt(mask0, scale, (RegTensor<int32_t> &)tmp0, maskValid);
    vshls(tmp1, scale, (int16_t)23, mask0);
    vmul(z1, dst, (RegTensor<float> &)tmp1, mask0);
    vsel(dst, z1, dst, mask0);

    pnot(mask0, mask0, maskValid);
    vdup(tmp0, 4194304, mask0, MODE_ZEROING); // set 0x0040 0000
    vabs(scale, scale, mask0);
    vshr(scale, (RegTensor<int32_t> &)tmp0, scale, mask0);
    vmul(z1, dst, (RegTensor<float> &)scale, mask0);
    vsel(dst, z1, dst, mask0);

    // get the position of nan
    vdup(tmp0, nan.i, mask, MODE_ZEROING);
    vcmp_ne(maskNan, src0Abs, src0Abs, mask);
    vcmp_ne(maskTmp, src1Abs, src1Abs, mask);
    por(maskNan, maskNan, maskTmp, mask);
    // set output with nan input to nan
    vsel(dst, (RegTensor<float> &)tmp0, dst, maskNan);
}

template <typename T, typename U>
PTO_INTERNAL void DivIEEE754HalfImpl(RegTensor<half> &dst, RegTensor<half> &src0, RegTensor<half> &src1, MaskReg &mask)
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
    HalfUnion normalizeScaleReduce;
    normalizeScaleReduce.i = 0x1400; // 2^-10

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
    MaskReg maskSrc0Normal;
    MaskReg maskSrc0Subnormal;
    MaskReg maskSrc1Normal;
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
    RegTensor<int16_t> scale;
    RegTensor<uint16_t> dstExponent;
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
    vcmp_eq(maskInf, (RegTensor<uint16_t> &)src0Abs, tmp0, mask);
    vcmp_eq(maskTmp, (RegTensor<uint16_t> &)src1Abs, tmp0, mask);
    por(maskValid, maskInf, maskTmp, mask);
    // get positions of 0 divisor or dividend
    vdup(tmp0, 0, mask, MODE_ZEROING);
    vcmp_eq(maskSrc0Zero, (RegTensor<uint16_t> &)src0Abs, tmp0, mask);
    // merge for positions of invalid numbers
    por(maskValid, maskValid, maskSrc0Zero, mask);
    vcmp_eq(maskSrc1Zero, (RegTensor<uint16_t> &)src1Abs, tmp0, mask);
    // negating for positions of valid numbers
    por(maskValid, maskValid, maskSrc1Zero, mask);
    pnot(maskValid, maskValid, mask);

    // normalize subnormal elements of src0
    // get positions of subnormal numbers in dividend
    vcmp_lt(maskSrc0Subnormal, src0Abs, maxSubnormal, mask);
    // negating for normal positions
    pnot(maskSrc0Normal, maskSrc0Subnormal, mask);
    // normalizatoin
    vmuls(src0Subnormal, src0, normalizeScaleEnlarge.f, maskSrc0Subnormal, MODE_ZEROING);

    // normalize subnormal elements of src1
    vcmp_lt(maskSrc1Subnormal, src1Abs, maxSubnormal, mask);
    pnot(maskSrc1Normal, maskSrc1Subnormal, mask);
    vmuls(src1Subnormal, src1, normalizeScaleEnlarge.f, maskSrc1Subnormal, MODE_ZEROING);

    // merge the normalized subnormal elements with normal elements
    vsel(src0All, src0, src0Subnormal, maskSrc0Normal);
    vsel(src1All, src1, src1Subnormal, maskSrc1Normal);

    // standardized the exponent bits of src0 vand src1
    // zero out the exponent bits 00000000
    vdup(tmp0, exponentExtractor, mask, MODE_ZEROING);
    vand((RegTensor<uint16_t> &)src0Norm, (RegTensor<uint16_t> &)src0All, tmp0, maskValid);
    vand((RegTensor<uint16_t> &)src1Norm, (RegTensor<uint16_t> &)src1All, tmp0, maskValid);
    // set the exponent bits to 01111111
    vdup(tmp0, exponentNormalizer, mask, MODE_ZEROING);
    vadd((RegTensor<uint16_t> &)src0Norm, (RegTensor<uint16_t> &)src0Norm, tmp0, maskValid);
    vadd((RegTensor<uint16_t> &)src1Norm, (RegTensor<uint16_t> &)src1Norm, tmp0, maskValid);
    vsel(src0Norm, src0Norm, src0All, maskValid);
    vsel(src1Norm, src1Norm, src1All, maskValid);
    vabs(src0AbsNorm, src0Norm, maskValid);
    vabs(src1AbsNorm, src1Norm, maskValid);
    vcmp_le(maskNorm, src0AbsNorm, src1AbsNorm, maskValid);

    vdiv(dst, src0Norm, src1Norm, mask, MODE_ZEROING);

    // subnormal dividend, normal divisor
    pand(mask0, maskSrc0Subnormal, maskSrc1Normal, mask);
    // normalization compensation
    vmuls(z1, dst, normalizeScaleReduce.f, mask0, MODE_ZEROING);
    vsel(dst, z1, dst, mask0);

    // normal dividend, subnormal divisor
    pand(mask0, maskSrc0Normal, maskSrc1Subnormal, mask);
    // normalization compensation
    vmuls(z1, dst, normalizeScaleEnlarge.f, mask0);
    // merge the compensated result
    vsel(dst, z1, dst, mask0);

    // preserve sign for error handling section below
    vdup(tmp0, signExtractor, mask, MODE_ZEROING);
    vand((RegTensor<uint16_t> &)dstSign, (RegTensor<uint16_t> &)dst, tmp0, mask);

    // ===========================================================
    // exponent operation
    // ===========================================================
    // extract the exponent section 0 11..11 00..00
    vdup(tmp0, F16_INF, mask, MODE_ZEROING);
    vand(src0Exponent, (RegTensor<uint16_t> &)src0All, tmp0, mask);
    vand(src1Exponent, (RegTensor<uint16_t> &)src1All, tmp0, mask);
    vand(dstExponent, (RegTensor<uint16_t> &)dst, tmp0, mask);

    // exponent subtraction (effectively fp number division)
    vshrs(src0Exponent, src0Exponent, (int16_t)10, mask);
    vshrs(src1Exponent, src1Exponent, (int16_t)10, mask);
    vshrs(dstExponent, dstExponent, (int16_t)10, mask);
    vsub(scale, (RegTensor<int16_t> &)src0Exponent, (RegTensor<int16_t> &)src1Exponent, mask);
    vadds(scale, scale, 15, mask);
    // ===========================================================
    // exception handling
    // ===========================================================
    // overflow (exponent over 31) underflow (exponent under -9) detection // FP16:1S + 5E + 9M
    vdup(tmp1, -9, mask, MODE_ZEROING);
    vcmp_eq(mask0, scale, (RegTensor<int16_t> &)tmp1, mask);
    pand(mask0, mask0, maskValid, mask);
    vdup(tmp0, min_denormal.i, mask0, MODE_ZEROING);
    vadd((RegTensor<uint16_t> &)z1, (RegTensor<uint16_t> &)dstSign, tmp0, mask0);
    vdup(tmp2, static_cast<uint16_t>(0), mask0, MODE_ZEROING);
    vadd((RegTensor<uint16_t> &)z2, (RegTensor<uint16_t> &)dstSign, tmp2, mask0);
    vsel(z1, z2, z1, maskNorm);
    vsel(dst, z1, dst, mask0);
    pnot(mask0, mask0, mask);
    pand(maskValid, mask0, maskValid, mask);
    // True if underflow/overflow
    vcmp_lt(mask0, scale, (RegTensor<int16_t> &)tmp1, mask);
    // set overflown/underflown result to infinity
    pand(mask0, mask0, maskValid, mask);
    vdup(tmp0, 0, mask, MODE_ZEROING); // set to 0
    vadd((RegTensor<uint16_t> &)z1, (RegTensor<uint16_t> &)dstSign, tmp0, mask0);
    vsel(dst, z1, dst, mask0);
    pnot(mask0, mask0, mask);
    pand(maskValid, mask0, maskValid, mask);

    vdup(tmp0, 31, mask, MODE_ZEROING);
    vcmp_eq(mask0, scale, (RegTensor<int16_t> &)tmp0, mask);
    pand(mask0, mask0, maskValid, mask);
    vdup(tmp1, 1, mask0, MODE_ZEROING);
    vsub(tmp1, scale, tmp1, mask0);
    vsel(scale, tmp1, scale, mask0);
    vmuls(z1, dst, 2, mask0, MODE_ZEROING);
    vsel(dst, z1, dst, mask0);

    vcmp_gt(mask0, scale, (RegTensor<int16_t> &)tmp0, mask);
    pand(mask0, mask0, maskValid, mask);
    vdup(tmp0, F16_INF, mask, MODE_ZEROING); // set to infinity
    vadd((RegTensor<uint16_t> &)z1, (RegTensor<uint16_t> &)dstSign, tmp0, mask0);
    vsel(dst, z1, dst, mask0);
    pnot(mask0, mask0, mask);
    pand(maskValid, mask0, maskValid, mask);

    vdup(tmp0, 0, maskValid, MODE_ZEROING);
    vcmp_gt(mask0, scale, (RegTensor<int16_t> &)tmp0, maskValid);
    vshls(tmp1, scale, (int16_t)10, mask0);
    vmul(z1, dst, (RegTensor<half> &)tmp1, mask0);
    vsel(dst, z1, dst, mask0);

    pnot(mask0, mask0, maskValid);
    vdup(tmp0, 512, mask0, MODE_ZEROING); // set 0x0200
    vabs(scale, scale, mask0);
    vshr(scale, (RegTensor<int16_t> &)tmp0, scale, mask0);
    vmul(z1, dst, (RegTensor<half> &)scale, mask0);
    vsel(dst, z1, dst, mask0);

    // get the position of nan
    vdup(tmp0, nan.i, mask, MODE_ZEROING);
    vcmp_ne(maskNan, src0Abs, src0Abs, mask);
    vcmp_ne(maskTmp, src1Abs, src1Abs, mask);
    por(maskNan, maskNan, maskTmp, mask);
    // set output with nan input to nan
    vsel(dst, (RegTensor<half> &)tmp0, dst, maskNan);
}
} // namespace pto
#endif // TINSERT_CUSTOM_HPP
