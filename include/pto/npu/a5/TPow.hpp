/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPOW_HPP
#define TPOW_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/common/type.hpp>

#include "utils.hpp"

namespace pto {
namespace PowF {
constexpr float LOG2_LOWEST_VALUE = 1.175494351e-38f;
constexpr float LOG2_LOWEST_VALUE_MULS = 8388608.0f;
constexpr float LOG2_REDUCE_COEFF1 = 0.70710678f;
constexpr int32_t LOG2_REDUCE_COEFF2 = 0xff800000;
constexpr float LOG2_REDUCE_FMAF_COEFF1 = 1.19209290e-7f;
constexpr float LOG2_BEST_FMAF_COEFF1 = 0.129394531f;
constexpr float LOG2_BEST_FMAF_COEFF2 = 0.141957462f;
constexpr float LOG2_BEST_FMAF_COEFF3 = 0.200015724f;
constexpr float LOG2_BEST_FMAF_COEFF4 = 0.333333254f;
constexpr float LOG2_HI1 = 6.93147182e-1f;
constexpr float LOG2_HI2 = -6.93147182e-1f;
constexpr float LOG2_LO = -1.90465421e-9f;
constexpr float EXP_OVFL_UNFL_F = -104.0f;
constexpr float EXP_MIN_F = 88.7228390f;
constexpr int32_t INF = 0x7F800000;
constexpr int32_t F32_NAN = 0x7fc00000;
constexpr int32_t R10_COEFF = 0x7F800000;
constexpr int32_t R12_COEFF = 0x7FFFFFFF;
constexpr int16_t COMPARE_ZERO_OFFSET = 31;
constexpr float F32_FRACTIONS = -23.0f;

PTO_INTERNAL void IsInfNum(MaskReg &infMask, RegTensor<float> &srcReg, RegTensor<int32_t> &tmpR12Reg, MaskReg &mask)
{
    RegTensor<float> tmpFloatReg;
    vand((RegTensor<int32_t> &)tmpFloatReg, (RegTensor<int32_t> &)srcReg, tmpR12Reg, mask);
    vcmps_eq(infMask, (RegTensor<int32_t> &)tmpFloatReg, INF, mask);
}

PTO_INTERNAL void IsNanNum(MaskReg &nanMask, RegTensor<float> &srcReg, MaskReg &mask)
{
    vcmp_ne(nanMask, srcReg, srcReg, mask);
}

PTO_INTERNAL void RFloor(RegTensor<float> &dstReg, RegTensor<float> &srcReg, MaskReg &mask)
{
    vtrc(dstReg, srcReg, ROUND_F, mask, MODE_ZEROING);
}

PTO_INTERNAL void ComputeExpoOddInt(MaskReg &oddMask, RegTensor<float> &expReg, RegTensor<float> &twoReg, MaskReg &mask)
{
    // calculate exp is odd or not: expo_odd_int = fmaf (-2.0f, floorf (0.5f * b), b) == 1.0f;
    RegTensor<float> tmpFloatReg;
    vmuls(tmpFloatReg, expReg, 0.5f, mask);
    RFloor(tmpFloatReg, tmpFloatReg, mask);
    vmadd(tmpFloatReg, twoReg, expReg, mask);
    vcmps_eq(oddMask, tmpFloatReg, 1.0f, mask);
}

PTO_INTERNAL void ProcessFloatSpecialCase(RegTensor<float> &dstReg, RegTensor<float> &baseReg, RegTensor<float> &expReg,
                                          RegTensor<int32_t> &tmpR10Reg, RegTensor<int32_t> &tmpR12Reg,
                                          RegTensor<float> &twoReg, MaskReg &mask)
{
    RegTensor<float> tmpFloatReg, tmpFloatReg2;
    MaskReg cmpMask1, cmpMask2, curMask;

    // 1. 基本情况
    // if (exp == 0.0f || base == 1.0f)
    //     return 1.0f;
    vcmps_eq(cmpMask1, expReg, 0.0f, mask);
    vcmps_eq(cmpMask2, baseReg, 1.0f, mask);
    por(cmpMask2, cmpMask1, cmpMask2, mask);
    vdup(dstReg, 1.0f, cmpMask2, MODE_MERGING);

    // 2. NaN处理
    // if (isnan(base) || isnan(exp))
    //     return NAN;
    pnot(curMask, cmpMask2, mask);
    IsNanNum(cmpMask1, baseReg, mask);
    IsNanNum(cmpMask2, expReg, mask);
    por(cmpMask2, cmpMask1, cmpMask2, curMask);
    vdup((RegTensor<int32_t> &)dstReg, F32_NAN, cmpMask2, MODE_MERGING);

    // 3. 无穷大和零处理
    // if (isinf(base) || base == 0.0f) {
    //     if (exp < 0.0f)
    //         base = base ^ 0x7F800000;          // 指数为负：翻转指数位
    //     return (exp is odd) ? base : (base & 0x7FFFFFFF); // 奇数指数保留符号，否则取绝对值
    // }
    pxor(curMask, cmpMask2, curMask, mask);
    IsInfNum(cmpMask1, baseReg, tmpR12Reg, curMask);
    vcmps_eq(cmpMask2, baseReg, 0.0f, curMask);
    por(cmpMask1, cmpMask1, cmpMask2, mask);
    vcmps_lt(cmpMask2, expReg, 0.0f, cmpMask1);
    vxor((RegTensor<int32_t> &)tmpFloatReg, (RegTensor<int32_t> &)baseReg, tmpR10Reg, curMask);
    vsel(tmpFloatReg, tmpFloatReg, baseReg, cmpMask2);
    vand((RegTensor<int32_t> &)tmpFloatReg2, (RegTensor<int32_t> &)tmpFloatReg, tmpR12Reg, curMask);
    ComputeExpoOddInt(cmpMask2, expReg, twoReg, mask);
    vsel(tmpFloatReg, tmpFloatReg, tmpFloatReg2, cmpMask2);
    vsel(dstReg, tmpFloatReg, dstReg, cmpMask1);

    // 4-1. 负数底数处理
    // if (base < 0.0f) {
    //     if (exp != floor(exp))
    //         return NAN;  // 负数的非整数幂为NaN
    //     if (is_odd(exp))
    //         return -result;
    // }
    pxor(curMask, cmpMask1, curMask, mask);
    pand(cmpMask2, cmpMask2, curMask, mask);
    vneg(tmpFloatReg, dstReg, curMask);
    vsel(tmpFloatReg, tmpFloatReg, dstReg, cmpMask2);
    RFloor(tmpFloatReg2, expReg, curMask);
    vcmp_ne(cmpMask1, expReg, tmpFloatReg2, curMask);
    vdup((RegTensor<int32_t> &)tmpFloatReg, F32_NAN, cmpMask1, MODE_MERGING);
    vcmps_lt(cmpMask2, baseReg, 0.0f, curMask);
    vsel(dstReg, tmpFloatReg, dstReg, cmpMask2);

    // 4-2. 特殊组合处理
    // if (base == -1.0f && isinf(exp))
    //     return 1.0f;
    IsInfNum(cmpMask1, expReg, tmpR12Reg, curMask);
    vcmps_eq(cmpMask2, baseReg, -1.0f, cmpMask1);
    vdup(dstReg, 1.0f, cmpMask2, MODE_MERGING);
}

template <typename T>
PTO_INTERNAL void LoadSrcData(RegTensor<float> &srcReg, __ubuf__ T *src, uint16_t offset, MaskReg &mask)
{
    if constexpr (isSupportType<T, half, bfloat16_t>) {
        RegTensor<T> tmpReg;
        vlds(tmpReg, src, offset, UNPK_B16);
        vcvt(srcReg, tmpReg, mask, PART_EVEN);
    } else {
        vlds(srcReg, src, offset, NORM);
    }
}

template <typename T>
PTO_INTERNAL void StoreDstData(RegTensor<float> &dstReg, __ubuf__ T *dst, uint16_t offset, MaskReg &mask)
{
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    if constexpr (isSupportType<T, half, bfloat16_t>) {
        RegTensor<T> tmpReg;
        MaskReg tmpMask;
        vcvt(tmpReg, dstReg, mask, ROUND_R, RS_DISABLE, PART_EVEN);
        vpack((RegTensor<uint16_t> &)tmpReg, (RegTensor<uint32_t> &)tmpReg, LOWER);
        ppack(tmpMask, mask, LOWER);
        vsts(tmpReg, dst, offset, distValue, tmpMask);
    } else {
        vsts(dstReg, dst, offset, distValue, mask);
    }
}

PTO_INTERNAL void GetTPowFloatCore(RegTensor<float> &dstReg, RegTensor<float> &baseReg, RegTensor<float> &expReg,
                                   MaskReg &mask)
{
    RegTensor<float> tmpReg;
    vabs(tmpReg, baseReg, mask);
    vln(tmpReg, tmpReg, mask, MODE_ZEROING);
    vmul(dstReg, expReg, tmpReg, mask, MODE_ZEROING);
    vexp(dstReg, dstReg, mask, MODE_ZEROING);
}

template <typename T, uint32_t DstStride, uint32_t BaseStride, uint32_t ExpStride>
PTO_INTERNAL void TPowFloat(__ubuf__ T *dst, __ubuf__ T *base, __ubuf__ T *exp, unsigned validRow, unsigned validCol)
{
    constexpr uint16_t nElemPerRpt = CCE_VL / sizeof(float);
    uint16_t repeatTime = CeilDivision(validCol, nElemPerRpt);

    __VEC_SCOPE__
    {
        unsigned sReg = validCol;
        MaskReg mask = CreatePredicate<float>(sReg);
        MaskReg tmpMask;
        RegTensor<float> baseReg, expReg, dstReg, twoReg;
        RegTensor<int32_t> tmpR10Reg, tmpR12Reg;
        vdup(tmpR10Reg, R10_COEFF, mask, MODE_ZEROING);
        vdup(tmpR12Reg, R12_COEFF, mask, MODE_ZEROING);
        vdup(twoReg, -2.0f, mask, MODE_ZEROING);

        for (uint16_t i = 0; i < (uint16_t)validRow; i++) {
            sReg = validCol;
            for (uint16_t j = 0; j < repeatTime; j++) {
                mask = CreatePredicate<float>(sReg);
                tmpMask = mask;
                LoadSrcData(baseReg, base, i * BaseStride + j * nElemPerRpt, mask);
                LoadSrcData(expReg, exp, i * ExpStride + j * nElemPerRpt, mask);
                GetTPowFloatCore(dstReg, baseReg, expReg, mask);
                ProcessFloatSpecialCase(dstReg, baseReg, expReg, tmpR10Reg, tmpR12Reg, twoReg, tmpMask);
                StoreDstData(dstReg, dst, i * DstStride + j * nElemPerRpt, mask);
            }
        }
    }
}

template <typename T, uint32_t DstStride, uint32_t BaseStride>
PTO_INTERNAL void TPowFloat(__ubuf__ T *dst, __ubuf__ T *base, T exp, unsigned validRow, unsigned validCol)
{
    constexpr uint16_t nElemPerRpt = CCE_VL / sizeof(float);
    uint16_t repeatTime = CeilDivision(validCol, nElemPerRpt);
    __VEC_SCOPE__
    {
        unsigned sReg = validCol;
        MaskReg mask = CreatePredicate<float>(sReg);
        MaskReg tmpMask;
        RegTensor<float> baseReg, expReg, dstReg, twoReg;
        RegTensor<int32_t> tmpR10Reg, tmpR12Reg;
        vdup(expReg, exp, mask, MODE_ZEROING);
        vdup(tmpR10Reg, R10_COEFF, mask, MODE_ZEROING);
        vdup(tmpR12Reg, R12_COEFF, mask, MODE_ZEROING);
        vdup(twoReg, -2.0f, mask, MODE_ZEROING);
        for (uint16_t i = 0; i < (uint16_t)validRow; i++) {
            sReg = validCol;
            for (uint16_t j = 0; j < repeatTime; j++) {
                mask = CreatePredicate<float>(sReg);
                tmpMask = mask;
                LoadSrcData(baseReg, base, i * BaseStride + j * nElemPerRpt, mask);
                GetTPowFloatCore(dstReg, baseReg, expReg, mask);
                ProcessFloatSpecialCase(dstReg, baseReg, expReg, tmpR10Reg, tmpR12Reg, twoReg, tmpMask);
                StoreDstData(dstReg, dst, i * DstStride + j * nElemPerRpt, mask);
            }
        }
    }
}

struct PowerLogParams {
    RegTensor<float> zeroReg;
    RegTensor<float> oneReg;
    RegTensor<float> fractionReg;
    RegTensor<float> subReg;
    RegTensor<int32_t> intReg;
    RegTensor<float> rReg;
    RegTensor<float> addReg1;
    RegTensor<float> addReg2;
    RegTensor<float> twoReg;
    RegTensor<int32_t> tmpR10Reg;
    RegTensor<int32_t> tmpR12Reg;
};

PTO_INTERNAL void PowerLogParamsInit(PowerLogParams &params, MaskReg &mask)
{
    vdup(params.zeroReg, 0.0f, mask, MODE_ZEROING);
    vdup(params.oneReg, 1.0f, mask, MODE_ZEROING);
    vdup(params.fractionReg, F32_FRACTIONS, mask, MODE_ZEROING);
    vdup(params.subReg, LOG2_REDUCE_COEFF1, mask, MODE_ZEROING);
    vdup(params.intReg, LOG2_REDUCE_COEFF2, mask, MODE_ZEROING);
    vdup(params.rReg, LOG2_BEST_FMAF_COEFF2, mask, MODE_ZEROING);
    vdup(params.addReg1, LOG2_BEST_FMAF_COEFF3, mask, MODE_ZEROING);
    vdup(params.addReg2, LOG2_BEST_FMAF_COEFF4, mask, MODE_ZEROING);
    vdup(params.tmpR10Reg, R10_COEFF, mask, MODE_ZEROING);
    vdup(params.tmpR12Reg, R12_COEFF, mask, MODE_ZEROING);
    vdup(params.twoReg, -2.0f, mask, MODE_ZEROING);
}

PTO_INTERNAL void GetLogFExtStepOne(RegTensor<float> &logHighReg, RegTensor<float> &logLowReg,
                                    RegTensor<float> &tmpResultReg, RegTensor<float> &baseReg, PowerLogParams &params,
                                    MaskReg &mask)
{
    RegTensor<float> tmpAReg, tmpFloatReg, absReg;
    RegTensor<int32_t> tmpEReg;
    MaskReg cmpMask;

    vabs(absReg, baseReg, mask);
    vcmps_lt(cmpMask, absReg, LOG2_LOWEST_VALUE, mask);
    vmuls(tmpAReg, absReg, LOG2_LOWEST_VALUE_MULS, mask);
    vsel(logHighReg, params.fractionReg, params.zeroReg, cmpMask);

    tmpFloatReg = params.subReg;
    vsub(tmpEReg, (RegTensor<int32_t> &)absReg, (RegTensor<int32_t> &)tmpFloatReg, mask);
    vand(tmpEReg, tmpEReg, params.intReg, mask);
    vsub((RegTensor<int32_t> &)logLowReg, (RegTensor<int32_t> &)absReg, tmpEReg, mask);
    vcvt(tmpFloatReg, tmpEReg, mask, ROUND_A);
    vaxpy(logHighReg, tmpFloatReg, LOG2_REDUCE_FMAF_COEFF1, mask);
    RegTensor<float> tmpPReg;
    vadds(tmpPReg, logLowReg, 1.0f, mask);
    vadds(logLowReg, logLowReg, -1.0f, mask);

    vdiv(tmpResultReg, params.oneReg, tmpPReg, mask);
}

PTO_INTERNAL void GetLogFExtStepTwo(RegTensor<float> &logHigh, RegTensor<float> &logLow, RegTensor<float> &tmpRReg,
                                    PowerLogParams &params, MaskReg &mask)
{
    RegTensor<float> tmpQHIReg, tmpQLOReg;
    RegTensor<float> tmpFloatReg, tmpFloatReg2;
    vmul(tmpQHIReg, logLow, tmpRReg, mask, MODE_ZEROING);
    vmuls(tmpFloatReg, tmpQHIReg, -2.0f, mask);
    vadd(tmpFloatReg, tmpFloatReg, logLow, mask);
    vneg(tmpFloatReg2, logLow, mask);
    vmadd(tmpFloatReg2, tmpQHIReg, tmpFloatReg, mask, MODE_ZEROING);
    vmul(tmpQLOReg, tmpRReg, tmpFloatReg2, mask, MODE_ZEROING);
    RegTensor<float> tmpSReg;
    vmul(tmpSReg, tmpQHIReg, tmpQHIReg, mask, MODE_ZEROING);
    tmpRReg = params.rReg;
    vaxpy(tmpRReg, tmpSReg, LOG2_BEST_FMAF_COEFF1, mask, MODE_ZEROING);
    vmadd(tmpRReg, tmpSReg, params.addReg1, mask, MODE_ZEROING);
    vmadd(tmpRReg, tmpSReg, params.addReg2, mask, MODE_ZEROING);
    vmul(tmpRReg, tmpRReg, tmpSReg, mask, MODE_ZEROING);
    vadd(tmpQHIReg, tmpQHIReg, tmpQHIReg, mask);
    vadd(tmpQLOReg, tmpQLOReg, tmpQLOReg, mask);
    RegTensor<float> tmpFHIReg, tmpFLOReg;
    vmuls(tmpFHIReg, logHigh, LOG2_HI1, mask);
    vadd(tmpFHIReg, tmpFHIReg, tmpQHIReg, mask);
    tmpFloatReg2 = tmpFHIReg;
    vaxpy(tmpFloatReg2, logHigh, LOG2_HI2, mask);
    vsub(tmpFLOReg, tmpQHIReg, tmpFloatReg2, mask);
    vmadd(tmpQHIReg, tmpRReg, tmpFLOReg, mask, MODE_ZEROING);
    vmuls(tmpFloatReg, tmpQLOReg, 3.0f, mask);
    vmula(tmpQLOReg, tmpFloatReg, tmpRReg, mask);
    vaxpy(tmpQLOReg, logHigh, LOG2_LO, mask);
    vadd(tmpQLOReg, tmpQLOReg, tmpQHIReg, mask);
    vadd(logHigh, tmpFHIReg, tmpQLOReg, mask);
    vsub(tmpFloatReg, tmpFHIReg, logHigh, mask);
    vadd(logLow, tmpFloatReg, tmpQLOReg, mask);
}

PTO_INTERNAL void GetExpCore(RegTensor<float> &dstReg, RegTensor<float> &logHighReg, RegTensor<float> &logLowReg,
                             RegTensor<float> &expReg, MaskReg &mask)
{
    RegTensor<float> tmPHIReg, tmPLOReg, tmpRReg;
    vmul(tmPHIReg, logHighReg, expReg, mask, MODE_ZEROING);
    RegTensor<float> tmpFloatReg, tmpFloatReg2;
    vneg(tmPLOReg, tmPHIReg, mask);
    vmula(tmPLOReg, logHighReg, expReg, mask);
    vmula(tmPLOReg, logLowReg, expReg, mask);
    vexp(tmpRReg, tmPHIReg, mask, MODE_ZEROING);
    vmula(tmpRReg, tmPLOReg, tmpRReg, mask);
    MaskReg cmpMask1, cmpMask2;
    vcmps_ge(cmpMask1, tmPHIReg, 0.0f, mask);
    vdup((RegTensor<int32_t> &)tmpFloatReg, INF, cmpMask1, MODE_ZEROING);
    vcmps_ge(cmpMask2, tmPHIReg, EXP_MIN_F, mask);
    vcmps_lt(cmpMask1, tmPHIReg, EXP_OVFL_UNFL_F, mask);
    por(cmpMask2, cmpMask1, cmpMask2, mask);
    vsel(dstReg, tmpFloatReg, tmpRReg, cmpMask2);
}

template <typename T, uint32_t DstStride, uint32_t BaseStride, uint32_t ExpStride>
PTO_INTERNAL void TPowFloatHighPrecisionImpl(__ubuf__ T *dst, __ubuf__ T *base, __ubuf__ T *exp, unsigned validRow,
                                             unsigned validCol)
{
    constexpr uint16_t nElemPerRpt = CCE_VL / sizeof(float);
    uint16_t repeatTime = CeilDivision(validCol, nElemPerRpt);

    __VEC_SCOPE__
    {
        unsigned sReg = validCol;
        MaskReg mask = CreatePredicate<float>(sReg);
        PowerLogParams params;
        PowerLogParamsInit(params, mask);

        RegTensor<float> baseReg, expReg, dstReg;
        RegTensor<float> logHighReg, logLowReg, tmpResultReg;
        for (uint16_t i = 0; i < (uint16_t)validRow; i++) {
            sReg = validCol;
            for (uint16_t j = 0; j < repeatTime; j++) {
                mask = CreatePredicate<float>(sReg);

                LoadSrcData(expReg, exp, i * ExpStride + j * nElemPerRpt, mask);
                LoadSrcData(baseReg, base, i * BaseStride + j * nElemPerRpt, mask);

                GetLogFExtStepOne(logHighReg, logLowReg, tmpResultReg, baseReg, params, mask);
                GetLogFExtStepTwo(logHighReg, logLowReg, tmpResultReg, params, mask);
                GetExpCore(dstReg, logHighReg, logLowReg, expReg, mask);
                ProcessFloatSpecialCase(dstReg, baseReg, expReg, params.tmpR10Reg, params.tmpR12Reg, params.twoReg,
                                        mask);

                StoreDstData(dstReg, dst, i * DstStride + j * nElemPerRpt, mask);
            }
        }
    }
}

template <typename T, uint32_t DstStride, uint32_t BaseStride>
PTO_INTERNAL void TPowFloatHighPrecisionImpl(__ubuf__ T *dst, __ubuf__ T *base, T exp, unsigned validRow,
                                             unsigned validCol)
{
    constexpr uint16_t nElemPerRpt = CCE_VL / sizeof(float);
    uint16_t repeatTime = CeilDivision(validCol, nElemPerRpt);

    __VEC_SCOPE__
    {
        unsigned sReg = validCol;
        MaskReg mask = CreatePredicate<float>(sReg);
        PowerLogParams param;
        PowerLogParamsInit(param, mask);

        RegTensor<float> baseReg, expReg, dstReg;
        RegTensor<float> logLowReg, logHighReg, tmpResultReg;
        vdup(expReg, exp, mask, MODE_ZEROING);
        for (uint16_t i = 0; i < (uint16_t)validRow; i++) {
            sReg = validCol;
            for (uint16_t j = 0; j < repeatTime; j++) {
                mask = CreatePredicate<float>(sReg);
                LoadSrcData(baseReg, base, i * BaseStride + j * nElemPerRpt, mask);

                GetLogFExtStepOne(logHighReg, logLowReg, tmpResultReg, baseReg, param, mask);
                GetLogFExtStepTwo(logHighReg, logLowReg, tmpResultReg, param, mask);
                GetExpCore(dstReg, logHighReg, logLowReg, expReg, mask);
                ProcessFloatSpecialCase(dstReg, baseReg, expReg, param.tmpR10Reg, param.tmpR12Reg, param.twoReg, mask);

                StoreDstData(dstReg, dst, i * DstStride + j * nElemPerRpt, mask);
            }
        }
    }
}

} // namespace PowF

namespace PowI {
constexpr int16_t SHIFT_ONE_BIT = 1;
constexpr int16_t BITS_PER_BYTE = 8;

template <typename T, typename ConvType>
PTO_INTERNAL void LoadSrcData(RegTensor<ConvType> &srcReg, __ubuf__ T *src, uint32_t offset, MaskReg &mask)
{
    if constexpr (sizeof(T) == 1) {
        RegTensor<T> tmpReg;
        vlds(tmpReg, src, offset, UNPK_B8);
        vcvt(srcReg, tmpReg, mask, PART_EVEN);
    } else {
        vlds(srcReg, src, offset, NORM);
    }
}

template <typename T, typename ConvType>
PTO_INTERNAL void StoreDstData(RegTensor<ConvType> &dstReg, __ubuf__ T *dst, uint32_t offset, MaskReg &mask)
{
    constexpr auto distValue =
        std::integral_constant<::DistVST, static_cast<::DistVST>(GetDistVst<T, DistVST::DIST_NORM>())>();
    if constexpr (sizeof(T) == 1) {
        RegTensor<T> tmpReg;
        MaskReg tmpMask;
        vpack((RegTensor<uint8_t> &)tmpReg, (RegTensor<uint16_t> &)dstReg, LOWER);
        ppack(tmpMask, mask, LOWER);
        vsts(tmpReg, dst, offset, distValue, tmpMask);
    } else {
        vsts(dstReg, dst, offset, distValue, mask);
    }
}

template <typename T>
PTO_INTERNAL void PowIntegerCore(T &dstReg, T &baseReg, T &expReg, MaskReg &mask)
{
    T selReg;
    MaskReg selMask;
    vdup(selReg, 1, mask, MODE_ZEROING);
    vand(selReg, expReg, selReg, mask);
    vcmps_eq(selMask, selReg, 1, mask);

    T tmpReg;
    vmul(tmpReg, dstReg, baseReg, mask, MODE_ZEROING);
    vsel(dstReg, tmpReg, dstReg, selMask);

    vshrs(expReg, expReg, SHIFT_ONE_BIT, mask);
    vmul(baseReg, baseReg, baseReg, mask, MODE_ZEROING);
}

template <typename ConvType, typename T>
PTO_INTERNAL void ProcessSpecialCaseForPowI(T &dstReg, T &baseReg, T &expReg, MaskReg &mask)
{
    T tmpReg;
    vdup(tmpReg, 1, mask, MODE_ZEROING);

    // if (exp == 0) || (base == 1)
    //     return 1;
    MaskReg cmpMask1, cmpMask2, condMask;
    vcmps_eq(cmpMask1, expReg, 0, mask);
    vcmps_eq(cmpMask2, baseReg, 1, mask);
    por(condMask, cmpMask1, cmpMask2, mask);
    vsel(dstReg, tmpReg, dstReg, condMask);
    pxor(mask, mask, condMask, mask);

    // if (base != -1 && exp < 0)
    //    return 0;
    if constexpr (std::is_signed_v<ConvType>) {
        vdup(tmpReg, 0, mask, MODE_ZEROING);
        vcmps_ne(cmpMask1, baseReg, -1, mask);
        vcmps_lt(cmpMask2, expReg, 0, mask);
        pand(condMask, cmpMask1, cmpMask2, mask);
        vsel(dstReg, tmpReg, dstReg, condMask);
    }
}

template <typename ConvType, typename T>
PTO_INTERNAL void TPowIntegerCompute(T &dstReg, T &baseReg, T &expReg, MaskReg &mask)
{
    // TODO: vcmax(dst, exp); maxLoop = __buildin_clz(dst[0])
    constexpr uint16_t maxLoop = sizeof(ConvType) * BITS_PER_BYTE;
    T tmpBaseReg = baseReg;
    T tmpExpReg = expReg;
    MaskReg tmpMask = mask;
    for (uint16_t j = 0; j < maxLoop; j++) {
        PowIntegerCore(dstReg, tmpBaseReg, tmpExpReg, mask);
    }
    ProcessSpecialCaseForPowI<ConvType>(dstReg, baseReg, expReg, tmpMask);
}

template <typename T, uint32_t DstStride, uint32_t BaseStride, uint32_t ExpStride>
PTO_INTERNAL void TPowInteger(__ubuf__ T *dst, __ubuf__ T *base, __ubuf__ T *exp, unsigned validRow, unsigned validCol)
{
    using ConvType = std::conditional_t<std::is_same_v<T, int8_t>, int16_t,
                                        std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, T>>;
    constexpr uint16_t nElemPerRpt = CCE_VL / sizeof(ConvType);
    uint16_t repeatTime = CeilDivision(validCol, nElemPerRpt);

    __VEC_SCOPE__
    {
        unsigned sReg = validCol;
        RegTensor<ConvType> baseReg, expReg;
        RegTensor<ConvType> initRetReg, dstReg;

        MaskReg mask = CreatePredicate<ConvType>(sReg);
        vdup(initRetReg, 1, mask, MODE_ZEROING);
        for (uint16_t i = 0; i < (uint16_t)validRow; i++) {
            sReg = validCol;
            for (uint16_t j = 0; j < repeatTime; j++) {
                mask = CreatePredicate<ConvType>(sReg);
                dstReg = initRetReg;

                LoadSrcData(baseReg, base, i * BaseStride + j * nElemPerRpt, mask);
                LoadSrcData(expReg, exp, i * ExpStride + j * nElemPerRpt, mask);
                TPowIntegerCompute<ConvType>(dstReg, baseReg, expReg, mask);
                StoreDstData(dstReg, dst, i * DstStride + j * nElemPerRpt, mask);
            }
        }
    }
}

template <typename T, uint32_t DstStride, uint32_t BaseStride>
PTO_INTERNAL void TPowInteger(__ubuf__ T *dst, __ubuf__ T *base, T exp, unsigned validRow, unsigned validCol)
{
    using ConvType = std::conditional_t<std::is_same_v<T, int8_t>, int16_t,
                                        std::conditional_t<std::is_same_v<T, uint8_t>, uint16_t, T>>;
    constexpr uint16_t nElemPerRpt = CCE_VL / sizeof(ConvType);
    uint16_t repeatTime = CeilDivision(validCol, nElemPerRpt);

    __VEC_SCOPE__
    {
        unsigned sReg = validCol;
        RegTensor<ConvType> initRetReg, dstReg;
        RegTensor<ConvType> baseReg, expReg;

        MaskReg mask = CreatePredicate<ConvType>(sReg);
        vdup(initRetReg, 1, mask, MODE_ZEROING);
        vdup(expReg, exp, mask, MODE_ZEROING);
        for (uint16_t i = 0; i < (uint16_t)validRow; i++) {
            sReg = validCol;
            for (uint16_t j = 0; j < repeatTime; j++) {
                mask = CreatePredicate<ConvType>(sReg);
                dstReg = initRetReg;

                LoadSrcData(baseReg, base, i * BaseStride + j * nElemPerRpt, mask);
                TPowIntegerCompute<ConvType>(dstReg, baseReg, expReg, mask);
                StoreDstData(dstReg, dst, i * DstStride + j * nElemPerRpt, mask);
            }
        }
    }
}

} // namespace PowI

template <typename T>
inline constexpr bool IsFloatNum = isSupportType<T, float, half, bfloat16_t>;
template <typename T>
inline constexpr bool IsIntegerNum = isSupportType<T, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t>;

template <PowAlgorithm algo, typename DstTile, typename BaseTile, typename ExpTile>
__tf__ PTO_INTERNAL void TPowImpl(typename DstTile::TileDType __out__ dstData,
                                  typename BaseTile::TileDType __in__ baseData,
                                  typename ExpTile::TileDType __in__ expData, unsigned validRow, unsigned validCol,
                                  unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename DstTile::DType;
    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *base = (__ubuf__ T *)__cce_get_tile_ptr(baseData);
    __ubuf__ T *exp = (__ubuf__ T *)__cce_get_tile_ptr(expData);

    if constexpr (IsFloatNum<T>) {
        if constexpr (algo == PowAlgorithm::DEFAULT) {
            PowF::TPowFloat<T, DstTile::RowStride, BaseTile::RowStride, ExpTile::RowStride>(dst, base, exp, validRow,
                                                                                            validCol);
        } else if (algo == PowAlgorithm::HIGH_PRECISION) {
            PowF::TPowFloatHighPrecisionImpl<T, DstTile::RowStride, BaseTile::RowStride, ExpTile::RowStride>(
                dst, base, exp, validRow, validCol);
        }
    } else if constexpr (IsIntegerNum<T>) {
        PowI::TPowInteger<T, DstTile::RowStride, BaseTile::RowStride, ExpTile::RowStride>(dst, base, exp, validRow,
                                                                                          validCol);
    }
}

template <PowAlgorithm algo, typename DstTile, typename BaseTile, typename ExpTile>
PTO_INTERNAL void PowCheckType()
{
    static_assert(DstTile::isRowMajor && BaseTile::isRowMajor && ExpTile::isRowMajor,
                  "TPOW: Not supported Layout type");
    static_assert(DstTile::Loc == TileType::Vec && BaseTile::Loc == TileType::Vec && ExpTile::Loc == TileType::Vec,
                  "TPOW: TileType of dst, base and exp tiles must be TileType::Vec.");
    static_assert(DstTile::ValidCol <= DstTile::Cols,
                  "TPOW: Number of dst's valid columns must not be greater than number of tile columns.");
    static_assert(DstTile::ValidRow <= DstTile::Rows,
                  "TPOW: Number of dst's valid rows must not be greater than number of tile rows.");
    static_assert(BaseTile::ValidCol <= BaseTile::Cols,
                  "TPOW: Number of base's valid columns must not be greater than number of tile columns.");
    static_assert(BaseTile::ValidRow <= BaseTile::Rows,
                  "TPOW: Number of base's valid rows must not be greater than number of tile rows.");
    static_assert(ExpTile::ValidCol <= ExpTile::Cols,
                  "TPOW: Number of exp's valid columns must not be greater than number of tile columns.");
    static_assert(ExpTile::ValidRow <= ExpTile::Rows,
                  "TPOW: Number of exp's valid rows must not be greater than number of tile rows.");

    using T = typename DstTile::DType;

    if constexpr (algo == PowAlgorithm::HIGH_PRECISION) {
        static_assert(isSupportType<T, float, half, bfloat16_t>,
                      "Type must be half/float/bfloat16 in high precision algorithm.");
    } else {
        static_assert(isSupportType<T, float, half, int32_t, uint32_t, int16_t, uint16_t, int8_t, uint8_t>,
                      "Type must be uint8/int8/uint16/int16/uint32/int32/half/float in default algorithm.");
    }
    static_assert(std::is_same_v<T, typename BaseTile::DType> && std::is_same_v<T, typename ExpTile::DType>,
                  "TPOW: The data type of dst, base and exp must be consistent");
}

template <PowAlgorithm algo, typename DstTile, typename BaseTile, typename ExpTile, typename TmpTile>
PTO_INTERNAL void TPOW_IMPL(DstTile &dst, BaseTile &base, ExpTile &exp, TmpTile &tmp)
{
    PowCheckType<algo, DstTile, BaseTile, ExpTile>();
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    PTO_ASSERT(validCol == base.GetValidCol(), "TPOW: Number of columns of base and dst must be same.");
    PTO_ASSERT(validRow == base.GetValidRow(), "TPOW: Number of rows of base and dst must be same.");
    PTO_ASSERT(validCol == exp.GetValidCol(), "TPOW: Number of columns of exp and dst must be same.");
    PTO_ASSERT(validRow == exp.GetValidRow(), "TPOW: Number of rows of exp and dst must be same.");

    TPowImpl<algo, DstTile, BaseTile, ExpTile>(dst.data(), base.data(), exp.data(), validRow, validCol);
}

template <PowAlgorithm algo, typename DstTile, typename BaseTile>
__tf__ PTO_INTERNAL void TPowSImpl(typename DstTile::TileDType __out__ dstData,
                                   typename BaseTile::TileDType __in__ baseData, typename DstTile::DType exp,
                                   unsigned validRow, unsigned validCol, unsigned version = VFImplKind::VFIMPL_DEFAULT)
{
    using T = typename DstTile::DType;
    __ubuf__ T *dst = (__ubuf__ T *)__cce_get_tile_ptr(dstData);
    __ubuf__ T *base = (__ubuf__ T *)__cce_get_tile_ptr(baseData);

    if constexpr (IsFloatNum<T>) {
        if constexpr (algo == PowAlgorithm::DEFAULT) {
            PowF::TPowFloat<T, DstTile::RowStride, BaseTile::RowStride>(dst, base, exp, validRow, validCol);
        } else if (algo == PowAlgorithm::HIGH_PRECISION) {
            PowF::TPowFloatHighPrecisionImpl<T, DstTile::RowStride, BaseTile::RowStride>(dst, base, exp, validRow,
                                                                                         validCol);
        }
    } else if constexpr (IsIntegerNum<T>) {
        PowI::TPowInteger<T, DstTile::RowStride, BaseTile::RowStride>(dst, base, exp, validRow, validCol);
    }
}

template <PowAlgorithm algo, typename DstTile, typename BaseTile, typename TmpTile>
PTO_INTERNAL void TPOWS_IMPL(DstTile &dst, BaseTile &base, typename DstTile::DType exp, TmpTile &tmp)
{
    PowCheckType<algo, DstTile, BaseTile, DstTile>();
    unsigned validRow = dst.GetValidRow();
    unsigned validCol = dst.GetValidCol();
    PTO_ASSERT(validCol == base.GetValidCol(), "TPOW: Number of columns of base and dst must be same.");
    PTO_ASSERT(validRow == base.GetValidRow(), "TPOW: Number of rows of base and dst must be same.");

    TPowSImpl<algo, DstTile, BaseTile>(dst.data(), base.data(), exp, validRow, validCol);
}
} // namespace pto
#endif