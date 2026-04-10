/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSQRTHP_HPP
#define TSQRTHP_HPP

namespace pto {
template <typename T, typename U>
PTO_INTERNAL void SqrtPrecisionImpl(U &dstReg, U &srcReg, MaskReg &mask)
{
    HalfUnion multiplyFactor0;
    multiplyFactor0.i = 0x6C00;
    HalfUnion multiplyFactor1;
    multiplyFactor1.i = 0x2400;
    HalfUnion subnormalThreshold;
    subnormalThreshold.i = 0x03FF;
    RegTensor<T> tmpReg;
    RegTensor<T> dstRegCopy;
    RegTensor<T> srcRegCopy = srcReg;
    MaskReg cmpMaskReg;

    vcmps_lt(cmpMaskReg, srcRegCopy, subnormalThreshold.f, mask);
    vmuls(tmpReg, srcRegCopy, multiplyFactor0.f, mask, MODE_ZEROING);
    vsel(srcRegCopy, tmpReg, srcRegCopy, cmpMaskReg);
    vsqrt(dstRegCopy, srcRegCopy, mask, MODE_ZEROING);
    vmuls(tmpReg, dstRegCopy, multiplyFactor1.f, mask, MODE_ZEROING);
    vsel(dstReg, tmpReg, dstRegCopy, cmpMaskReg);
}

template <typename T, typename U>
PTO_INTERNAL void SqrtFloatImpl(RegTensor<float> &dst, RegTensor<float> &src, MaskReg &mask)
{
    constexpr float subnormalBound = 1;
    constexpr float halfFactor = 0.5f;
    constexpr float negOne = -1.0f;
    constexpr float multiplyFactor0 = 16777216.0f;
    constexpr float multiplyFactor1 = 0.000244140625f;
    constexpr uint32_t posInf = 0x7f800000u;
    constexpr uint32_t negZero = 0x80000000u;
    RegTensor<T> regOne;
    RegTensor<T> tmpReg;
    RegTensor<T> errReg;
    RegTensor<T> resReg;
    RegTensor<T> dstRegCopy;
    RegTensor<T> srcRegCopy = src;
    RegTensor<uint32_t> regNegOne;
    RegTensor<uint32_t> zeroReg;

    MaskReg cmpMaskReg;
    MaskReg isInfPreg;
    MaskReg isZeroPreg;
    MaskReg maskFull;
    maskFull = pset_b8(PAT_ALL);

    vcmps_lt(cmpMaskReg, srcRegCopy, subnormalBound, mask);
    vmuls(tmpReg, srcRegCopy, multiplyFactor0, mask, MODE_ZEROING);
    vsel(srcRegCopy, tmpReg, srcRegCopy, cmpMaskReg);

    vdup(regOne, 1.0f, maskFull, MODE_ZEROING);
    vsqrt(tmpReg, srcRegCopy, mask, MODE_ZEROING);
    vdiv(dstRegCopy, regOne, tmpReg, mask, MODE_ZEROING);

    vmuls(tmpReg, dstRegCopy, negOne, mask, MODE_ZEROING);
    vmul(errReg, dstRegCopy, srcRegCopy, mask, MODE_ZEROING);
    vmula(regOne, errReg, tmpReg, mask, MODE_ZEROING);
    vmuls(tmpReg, dstRegCopy, halfFactor, mask, MODE_ZEROING);
    vmula(dstRegCopy, regOne, tmpReg, mask, MODE_ZEROING);

    vmul(resReg, dstRegCopy, srcRegCopy, mask, MODE_ZEROING);
    vmuls(tmpReg, resReg, negOne, mask, MODE_ZEROING);
    vmov(errReg, srcRegCopy);
    vmula(errReg, resReg, tmpReg, mask, MODE_ZEROING);
    vmuls(tmpReg, dstRegCopy, halfFactor, mask, MODE_ZEROING);
    vmadd(tmpReg, errReg, resReg, mask, MODE_ZEROING);

    vmuls(dstRegCopy, tmpReg, multiplyFactor1, mask, MODE_ZEROING);
    vsel(tmpReg, dstRegCopy, tmpReg, cmpMaskReg);

    vcmps_eq(isInfPreg, (vector_u32 &)srcRegCopy, posInf, mask);
    vdup(regNegOne, negZero, maskFull, MODE_ZEROING);
    vor(zeroReg, (vector_u32 &)srcRegCopy, regNegOne, mask, MODE_ZEROING);
    vcmps_eq(isZeroPreg, zeroReg, negZero, mask);
    por(cmpMaskReg, isZeroPreg, isInfPreg, mask);
    vsel(dst, srcRegCopy, tmpReg, cmpMaskReg);
}
} // namespace pto
#endif // TINSERT_CUSTOM_HPP
